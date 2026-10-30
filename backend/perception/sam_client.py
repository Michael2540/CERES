"""SAM client (M2): Meta sam-3.1 over the Responses API (`api`) and a deterministic HSV fallback (`hsv`).

The request/response format is the one verified live in SPIKES.md S1. Masks come back cropped to their box,
so they are pasted into a full-frame raster at rec.bounds.left/top.

Integrity rules for the `api` backend:
- the stream must complete with no error diagnostics, else `SamStreamError` (retryable, never cached,
  and the frame gets no masks row);
- spend is recorded exactly once per completed paid call, before decoding, so a later decode failure is
  still ledgered;
- the cache stores the verified raw output text keyed by (raw frame sha256, fault_hash, prompt, backend), and masks
  are decoded from it. A retry after a decode failure therefore re-decodes from disk and never re-bills.
  A cache hit makes no backend call and records no spend;
- budget is reserved under an asyncio lock at check time, so concurrent calls cannot overshoot the cap.

The `hsv` backend is clearly NOT SAM (model label "hsv-fallback"): offline demo and tests only. SAM masks
are the primary source for the metrics.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import certifi
import cv2
import httpx
import numpy as np

from backend.config import ROOT, load_settings
from backend.perception.metrics import rle_encode
from backend.spend import check_budget, cost_for, record_spend

CACHE_DIR = ROOT / ".sam_cache"
SAM_MODEL = "sam-3.1"
BASE_URL = "https://api.meta.ai/v1"
HSV_MODEL = "hsv-fallback"
log = logging.getLogger(__name__)


@dataclass
class Mask:
    object_id: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame px
    score: float
    rle: dict  # full-frame mask, see metrics.rle_encode

    def raster(self) -> np.ndarray:
        from backend.perception.metrics import rle_decode

        return rle_decode(self.rle)


class SamStreamError(RuntimeError):
    """The sam-3.1 stream was truncated, failed or unparseable. Retryable; never cached."""


class SamBackend(Protocol):
    """acquire() does the (possibly paid) call and returns a payload that is safe to cache; adecode()
    turns a payload back into masks with no network and no spend."""

    name: str
    model: str

    async def acquire(self, image_path: str | Path, prompt: str) -> str: ...

    async def adecode(self, payload: str, image_path: str | Path) -> list[Mask]: ...

    def segment(self, image_path: str | Path, prompt: str) -> list[Mask]: ...

    async def asegment(self, image_path: str | Path, prompt: str) -> list[Mask]: ...


class _Convenience:
    async def asegment(self, image_path: str | Path, prompt: str = "plant") -> list[Mask]:
        return await self.adecode(await self.acquire(image_path, prompt), image_path)  # type: ignore[attr-defined]

    def segment(self, image_path: str | Path, prompt: str = "plant") -> list[Mask]:
        return asyncio.run(self.asegment(image_path, prompt))


class HsvBackend(_Convenience):
    """Deterministic HSV threshold + connected components. Not a model; free.

    The mask covers green and yellow hues (OpenCV hue 15-90, S/V floors) so chlorosis_frac is measurable
    on it. SAM masks (api backend) are the primary source; this is the offline fallback.
    """

    name = "hsv"
    model = HSV_MODEL

    def __init__(self, min_area: int = 150, join_px: int = 15) -> None:
        self.min_area, self.join_px = min_area, join_px

    def _segment(self, image_path: str | Path) -> list[Mask]:
        from backend.sim.faults import plant_mask

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        green = plant_mask(img)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.join_px, self.join_px))
        joined = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        out: list[Mask] = []
        for i in range(1, n):
            m = (labels == i) & green
            if m.sum() < self.min_area:
                continue
            x, y, w, h = (int(v) for v in stats[i, :4])
            out.append(Mask(str(len(out)), (x, y, x + w, y + h), 1.0, rle_encode(m)))
        return out

    async def acquire(self, image_path: str | Path, prompt: str = "plant") -> str:
        masks = await asyncio.to_thread(self._segment, image_path)
        return json.dumps([asdict(m) for m in masks])

    async def adecode(self, payload: str, image_path: str | Path) -> list[Mask]:
        return [Mask(m["object_id"], tuple(m["box"]), m["score"], m["rle"]) for m in json.loads(payload)]


_HEADER = re.compile(r"<[0-9a-zA-Z]{1,4}>")
_WH = re.compile(r";w=(\d+);h=(\d+)")


async def _events(stream: Any) -> AsyncIterator[Any]:
    async for ev in stream:
        yield ev


def _fake_stream(text: str) -> AsyncIterator[Any]:
    base = {"item_id": "cached", "output_index": 0, "content_index": 0}

    async def gen() -> AsyncIterator[Any]:
        yield SimpleNamespace(type="response.output_text.delta", delta=text, **base)
        yield SimpleNamespace(type="response.output_text.done", text=text, **base)
        yield SimpleNamespace(type="response.completed")

    return gen()


def _check_result(result: Any) -> None:
    if getattr(result.outcome, "status", None) != "completed":
        raise SamStreamError(f"sam stream did not complete: {result.outcome!r}")
    errs = [d for d in result.diagnostics if d.severity == "error"]
    if errs:
        raise SamStreamError(f"sam stream had error diagnostics: {[d.code for d in errs]}")


def _paste(crop: np.ndarray, bounds: Any, full: np.ndarray) -> None:
    bh, bw = bounds.bottom - bounds.top, bounds.right - bounds.left
    if crop.shape != (bh, bw):  # mask dims differ from its box: nearest-neighbour to the box
        crop = cv2.resize(crop.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST).astype(bool)
    height, width = full.shape
    y0, x0 = max(bounds.top, 0), max(bounds.left, 0)
    y1, x1 = min(bounds.bottom, height), min(bounds.right, width)
    if y1 > y0 and x1 > x0:
        full[y0:y1, x0:x1] = crop[y0 - bounds.top : y1 - bounds.top, x0 - bounds.left : x1 - bounds.left]


def masks_from_records(
    records: Any, shape: tuple[int, int], source_wh: tuple[int, int] | None = None
) -> list[Mask]:
    """Paste box-cropped SAM mask records into full-frame masks of `shape` (h, w).

    Boxes and masks live in the parser's w/h space (source_wh); when that differs from the image they are
    scaled to the image with nearest-neighbour. A mask whose dims differ from its box is resized to the box.
    """
    from meta_sam_parser import SegmentationMaskRecord, decode_mask_to_raster

    h, w = shape
    sw, sh = source_wh or (w, h)
    out: list[Mask] = []
    for rec in records:
        if not isinstance(rec, SegmentationMaskRecord):
            continue
        mh, mw = int(rec.mask.height), int(rec.mask.width)
        crop = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(mh, mw).astype(bool)
        full = np.zeros((sh, sw), dtype=bool)
        _paste(crop, rec.bounds, full)
        b = rec.bounds
        box = (b.left, b.top, b.right, b.bottom)
        if (sw, sh) != (w, h):
            full = cv2.resize(full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            fx, fy = w / sw, h / sh
            box = (round(box[0] * fx), round(box[1] * fy), round(box[2] * fx), round(box[3] * fy))
        out.append(Mask(str(rec.object_id), box, float(rec.confidence or 0.0), rle_encode(full)))
    return out


class ApiBackend(_Convenience):
    """Meta sam-3.1 via AsyncOpenAI Responses streaming. Budget-reserved and ledgered per paid call."""

    name = "api"
    model = SAM_MODEL

    def __init__(self, conn: sqlite3.Connection, client: Any | None = None) -> None:
        self.conn = conn
        self._client = client
        self._lock = asyncio.Lock()
        self._reserved = 0.0

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            http = httpx.AsyncClient(verify=certifi.where(), timeout=120)
            self._client = AsyncOpenAI(
                base_url=BASE_URL, api_key=os.environ["META_API_KEY"], http_client=http
            )
        return self._client

    async def acquire(self, image_path: str | Path, prompt: str = "plant") -> str:
        """Call sam-3.1 and return the verified raw output text. Records spend once, on completion."""
        from meta_sam_parser import ResponsesStreamError, image_segmentation_format, parse_responses_stream

        load_settings()  # loads .env into the environment
        cost = cost_for(SAM_MODEL, images=1)
        async with self._lock:  # check and reserve atomically so concurrent calls cannot overshoot
            check_budget(self.conn, self._reserved + cost)
            self._reserved += cost
        try:
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            stream = await self._get_client().responses.create(
                model=SAM_MODEL,
                stream=True,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                        ],
                    }
                ],
                metadata={"include_confidence": "true"},
            )
            try:
                parsed = parse_responses_stream(_events(stream), image_segmentation_format())
                result = await parsed.final_result()
            except ResponsesStreamError as e:
                raise SamStreamError(f"sam stream failed: {e}") from e
            if getattr(result.outcome, "status", None) == "completed":
                record_spend(self.conn, "perception", "meta", SAM_MODEL, images=1)  # billed once
            _check_result(result)
            raw = result.raw_output
            has_mask = any(r.__class__.__name__ == "SegmentationMaskRecord" for r in result.records)
            if not has_mask:
                if _HEADER.sub("", raw).strip():
                    raise SamStreamError(f"sam output has no masks and is not empty: {raw[:80]!r}")
                log.warning("sam-3.1 returned zero matches for %s (prompt %r)", image_path, prompt)
            return raw
        finally:
            self._reserved -= cost

    async def adecode(self, payload: str, image_path: str | Path) -> list[Mask]:
        from meta_sam_parser import ResponsesStreamError, image_segmentation_format, parse_responses_stream

        img = cv2.imread(str(image_path))
        try:
            parsed = parse_responses_stream(_fake_stream(payload), image_segmentation_format())
            result = await parsed.final_result()
        except ResponsesStreamError as e:
            raise SamStreamError(f"cannot re-parse sam output: {e}") from e
        _check_result(result)
        m = _WH.search(payload)
        source = (int(m[1]), int(m[2])) if m else None
        return masks_from_records(result.records, img.shape[:2], source)


def get_backend(name: str | None, conn: sqlite3.Connection) -> SamBackend:
    name = name or load_settings().sam_backend
    if name == "api":
        return ApiBackend(conn)
    if name == "hsv":
        return HsvBackend()
    raise ValueError(f"unknown SAM_BACKEND {name!r} (api|hsv)")


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_cache_key(image_path: str | Path, prompt: str, backend_name: str) -> str:
    """The pre-fault_hash key: sha256 of the (possibly re-encoded derived) image bytes. Kept for migration."""
    return hashlib.sha256(f"{_sha(image_path)}|{prompt}|{backend_name}".encode()).hexdigest()


def cache_key(
    image_path: str | Path,
    prompt: str,
    backend_name: str,
    raw_path: str | Path | None = None,
    fault_hash: str | None = None,
) -> str:
    """Key on (raw frame sha, fault_hash, prompt, backend). A derived frame (tray B/C fault) passes its raw
    ALED frame and the frame's fault_hash, so re-encoding the derived JPEG never changes the key while a
    changed fault (new FAULTS_VERSION or parameters) does. Without a fault_hash the key is the image sha
    alone, identical to the legacy key, so dataset frames need no migration."""
    h = _sha(raw_path if raw_path is not None else image_path)
    fh = f"|{fault_hash}" if fault_hash else ""
    return hashlib.sha256(f"{h}{fh}|{prompt}|{backend_name}".encode()).hexdigest()


def cache_path(
    image_path: str | Path,
    prompt: str,
    backend_name: str,
    cache_dir: Path = CACHE_DIR,
    raw_path: str | Path | None = None,
    fault_hash: str | None = None,
) -> Path:
    return cache_dir / f"{cache_key(image_path, prompt, backend_name, raw_path, fault_hash)}.json"


async def segment_cached(
    backend: SamBackend,
    image_path: str | Path,
    prompt: str = "plant",
    cache_dir: Path = CACHE_DIR,
    raw_path: str | Path | None = None,
    fault_hash: str | None = None,
) -> tuple[list[Mask], bool]:
    """(masks, from_cache). Only a verified payload is cached, before decoding, so a decode failure after a
    paid call is retried from disk without a second call. A hit makes no backend call and records no spend."""
    p = cache_path(image_path, prompt, backend.name, cache_dir, raw_path, fault_hash)
    cached = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    hit = isinstance(cached, dict) and "payload" in cached  # older cache formats count as a miss
    if hit:
        payload = cached["payload"]
    else:
        payload = await backend.acquire(image_path, prompt)  # SamStreamError: nothing is cached
        cache_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"payload": payload}), encoding="utf-8")
    return await backend.adecode(payload, image_path), hit
