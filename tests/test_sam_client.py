import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster

from backend.perception.sam_client import ApiBackend, HsvBackend, Mask, SamStreamError, segment_cached
from backend.spend import BudgetExceeded, total_spend


def canned_output() -> str:
    """Two instances in the SPIKES.md S1 format; masks are cropped to their boxes."""
    parts = ["<0f>"]
    # x2/y2 are inclusive and data=<height>,<width>,<payload> (S1: box 24x32 px carried data=32,24)
    for oid, (x1, y1, x2, y2) in enumerate([(10, 20, 17, 25), (40, 30, 45, 39)]):
        w, h = x2 - x1 + 1, y2 - y1 + 1
        payload = _encode_raster(bytes([1]) * (w * h), w, h)
        parts.append(
            f"{oid}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=64;h=48;c=0.86|>"
            f"<|mask;x=0;y=0;c=0.86;data={h},{w},{payload}|>"
        )
    return "".join(parts)


def fake_stream(text: str):
    base = {"item_id": "i1", "output_index": 0, "content_index": 0}

    async def gen():
        for k in range(0, len(text), 40):
            yield SimpleNamespace(type="response.output_text.delta", delta=text[k : k + 40], **base)
        yield SimpleNamespace(type="response.output_text.done", text=text, **base)
        yield SimpleNamespace(type="response.completed")

    return gen()


class FakeClient:
    def __init__(self, text: str):
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)
        self._text = text

    async def _create(self, **kw):
        self.calls.append(kw)
        return fake_stream(self._text)


@pytest.fixture
def jpg(tmp_path):
    p = tmp_path / "f.jpg"
    cv2.imwrite(str(p), np.full((48, 64, 3), 80, np.uint8))
    return p


def test_api_backend_parses_and_pastes(conn, jpg):
    client = FakeClient(canned_output())
    masks = ApiBackend(conn, client).segment(jpg, "plant")
    assert len(masks) == 2
    assert masks[0].box == (10, 20, 18, 26) and masks[0].score == pytest.approx(0.86)
    r = masks[0].raster()
    assert r.shape == (48, 64) and r.sum() == 8 * 6 and r[20:26, 10:18].all() and not r[:20].any()
    kw = client.calls[0]
    assert (
        kw["model"] == "sam-3.1" and kw["stream"] is True and kw["metadata"] == {"include_confidence": "true"}
    )
    assert kw["input"][0]["content"][0]["text"] == "plant"
    assert total_spend(conn) == pytest.approx(0.0025)


def test_cache_hit_avoids_call_and_spend(conn, jpg, tmp_path):
    client = FakeClient(canned_output())
    be, cache = ApiBackend(conn, client), tmp_path / "cache"
    m1, hit1 = asyncio.run(segment_cached(be, jpg, "plant", cache))
    m2, hit2 = asyncio.run(segment_cached(be, jpg, "plant", cache))
    assert (hit1, hit2) == (False, True) and len(client.calls) == 1
    assert [m.rle for m in m1] == [m.rle for m in m2] and isinstance(m2[0], Mask)
    assert total_spend(conn) == pytest.approx(0.0025)


def test_budget_guard_blocks_before_call(conn, jpg, monkeypatch):
    monkeypatch.setenv("CERES_BUDGET_USD", "0.001")
    client = FakeClient(canned_output())
    with pytest.raises(BudgetExceeded):
        ApiBackend(conn, client).segment(jpg, "plant")
    assert client.calls == []


def test_hsv_backend(tmp_path):
    img = np.full((100, 100, 3), (10, 15, 20), np.uint8)
    img[10:40, 10:40] = (30, 170, 60)
    img[60:90, 55:95] = (30, 170, 60)
    p = tmp_path / "g.jpg"
    cv2.imwrite(str(p), img)
    masks = HsvBackend().segment(p, "plant")
    assert len(masks) == 2 and HsvBackend.model == "hsv-fallback"


def _events(text: str | None, *, complete: bool = True, done: bool = True):
    base = {"item_id": "i1", "output_index": 0, "content_index": 0}

    async def gen():
        if text:
            yield SimpleNamespace(type="response.output_text.delta", delta=text, **base)
        if done and text is not None:
            yield SimpleNamespace(type="response.output_text.done", text=text, **base)
        if complete:
            yield SimpleNamespace(type="response.completed")

    return gen()


class EventsClient(FakeClient):
    def __init__(self, make, delay: float = 0.0):
        super().__init__("")
        self._make, self._delay = make, delay

    async def _create(self, **kw):
        self.calls.append(kw)
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._make()


def _acquire_fails(conn, jpg, tmp_path, make, billed: float):
    client = EventsClient(make)
    with pytest.raises(SamStreamError):
        asyncio.run(segment_cached(ApiBackend(conn, client), jpg, "plant", tmp_path / "c"))
    assert not list((tmp_path / "c").glob("*.json")) if (tmp_path / "c").exists() else True
    assert total_spend(conn) == pytest.approx(billed)


def test_truncated_stream_not_cached_not_billed(conn, jpg, tmp_path):
    _acquire_fails(
        conn, jpg, tmp_path, lambda: _events(canned_output()[:70], complete=False, done=False), 0.0
    )


def test_no_events_not_cached(conn, jpg, tmp_path):
    _acquire_fails(conn, jpg, tmp_path, lambda: _events(None, complete=False), 0.0)


def test_garbage_text_not_cached(conn, jpg, tmp_path):
    _acquire_fails(
        conn, jpg, tmp_path, lambda: _events("sorry, I cannot do that"), 0.0025
    )  # completed = billed


def test_completed_empty_is_cached_and_logged(conn, jpg, tmp_path, caplog):
    client = EventsClient(lambda: _events(""))
    masks, hit = asyncio.run(segment_cached(ApiBackend(conn, client), jpg, "plant", tmp_path / "c"))
    assert masks == [] and not hit and len(list((tmp_path / "c").glob("*.json"))) == 1
    assert "zero matches" in caplog.text


def test_decode_failure_after_completion_billed_once(conn, jpg, tmp_path):
    client = EventsClient(lambda: _events(canned_output()))
    be, cache = ApiBackend(conn, client), tmp_path / "c"
    real = be.adecode

    async def boom(payload, image_path):
        raise ValueError("decode exploded")

    be.adecode = boom
    with pytest.raises(ValueError):
        asyncio.run(segment_cached(be, jpg, "plant", cache))
    assert total_spend(conn) == pytest.approx(0.0025) and len(client.calls) == 1
    be.adecode = real
    masks, hit = asyncio.run(segment_cached(be, jpg, "plant", cache))
    assert hit and len(masks) == 2 and len(client.calls) == 1
    assert total_spend(conn) == pytest.approx(0.0025)  # the retry did not bill


def test_mask_dims_differ_from_box_and_source_size(conn, tmp_path):
    # box 8x6 px but the mask is 4x3: resized to the box. Source space is 64x48, the image is 128x96.
    payload = _encode_raster(bytes([1]) * 12, 4, 3)
    text = f"<0f>0<|box;x1=10;y1=20;x2=17;y2=25;w=64;h=48;c=0.9|><|mask;x=0;y=0;c=0.9;data=3,4,{payload}|>"
    p = tmp_path / "big.jpg"
    cv2.imwrite(str(p), np.full((96, 128, 3), 80, np.uint8))
    (m,) = asyncio.run(ApiBackend(conn, FakeClient(text)).adecode(text, p))
    r = m.raster()
    assert r.shape == (96, 128) and r.sum() == 16 * 12  # 8x6 box scaled x2
    assert m.box == (20, 40, 36, 52) and r[40:52, 20:36].all()


def test_budget_reservation_under_concurrency(conn, tmp_path, monkeypatch):
    from backend.config import load_trays
    from backend.perception.segment import segment_frames

    monkeypatch.setenv("CERES_BUDGET_USD", "0.003")
    for i in range(6):
        p = tmp_path / f"A_{i}.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), 60 + 10 * i, np.uint8))
        conn.execute(
            "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source) VALUES (?, 'A', ?, ?, 0.01, 'nominal', 'dataset')",
            (f"A:{i}", f"2016-01-0{i + 1}T00:00:00Z", p.as_posix()),
        )
    conn.commit()
    client = EventsClient(lambda: _events(canned_output()), delay=0.02)
    stats = segment_frames(
        conn, load_trays(), ApiBackend(conn, client), concurrency=4, cache_dir=tmp_path / "c"
    )
    assert len(client.calls) == 1 and stats["segmented"] == 1 and stats["budget_blocked"] == 5
    assert total_spend(conn) <= 0.003


def test_hsv_mask_includes_yellow_so_chlorosis_is_measurable(tmp_path):
    from backend.perception.metrics import chlorosis_frac

    img = np.full((100, 100, 3), (10, 15, 20), np.uint8)
    img[10:50, 10:50] = (0, 160, 230)  # yellow leaf (BGR), hue ~42 degrees
    p = tmp_path / "y.jpg"
    cv2.imwrite(str(p), img)
    (m,) = HsvBackend().segment(p, "plant")
    assert chlorosis_frac(img, m.raster()) > 0.8
