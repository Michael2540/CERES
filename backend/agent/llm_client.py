"""LLM client: role routing, EDGE_MODE switch, spend ledger and budget refusal (M4a).

One call path, `LLMClient.chat`, for every role. Facts verified live in SPIKES.md S2:
- tool_choice is only ever "auto", and only sent when tools are given; the final structured call omits tools;
- strict json_schema goes in `response_format`; reasoning_effort goes through `extra_body`;
- the OpenAI SDK uses an httpx client with verify=certifi.where() (the global truststore crashes on Windows).
Before each call the budget is checked with a conservative estimate; after it the real usage is ledgered.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from backend.config import load_settings
from backend.spend import check_budget, cost_for, record_spend

log = logging.getLogger(__name__)

META_BASE_URL = "https://api.meta.ai/v1"
MAX_RETRIES = 3
BACKOFF_S = 1.0
EST_OUT_TOKENS = 4096  # conservative output allowance for the pre-call budget check
EST_TOKENS_PER_IMAGE = 1500  # ~1k measured (SPIKES S2), padded
EST_CHARS_PER_TOKEN = 2.0  # JSON-heavy tool text measured at roughly 2-3 chars per token


@dataclass(frozen=True)
class RoleConfig:
    provider: str  # meta | glimmer
    model: str
    reasoning_effort: str | None
    max_tokens: int
    image_cap: int


# agronomist: full model at low effort (S2: 5.5 s for 6 images). Env CERES_AGRONOMIST_MODEL overrides the model.
# edge: placeholder entry (spark_contributor); resolve_role() applies EDGE_MODE.
ROLES: dict[str, RoleConfig] = {
    "agronomist": RoleConfig("meta", "muse-spark-1.3", "low", 16384, 6),
    "edge": RoleConfig("meta", "muse-spark-1.3-contributor", "low", 16384, 1),
    "video": RoleConfig("meta", "muse-spark-1.3", "low", 16384, 6),  # M9 decides the frame budget
    "voice": RoleConfig("meta", "muse-spark-1.3-contributor", "low", 16384, 0),
}


def resolve_role(role: str) -> RoleConfig:
    """The role's config after env overrides and EDGE_MODE."""
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}; one of {sorted(ROLES)}")
    cfg = ROLES[role]
    if role == "agronomist":
        model = os.getenv("CERES_AGRONOMIST_MODEL")
        return replace(cfg, model=model) if model else cfg
    if role == "edge":
        mode = load_settings().edge_mode
        if mode == "recorded":
            raise NotImplementedError(
                "EDGE_MODE=recorded replays data/demo/edge_runs and makes no live call (M8: backend.link.edge.triage "
                "handles it before any client exists). Set EDGE_MODE=spark_contributor or remote_glimmer for a live call."
            )
        if mode == "spark_contributor":
            return replace(cfg, provider="meta", model="muse-spark-1.3-contributor", image_cap=1)
        return replace(cfg, provider="glimmer", model=os.getenv("GLIMMER_MODEL") or "muse-glimmer", image_cap=1)
    return cfg


def count_images(messages: list[dict]) -> int:
    """Number of image parts across all messages."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") in ("image_url", "input_image"))
    return n


def check_image_cap(messages: list[dict], cap: int) -> int:
    n = count_images(messages)
    if n > cap:
        raise ValueError(f"{n} images in the request, but the role's image_cap is {cap}")
    return n


def estimate_tokens(messages: list[dict]) -> int:
    """Rough input tokens: text length / 3 (image data URLs are counted as EST_TOKENS_PER_IMAGE instead)."""
    chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            chars += sum(len(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
        if m.get("tool_calls"):
            chars += len(json.dumps(m["tool_calls"]))
    return int(chars / EST_CHARS_PER_TOKEN) + EST_TOKENS_PER_IMAGE * count_images(messages)


@dataclass
class ChatResult:
    message: dict  # assistant message ready to append: {role, content, tool_calls?}
    usage: dict  # {input_tokens, output_tokens}
    latency_ms: int
    cost_usd: float
    model: str


class LLMClient:
    """OpenAI-SDK client with a spend ledger. `openai_client` (any object with chat.completions.create) is for tests."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        openai_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.conn = conn
        self._override = openai_client
        self._clients: dict[str, Any] = {}
        self._sleep = sleep

    def _client(self, provider: str) -> Any:
        if self._override is not None:
            return self._override
        if provider not in self._clients:
            import certifi
            import httpx
            from openai import OpenAI

            load_settings()  # loads .env into the environment
            http = httpx.Client(verify=certifi.where(), timeout=180)
            if provider == "meta":
                self._clients[provider] = OpenAI(
                    base_url=META_BASE_URL, api_key=os.environ["META_API_KEY"], http_client=http
                )
            else:  # glimmer: any OpenAI-compatible endpoint
                self._clients[provider] = OpenAI(
                    base_url=os.environ["GLIMMER_BASE_URL"], api_key=os.environ["GLIMMER_API_KEY"], http_client=http
                )
        return self._clients[provider]

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status == 429 or status >= 500
        return type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout")

    def chat(
        self,
        role: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        cfg = resolve_role(role)
        check_image_cap(messages, cfg.image_cap)
        est_in = estimate_tokens(messages)
        est_cost = cost_for(cfg.model, est_in, min(cfg.max_tokens, EST_OUT_TOKENS))
        with _BUDGET_LOCK:
            check_budget(self.conn, est_cost + _RESERVED[0])
            _RESERVED[0] += est_cost
        kwargs: dict[str, Any] = {"model": cfg.model, "messages": messages, "max_tokens": cfg.max_tokens}
        if cfg.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": cfg.reasoning_effort}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"  # the only value the API accepts (SPIKES S2)
        if response_format:
            kwargs["response_format"] = response_format
        released = False
        try:
            client = self._client(cfg.provider)
            t0 = time.perf_counter()
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = client.chat.completions.create(**kwargs)
                    break
                except Exception as exc:
                    if attempt >= MAX_RETRIES or not self._retryable(exc):
                        raise
                    delay = BACKOFF_S * 2**attempt
                    log.warning("LLM call failed (%s); retry %d in %.1fs", exc, attempt + 1, delay)
                    self._sleep(delay)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            u = getattr(resp, "usage", None)
            usage = {
                "input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            }
            with _BUDGET_LOCK:  # ledger the real cost and drop the reservation in one step
                cost = record_spend(self.conn, role, cfg.provider, cfg.model, usage["input_tokens"], usage["output_tokens"])
                _RESERVED[0] -= est_cost
                released = True
        finally:
            if not released:
                with _BUDGET_LOCK:
                    _RESERVED[0] -= est_cost
        return ChatResult(_assistant_message(resp.choices[0].message), usage, latency_ms, cost, cfg.model)


# Process-wide budget reservation (mirrors sam_client): concurrent assessments in the API's worker threads must not
# each pass the check against the same ledger total. A call reserves its estimate until its real cost is ledgered.
_BUDGET_LOCK = threading.Lock()
_RESERVED = [0.0]


def _assistant_message(msg: Any) -> dict:
    out: dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", None) or ""}
    calls = getattr(msg, "tool_calls", None)
    if calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"},
            }
            for c in calls
        ]
    return out
