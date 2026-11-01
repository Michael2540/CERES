# ruff: noqa: F811
"""M9: offline demo. Seed round-trip, DEMO_MODE replay of the Flight Agronomist (no network), a missing recording, the
badges/footer text rules. Replay tests run with the LLM client constructor patched to raise."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest
from test_agent import ALERT, AS, AS_OF, FakeLLM, api, base_db, db, happy_script, row  # noqa: F401

from backend.agent import agronomist, llm_client, replay
from backend.api import assessments as assess_api
from backend.api.alerts import visible_alerts
from backend.db import connect
from backend.link import queue, service

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runs(tmp_path, monkeypatch):
    d = tmp_path / "agent_runs"
    d.mkdir()
    monkeypatch.setattr(replay, "RUNS_DIR", d)
    return d


def record(conn, runs_dir):
    recorder = replay.RecordingLLM(FakeLLM(happy_script()))
    aid = agronomist.assess(conn, ALERT, AS, llm=recorder)
    rec = replay.export_recording(conn, aid, recorder)
    replay.recording_path(ALERT, runs_dir).write_text(json.dumps(rec), encoding="utf-8")
    return aid, rec


def demo_mode(monkeypatch, sockets=True):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("CERES_REPLAY_PACE_S", "0")

    def boom(*a, **k):
        raise AssertionError("a client was constructed or a socket opened in DEMO_MODE")

    monkeypatch.setattr(llm_client.LLMClient, "__init__", boom)
    monkeypatch.setattr(agronomist, "LLMClient", boom)
    if sockets:  # the ASGI test client itself uses loopback sockets
        monkeypatch.setattr(socket.socket, "connect", boom)


def spend_rows(conn):
    return conn.execute("SELECT COUNT(*) FROM spend").fetchone()[0]


def test_replay_matches_live_with_no_network_and_writes_no_spend(db, runs, monkeypatch):
    live_aid, rec = record(db, runs)
    assert rec["role"] == "agronomist" and rec["model"] == "muse-spark-1.3" and rec["valid"]
    before = spend_rows(db)
    demo_mode(monkeypatch)
    aid = agronomist.assess(db, ALERT, AS)  # the SAME orchestrator, replayed
    live, rep = row(db, live_aid), row(db, aid)
    assert aid != live_aid and rep["recorded"] == 1 and live["recorded"] == 0
    assert rep["model"] == "muse-spark-1.3" and rep["cost_usd"] == 0.0 and spend_rows(db) == before
    assert rep["valid"] == 1 and json.loads(rep["citation_errors"]) == []  # validators re-ran and passed
    assert live["json"].replace(live_aid, "X") == rep["json"].replace(aid, "X")
    acts = db.execute("SELECT status FROM actions WHERE assessment_id = ?", (aid,)).fetchall()
    assert [a["status"] for a in acts] == ["pending"]  # propose_action re-ran through the real handler


def test_replay_revalidates_and_never_trusts_a_stored_verdict(db, runs, monkeypatch):
    _, rec = record(db, runs)
    final = json.loads(rec["responses"][-1]["message"]["content"])
    final["findings"][0]["evidence_ids"] = ["B:soil_vwc:2099-01-01T00:00:00Z"]  # a citation no tool returned
    rec["responses"][-1]["message"]["content"] = json.dumps(final)
    rec["responses"] += [rec["responses"][-1]]  # the one retry of the structured final call
    rec["valid"], rec["citation_errors"] = True, []  # the stored verdict claims it is fine
    replay.recording_path(ALERT, runs).write_text(json.dumps(rec), encoding="utf-8")
    demo_mode(monkeypatch)
    aid = agronomist.assess(db, ALERT, AS)
    r = row(db, aid)
    assert r["valid"] == 0 and any("2099" in e for e in json.loads(r["citation_errors"]))


def test_a_recording_that_no_longer_fits_the_code_is_refused(db, runs, monkeypatch):
    _, rec = record(db, runs)
    rec["tool_calls"][2]["args"] = {"tray_id": "A", "timestamps": []}
    replay.recording_path(ALERT, runs).write_text(json.dumps(rec), encoding="utf-8")
    demo_mode(monkeypatch)
    with pytest.raises(replay.ReplayMismatch):
        agronomist.assess(db, ALERT, AS)
    bad = db.execute("SELECT valid, citation_errors FROM assessments ORDER BY rowid DESC LIMIT 1").fetchone()
    assert bad["valid"] == 0 and "ReplayMismatch" in bad["citation_errors"]


def test_missing_recording_is_a_clear_error_and_writes_nothing(db, runs, monkeypatch):
    demo_mode(monkeypatch)
    n = db.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    with pytest.raises(replay.NoAgentRecording, match="no recorded run for this alert in demo mode"):
        agronomist.assess(db, ALERT, AS)
    assert db.execute("SELECT COUNT(*) FROM assessments").fetchone()[0] == n


def test_recording_after_as_of_is_not_replayed(db, runs, monkeypatch):
    _, rec = record(db, runs)
    rec["as_of"] = "2016-01-06T15:00:00Z"
    replay.recording_path(ALERT, runs).write_text(json.dumps(rec), encoding="utf-8")
    demo_mode(monkeypatch)
    with pytest.raises(replay.NoAgentRecording, match="later than as_of"):
        agronomist.assess(db, ALERT, AS)


def test_live_client_cannot_be_built_in_demo_mode(db, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    with pytest.raises(RuntimeError, match="DEMO_MODE"):
        llm_client.LLMClient(db)


def test_api_replays_and_409s_without_a_recording(api, runs, monkeypatch):
    demo_mode(monkeypatch, sockets=False)
    r = api.post(f"/alerts/{ALERT}/assess")
    assert r.status_code == 409 and "no recorded run for this alert in demo mode" in r.json()["detail"]
    conn = connect(os.environ["DB_PATH"])
    assert conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0] == 0
    monkeypatch.setenv("DEMO_MODE", "0")  # live mode only to MAKE a recording from a scripted client
    record(conn, runs)
    conn.close()
    demo_mode(monkeypatch, sockets=False)
    resp = api.post(f"/alerts/{ALERT}/assess", params={"force": 1})
    assert resp.status_code == 200, resp.text
    aid = resp.json()["assessment_id"]
    assess_api.THREADS[aid].join(20)
    body = api.get(f"/assessments/{aid}").json()
    assert body["recorded"] is True and body["valid"] is True and body["model"] == "muse-spark-1.3" and body["cost_usd"] == 0.0
    assert '"recorded": true' in api.get(f"/assessments/{aid}/stream").text
    link = api.get("/link").json()
    assert link["demo_mode"] is True and ALERT in link["agent_recordings"]


def test_drain_without_a_recording_fails_clearly(db, runs, monkeypatch):
    demo_mode(monkeypatch)
    alert = next(a for a in visible_alerts(db, AS) if a["id"] == ALERT)
    iid = queue.enqueue(db, alert, AS, time.time())
    queue.update(db, iid, status="edge_done")
    out = service.drain(db, now=time.time() + 1, as_of=AS)
    assert out[0]["status"] == "failed" and "no recorded run for this alert in demo mode" in out[0]["error"]


# ------------------------------------------------------------------ seed round trip
def test_seed_round_trip_gives_the_same_tray_summary(api, tmp_path, monkeypatch):
    from scripts import build_demo_seed, seed_demo

    src = os.environ["DB_PATH"]
    out = tmp_path / "bundle"
    assert build_demo_seed.main(["--source-db", src, "--out-dir", str(out), "--width", "320"]) == 0
    assert (out / "seed.db.gz").is_file() and list((out / "frames").glob("*.jpg"))
    demo_db = seed_demo.seed(tmp_path / "demo.db", out / "seed.db.gz")

    def snap():
        p = {"as_of": AS_OF}
        s = {u: api.get(u, params=p).json() for u in ("/trays", "/alerts", "/trays/B")}
        s["series"] = api.get("/trays/B/series", params={"metric": "area_cm2", "hours": 48, **p}).json()
        return s

    want = snap()
    monkeypatch.setenv("DB_PATH", str(demo_db))
    monkeypatch.setenv("IMAGES_ROOT", str(out / "frames"))
    assert snap() == want
    fid = f"B:{AS_OF}"
    for kind in ("image", "overlay"):
        assert api.get(f"/trays/B/frames/{fid}/{kind}", params={"as_of": AS_OF}).status_code == 200


# ------------------------------------------------------------------ footer, badges, wording
FOOTER = "Built with Muse Spark 1.3, SAM 3.1 (Meta Model API) · plant images: ALED, CC BY 4.0"


def web_sources():
    for d in ("app", "components", "lib"):
        yield from (ROOT / "web" / d).rglob("*.ts*")


def test_footer_text_and_forbidden_model_names():
    assert FOOTER in (ROOT / "web" / "components" / "Footer.tsx").read_text(encoding="utf-8")
    assert "<Footer" in (ROOT / "web" / "app" / "layout.tsx").read_text(encoding="utf-8")
    for p in web_sources():
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            assert "dinov3" not in line.lower(), f"{p}:{n} mentions DINOv3"
            if "glimmer" in line.lower():  # only the EDGE_MODE setting (remote_glimmer) may name it
                assert "remote_glimmer" in line or "EDGE_MODE" in line, f"{p}:{n} mentions Glimmer"


def test_badges_exist_for_every_model_result():
    web = ROOT / "web"
    agent = (web / "components" / "AgentPanel.tsx").read_text(encoding="utf-8")
    mars = (web / "components" / "Mars.tsx").read_text(encoding="utf-8")
    trust = (web / "app" / "trust" / "page.tsx").read_text(encoding="utf-8")
    assert "RECORDED" in agent and "LIVE" in agent
    assert "earth-recorded" in mars and "edge-recorded" in mars
    assert trust.count("<RunBadge") >= 2 and "RECORDED RUN" in trust and "run_id" in trust


def test_no_secrets_or_local_paths_in_the_demo_bundle():
    bad = ("api_key", "bearer ", "c:/users", "c:\\\\users")
    for p in (ROOT / "data" / "demo").rglob("*.json"):
        t = p.read_text(encoding="utf-8").lower()
        assert not any(b in t for b in bad), p
