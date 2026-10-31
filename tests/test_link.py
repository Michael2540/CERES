"""Mars mode (M8): link state machine, queue, edge triage (live with a fake LLM, repair, token budget, recorded replay
with no network), audit, auto-queueing, the clock hook and the API. No network, no paid call."""

from __future__ import annotations

import copy
import json
import shutil
import socket
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from test_agent import ALERT, AREA, AS, SOIL, FakeLLM, amsg, base_db, db, happy_script, tcall
from test_api import fixture, shape_errors
from test_pipeline import AS_OF

from backend.agent import llm_client
from backend.agent.agronomist import find_alert
from backend.agent.llm_client import estimate_tokens
from backend.api import assessments as assess_api
from backend.api import evaluation
from backend.api.alerts import visible_alerts
from backend.app import app
from backend.link import audit, delay_sim, edge, queue, service
from backend.sim.clock import CLOCK, MissionClock
from backend.spend import BudgetExceeded
from backend.timeutil import iso_z, parse_ts


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "edge_runs"
    monkeypatch.setattr(edge, "RUNS_DIR", d)
    monkeypatch.setenv("EDGE_MODE", "spark_contributor")
    return d


# ------------------------------------------------------------------ the link state machine
def test_default_state_is_up_and_transitions_are_explicit(db):
    assert delay_sim.current(db) == delay_sim.LinkState()
    assert delay_sim.current(db).state == "up"
    with pytest.raises(delay_sim.InvalidTransition):
        delay_sim.transition(db, "up", None, AS)  # up -> up
    s = delay_sim.transition(db, "delayed", 30, AS, now=100.0)
    assert (s.state, s.delay_s, s.since, s.since_wall) == ("delayed", 30.0, iso_z(AS), 100.0)
    with pytest.raises(delay_sim.InvalidTransition):
        delay_sim.transition(db, "delayed", 30, AS)  # same delay again
    assert delay_sim.transition(db, "delayed", 60, AS).delay_s == 60.0  # delayed -> delayed changes the delay
    assert delay_sim.transition(db, "down", None, AS).delay_s == 0.0
    with pytest.raises(delay_sim.InvalidTransition):
        delay_sim.transition(db, "down", None, AS)
    assert delay_sim.transition(db, "delayed", 5, AS).state == "delayed"  # down -> delayed
    assert delay_sim.transition(db, "up", None, AS).state == "up"  # delayed -> up
    assert [r["state"] for r in db.execute("SELECT state FROM link_events ORDER BY id")] == [
        "delayed", "delayed", "down", "delayed", "up"]  # every change is logged


@pytest.mark.parametrize("state,delay", [("delayed", None), ("delayed", 0), ("delayed", -1), ("delayed", float("nan")),
                                         ("delayed", 1e9), ("up", 5), ("down", 5), ("sideways", None)])
def test_bad_arguments_are_rejected(db, state, delay):
    with pytest.raises(ValueError) as e:
        delay_sim.transition(db, state, delay, AS)
    assert not isinstance(e.value, delay_sim.InvalidTransition)
    assert delay_sim.current(db).state == "up"  # nothing was stored


def test_state_survives_a_restart_and_is_time_gated(base_db, tmp_path):
    from backend.db import connect

    path = tmp_path / "persist.db"
    shutil.copy(base_db, path)
    c1 = connect(path)
    delay_sim.transition(c1, "delayed", 45, AS, now=5.0)
    c1.close()
    c2 = connect(path)  # a new connection stands in for a restart
    cur = delay_sim.current(c2)
    assert (cur.state, cur.delay_s) == ("delayed", 45.0)
    assert delay_sim.state_at(c2, AS - timedelta(hours=1)).state == "up"  # before the change
    assert delay_sim.state_at(c2, AS).state == "delayed"
    c2.close()


def test_deliverable_rule():
    up, down = delay_sim.LinkState("up"), delay_sim.LinkState("down")
    dl = delay_sim.LinkState("delayed", 30.0)
    assert delay_sim.deliverable(up, 100.0, 100.0) and not delay_sim.deliverable(down, 100.0, 1e9)
    assert not delay_sim.deliverable(dl, 100.0, 129.9) and delay_sim.deliverable(dl, 100.0, 130.0)
    assert delay_sim.seconds_until(up, 100.0, 100.0) == 0.0 and delay_sim.seconds_until(down, 100.0, 100.0) is None
    assert delay_sim.seconds_until(dl, 100.0, 110.0) == 20.0 and delay_sim.seconds_until(dl, 100.0, 500.0) == 0.0


# ------------------------------------------------------------------ queue order and delay
def _alert(conn, aid=ALERT, as_of=AS):
    return find_alert(conn, aid, as_of)


def test_queue_order_delay_and_duplicates(db):
    a = _alert(db)
    i1 = queue.enqueue(db, a, AS, now=100.0)
    with pytest.raises(queue.AlreadyQueued):
        queue.enqueue(db, a, AS, now=101.0)  # one open item per alert
    # two more items with other alert ids, queued earlier in mission time but later in wall time
    for n, (ts, wall) in enumerate([("2016-01-06T13:00:00Z", 90.0), ("2016-01-06T12:00:00Z", 95.0)]):
        db.execute("INSERT INTO link_queue(id, ts, kind, payload, status, queued_at) VALUES (?,?,?,?,?,?)",
                   (f"x{n}", ts, "assess", json.dumps({"alert_id": f"a{n}", "as_of": ts}), "queued", wall))
    db.commit()
    order = [r["id"] for r in queue.items(db)]
    assert order == ["x1", "x0", i1]  # ts order, not insertion or wall order
    delayed = delay_sim.LinkState("delayed", 10.0)
    assert queue.deliverable_items(db, delayed, 99.9) == []  # nothing has waited 10 s yet
    assert [r["id"] for r in queue.deliverable_items(db, delayed, 100.0)] == ["x0"]  # queued at wall 90
    assert [r["id"] for r in queue.deliverable_items(db, delayed, 105.0)] == ["x1", "x0"]  # + wall 95, still in ts order
    assert [r["id"] for r in queue.deliverable_items(db, delayed, 110.0)] == ["x1", "x0", i1]
    assert queue.deliverable_items(db, delay_sim.LinkState("down"), 1e9) == []
    assert len(queue.deliverable_items(db, delay_sim.LinkState("up"), 0.0)) == 3
    assert queue.depth(db, AS) == 3 and queue.depth(db, parse_ts("2016-01-06T12:30:00Z")) == 1  # time-gated
    assert [r["id"] for r in queue.items(db, parse_ts("2016-01-06T12:30:00Z"))] == ["x1"]


# ------------------------------------------------------------------ audit
@pytest.mark.parametrize("urg,status", [("now", "alert"), ("next_check", "watch"), ("log_only", "nominal")])
def test_urgency_maps_to_status(urg, status):
    assert audit.URGENCY_STATUS[urg] == status and audit.edge_status(urg) == status
    assert audit.edge_status("other") is None and audit.edge_status(None) is None


def _earth(top="underwatering", status="alert"):
    return {"status": status, "probable_causes": [
        {"cause": "sensor_fault", "likelihood": 0.1}, {"cause": top, "likelihood": 0.7}]}


@pytest.mark.parametrize("urg,cause,status,top,agree,delta,match", [
    ("now", "underwatering", "alert", "underwatering", True, 0, True),
    ("now", "underwatering", "watch", "underwatering", False, 1, True),  # edge more urgent
    ("log_only", "underwatering", "alert", "underwatering", False, -2, True),  # edge under-called
    ("next_check", "low_light", "watch", "underwatering", False, 0, False),
    ("now", "low_light", "nominal", "underwatering", False, 2, False),
])
def test_compute_audit(urg, cause, status, top, agree, delta, match):
    a = audit.compute_audit({"urgency": urg, "likely_cause": cause}, _earth(top, status))
    assert (a["agree"], a["urgency_delta"], a["cause_match"]) == (agree, delta, match)
    assert (a["edge_cause"], a["earth_top_cause"], a["edge_urgency"], a["earth_status"]) == (cause, top, urg, status)
    assert a["notes"].startswith(f"Edge: {urg} ({audit.URGENCY_STATUS[urg]}) for {cause}. Earth: {status} for {top}.")
    assert set(a) == {"agree", "urgency_delta", "cause_match", "edge_cause", "earth_top_cause", "edge_urgency",
                      "earth_status", "notes"}


def test_audit_with_a_missing_side_compares_nothing():
    a = audit.compute_audit(None, _earth())
    assert a["agree"] is None and a["urgency_delta"] is None and "edge triage" in a["notes"]
    assert audit.compute_audit({"urgency": "now", "likely_cause": "x"}, None)["agree"] is None


# ------------------------------------------------------------------ edge triage (live, fake model)
def edge_final(cites=(SOIL, ALERT), **over):
    e = {"tray_id": "B", "urgency": "now", "likely_cause": "underwatering", "evidence_ids": list(cites),
         "crew_instruction": "Check substrate moisture in tray B and water if dry.",
         "handoff_note_for_earth": "B shrinking vs control A; soil moisture low."}
    e.update(over)
    return e


def edge_script(final=None, round1=True):
    items = []
    if round1:
        items.append(amsg([tcall("e1", "get_sensor_series", {"tray_id": "B", "metric": "soil_vwc", "hours": 24}),
                           tcall("e2", "compute_vpd", {"temp_c": 24, "rh_pct": 60})]))
        items.append(amsg(content="done"))
    items.append(amsg(content=json.dumps(final or edge_final())))
    return items


def run_edge(conn, script, **kw):
    llm = FakeLLM(script)
    return edge.triage(conn, ALERT, AS, llm=llm, **kw), llm


def arow(conn, aid):
    return conn.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()


def test_live_triage_writes_edge_assessment_and_tool_log(db):
    aid, llm = run_edge(db, edge_script())
    r = arow(db, aid)
    assert r["role"] == "edge" and r["valid"] == 1 and r["recorded"] == 0 and aid.startswith("edge_B_")
    assert (r["model"], r["provider"]) == ("muse-spark-1.3-contributor", "meta")
    assert json.loads(r["json"])["urgency"] == "now" and json.loads(r["warnings"] or "[]") == []
    assert r["cost_usd"] == pytest.approx(0.003)
    rows = db.execute("SELECT round, tool FROM tool_log WHERE assessment_id = ? ORDER BY rowid", (aid,)).fetchall()
    assert [(x["round"], x["tool"]) for x in rows] == [
        (0, "get_tray_overview"), (0, "get_frames"), (1, "get_sensor_series"), (1, "compute_vpd")]
    assert all(db.execute("SELECT result FROM tool_log WHERE assessment_id = ?", (aid,)).fetchall())
    # only the three edge tools are offered, plus exactly one image in the first user message
    assert {t["function"]["name"] for t in llm.calls[0]["tools"]} == set(edge.EDGE_TOOLS)
    assert llm.calls[-1]["tools"] is None and llm.calls[-1]["rf"]["json_schema"]["name"] == "edge_triage"
    assert llm_client.count_images(llm.calls[0]["messages"]) == 1
    text = llm.calls[0]["messages"][1]["content"][0]["text"]
    assert ALERT in text and "{{" not in text and "LINK up" in text and "[one tray image attached]" in text
    assert db.execute("SELECT COUNT(*) FROM assessments WHERE role IS NULL OR role = 'agronomist'").fetchone()[0] == 0
    assert visible_alerts(db, AS)[0]["assessment_id"] is None or arow(db, visible_alerts(db, AS)[0]["assessment_id"])["role"] != "edge"


def test_edge_refuses_a_tool_it_does_not_have_and_logs_it(db):
    script = [amsg([tcall("e1", "get_crew_notes", {"hours": 24})]), amsg(content="x"), amsg(content=json.dumps(edge_final()))]
    aid, llm = run_edge(db, script)
    log = db.execute("SELECT args FROM tool_log WHERE assessment_id = ? AND tool = 'get_crew_notes'", (aid,)).fetchone()
    assert "not available on board" in json.loads(log["args"])["_error"]
    tool_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "not available on board" in tool_msg["content"]


def test_repair_call_fixes_a_bad_citation(db):
    bad = edge_final(cites=("B:soil_vwc:2016-01-06T15:00:00Z",))
    script = edge_script(bad) + [amsg(content=json.dumps(edge_final()))]
    aid, llm = run_edge(db, script)
    assert arow(db, aid)["valid"] == 1 and len(llm.calls) == 4
    fb = llm.calls[-1]["messages"][-1]
    assert fb["role"] == "user" and "citation:" in fb["content"] and "2016-01-06T15:00:00Z" in fb["content"]
    assert llm.calls[-1]["tools"] is None


def test_second_failure_is_stored_invalid_without_a_third_try(db):
    bad = edge_final(cites=("B:soil_vwc:2016-01-06T15:00:00Z",))
    aid, llm = run_edge(db, edge_script(bad) + [amsg(content=json.dumps(bad))])
    r = arow(db, aid)
    assert r["valid"] == 0 and len(llm.calls) == 4 and not llm.script
    assert any(e.startswith("citation:") for e in json.loads(r["citation_errors"]))
    assert json.loads(r["json"])["tray_id"] == "B"  # still stored, shown as invalid


def test_unparseable_then_schema_error_paths(db):
    script = [amsg(content="not json at all"), amsg(content="still {not json"),
              amsg(content=json.dumps({"tray_id": "B", "urgency": "panic"}))]
    aid, llm = run_edge(db, script)
    r = arow(db, aid)
    assert r["valid"] == 0 and len(llm.calls) == 3  # plain answer, structured final, ONE repair
    assert "not valid JSON" in llm.calls[2]["messages"][-1]["content"]
    assert any(e.startswith("schema:") for e in json.loads(r["citation_errors"]))


def test_wrong_tray_is_an_error(db):
    aid, _ = run_edge(db, edge_script(edge_final(tray_id="A")) + [amsg(content=json.dumps(edge_final(tray_id="A")))])
    assert any("differs from the alerted tray" in e for e in json.loads(arow(db, aid)["citation_errors"]))


def test_grounding_failure_triggers_the_repair_call_then_invalid(db):
    hallucinated = edge_final(crew_instruction="Soil is at 0.4321 and VPD is 3.77 kPa; water now.")
    aid, llm = run_edge(db, edge_script(hallucinated) + [amsg(content=json.dumps(hallucinated))])
    r = arow(db, aid)
    errs = json.loads(r["citation_errors"])
    assert r["valid"] == 0 and len(llm.calls) == 4
    assert any(e.startswith("grounding:") and "0.4321" in e for e in errs)
    assert {w["number"] for w in json.loads(r["warnings"])} == {"0.4321", "3.77"}
    assert "grounding:" in llm.calls[-1]["messages"][-1]["content"]


def test_grounded_numbers_pass_and_the_repair_removes_bad_ones(db):
    soil = None

    def final(messages):
        nonlocal soil
        series = next(json.loads(m["content"]) for m in messages if m["role"] == "tool" and '"soil_vwc"' in m["content"])
        soil = series["v"][-1]
        return amsg(content=json.dumps(edge_final(crew_instruction=f"Soil moisture reads {soil:.2f}; check water.")))

    aid, _ = run_edge(db, edge_script()[:2] + [final])
    assert arow(db, aid)["valid"] == 1 and soil is not None
    fixed = amsg(content=json.dumps(edge_final()))
    aid2, llm = run_edge(db, edge_script(edge_final(crew_instruction="VPD is 9.99 kPa.")) + [fixed])
    assert arow(db, aid2)["valid"] == 1 and len(llm.calls) == 4


def test_token_budget_refuses_an_oversize_prompt(db, monkeypatch):
    monkeypatch.setattr(edge, "MAX_PROMPT_TOKENS", 200)
    llm = FakeLLM([])
    with pytest.raises(edge.TokenBudgetExceeded):
        edge.triage(db, ALERT, AS, llm=llm)
    assert not llm.calls  # refused before any call
    r = db.execute("SELECT * FROM assessments WHERE role = 'edge'").fetchone()
    assert r["valid"] == 0 and json.loads(r["citation_errors"])[0].startswith("error: TokenBudgetExceeded")


def test_token_budget_omits_results_that_do_not_fit_and_uncites_them(db, monkeypatch):
    _, probe = run_edge(db, edge_script())
    n0 = estimate_tokens(probe.calls[0]["messages"])
    limit = n0 + edge.FINAL_RESERVE_TOKENS + 50
    monkeypatch.setattr(edge, "MAX_PROMPT_TOKENS", limit)
    script = [amsg([tcall("e1", "get_sensor_series", {"tray_id": "B", "metric": "soil_vwc", "hours": 72})]),
              amsg(content=json.dumps(edge_final()))]
    aid, llm = run_edge(db, script)
    assert all(estimate_tokens(c["messages"]) <= limit for c in llm.calls), "no call may exceed the budget"
    tool_msg = next(m for m in llm.calls[-1]["messages"] if m["role"] == "tool")
    assert "omitted" in tool_msg["content"]
    ids = db.execute("SELECT returned_ids FROM tool_log WHERE assessment_id = ? AND tool = 'get_sensor_series'", (aid,)).fetchone()
    assert json.loads(ids[0]) == []  # the model never saw them, so they are not citable
    assert arow(db, aid)["valid"] == 1


def test_unparseable_repair_reports_grounding_as_not_evaluated_never_ok(db):
    hallucinated = edge_final(crew_instruction="Soil is at 0.4321; water now.")
    aid, llm = run_edge(db, edge_script(hallucinated) + [amsg(content="sorry {not json")])
    r = arow(db, aid)
    assert r["valid"] == 0 and len(llm.calls) == 4
    assert "not valid JSON" in " ".join(json.loads(r["citation_errors"]))
    assert r["warnings"] is None and r["json"] is None  # the FINAL reply was never evaluated: no "ok", no stale answer
    iid = queue.enqueue(db, _alert(db), AS, now=1.0)
    queue.update(db, iid, edge_assessment_id=aid)
    view = queue.view(db, queue.get(db, iid), delay_sim.current(db), 2.0)
    assert view["grounding"] is None and view["edge_valid"] is False


def test_evaluated_clean_edge_run_stores_an_empty_grounding_list(db):
    aid, _ = run_edge(db, edge_script())
    assert arow(db, aid)["warnings"] == "[]"


def test_compute_vpd_arguments_do_not_ground_numbers(db):
    script = [amsg([tcall("e1", "compute_vpd", {"temp_c": 7.7, "rh_pct": 99.9})]), amsg(content="done")]
    cheat = edge_final(crew_instruction="Air is 7.7 C at 99.9 % RH; check the vent.")
    aid, _llm = run_edge(db, script + [amsg(content=json.dumps(cheat)), amsg(content=json.dumps(cheat))])
    r = arow(db, aid)
    assert r["valid"] == 0
    assert {w["number"] for w in json.loads(r["warnings"])} == {"7.7", "99.9 %"}
    result = json.loads(db.execute("SELECT result FROM tool_log WHERE assessment_id = ? AND tool = 'compute_vpd'", (aid,)).fetchone()[0])
    assert result["temp_c"] == 7.7  # the result does echo its inputs; the pool must not use them
    vpd = result["vpd_kpa"]
    ok = edge_final(crew_instruction=f"Computed VPD is {vpd} kPa; check the vent.")
    aid2, _ = run_edge(db, script + [amsg(content=json.dumps(ok))])
    assert arow(db, aid2)["valid"] == 1


def test_unbacked_crew_action_claim_is_an_edge_error(db):
    claim = edge_final(handoff_note_for_earth="Tray B dry. Crew watered B.")
    aid, llm = run_edge(db, edge_script(claim) + [amsg(content=json.dumps(claim))])
    r = arow(db, aid)
    assert r["valid"] == 0 and any("unbacked crew-action claim" in e for e in json.loads(r["citation_errors"]))
    assert "unbacked crew-action claim" in llm.calls[-1]["messages"][-1]["content"]  # fed to the repair call
    advice = edge_final(handoff_note_for_earth="Tray B dry. Crew should water B.")
    aid2, _ = run_edge(db, edge_script(advice))
    assert arow(db, aid2)["valid"] == 1


def test_invalid_edge_triage_is_edge_invalid_and_never_edge_done(db):
    bad = edge_final(crew_instruction="Soil is at 0.4321; water now.")
    script = edge_script(bad) + [amsg(content=json.dumps(bad))]
    delay_sim.transition(db, "down", None, AS)
    iid = service.enqueue_and_triage(db, _alert(db), AS, llm_factory=lambda c: FakeLLM(script), now=1000.0)
    r = queue.get(db, iid)
    assert r["status"] == "edge_invalid" and r["edge_json"] is not None  # errors and the answer stay visible
    assert r["error"].startswith("edge triage invalid:") and "0.4321" in r["error"]
    v = queue.view(db, r, delay_sim.current(db), 2000.0)
    assert v["edge_valid"] is False and v["citation_errors"] and v["status"] == "edge_invalid"
    assert queue.depth(db, AS) == 1  # still waiting for Earth
    assert [x["id"] for x in queue.deliverable_items(db, delay_sim.LinkState("up"), 0.0)] == [iid]
    delay_sim.transition(db, "up", None, AS)
    out = service.drain(db, now=2.0, llm_factory=lambda c: FakeLLM(happy_script()))
    assert out == [{"id": iid, "status": "audited", "error": None}]  # the Earth run still happens
    a = json.loads(queue.get(db, iid)["audit_json"])
    assert a["agree"] is None and a["cause_match"] is None and a["urgency_delta"] is None
    assert a["notes"].startswith("edge triage invalid: ") and a["notes"].endswith("; audit not meaningful")
    assert a["edge_cause"] == "underwatering" and a["earth_top_cause"] == "underwatering"  # shown, never compared


def test_invalid_earth_assessment_makes_the_audit_not_meaningful(db):
    delay_sim.transition(db, "down", None, AS)
    iid = service.enqueue_and_triage(db, _alert(db), AS, llm_factory=lambda c: FakeLLM(edge_script()), now=1.0)
    assert queue.get(db, iid)["status"] == "edge_done"
    delay_sim.transition(db, "up", None, AS)
    bad = json.loads(happy_script()[-1]["content"])
    bad["findings"][0]["evidence_ids"] = ["B:soil_vwc:2016-01-06T15:00:00Z"]  # never returned by a tool
    script = happy_script()[:-1] + [amsg(content=json.dumps(bad)), amsg(content=json.dumps(bad))]
    out = service.drain(db, now=2.0, llm_factory=lambda c: FakeLLM(script))
    assert out[0]["status"] == "audited"
    a = json.loads(queue.get(db, iid)["audit_json"])
    assert a["agree"] is None and "Earth assessment invalid" in a["notes"] and a["notes"].endswith("audit not meaningful")
    assert "edge triage invalid" not in a["notes"]


# ------------------------------------------------------------------ recorded replay: no network
def _record(db, runs_dir):
    """A live run with a recorder, written to the runs dir. Returns (live assessment id, recording dict)."""
    def final(messages):
        calc = next(json.loads(m["content"])["id"] for m in messages if m["role"] == "tool" and '"calc:vpd:' in m["content"])
        return amsg(content=json.dumps(edge_final(cites=(SOIL, calc, ALERT))))

    recorder = edge.RecordingLLM(FakeLLM(edge_script()[:2] + [final]))
    aid = edge.triage(db, ALERT, AS, llm=recorder)
    rec = edge.export_recording(db, aid, recorder, "spark_contributor")
    runs_dir.mkdir(parents=True, exist_ok=True)
    edge.recording_path(ALERT, runs_dir).write_text(json.dumps(rec), encoding="utf-8")
    return aid, rec


def _forbid_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a network call or client construction was attempted in recorded mode")

    monkeypatch.setattr(edge, "LLMClient", boom)
    monkeypatch.setattr(llm_client.LLMClient, "__init__", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)


def _log(conn, aid):
    return [(r["round"], r["tool"], r["args"], r["returned_ids"], r["result"]) for r in
            conn.execute("SELECT * FROM tool_log WHERE assessment_id = ? ORDER BY rowid", (aid,))]


def test_recorded_replay_matches_the_live_run_with_no_network(db, runs_dir, monkeypatch):
    live_aid, rec = _record(db, runs_dir)
    assert rec["model"] == "muse-spark-1.3-contributor" and rec["provider"] == "meta" and rec["valid"]
    assert len(rec["responses"]) == 3 and [t["tool"] for t in rec["tool_calls"]][:2] == ["get_tray_overview", "get_frames"]
    monkeypatch.setenv("EDGE_MODE", "recorded")
    _forbid_network(monkeypatch)
    aid = edge.triage(db, ALERT, AS)  # the SAME code path, replayed
    assert aid != live_aid
    live, rep = arow(db, live_aid), arow(db, aid)
    assert rep["recorded"] == 1 and live["recorded"] == 0 and rep["role"] == "edge"
    assert (rep["model"], rep["provider"]) == ("muse-spark-1.3-contributor", "meta") and rep["cost_usd"] == 0.0
    assert rep["valid"] == 1 and json.loads(rep["citation_errors"]) == []  # validators ran and passed on the replay
    def norm(rows, run_id):  # calc ids carry the run id; the replay re-keys them to the new run
        return [tuple(x.replace(run_id, "AID") if isinstance(x, str) else x for x in row) for row in rows]

    assert live["json"].replace(live_aid, "AID") == rep["json"].replace(aid, "AID")
    assert norm(_log(db, live_aid), live_aid) == norm(_log(db, aid), aid)  # same tool_log rows, results included
    cited = json.loads(rep["json"])["evidence_ids"]
    assert any(c.startswith(f"calc:vpd:{aid}") for c in cited)  # the calc id resolves for the new run


def test_a_recording_made_after_as_of_is_not_replayed(db, runs_dir, monkeypatch):
    _, rec = _record(db, runs_dir)
    rec["as_of"] = "2016-01-06T15:00:00Z"  # made one hour after the time we ask for
    edge.recording_path(ALERT, runs_dir).write_text(json.dumps(rec), encoding="utf-8")
    monkeypatch.setenv("EDGE_MODE", "recorded")
    _forbid_network(monkeypatch)
    with pytest.raises(edge.NoRecording, match="later than as_of"):
        edge.triage(db, ALERT, AS)  # no future data
    rec["as_of"] = "2016-01-06T13:00:00Z"  # made earlier: fine, and the item keeps its own (later) as_of
    edge.recording_path(ALERT, runs_dir).write_text(json.dumps(rec), encoding="utf-8")
    assert arow(db, edge.triage(db, ALERT, AS))["recorded"] == 1


def test_no_recording_is_reported_honestly_never_faked(db, monkeypatch):
    monkeypatch.setenv("EDGE_MODE", "recorded")
    _forbid_network(monkeypatch)
    with pytest.raises(edge.NoRecording, match="no recorded edge run for this alert"):
        edge.triage(db, ALERT, AS)
    assert db.execute("SELECT COUNT(*) FROM assessments WHERE role = 'edge'").fetchone()[0] == 0
    a = _alert(db)
    iid = service.enqueue_and_triage(db, a, AS, now=1.0)
    r = queue.get(db, iid)
    assert r["status"] == "queued" and r["edge_json"] is None and r["error"] == "no recorded edge run for this alert"


def test_a_corrupt_or_foreign_recording_is_not_replayed(db, runs_dir, monkeypatch):
    runs_dir.mkdir(parents=True)
    edge.recording_path(ALERT, runs_dir).write_text("{broken", encoding="utf-8")
    monkeypatch.setenv("EDGE_MODE", "recorded")
    with pytest.raises(edge.NoRecording, match="unusable"):
        edge.triage(db, ALERT, AS)


def test_a_recording_that_does_not_match_the_code_fails_the_replay(db, runs_dir, monkeypatch):
    _, rec = _record(db, runs_dir)
    tampered = copy.deepcopy(rec)
    next(t for t in tampered["tool_calls"] if t["tool"] == "get_sensor_series")["args"]["hours"] = 5
    edge.recording_path(ALERT, runs_dir).write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setenv("EDGE_MODE", "recorded")
    _forbid_network(monkeypatch)
    with pytest.raises(edge.ReplayMismatch):
        edge.triage(db, ALERT, AS)
    r = db.execute("SELECT * FROM assessments WHERE role = 'edge' ORDER BY rowid DESC").fetchone()
    assert r["valid"] == 0 and "ReplayMismatch" in r["citation_errors"]


# ------------------------------------------------------------------ badge truthfulness
def test_badge_states_recorded_or_remote_api_and_the_real_model(db, runs_dir, monkeypatch):
    live_aid, _ = _record(db, runs_dir)
    live = queue.provenance(arow(db, live_aid))
    assert live["recorded"] is False and live["model"] == "muse-spark-1.3-contributor" and live["provider"] == "meta"
    assert live["edge_location"] == "remote_api"
    assert live["badge"] == "EDGE (muse-spark-1.3-contributor, remote API: stands in for an on-board model)"
    assert "glimmer" not in json.dumps(live).lower()
    monkeypatch.setenv("EDGE_MODE", "recorded")
    rep = queue.provenance(arow(db, edge.triage(db, ALERT, AS)))
    assert rep["recorded"] is True and rep["edge_location"] == "recorded"
    assert rep["badge"] == "RECORDED · EDGE (muse-spark-1.3-contributor)" and "glimmer" not in json.dumps(rep).lower()
    monkeypatch.setenv("EDGE_MODE", "remote_glimmer")
    monkeypatch.setenv("GLIMMER_MODEL", "glim-test")
    gl = queue.provenance(arow(db, run_edge(db, edge_script())[0]))
    assert gl["provider"] == "glimmer" and gl["model"] == "glim-test" and gl["edge_location"] == "remote_api"
    assert gl["badge"] == "EDGE (glim-test, remote API: stands in for an on-board model)"
    for p in (live, rep, gl):  # nothing is served on the habitat computer, so ON-BOARD never labels a model
        assert "ON-BOARD" not in p["badge"]
    assert queue.provenance(None) is None


# ------------------------------------------------------------------ service: enqueue, drain, audit
def test_enqueue_triage_then_drain_computes_the_audit_in_code(db):
    delay_sim.transition(db, "down", None, AS)
    iid = service.enqueue_and_triage(db, _alert(db), AS, llm_factory=lambda c: FakeLLM(edge_script()), now=1000.0)
    r = queue.get(db, iid)
    assert r["status"] == "edge_done" and json.loads(r["edge_json"])["likely_cause"] == "underwatering"
    assert service.drain(db, now=1001.0, llm_factory=lambda c: FakeLLM([])) == []  # link down: nothing leaves
    delay_sim.transition(db, "delayed", 30, AS)
    assert service.drain(db, now=1010.0, llm_factory=lambda c: FakeLLM([])) == []  # delay not elapsed
    out = service.drain(db, now=1031.0, llm_factory=lambda c: FakeLLM(happy_script()))
    assert out == [{"id": iid, "status": "audited", "error": None}]
    r = queue.get(db, iid)
    a = json.loads(r["audit_json"])
    assert a["agree"] is True and a["urgency_delta"] == 0 and a["cause_match"] is True
    assert (a["edge_cause"], a["earth_top_cause"], a["edge_urgency"], a["earth_status"]) == (
        "underwatering", "underwatering", "now", "alert")
    earth = arow(db, r["earth_assessment_id"])
    assert earth["role"] == "agronomist" and earth["ts"] == iso_z(AS)
    assert visible_alerts(db, AS)[0]["assessment_id"] == earth["id"]  # the edge row never shadows the Earth assessment
    assert service.drain(db, now=2000.0, llm_factory=lambda c: FakeLLM([])) == []  # audited items do not drain again


def test_drain_marks_failures_and_budget_refusals(db):
    delay_sim.transition(db, "down", None, AS)
    iid = service.enqueue_and_triage(db, _alert(db), AS, llm_factory=lambda c: FakeLLM(edge_script()), now=1.0)
    delay_sim.transition(db, "up", None, AS)

    class Broke:
        def chat(self, *a, **k):
            raise BudgetExceeded("over the cap")

    out = service.drain(db, now=2.0, llm_factory=lambda c: Broke())
    assert out[0]["status"] == "failed" and out[0]["error"].startswith("BudgetExceeded")
    assert queue.get(db, iid)["audit_json"] is None


def test_drain_without_an_edge_result_still_reaches_earth_with_a_null_audit(db, monkeypatch):
    monkeypatch.setenv("EDGE_MODE", "recorded")
    delay_sim.transition(db, "down", None, AS)
    iid = service.enqueue_and_triage(db, _alert(db), AS, now=1.0)
    delay_sim.transition(db, "up", None, AS)
    out = service.drain(db, now=2.0, llm_factory=lambda c: FakeLLM(happy_script()))
    assert out[0]["status"] == "audited"
    a = json.loads(queue.get(db, iid)["audit_json"])
    assert a["agree"] is None and "edge triage" in a["notes"]


# ------------------------------------------------------------------ automatic queueing and the clock hook
def test_auto_enqueue_only_when_the_link_is_not_up_and_only_new_alerts(db):
    first = parse_ts(db.execute("SELECT first_ts FROM alerts WHERE id = ?", (ALERT,)).fetchone()[0])
    prev, new = first - timedelta(hours=1), AS
    assert service.auto_enqueue(db, prev, new, llm_factory=lambda c: FakeLLM(edge_script())) == []  # link up
    delay_sim.transition(db, "down", None, AS)
    assert service.auto_enqueue(db, new, prev) == []  # backwards
    assert service.auto_enqueue(db, first + timedelta(hours=2), first + timedelta(hours=3)) == []  # nothing new there
    made = service.auto_enqueue(db, prev, new, llm_factory=lambda c: FakeLLM(edge_script()))
    assert queue.item_id(ALERT, new) in made
    assert queue.get(db, queue.item_id(ALERT, new))["status"] == "edge_done"
    assert service.auto_enqueue(db, prev, new, llm_factory=lambda c: FakeLLM(edge_script())) == []  # no duplicates


def test_auto_enqueue_caps_live_triage_and_keeps_the_rest_queued(db, monkeypatch):
    monkeypatch.setattr(service, "AUTO_MAX_TRIAGE", 0)
    delay_sim.transition(db, "down", None, AS)
    first = parse_ts(db.execute("SELECT first_ts FROM alerts WHERE id = ?", (ALERT,)).fetchone()[0])

    def factory(_c):
        raise AssertionError("a live edge call was made past the cap")

    made = service.auto_enqueue(db, first - timedelta(hours=1), AS, llm_factory=factory)
    assert made and all(queue.get(db, i)["status"] == "queued" for i in made)


def test_clock_take_advance():
    t = [0.0]
    a0 = parse_ts("2016-01-03T00:00:00Z")
    c = MissionClock(as_of=a0, speed=3600, running=True, time_fn=lambda: t[0])
    assert c.take_advance() is None
    t[0] = 10.0
    assert c.take_advance() == (a0, a0 + timedelta(hours=10))
    assert c.take_advance() is None
    t[0] = 20.0
    c.now()
    c.set(running=False)
    c.set(as_of="2016-01-05T00:00:00Z")  # a paused forward jump counts too
    adv = c.take_advance()
    assert adv[0] == a0 + timedelta(hours=10) and adv[1] == parse_ts("2016-01-05T00:00:00Z")
    c.set(as_of="2016-01-04T00:00:00Z")  # backwards: nothing to trigger
    assert c.take_advance() is None


# ------------------------------------------------------------------ API
@pytest.fixture
def api(base_db, tmp_path, monkeypatch):
    path = tmp_path / "api.db"
    shutil.copy(base_db, path)
    monkeypatch.setenv("DB_PATH", str(path))
    CLOCK.set(as_of=AS_OF, speed=720, running=False)
    scripts: list = []
    monkeypatch.setattr(assess_api, "llm_factory", lambda conn: scripts.pop(0))
    c = TestClient(app)
    c.scripts = scripts
    yield c
    for t in list(__import__("backend.api.link", fromlist=["x"]).DRAIN_THREADS):
        t.join(30)
    CLOCK.set(as_of="2016-01-03T12:00:00Z", speed=720, running=False)


def j(r, code=200):
    assert r.status_code == code, (r.status_code, r.text)
    return r.json()


def test_get_link_default_matches_fixture_and_never_says_glimmer(api):
    body = j(api.get("/link"))
    assert not shape_errors(body, fixture("link.json")), shape_errors(body, fixture("link.json"))
    assert body["state"] == "up" and body["simulated"] is True and "3-22 min" in body["light_delay_note"]
    assert body["edge_mode"] == "spark_contributor" and body["edge_model"] == "muse-spark-1.3-contributor"
    assert body["edge_location"] == "remote_api" and body["earth_model"] == "muse-spark-1.3"
    assert "glimmer" not in json.dumps(body).lower()


def test_post_link_transitions_200_409_422_and_time_gate(api):
    j(api.post("/link", json={"state": "up"}), 409)
    j(api.post("/link", json={"state": "delayed"}), 422)
    j(api.post("/link", json={"state": "delayed", "delay_s": -3}), 422)
    j(api.post("/link", json={"state": "sideways"}), 422)
    j(api.post("/link", json={"state": "down", "delay_s": 5}), 422)
    body = j(api.post("/link", json={"state": "delayed", "delay_s": 30}))
    assert (body["state"], body["delay_s"], body["since"]) == ("delayed", 30.0, AS_OF)
    j(api.post("/link", json={"state": "delayed", "delay_s": 30}), 409)
    assert j(api.get("/link"))["state"] == "delayed"
    assert j(api.get("/link", params={"as_of": "2016-01-06T10:00:00Z"}))["state"] == "up"  # before the change
    assert j(api.post("/link", json={"state": "down"}))["state"] == "down"


def test_queue_post_404_409_200_and_get_shape(api):
    api.scripts.append(FakeLLM(edge_script()))
    j(api.post("/link/queue", json={"alert_id": ALERT}), 409)  # link is up
    j(api.post("/link", json={"state": "down"}))
    j(api.post("/link/queue", json={"alert_id": "alert_nope"}), 404)
    j(api.post("/link/queue", json={"alert_id": ALERT}, params={"as_of": "2016-01-01T00:00:00Z"}), 404)  # not yet raised
    j(api.post("/link/queue", json={"alert_id": ALERT}, params={"as_of": "2016-01-09T00:00:00Z"}), 409)  # resolved by then
    item = j(api.post("/link/queue", json={"alert_id": ALERT}))
    assert item["status"] == "edge_done" and item["edge_json"]["likely_cause"] == "underwatering"
    assert item["perception"] == "frames segmented before the link dropped"
    assert item["provenance"]["recorded"] is False and item["provenance"]["model"] == "muse-spark-1.3-contributor"
    j(api.post("/link/queue", json={"alert_id": ALERT}), 409)  # already queued
    q = j(api.get("/link/queue"))
    assert [i["id"] for i in q["items"]] == [item["id"]]
    assert not shape_errors(q, fixture("link_queue.json")), shape_errors(q, fixture("link_queue.json"))
    assert j(api.get("/link/queue", params={"as_of": "2016-01-06T10:00:00Z"}))["items"] == []  # time-gated
    assert j(api.get("/link"))["queue_depth"] == 1
    assert "glimmer" not in json.dumps(q).lower()


def test_queue_in_recorded_mode_without_a_recording_says_so(api, monkeypatch):
    monkeypatch.setenv("EDGE_MODE", "recorded")
    j(api.post("/link", json={"state": "down"}))
    item = j(api.post("/link/queue", json={"alert_id": ALERT}))
    assert item["edge_json"] is None and item["provenance"] is None and item["status"] == "queued"
    assert item["error"] == "no recorded edge run for this alert"
    body = j(api.get("/link"))
    assert body["edge_mode"] == "recorded" and body["edge_model"] is None and "glimmer" not in json.dumps(body).lower()
    assert body["edge_location"] == "recorded"


def test_drain_plan_confirm_and_audit(api):
    assert j(api.post("/link/drain", json={"confirm": True}))["started"] is False  # link up, queue empty
    j(api.post("/link", json={"state": "down"}))
    j(api.post("/link/drain", json={}), 409)  # link down
    api.scripts.append(FakeLLM(edge_script()))
    item = j(api.post("/link/queue", json={"alert_id": ALERT}))
    j(api.post("/link", json={"state": "delayed", "delay_s": 3600}))
    waiting = j(api.post("/link/drain", json={"confirm": True}))
    assert waiting["started"] is False and "delay not elapsed" in waiting["detail"]
    j(api.post("/link", json={"state": "up"}))
    plan = j(api.post("/link/drain", json={}))
    assert plan["started"] is False and plan["confirm_required"] is True and plan["item_ids"] == [item["id"]]
    assert plan["est_cost_usd"] == {"low": 0.1, "high": 0.2}
    assert j(api.get("/link/queue"))["items"][0]["status"] == "edge_done"  # the plan ran nothing
    api.scripts.append(FakeLLM(happy_script()))
    run = j(api.post("/link/drain", json={"confirm": True}))
    assert run["started"] is True and run["confirm_required"] is False
    assert not shape_errors(run, fixture("link_drain.json"))
    from backend.api import link as link_api

    for t in link_api.DRAIN_THREADS:
        t.join(30)
    done = j(api.get("/link/queue"))["items"][0]
    assert done["status"] == "audited" and done["audit_json"]["agree"] is True and done["earth_assessment_id"]
    asm = j(api.get(f"/assessments/{done['edge_assessment_id']}"))
    assert asm["role"] == "edge" and asm["warnings"] == [] and asm["model"] == "muse-spark-1.3-contributor"


def test_drain_is_refused_past_the_budget(api, monkeypatch):
    j(api.post("/link", json={"state": "down"}))
    api.scripts.append(FakeLLM(edge_script()))
    j(api.post("/link/queue", json={"alert_id": ALERT}))
    j(api.post("/link", json={"state": "up"}))
    monkeypatch.setenv("CERES_BUDGET_USD", "0.05")
    r = api.post("/link/drain", json={"confirm": True})
    assert r.status_code == 402 and "budget" in r.text.lower()
    assert j(api.get("/link/queue"))["items"][0]["status"] == "edge_done"


def test_eval_latest_role_edge_404_then_200(api, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "LATEST_EDGE_PATH", tmp_path / "none.json")
    assert api.get("/eval/latest", params={"role": "edge"}).status_code == 404
    p = tmp_path / "latest_edge.json"
    p.write_text(json.dumps({"run_id": "eval_edge_x", "role": "edge"}), encoding="utf-8")
    monkeypatch.setattr(evaluation, "LATEST_EDGE_PATH", p)
    assert j(api.get("/eval/latest", params={"role": "edge"}))["run_id"] == "eval_edge_x"
    assert api.get("/eval/latest", params={"role": "nope"}).status_code == 422



def test_drain_plan_and_items_are_as_of_gated_like_the_reads(api):
    j(api.post("/link", json={"state": "down"}))
    api.scripts.append(FakeLLM(edge_script()))
    item = j(api.post("/link/queue", json={"alert_id": ALERT}))
    j(api.post("/link", json={"state": "up"}))
    early = j(api.post("/link/drain", json={"confirm": True}, params={"as_of": "2016-01-06T10:00:00Z"}))
    assert early["item_ids"] == [] and early["started"] is False  # the item was queued later than this as_of
    assert j(api.get("/link/queue", params={"as_of": "2016-01-06T10:00:00Z"}))["items"] == []
    assert j(api.get("/link/queue"))["items"][0]["status"] == "edge_done"  # nothing was drained
    plan = j(api.post("/link/drain", json={}))
    assert plan["item_ids"] == [item["id"]]


def test_service_drain_skips_items_after_as_of(db):
    delay_sim.transition(db, "down", None, AS)
    iid = service.enqueue_and_triage(db, _alert(db), AS, llm_factory=lambda c: FakeLLM(edge_script()), now=1.0)
    delay_sim.transition(db, "up", None, AS)
    assert service.drain(db, now=2.0, as_of=AS - timedelta(hours=1), llm_factory=lambda c: FakeLLM([])) == []
    assert queue.get(db, iid)["status"] == "edge_done"
