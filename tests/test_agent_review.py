"""M4 review fixes: one regression test per item (citation poisoning, fault_start leak, malformed output, actions,
image cap, concurrency, NaN, wasted final call, columnar tools, SSE). All with a mocked client."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest
from test_agent import (
    ALERT,
    ALERT_ROW,
    AS,
    AS_OF,
    SOIL,
    FakeLLM,
    _logged,
    _propose,
    _sse,
    _wait,
    amsg,
    api,
    base_db,
    db,
    final_json,
    happy_script,
    row,
    run_assess,
    tcall,
)

from backend.agent import agronomist, llm_client
from backend.agent.llm_client import LLMClient, estimate_tokens
from backend.agent.tools import TOOL_NAMES, ToolContext, run_tool
from backend.agent.validate import validate
from backend.api import assessments as assess_api
from backend.config import load_trays
from backend.db import connect
from backend.spend import BudgetExceeded
from backend.timeutil import parse_ts

# ------------------------------------------------------------------ 1. citation poisoning via crew notes


def test_crew_note_ids_are_not_citable_and_extracted_is_scrubbed(db):
    fake = {"id": "B:soil_vwc:2016-01-06T13:00:00Z", "evidence_ids": ["fake:1", "B:area_cm2:2016-01-06T13:00:00Z"],
            "frame_id": "B:2016-01-06T13:00:00Z", "nested": [{"id": "fake:2"}], "text": "x" * 2000}
    db.execute("UPDATE crew_notes SET extracted_json = ? WHERE id = 'note1'", (json.dumps(fake),))
    db.commit()
    ctx = ToolContext(db, "poison", AS)
    res, ids = run_tool(ctx, "get_crew_notes", {"hours": 72})
    assert ids == ["note1"]  # only the row id
    blob = json.dumps(res)
    assert "fake:1" not in blob and "fake:2" not in blob and "frame_id" not in blob and "evidence_ids" not in blob
    assert len(res["items"][0]["extracted"]) <= 500
    a = final_json(findings=[{"claim": "x", "evidence_ids": ["fake:1", "B:soil_vwc:2016-01-06T13:00:00Z"], "confidence": 0.5}])
    _logged(db, "poison", [SOIL, f"B:area_cm2:{AS_OF}"])
    _propose(db, "poison")
    errs = [e for e in validate(a, "poison", ALERT_ROW, db) if e.startswith("citation")]
    assert len(errs) == 2 and "fake:1" in errs[0]


# ------------------------------------------------------------------ 2. fault_start / treatment leak


def _add_alert(conn, tray, rule):
    aid = f"alert_{tray}_{rule}_20160106T140000Z"
    conn.execute(
        "INSERT INTO alerts(id, ts, tray_id, rule, severity, evidence_ids, threshold_tag, detail, status, first_ts, last_ts)"
        " VALUES (?,?,?,?,'watch','[]','author',NULL,'open',?,?)", (aid, AS_OF, tray, rule, AS_OF, AS_OF))
    conn.execute("INSERT INTO alert_events(alert_id, ts, evidence_ids, detail) VALUES (?,?,?,?)", (aid, AS_OF, "[]", ""))
    conn.commit()
    return aid


@pytest.mark.parametrize("tray, rule", [("B", "WILT_RISK"), ("C", "GROWTH_LAG"), ("L", "LOW_VPD_RISK")])
def test_rendered_prompts_do_not_leak_fault_start_or_treatment(db, tray, rule):
    aid_alert = ALERT if tray == "B" else _add_alert(db, tray, rule)
    llm = FakeLLM([amsg(content="ok"), amsg(content="{}"), amsg(content="{}")])
    try:
        agronomist.assess(db, aid_alert, AS, llm=llm)
    except Exception:  # noqa: BLE001, S110
        pass
    text = " ".join(m["content"] for m in llm.calls[0]["messages"][:2])
    for t in load_trays().trays:
        if t.fault_start:
            assert t.fault_start not in text and t.fault_start[:10] not in text
        assert t.treatment == "control" or t.treatment not in text  # tray A is openly the control
    assert "{{" not in text and "fault_start" not in text and "own pre-" not in text


# ------------------------------------------------------------------ 3. malformed model output


@pytest.mark.parametrize("name", [n for n in TOOL_NAMES])
def test_tools_survive_list_valued_arguments(db, name):
    ctx = ToolContext(db, "mal", AS)
    for args in ({"tray_id": ["A"], "metric": ["soil_vwc"], "hours": [1], "timestamps": [["x"]], "crop": ["a"],
                  "temp_c": ["1"], "rh_pct": {}, "action": ["x"], "evidence_ids": [["x"]]}, {"tray_id": {"a": 1}}, {}):
        res, ids = run_tool(ctx, name, args)  # must not raise
        assert isinstance(res, dict)
        if name not in ("get_setpoints", "get_crew_notes"):
            assert "error" in res and ids == []


def test_run_tool_turns_unexpected_exceptions_into_errors(db, monkeypatch):
    from backend.agent import tools

    def boom(_ctx, _args):
        raise KeyError("oops")

    monkeypatch.setitem(tools.HANDLERS, "get_setpoints", boom)
    res, ids = run_tool(ToolContext(db, "x", AS), "get_setpoints", {"crop": "a"})
    assert res == {"error": "KeyError: 'oops'"} and ids == []


def test_validator_handles_non_string_evidence_ids_and_odd_types(db):
    _logged(db, "m1", [SOIL])
    _propose(db, "m1")
    a = final_json(findings=[{"claim": "x", "evidence_ids": [["x"], {"a": 1}, 5], "confidence": 0.5}])
    errs = validate(a, "m1", ALERT_ROW, db)  # no TypeError
    assert sum("non-string evidence id" in e for e in errs) == 3
    odd = final_json(tray_id=["B"], causes_considered="sensor_fault")
    assert validate(odd, "m1", ALERT_ROW, db)
    assert validate({"tray_id": [1], "recommended_actions": [{"action": "x"}], "probable_causes": [7]}, "m1", ALERT_ROW, db)
    assert validate([], "m1", ALERT_ROW, db)


def test_tool_loop_survives_list_tray_id_end_to_end(db):
    script = [
        amsg([tcall("c1", "get_sensor_series", {"tray_id": ["A"], "metric": "soil_vwc", "hours": 3})]),
        amsg(content="ok"),
        amsg(content=json.dumps(final_json(recommended_actions=[]))),
    ]
    aid, llm, _ = run_assess(db, script)
    assert "error" in json.loads(next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")["content"])
    assert row(db, aid)["valid"] == 1


# ------------------------------------------------------------------ 4. actions


def test_propose_action_rejects_ids_not_returned_in_this_run(db):
    ctx = ToolContext(db, "pa", AS, alert_id=ALERT)
    run_tool(ctx, "get_sensor_series", {"tray_id": "B", "metric": "soil_vwc", "hours": 2})
    bad = {"tray_id": "B", "action": "Water", "magnitude": "", "rationale": "", "evidence_ids": [SOIL, "B:soil_vwc:nope", "fake"]}
    res, _ = run_tool(ctx, "propose_action", bad)
    assert "error" in res and "B:soil_vwc:nope" in res["error"] and "fake" in res["error"] and SOIL not in res["error"]
    assert db.execute("SELECT COUNT(*) FROM actions WHERE assessment_id = 'pa'").fetchone()[0] == 0
    ok = {**bad, "evidence_ids": [SOIL, ALERT]}  # the alert id counts as known
    assert run_tool(ctx, "propose_action", ok)[0]["status"] == "pending_human_approval"


def test_failed_or_invalid_runs_supersede_pending_actions(db):
    bad = final_json(findings=[{"claim": "x", "evidence_ids": ["B:soil_vwc:2016-01-06T15:00:00Z"], "confidence": 0.5}])
    script = [*happy_script(bad)[:2], amsg(content=json.dumps(bad)), amsg(content=json.dumps(bad))]
    aid, _, _ = run_assess(db, script)
    assert row(db, aid)["valid"] == 0
    assert [r["status"] for r in db.execute("SELECT status FROM actions WHERE assessment_id = ?", (aid,))] == ["superseded"]
    # a crash mid-run (here: the model call raises after a proposal was stored)
    calls = {"n": 0}

    class Dying(FakeLLM):
        def chat(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("provider down")
            return super().chat(*a, **k)

    with pytest.raises(RuntimeError):
        agronomist.assess(db, ALERT, AS, llm=Dying(happy_script()))
    last = db.execute("SELECT id FROM assessments ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    st = [r["status"] for r in db.execute("SELECT status FROM actions WHERE assessment_id = ?", (last,))]
    assert st == ["superseded"]
    good, _, _ = run_assess(db, happy_script())
    assert [r["status"] for r in db.execute("SELECT status FROM actions WHERE assessment_id = ?", (good,))] == ["pending"]


def test_api_lists_superseded_actions_and_refuses_to_decide_them(api, monkeypatch):
    bad = final_json(findings=[{"claim": "x", "evidence_ids": ["nope"], "confidence": 0.5}])
    script = [*happy_script(bad)[:2], amsg(content=json.dumps(bad)), amsg(content=json.dumps(bad))]
    monkeypatch.setattr(assess_api, "llm_factory", lambda _c: FakeLLM(script))
    aid = api.post(f"/alerts/{ALERT}/assess").json()["assessment_id"]
    _wait(aid)
    body = api.get(f"/assessments/{aid}").json()
    assert body["valid"] is False and [a["status"] for a in body["actions"]] == ["superseded"]
    act = body["actions"][0]["action_id"]
    assert api.post(f"/actions/{act}", json={"decision": "approve"}).status_code == 409
    assert api.post("/actions/nope", json={"decision": "approve"}).status_code == 404
    good = api.post(f"/alerts/{ALERT}/assess", params={"force": 1}).json()["assessment_id"]
    assert good != aid
    _wait(good)


def test_api_decides_a_pending_action_once(api, monkeypatch):
    monkeypatch.setattr(assess_api, "llm_factory", lambda _c: FakeLLM(happy_script()))
    aid = api.post(f"/alerts/{ALERT}/assess").json()["assessment_id"]
    _wait(aid)
    act = api.get(f"/assessments/{aid}").json()["actions"][0]["action_id"]
    assert api.post(f"/actions/{act}", json={"decision": "approve"}).json() == {"action_id": act, "status": "approved"}
    assert api.post(f"/actions/{act}", json={"decision": "reject"}).status_code == 409


# ------------------------------------------------------------------ 5. image cap (see also test_image_cap_across_the_run)


def test_unattached_frames_cannot_be_cited(db):
    stamps = [f"2016-01-06T{h:02d}:00:00Z" for h in range(9, 15)]
    script = [
        amsg([tcall("c1", "get_frames", {"tray_id": "B", "timestamps": stamps}),
              tcall("c2", "get_frames", {"tray_id": "A", "timestamps": stamps})]),
        amsg(content="ok"),
    ]
    unseen = "B:2016-01-06T13:00:00Z"  # not attached (round-robin gives 09-11 to each tray)
    bad = final_json(recommended_actions=[], findings=[{"claim": "x", "evidence_ids": [unseen], "confidence": 0.5}])
    good = final_json(recommended_actions=[])
    aid, llm, _ = run_assess(db, [*script, amsg(content=json.dumps(bad)), amsg(content=json.dumps(good))])
    r = row(db, aid)
    assert r["valid"] == 1 and len(llm.calls) == 4  # the first answer cited a frame it never saw and was sent back
    assert unseen in llm.calls[-1]["messages"][-1]["content"]


def test_frames_seen_in_an_earlier_round_do_not_use_the_cap_again(db):
    st = ["2016-01-06T09:00:00Z", "2016-01-06T10:00:00Z", "2016-01-06T11:00:00Z"]
    script = [
        amsg([tcall("c1", "get_frames", {"tray_id": "B", "timestamps": st})]),
        amsg([tcall("c2", "get_frames", {"tray_id": "B", "timestamps": [*st, "2016-01-06T12:00:00Z", "2016-01-06T13:00:00Z", "2016-01-06T14:00:00Z"]})]),
        amsg(content="ok"),
        amsg(content=json.dumps(final_json(recommended_actions=[]))),
    ]
    _, llm, _ = run_assess(db, script)
    tool2 = json.loads([m for m in llm.calls[2]["messages"] if m["role"] == "tool"][1]["content"])
    assert all(r["attached"] for r in tool2["frame_refs"])  # 3 old + 3 new = 6 = cap


# ------------------------------------------------------------------ 6. concurrency


def test_vpd_calc_ids_are_per_assessment_and_resolve(api):
    c = connect(__import__("os").environ["DB_PATH"])
    a1, a2 = agronomist.begin(c, ALERT, AS), agronomist.begin(c, ALERT, AS)
    ids = []
    for a in (a1, a2):
        ctx = ToolContext(c, a, AS)
        ids += [run_tool(ctx, "compute_vpd", {"temp_c": 25, "rh_pct": 50})[1][0], run_tool(ctx, "compute_vpd", {"temp_c": 20, "rh_pct": 50})[1][0]]
    c.close()
    assert ids == [f"calc:vpd:{a1}:1", f"calc:vpd:{a1}:2", f"calc:vpd:{a2}:1", f"calc:vpd:{a2}:2"]
    vals = [api.get(f"/evidence/{i}").json()["value"] for i in ids]
    assert vals[0] == pytest.approx(1.584, abs=1e-3) and vals[1] < vals[0] and vals[2] == vals[0]


def test_post_assess_returns_existing_run_unless_forced(api, monkeypatch):
    gate = []

    def slow(_m):
        while not gate:
            time.sleep(0.02)
        return amsg(content=json.dumps(final_json(recommended_actions=[])))

    monkeypatch.setattr(assess_api, "llm_factory", lambda _c: FakeLLM([slow]))
    a1 = api.post(f"/alerts/{ALERT}/assess").json()["assessment_id"]
    assert api.post(f"/alerts/{ALERT}/assess").json()["assessment_id"] == a1  # in flight
    gate.append(1)
    _wait(a1)
    assert api.post(f"/alerts/{ALERT}/assess").json()["assessment_id"] == a1  # finished
    monkeypatch.setattr(assess_api, "llm_factory", lambda _c: FakeLLM([amsg(content=json.dumps(final_json(recommended_actions=[])))]))
    a2 = api.post(f"/alerts/{ALERT}/assess", params={"force": 1}).json()["assessment_id"]
    assert a2 != a1
    _wait(a2)
    monkeypatch.setattr(assess_api, "llm_factory", lambda _c: FakeLLM([amsg(content=json.dumps(final_json(recommended_actions=[])))]))
    a3 = api.post(f"/alerts/{ALERT}/assess", params={"as_of": "2016-01-06T14:30:00Z"}).json()["assessment_id"]
    assert a3 not in (a1, a2)  # another as_of is another run
    _wait(a3)  # never leave a worker running past the test: it would outlive the monkeypatched client


def test_failed_run_can_be_started_again_without_force(api, monkeypatch):
    def boom(_c):
        raise BudgetExceeded("budget")

    monkeypatch.setattr(assess_api, "llm_factory", boom)
    a1 = api.post(f"/alerts/{ALERT}/assess").json()["assessment_id"]
    _wait(a1)
    monkeypatch.setattr(assess_api, "llm_factory", lambda _c: FakeLLM(happy_script()))
    a2 = api.post(f"/alerts/{ALERT}/assess").json()["assessment_id"]
    assert a2 != a1
    _wait(a2)


def test_budget_reservation_is_shared_across_threads(tmp_path, monkeypatch):
    from backend.db import init_db

    path = tmp_path / "b.db"
    init_db(path).close()
    monkeypatch.setenv("CERES_BUDGET_USD", "0.03")  # one reserved estimate (~$0.0175) fits, two do not
    started, release = threading.Event(), threading.Event()
    called = []

    def create(**_kw):
        called.append(1)
        started.set()
        release.wait(5)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    msgs = [{"role": "user", "content": "hi"}]
    out = {}

    def first():
        c = connect(path)
        out["first"] = LLMClient(c, fake).chat("agronomist", msgs)
        c.close()

    t = threading.Thread(target=first)
    t.start()
    assert started.wait(5)
    c2 = connect(path)
    with pytest.raises(BudgetExceeded):  # the in-flight call holds a reservation although nothing is ledgered yet
        LLMClient(c2, fake).chat("agronomist", msgs)
    release.set()
    t.join(5)
    assert llm_client._RESERVED[0] == pytest.approx(0.0, abs=1e-12)
    assert len(called) == 1 and out["first"].cost_usd > 0
    LLMClient(c2, fake).chat("agronomist", msgs)  # the reservation was released and the real cost is tiny
    c2.close()


def test_reservation_is_released_when_the_call_fails(tmp_path):
    from backend.db import init_db

    c = init_db(tmp_path / "r.db")

    class Err(Exception):
        status_code = 400

    def create(**_kw):
        raise Err()

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(Err):
        LLMClient(c, fake).chat("agronomist", [{"role": "user", "content": "hi"}])
    assert llm_client._RESERVED[0] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ 7. NaN / Infinity


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_parse_rejects_non_finite_json(bad):
    obj, err = agronomist._parse_json('{"a": ' + bad + "}")
    assert obj is None and "valid JSON" in err


def test_validator_rejects_non_finite_likelihood_and_confidence(db):
    _logged(db, "n1", [SOIL])
    _propose(db, "n1")
    a = final_json()
    a["probable_causes"][0]["likelihood"] = float("nan")
    a["findings"][0]["confidence"] = float("inf")
    errs = validate(a, "n1", ALERT_ROW, db)
    assert any("probable_causes[0].likelihood is not a finite number" in e for e in errs)
    assert any("findings[0].confidence is not a finite number" in e for e in errs)


def test_nan_in_the_model_answer_triggers_the_retry(db):
    nan_text = json.dumps(final_json()).replace('"likelihood": 0.7', '"likelihood": NaN')
    assert "NaN" in nan_text
    script = [*happy_script()[:2], amsg(content=nan_text), amsg(content=json.dumps(final_json()))]
    aid, llm, _ = run_assess(db, script)
    assert row(db, aid)["valid"] == 1 and len(llm.calls) == 4


# ------------------------------------------------------------------ 8. wasted final call


def test_valid_plain_answer_in_the_tool_loop_skips_the_structured_call(db):
    script = happy_script(final_json())
    del script[1]  # the round-2 message IS the final answer
    aid, llm, events = run_assess(db, script)
    assert len(llm.calls) == 2 and all(c["tools"] for c in llm.calls) and row(db, aid)["valid"] == 1
    assert events[-1][0] == "assessment" and events[-1][1]["valid"] is True


def test_invalid_plain_answer_still_gets_the_structured_call(db):
    bad = final_json(findings=[{"claim": "x", "evidence_ids": ["nope"], "confidence": 0.5}])
    script = happy_script(bad)
    script[1] = amsg(content=json.dumps(bad))
    script.insert(2, amsg(content=json.dumps(final_json())))
    aid, llm, _ = run_assess(db, script)
    assert len(llm.calls) == 3 and llm.calls[-1]["tools"] is None and row(db, aid)["valid"] == 1


def test_token_estimate_uses_two_chars_per_token():
    assert llm_client.EST_CHARS_PER_TOKEN == 2.0
    assert estimate_tokens([{"role": "user", "content": "x" * 200}]) == 100


# ------------------------------------------------------------------ 9. columnar, trimmed tools


def test_series_and_canopy_are_columnar_with_exact_ids(db):
    ctx = ToolContext(db, "col", AS, rule="WILT_RISK")
    res, ids = run_tool(ctx, "get_sensor_series", {"tray_id": "B", "metric": "soil_vwc", "hours": 6})
    assert set(res) >= {"tray_id", "metric", "unit", "id_format", "ts", "v"} and "items" not in res
    assert res["id_format"] == "B:soil_vwc:{ts}" and ids == [res["id_format"].format(ts=t) for t in res["ts"]]
    assert len(res["ts"]) == len(res["v"])
    res, ids = run_tool(ctx, "get_sensor_series", {"tray_id": "B", "metric": "rh_pct", "hours": 6})
    assert res["id_format"] == "air:rh_pct:{ts}" and res["sensor_tray"] == "air"
    res, ids = run_tool(ctx, "get_canopy_metrics", {"tray_id": "B", "hours": 72})
    cols = [k for k in ("area_cm2", "growth_rate", "exg", "chlorosis_frac") if k in res]
    assert cols and "items" not in res and all(len(res[k]) == len(res["ts"]) for k in cols)
    assert "frame_id" not in json.dumps(res) and "metric_ids" not in json.dumps(res) and '"source"' not in json.dumps(res)
    want = {res["id_format"].format(metric=m, ts=t) for m in cols for t, v in zip(res["ts"], res[m], strict=True) if v is not None}
    assert set(ids) == want
    assert len(res["ts"]) <= 48


def test_get_setpoints_is_trimmed_to_the_alert_rule(db):
    wilt, _ = run_tool(ToolContext(db, "s1", AS, rule="WILT_RISK"), "get_setpoints", {"crop": "arabidopsis_thaliana"})
    assert {t["rule"] for t in wilt["rule_thresholds"]} == {"WILT_RISK"}
    full, _ = run_tool(ToolContext(db, "s2", AS), "get_setpoints", {"crop": "arabidopsis_thaliana"})
    assert len(wilt["rule_thresholds"]) < len(full["rule_thresholds"])
    assert all(r["applies_to"] == "arabidopsis" for r in wilt["setpoints"])


def test_images_are_downscaled_to_640(db):
    import base64

    import cv2
    import numpy as np

    path = db.execute("SELECT path FROM frames WHERE tray_id = 'B' LIMIT 1").fetchone()[0]
    part = agronomist.image_part(path)
    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), 1).shape[1] == agronomist.MAX_IMAGE_WIDTH == 640


def test_system_prompt_explains_id_format():
    text = (agronomist.PROMPTS / "agronomist.system.md").read_text(encoding="utf-8")
    assert "id_format" in text and "Return only JSON" in text


# ------------------------------------------------------------------ 10. SSE


def test_bad_argument_calls_are_logged_and_replay_matches_the_live_stream(db):
    script = [
        amsg([{"id": "c1", "type": "function", "function": {"name": "get_frames", "arguments": "{not json"}},
              tcall("c2", "get_sensor_series", {"tray_id": "B", "metric": "soil_vwc", "hours": 2})]
             + [tcall(f"x{i}", "compute_vpd", {"temp_c": 20, "rh_pct": 50}) for i in range(8)]),
        amsg(content="ok"),
        amsg(content=json.dumps(final_json(recommended_actions=[]))),
    ]
    aid, _, events = run_assess(db, script)
    rows = db.execute("SELECT round, tool, args, returned_ids FROM tool_log WHERE assessment_id = ? ORDER BY rowid", (aid,)).fetchall()
    assert any(json.loads(r["args"]).get("_error", "").startswith("bad tool arguments") for r in rows)
    assert any("at most" in json.loads(r["args"]).get("_error", "") for r in rows)
    replay = [(e["event"], json.loads(e["data"])) for e in assess_api._tool_events(rows)]
    live = [(e, d) for e, d in events if e in ("tool_call", "tool_result")]
    assert [(e, d["round"], d["tool"]) for e, d in replay] == [(e, d["round"], d["tool"]) for e, d in live]
    assert [d.get("ids") for e, d in replay if e == "tool_result"] == [d.get("ids") for e, d in live if e == "tool_result"]


def test_stream_reports_a_stalled_run(api, monkeypatch):
    c = connect(__import__("os").environ["DB_PATH"])
    aid = agronomist.begin(c, ALERT, AS)
    c.execute("UPDATE assessments SET updated_at = ? WHERE id = ?", (time.time() - 1000, aid))
    c.commit()
    c.close()
    ev = _sse(api.get(f"/assessments/{aid}/stream").text)
    assert ev[-1] == ("error", {"message": "run stalled"})


def test_stream_with_fresh_heartbeat_is_not_stalled_and_honours_as_of(api):
    c = connect(__import__("os").environ["DB_PATH"])
    aid = agronomist.begin(c, ALERT, AS)
    c.close()
    assert api.get(f"/assessments/{aid}/stream", params={"as_of": "2016-01-06T10:00:00Z"}).status_code == 404
    c = connect(__import__("os").environ["DB_PATH"])  # finish it so the open stream ends
    c.execute("UPDATE assessments SET valid = 1, json = '{}', citation_errors = '[]', latency_ms = 5 WHERE id = ?", (aid,))
    c.commit()
    c.close()
    ev = _sse(api.get(f"/assessments/{aid}/stream").text)
    assert ev[-1][0] == "assessment" and parse_ts(AS_OF) == AS
