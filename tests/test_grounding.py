"""Numeric grounding (backend/agent/grounding.py): every rule R1-R8 of the module docstring has a case here, and every
attack string of the independent hallucination review (atk2.py / aud.py) has a regression test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_agent import ALERT, AS, SOIL, amsg, base_db, db, final_json, happy_script, row, run_assess, tcall

from backend.agent import grounding as g
from backend.timeutil import parse_ts


def nums(text: str, **kw) -> list[str]:
    return [n["text"] for n in g.extract(text, **kw)]


def flagged(text: str, pool: g.Pool | None = None, ids=()) -> list[str]:
    p = pool or POOL
    p = g.Pool(p.values, p.durations, set(ids) | p.ids)
    return [b["number"] for b in g.ungrounded({"x": text}, p)]


# a pool that holds none of the fabricated numbers below (91, 41, 37, 9.9, 17, 20, 38, ...)
POOL = g.Pool(values=[1.2, 24.0, 0.7], durations=[17.0, 40.0, 72.0])


# ---------------------------------------------------------------- extraction (R1-R3)
@pytest.mark.parametrize("text,expected", [
    ("Soil is 0.38 and VPD 1.2 kPa", ["0.38", "1.2"]),
    ("alert_B_WILT_RISK_20160106T140000Z fired", []),  # R1 compact stamp inside an id
    ("see B:soil_vwc:2016-01-06T14:00:00Z for 0.31", ["0.31"]),  # R1 ISO timestamp, then a real number
    ("at 2016-01-06T14:00:00Z the value was 12.5", ["12.5"]),
    ("on 2016-01-06 growth was 4.7", ["4.7"]),
    ("at 14:00 and 14:30:15 UTC it read 55", ["55"]),  # R1 clock times
    ("CO2 reached 1450 ppm in pot12", ["1450"]),  # R1 CO2 and pot ids
    ("cause co2_anomaly or low_vpd", []),  # R1 enum names
    ("2 pots and step 3, then 1 check", []),  # R2 counting words
    ("0 leaves, 3 trays, but 4 pots", ["4"]),  # R2 boundary: 3 ignored, 4 kept
    ("a rise of 0.2 and 2.0 kPa", ["0.2", "2.0"]),  # R2 only bare integers
    ("1,200 ppm", ["1200"]),  # R3 thousands
    ("growth -0.4 per day", ["-0.4"]),  # R3 sign kept in text
    ("between 5-6 hours", ["5", "6"]),  # a range is two numbers
    ("38 % of capacity", ["38 %"]),  # percent captured
    ("rose .5 kPa", [".5"]),  # no leading zero
    ("", []),
])
def test_extract(text, expected):
    assert nums(text) == expected


def test_extract_keeps_counting_numbers_for_the_pool():
    assert nums("2 pots", drop_counting=False) == ["2"]


def test_exact_ids_of_the_run_are_masked_and_nothing_else_is():
    cid = "calc:vpd:asmt_B_20160106T100000Z:12"
    assert nums(f"VPD from {cid} is 0.4", ids=[cid]) == ["0.4"]
    assert nums(f"VPD from {cid}.", ids=[cid]) == []  # trailing sentence punctuation
    assert nums("VPD from calc:vpd:asmt_B_20160106T100000Z:12 is 0.4") == ["12", "0.4"]  # an id the run never saw is read


# ---------------------------------------------------------------- finding 2: masks must not hide fabricated numbers
@pytest.mark.parametrize("text,expected", [
    ("VPD:9.9 kPa", ["9.9"]),
    ("soil_vwc_0.07 at B", ["0.07"]),
    ("RH97 in B", ["97"]),
    ("temp T31 C", ["31"]),
    ("ratio 17:83", ["17", "83"]),  # 83 is not a minute: not a clock time
    ("opened at 17:30 and 9:05", []),  # real clock times are masked
    ("x91", ["91"]),
    ("soil_vwc fell, rh_pct rose, tray B", []),  # allowlisted names carry no number
])
def test_glued_numbers_are_extracted(text, expected):
    assert nums(text) == expected
    assert flagged(text) == expected  # and none of them is in the pool


# ---------------------------------------------------------------- finding 3: past claims are not plans
@pytest.mark.parametrize("text", [
    "soil dry for 91 hours", "soil fell in 41 hours", "soil fell within 37 hours to dry", "for 91 minutes soil was dry",
    "every 91 hours it dried", "it has been dry for 91 hours", "The soil, which was wet, dried over 91 hours",
    "approx. 91 hrs of dryness", "91h", "dry 91 h", "91-hour dryness", "91‑hour",
])
def test_past_durations_are_checked(text):
    assert flagged(text) == ["91"] or flagged(text) == ["41"] or flagged(text) == ["37"], text


@pytest.mark.parametrize("text", [
    "Recheck soil in 12 hours.", "Water and recheck in 12 hours", "re-image in 6-12h; recovery confirms deficit",
    "Continue watching growth for 48 hours.", "Monitor, and repeat after 2 days", "check again within 30 min",
    "Wait for 6 hours then review", "Re-check in twelve hours", "compared to 2016-01-04/05",
])
def test_forward_looking_intervals_are_plans(text):
    assert flagged(text) == [], text


def test_plan_needs_a_plan_verb_in_its_own_clause():
    assert flagged("Recheck soil and turgor in 12 hours.") == []  # one clause: the plan verb covers the interval
    assert flagged("Soil fell and stayed dry in 41 hours.") == ["41"]  # no plan verb in the clause
    assert flagged("Soil was checked within 37 hours.") == ["37"]  # "checked" is past, not a plan verb
    assert flagged("Continue to monitor; soil dry for 91 hours") == ["91"]  # ";" ends the plan clause
    assert flagged("Soil dry for 48 hours, recheck later") == ["48"]  # "for" needs watch|monitor|continue|wait


# ---------------------------------------------------------------- finding 4: small integers
@pytest.mark.parametrize("text,bad", [
    ("VPD of 3 kPa", ["3"]), ("dropped 2 units", ["2"]), ("over the last 3 days", ["3"]), ("about 2 C warmer", ["2"]),
    ("2 pots", []), ("1 tray", []), ("step 3", []), ("3 plants and 2 leaves", []), ("4 pots", ["4"]),
])
def test_small_integers(text, bad):
    pool = g.Pool(values=[1.2], durations=[17.0])
    assert flagged(text, pool) == bad


# ---------------------------------------------------------------- finding 5: spelled numbers
@pytest.mark.parametrize("text,bad", [
    ("twenty kPa", ["twenty"]), ("thirty-eight percent", ["thirty-eight percent"]), ("ninety-one hours", ["ninety-one"]),
    ("one hundred and five ppm", ["one hundred and five"]), ("Thirty eight %", ["Thirty eight %"]),
    ("ninety-one per cent", ["ninety-one per cent"]),
    ("two pots", []), ("one tray", []), ("someone moved on", []),
])
def test_spelled_numbers(text, bad):
    assert flagged(text) == bad


def test_spelled_values_convert():
    assert [n["value"] for n in g.extract("twenty-four hours, one hundred and five ppm, ninety-nine kPa")] == [24, 105, 99]
    assert flagged("twenty-four kPa") == []  # 24.0 is in the pool
    assert flagged("thirty-eight percent", g.Pool(values=[0.38])) == []  # percent grounds a fraction


@pytest.mark.parametrize("text", [
    "91 percent", "91 pct", "91 per cent", "91.0e0", "1,91", "９１", "soil 91 %", "dropped by 77.7 mm",
])
def test_review_attack_strings_are_flagged(text):
    assert flagged(text), text


def test_nine_one_and_superscript_never_hide_digits():
    assert flagged("9 1") == ["9"]
    assert flagged("¹91") == ["191"]  # NFKC reads the superscript as a digit


# ---------------------------------------------------------------- finding 6: durations
def test_durations_check_only_the_duration_pool():
    pool = g.Pool(values=[36.0], durations=[17.0])
    assert flagged("soil flat for 36 hours", pool) == ["36"]  # 36 is a sensor value, not a duration
    assert flagged("soil flat for 17 hours", pool) == []
    assert flagged("flat for 17.4 hours", pool) == ["17.4"]  # tolerance is half a unit of the precision written
    assert flagged("flat for 17.4 hours", g.Pool(durations=[17.4])) == []
    assert flagged("dry for 1020 min", pool) == []  # 17 h in minutes
    assert flagged("dry for 0.7 days", pool) == []
    assert flagged("dry for 2 days", pool) == ["2"]
    assert flagged("dry for 18 hours", pool) == ["18"]  # no 2 % rule for durations
    assert flagged("about 18 hours", pool) == []  # "~"/"about"/"around" allow 10 %
    assert flagged("~16 h", pool) == []
    assert flagged("around 24 h", pool) == ["24"]


def test_l_recording_real_24h_claim_fails_with_that_recordings_data():
    rec = json.loads(Path("data/demo/edge_runs/alert_L_LOW_VPD_RISK_20160108T190000Z.json").read_text(encoding="utf-8"))
    calls = [{"tool": t["tool"], "args": t["args"], "result": t["result"], "returned_ids": t["returned_ids"]}
             for t in rec["tool_calls"]]
    alert = {"id": rec["alert_id"], "rule": "LOW_VPD_RISK"}
    pool = g.pool_from_calls(calls, parse_ts(rec["as_of"]), alert, "2016-01-08T19:00:00Z")
    assert any(abs(d - 17.0) < 1e-6 for d in pool.durations)  # alert 01-08T19:00 -> as_of 01-09T12:00
    # the handoff of the first recording (2026-10-31), before the grounding fixes; the file has since been re-recorded
    note = "Tray L VPD below threshold with RH high vs control; LOW_VPD_RISK on L real ~24h. Plants look healthy."
    assert [b["number"] for b in g.ungrounded({"handoff": note}, pool)] == ["24"]
    assert g.ungrounded({"handoff": note.replace("~24h", "17 h")}, pool) == []  # the true age grounds


# ---------------------------------------------------------------- matching (R4, R5)
def test_matches_at_precision_and_relative():
    pool = [0.3812, 24.0, 1.234]
    ok = lambda t: g.matches(g.extract(t)[0], pool)
    assert ok("0.38") and ok("0.4") and ok("24") and ok("24.0") and ok("1.23") and ok("1.2")
    assert not ok("0.45") and not ok("0.37") and not ok("0.39") and not ok("25")
    assert g.matches(g.extract("1.25")[0], [1.234])  # within 2 % relative
    assert not g.matches(g.extract("1.27")[0], [1.234])


def test_percent_and_sign_rules():
    assert g.matches(g.extract("38 %")[0], [0.38])  # R5
    assert g.matches(g.extract("38 %")[0], [38.4])
    assert not g.matches(g.extract("38")[0], [0.38])
    assert g.matches(g.extract("dropped by 0.4")[0], [0.4])
    assert g.collect({"v": -0.4}) == [0.4]


# ---------------------------------------------------------------- pool
def test_collect_reads_leaves_and_numbers_inside_strings_and_durations():
    dur: list[float] = []
    pool = g.collect({"v": [0.1, 0.25], "detail": "rh_pct flatline 55.0 for 6 h", "ok": True, "n": None,
                      "when": "2016-01-06T14:00:00Z", "_error": "ignored 99"}, dur=dur)
    assert 0.1 in pool and 0.25 in pool and 55.0 in pool and 6.0 in pool
    assert 99.0 not in pool and 2016.0 not in pool and 14.0 not in pool
    assert dur == [6.0]


def test_pool_from_calls_ignores_arguments_and_vpd_echo():
    vpd = {"id": "calc:vpd:a:1", "temp_c": 7.7, "rh_pct": 99.9, "vpd_kpa": 0.0009, "unit": "kPa (air VPD)"}
    series = {"tray_id": "B", "metric": "soil_vwc", "hours": 24.0, "ts": ["2016-01-06T10:00:00Z"], "v": [0.31]}
    calls = [
        {"tool": "compute_vpd", "args": {"temp_c": 7.7, "rh_pct": 99.9}, "result": vpd, "returned_ids": ["calc:vpd:a:1"]},
        {"tool": "get_sensor_series", "args": {"hours": 24, "tray_id": "B", "metric": "soil_vwc", "junk": 5.55},
         "result": series, "returned_ids": []},
        {"tool": "get_frames", "args": {}, "result": {"frame_refs": [{"id": "B:f", "ts": "2016-01-06T10:00:00Z"}]},
         "returned_ids": ["B:f"]},
    ]
    p = g.pool_from_calls(calls, parse_ts("2016-01-06T12:00:00Z"), {"id": "alert_x", "rule": "WILT_RISK"})
    assert 0.0009 in p.values and 0.31 in p.values
    assert 7.7 not in p.values and 99.9 not in p.values and 5.55 not in p.values
    assert flagged("air at 7.7 C and 99.9 %", p) == ["7.7", "99.9 %"]
    assert 24.0 in p.durations and any(abs(d - 2.0) < 1e-6 for d in p.durations)  # window arg, and the ts field
    assert "calc:vpd:a:1" in p.ids and "alert_x" in p.ids
    # the sample-grid array of a series does not ground every whole hour
    wide = {"hours": 72.0, "ts": [f"2016-01-0{d}T{h:02d}:00:00Z" for d in (4, 5) for h in range(24)]}
    p2 = g.pool_from_calls([{"tool": "get_sensor_series", "args": {}, "result": wide, "returned_ids": []}],
                           parse_ts("2016-01-06T12:00:00Z"), {"id": "a", "rule": "WILT_RISK"})
    assert flagged("dry for 33 hours", p2) == ["33"]


def test_ungrounded_reports_number_field_and_context():
    pool = g.Pool(values=g.collect({"soil": 0.3812, "vpd": 1.34}))
    bad = g.ungrounded({"crew_instruction": "Soil at 0.38 and VPD 0.45 kPa; water 2 pots.",
                        "handoff_note_for_earth": "Rose to 7.9 overnight."}, pool)
    assert [(b["field"], b["number"]) for b in bad] == [("crew_instruction", "0.45"), ("handoff_note_for_earth", "7.9")]
    assert "VPD 0.45 kPa" in bad[0]["context"]
    assert g.ungrounded({"x": "all fine, check 2 pots"}, pool) == []
    assert "0.45" in g.format_errors(bad)[0] and "crew_instruction" in g.format_errors(bad)[0]


# ---------------------------------------------------------------- fields (finding 8)
def test_texts_of_assessment_and_edge():
    a = final_json()
    t = g.assessment_texts(a)
    assert set(t) == {
        "findings[0].claim", "probable_causes[0].discriminating_check", "probable_causes[1].discriminating_check",
        "recommended_actions[0].rationale", "recommended_actions[0].action", "recommended_actions[0].magnitude",
        "causes_considered[0]", "causes_considered[1]", "causes_considered[2]", "crew_brief"}
    a["setpoint_caveats"] = ["VPD target 0.8 kPa is from the author"]
    a["missing_evidence"] = ["No reading for 91 hours"]
    t = g.assessment_texts(a)
    assert t["setpoint_caveats[0]"] and t["missing_evidence[0]"]
    assert g.edge_texts({"crew_instruction": "a", "handoff_note_for_earth": "b", "urgency": "now"}) == {
        "crew_instruction": "a", "handoff_note_for_earth": "b"}
    assert g.assessment_texts(None) == {}


# ---------------------------------------------------------------- against a real run's tool_log
def test_pool_comes_from_tool_log_results_and_alert(db):
    aid, _llm, _ = run_assess(db, happy_script())
    alert = {"id": ALERT, "rule": "WILT_RISK", "detail": "area fell 18 % in 24 h", "threshold_tag": ""}
    pool = g.build_pool(db, aid, alert)
    assert pool is not None
    assert 24.0 in pool.values and 18.0 in pool.values  # the result echoes hours; the alert detail
    assert 24.0 in pool.durations  # the look-back window of the call
    assert ALERT in pool.ids and any(i.startswith("B:soil_vwc:") for i in pool.ids)
    soil = json.loads(db.execute("SELECT result FROM tool_log WHERE assessment_id = ? AND tool = 'get_sensor_series'",
                                 (aid,)).fetchone()[0])["v"][-1]
    assert g.matches(g.extract(f"soil moisture {soil:.2f}")[0], pool)
    assert g.check(db, aid, alert, {"x": "the soil is at 0.9137 now"}) == [
        {"field": "x", "number": "0.9137", "context": "the soil is at 0.9137 now"}]


def test_run_without_stored_results_is_unavailable_not_passed(db):
    aid, _llm, _ = run_assess(db, happy_script())
    db.execute("UPDATE tool_log SET result = NULL WHERE assessment_id = ?", (aid,))
    assert g.build_pool(db, aid, {"id": ALERT, "rule": "WILT_RISK"}) is None
    assert g.check(db, aid, {"id": ALERT, "rule": "WILT_RISK"}, {"x": "0.5"}) is None


def test_agronomist_gets_warnings_but_validity_is_unchanged(db):
    clean_aid, _l, _e = run_assess(db, happy_script())
    assert row(db, clean_aid)["valid"] == 1 and json.loads(row(db, clean_aid)["warnings"]) == []
    bad = final_json(crew_brief="Water tray B; soil is at 0.4321 and VPD 3.77 kPa.")
    aid, _llm, _ = run_assess(db, happy_script(bad))
    r = row(db, aid)
    assert r["valid"] == 1 and json.loads(r["citation_errors"]) == []  # a warning, never a validity failure
    warned = json.loads(r["warnings"])
    assert {w["number"] for w in warned} >= {"3.77"} and all(w["field"] == "crew_brief" for w in warned)
    assert r["role"] == "agronomist"


def test_agronomist_free_text_beyond_the_old_fields_is_checked(db):
    bad = final_json(setpoint_caveats=["the VPD target of 0.8 kPa is a guess"], missing_evidence=["no data for 91 hours"])
    bad["recommended_actions"][0].update(action="Irrigate 777 ml", magnitude="add 4321 ml")
    bad["causes_considered"] = ["underwatering", "sensor_fault", "low_light"]
    script = happy_script(bad)
    script[0]["tool_calls"][3] = tcall("c4", "propose_action", {  # the final action must match the proposal
        "tray_id": "B", "action": "Irrigate 777 ml", "magnitude": "add 4321 ml", "rationale": "dry", "evidence_ids": [SOIL]})
    aid, _llm, _ = run_assess(db, script)
    r = row(db, aid)
    warned = {(w["field"], w["number"]) for w in json.loads(r["warnings"])}
    assert ("setpoint_caveats[0]", "0.8") in warned and ("missing_evidence[0]", "91") in warned  # a claimed gap
    assert ("recommended_actions[0].action", "777") in warned and ("recommended_actions[0].magnitude", "4321") in warned
    assert r["valid"] == 1  # still only warnings


# ---------------------------------------------------------------- crew-action claims
@pytest.mark.parametrize("text,claims", [
    ("Crew watered B", 1), ("The astronaut replaced the lamp. Soil is dry.", 1), ("B was refilled by the crew", 1),
    ("Someone has checked the cable", 1),
    ("Crew should water B", 0), ("Please have the crew check the lamp", 0), ("Crew can water B now", 0),
    ("Crew has not watered B", 0), ("If the crew watered B, recheck", 0), ("Water tray B and verify", 0),
    ("Tray B watered itself?", 0), ("", 0),
])
def test_crew_claims_both_directions(text, claims):
    assert len(g.crew_claims(text)) == claims


def test_crew_claim_backing():
    texts = {"handoff": "Crew watered B."}
    assert len(g.crew_errors(texts, backed=False)) == 1 and "unbacked crew-action claim" in g.crew_errors(texts, False)[0]
    assert g.crew_errors(texts, backed=True) == []
    assert g.crew_errors({"handoff": "Crew should water B."}, backed=False) == []


def test_agronomist_unbacked_crew_claim_is_a_warning(db):
    bad = final_json(crew_brief="Crew watered B yesterday; recheck.")
    aid, _llm, _ = run_assess(db, happy_script(bad))
    r = row(db, aid)
    warned = json.loads(r["warnings"])
    assert any(w["number"] == "unbacked crew-action claim" and w["field"] == "crew_brief" for w in warned)
    assert r["valid"] == 1


def test_crew_backed_reads_the_crew_notes_result(db):
    aid, _llm, _ = run_assess(db, happy_script())
    assert g.crew_backed(db, aid) is False
    db.execute("INSERT INTO tool_log(assessment_id, round, tool, args, returned_ids, result) VALUES (?,?,?,?,?,?)",
               (aid, 1, "get_crew_notes", "{}", "[]", json.dumps({"items": [{"id": "n1", "transcript": "watered B"}]})))
    assert g.crew_backed(db, aid) is True


@pytest.mark.parametrize("field, text, bad", [
    ("setpoint_caveats[0]", "wilt_window_h 24h is author-set", []),  # a duration setpoint the run fetched
    ("recommended_actions[0].magnitude", "5 min visual check", []),  # prescriptive: an effort, not a measurement
    ("crew_brief", "Plants are not at risk in the next 48 hours.", []),  # a forecast horizon
    ("missing_evidence[0]", "high-resolution 1-min raw air rh_pct series", []),  # a sampling resolution
    ("missing_evidence[0]", "no data for 91 hours", ["91"]),  # a claimed gap: checked
    ("findings[0].claim", "soil dry for 91 hours", ["91"]),
    ("crew_instruction", "water to 0.55 m3/m3 within 2 hours", ["0.55"]),  # plan interval skipped, target checked
    ("findings[0].claim", "a 91-hour drought", ["91"]),
])
def test_prescriptive_fields_horizons_and_resolution(field, text, bad):
    dur: list[float] = []
    vals = g.collect({"rows": [{"name": "wilt_window_h", "value": 24}]}, [], dur)
    pool = g.Pool(values=vals + [0.38], durations=dur + [17.0])
    assert [b["number"] for b in g.ungrounded({field: text}, pool)] == bad
