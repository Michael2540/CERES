import json
from pathlib import Path

DIR = Path(__file__).parent / "fixtures" / "api"
FILES = sorted(DIR.glob("*.json"))


def load(name):
    return json.loads((DIR / name).read_text(encoding="utf-8"))


def known_ids() -> set[str]:
    ids = set(load("evidence.json"))
    for f in DIR.glob("series_*.json"):
        ids |= {i["id"] for i in json.loads(f.read_text(encoding="utf-8"))["items"]}
    ids |= {i["id"] for i in load("frames_B.json")["items"]}
    ids |= {i["id"] for i in load("alerts.json")["items"]}
    return ids


def cited(obj) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "evidence_ids":
                out |= set(v)
            else:
                out |= cited(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= cited(v)
    return out


def test_all_valid_json():
    assert len(FILES) >= 15
    for f in FILES:
        json.loads(f.read_text(encoding="utf-8"))


def test_assessment_evidence_resolves():
    asmt = load("assessment.json")
    ids = cited(asmt)
    assert ids
    assert ids <= known_ids(), ids - known_ids()


def test_stream_ids_resolve_and_match_assessment():
    stream = load("assessment_stream.json")
    returned = set()
    for ev in stream:
        if ev["event"] == "tool_result":
            returned |= set(ev["data"]["ids"])
    assert returned <= known_ids()
    asmt = load("assessment.json")
    # every cited id is in the tool window (citation validity)
    assert cited(asmt["assessment"]) <= returned | {asmt["alert_id"]}
    assert stream[-1]["event"] == "assessment"
    assert stream[-1]["data"]["assessment_id"] == asmt["assessment_id"]


def test_provenance_enums():
    for f in DIR.glob("series_*.json"):
        for i in json.loads(f.read_text(encoding="utf-8"))["items"]:
            assert i["source"] in {"sensor", "replay", "simulated", "derived"}
    for i in load("frames_B.json")["items"]:
        assert i["source"] in {"dataset", "fault_transform", "fault_timewarp"} and i["calib_flag"]
