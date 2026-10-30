"""Read API: response shapes match tests/fixtures/api/*.json, as_of hides the future, evidence resolves."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_pipeline import AS_OF, _reading, build, cycle

from backend.app import app
from backend.db import init_db
from backend.science.vpd import vpd_kpa
from backend.sim.clock import CLOCK, MissionClock
from backend.spend import record_spend

FIX = Path(__file__).parent / "fixtures" / "api"


def fixture(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def shape_errors(actual, expected, path="$") -> list[str]:
    """Keys and JSON types must agree; a null on either side matches anything; lists compare every actual
    item against the first expected item; ints and floats are one 'number' type."""
    if actual is None or expected is None:
        return []
    num = (int, float)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if isinstance(actual, bool) and isinstance(expected, bool) else [f"{path}: bool mismatch"]
    if isinstance(expected, num):
        return [] if isinstance(actual, num) else [f"{path}: want number, got {type(actual).__name__}"]
    if isinstance(expected, str):
        return [] if isinstance(actual, str) else [f"{path}: want str, got {type(actual).__name__}"]
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: want list"]
        return [e for a in actual for e in shape_errors(a, expected[0], path + "[]")] if expected else []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: want object"]
        errs = [f"{path}: keys {sorted(actual)} != {sorted(expected)}"] if set(actual) != set(expected) else []
        for k in set(actual) & set(expected):
            errs += shape_errors(actual[k], expected[k], f"{path}.{k}")
        return errs
    return [f"{path}: unsupported fixture type"]


def test_shape_comparator_catches_differences():
    assert shape_errors({"a": 1}, {"a": 2.5}) == []
    assert shape_errors({"a": 1, "b": 2}, {"a": 1})
    assert shape_errors({"a": "x"}, {"a": 1})
    assert shape_errors({"a": None}, {"a": 1}) == []
    assert shape_errors({"items": [{"a": 1}, {"a": "x"}]}, {"items": [{"a": 1}]})


@pytest.fixture
def api(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    conn = init_db(db)
    (tmp_path / "imgs").mkdir()
    build(conn, tmp_path / "imgs")
    for m in range(0, 180, 10):  # tray L has its own climate under the dome: warm and very humid
        ts = f"2016-01-06T{11 + m // 60:02d}:{m % 60:02d}:00Z"
        _reading(conn, "L", "temp_c", ts, 24.0 + (m % 20) / 100)
        _reading(conn, "L", "rh_pct", ts, 90.0 + (m % 30) / 10)
    cycle(conn, tmp_path)
    record_spend(conn, "sam", "meta", "sam-3.1", images=2)
    conn.execute(
        "INSERT INTO actions(id, ts, tray_id, assessment_id, proposal_json, status) VALUES ('act1', ?, 'B', NULL, '{}', 'pending')",
        (AS_OF,),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("DB_PATH", str(db))
    CLOCK.set(as_of=AS_OF, speed=720, running=False)
    yield TestClient(app)
    CLOCK.set(as_of="2016-01-03T12:00:00Z", speed=720, running=False)


def get(c, url, **params):
    r = c.get(url, params=params) if params else c.get(url)
    assert r.status_code == 200, (url, r.status_code, r.text)
    return r.json()


def test_endpoint_shapes_match_fixtures(api):
    checks = {
        "/clock": "clock.json",
        "/spend": "spend.json",
        "/trays": "trays.json",
        "/trays/B": "tray_B.json",
        "/trays/B/frames": "frames_B.json",
        "/alerts": "alerts.json",
        "/trays/B/series?metric=area_cm2": "series_B_area_cm2.json",
        "/trays/B/series?metric=soil_vwc": "series_B_soil_vwc.json",
        "/trays/B/series?metric=chlorosis_frac": "series_B_chlorosis_frac.json",
        "/trays/air/series?metric=rh_pct": "series_air_rh_pct.json",
    }
    for url, name in checks.items():
        body = get(api, url)
        errs = shape_errors(body, fixture(name))
        if name == "spend.json":  # by_role is keyed by free-form role names
            errs = [e for e in errs if "by_role" not in e]
        assert not errs, (url, errs)
    assert get(api, "/trays/B/series?metric=area_cm2")["items"], "series must not be empty for the check"
    assert get(api, "/alerts")["items"] and get(api, "/trays/B")["open_alerts"]
    fx = fixture("evidence.json")  # one resolved id per kind, compared with the fixture entry of that kind
    ids = {
        "frame": f"B:{AS_OF}",
        "reading": f"B:soil_vwc:{AS_OF}",
        "metric": f"B:area_cm2:{AS_OF}",
        "alert": "alert_B_WILT_RISK_20160106T140000Z",
    }
    for kind, eid in ids.items():
        want = next(v for v in fx.values() if v["kind"] == kind)
        got = get(api, f"/evidence/{eid}")
        assert got["kind"] == kind and not shape_errors(got, want), (kind, got)


def test_trays_summary_and_detail(api):
    trays = {t["id"]: t for t in get(api, "/trays")["items"]}
    assert set(trays) == {"A", "B", "C", "L"}
    assert trays["B"]["open_alerts"] == 1 and trays["B"]["status"] == "alert"
    assert trays["A"]["open_alerts"] == 0 and trays["A"]["status"] == "nominal"
    assert trays["B"]["latest_frame_id"] == f"B:{AS_OF}" and trays["C"]["latest_frame_id"] is None
    b = get(api, "/trays/B")
    assert b["mission_day"] == 24 and b["age_days"] == 23 and b["latest"]["ts"] == AS_OF
    assert b["latest"]["area_px"] > 0 and b["latest"]["soil_vwc"] is not None
    assert b["latest"]["vpd_kpa"] == pytest.approx(vpd_kpa(b["latest"]["temp_c"], b["latest"]["rh_pct"]))
    assert api.get("/trays/Z").status_code == 404
    assert get(api, "/trays/B", as_of="2016-01-06T17:00:00Z")["open_alerts"]  # within the 14 h plant episode gap
    late = "2016-01-07T05:00:00Z"  # 15 h on: status is judged at as_of, whatever the last cycle stored
    assert get(api, "/trays/B", as_of=late)["open_alerts"] == []
    assert {a["status"] for a in get(api, "/alerts", as_of=late)["items"]} == {"resolved"}


def test_as_of_hides_the_future(api):
    early = "2016-01-06T11:00:00Z"
    frames = get(api, "/trays/B/frames", as_of=early)["items"]
    assert [f["ts"] for f in frames] == ["2016-01-06T11:00:00Z", "2016-01-06T10:00:00Z", "2016-01-06T09:00:00Z"]
    sensor = get(api, "/trays/B/series", metric="soil_vwc", hours=500, as_of=early)["items"]
    assert sensor and max(p["ts"] for p in sensor) <= early
    canopy = get(api, "/trays/B/series", metric="area_cm2", hours=500, as_of=early)["items"]
    assert max(p["ts"] for p in canopy) == early
    assert get(api, "/trays/B", as_of=early)["latest"]["ts"] == early
    assert get(api, "/alerts", as_of=early)["items"] == []  # the alerts were raised at 14:00
    assert get(api, "/trays", as_of=early)["items"][1]["open_alerts"] == 0
    assert api.get(f"/evidence/B:{AS_OF}", params={"as_of": early}).status_code == 404
    assert api.get(f"/evidence/B:soil_vwc:{AS_OF}", params={"as_of": early}).status_code == 404
    assert get(api, f"/evidence/B:soil_vwc:{AS_OF}")["kind"] == "reading"
    # the default is the mission clock
    CLOCK.set(as_of=early)
    assert get(api, "/trays/B/frames")["items"][0]["ts"] == early
    assert api.get("/trays/B/series", params={"metric": "temp_c", "as_of": "garbage"}).status_code == 422


def test_frames_pagination(api):
    p1 = get(api, "/trays/B/frames", limit=4)
    assert [f["ts"][11:13] for f in p1["items"]] == ["14", "13", "12", "11"] and p1["next_cursor"] == p1["items"][-1]["ts"]
    p2 = get(api, "/trays/B/frames", limit=4, cursor=p1["next_cursor"])
    assert [f["ts"][11:13] for f in p2["items"]] == ["10", "09"] and p2["next_cursor"] is None
    f = p1["items"][0]
    assert f["image_url"] == f"/trays/B/frames/{f['id']}/image" and f["source"] == "dataset"


def test_series_vpd_matches_vpd_module(api):
    air = get(api, "/trays/air/series", metric="vpd_kpa", hours=6)["items"]
    t = {p["ts"]: p["value"] for p in get(api, "/trays/air/series", metric="temp_c", hours=6)["items"]}
    rh = {p["ts"]: p["value"] for p in get(api, "/trays/air/series", metric="rh_pct", hours=6)["items"]}
    assert len(air) == len(rh) > 10
    for p in air:
        assert p["value"] == pytest.approx(vpd_kpa(t[p["ts"]], rh[p["ts"]])) and p["source"] == "derived"
        assert p["id"] == f"air:vpd_kpa:{p['ts']}"
    low = get(api, "/trays/L/series", metric="vpd_kpa", hours=6)["items"]
    assert low and all(p["value"] < 0.4 for p in low)  # L uses its own climate sensor
    assert low[0]["value"] == pytest.approx(vpd_kpa(24.0, 90.0))
    assert all(p["value"] > 0.5 for p in get(api, "/trays/B/series", metric="vpd_kpa", hours=6)["items"])
    assert api.get("/trays/B/series", params={"metric": "nope"}).status_code == 422


def test_canopy_series_values_and_ids(api):
    pts = get(api, "/trays/B/series", metric="area_px", hours=24)["items"]
    assert pts[0]["id"] == "B:area_px:2016-01-06T09:00:00Z"
    assert pts[0]["value"] > pts[-1]["value"] > 0  # shrinking
    gr = get(api, "/trays/B/series", metric="growth_rate", hours=24)["items"]
    assert gr and all(p["value"] < 0 for p in gr)
    ev = get(api, f"/evidence/{gr[-1]['id']}")
    assert ev["kind"] == "metric" and ev["value"] == pytest.approx(gr[-1]["value"]) and ev["frame_id"] == f"B:{gr[-1]['ts']}"


def test_every_alert_evidence_id_resolves(api):
    alerts = get(api, "/alerts")["items"]
    assert {(a["tray_id"], a["rule"]) for a in alerts} == {("B", "WILT_RISK"), ("air", "SENSOR_FAULT")}
    for a in alerts:
        assert a["evidence_ids"] and a["status"] == "open" and a["assessment_id"] is None
        for eid in a["evidence_ids"]:
            e = get(api, f"/evidence/{eid}")
            assert e["id"] == eid and e["kind"] in {"reading", "metric"}
    assert [a["rule"] for a in get(api, "/alerts", tray_id="B")["items"]] == ["WILT_RISK"]
    assert get(api, "/alerts", since="2016-01-07T00:00:00Z")["items"] == []
    assert api.get("/evidence/nope").status_code == 404


def test_actions_change_status(api):
    assert api.post("/actions/act1", json={"decision": "approve"}).json() == {"action_id": "act1", "status": "approved"}
    assert api.post("/actions/act1", json={"decision": "reject"}).status_code == 409  # only a pending action can be decided
    assert api.post("/actions/act1", json={"decision": "maybe"}).status_code == 422
    assert api.post("/actions/nope", json={"decision": "approve"}).status_code == 404


def test_spend_endpoint(api):
    s = get(api, "/spend")
    assert s["total_usd"] == pytest.approx(0.005) and s["by_role"] == {"sam": pytest.approx(0.005)}
    assert s["budget_usd"] > 0


def test_clock_endpoints(api):
    c = get(api, "/clock")
    assert c == {"as_of": AS_OF, "speed": 720, "mission_day": 24, "running": False}
    r = api.post("/clock", json={"as_of": "2016-01-01T06:00:00Z", "speed": 3600}).json()
    assert r["as_of"] == "2016-01-01T06:00:00Z" and r["speed"] == 3600 and r["mission_day"] == 19
    assert get(api, "/clock") == r
    api.post("/clock", json={"speed": 1_000_000, "running": True})
    time.sleep(0.05)
    moved = get(api, "/clock")
    assert moved["running"] and moved["as_of"] > "2016-01-01T06:00:00Z"
    api.post("/clock", json={"running": False})
    assert api.post("/clock", json={"speed": 0}).status_code == 422
    assert api.post("/clock", json={"as_of": "x"}).status_code == 422


def test_mission_clock_unit():
    now = [100.0]
    c = MissionClock(as_of=datetime(2016, 1, 3, 12, tzinfo=UTC), speed=720, time_fn=lambda: now[0])
    assert c.state() == {"as_of": "2016-01-03T12:00:00Z", "speed": 720.0, "mission_day": 21, "running": False}
    now[0] += 50
    assert c.now() == datetime(2016, 1, 3, 12, tzinfo=UTC)  # stopped: no movement
    c.set(running=True)
    now[0] += 5
    assert c.now() == datetime(2016, 1, 3, 13, tzinfo=UTC)  # 1 mission hour per 5 s
    c.set(speed=3600)
    now[0] += 3600  # 3600 s * 3600 = way past the end
    assert c.now() == c.end and not c.running  # clamped, and the replay stops
    c.set(as_of="2015-12-01T00:00:00Z")
    assert c.now() == c.start and c.mission_day == 1


async def test_background_loop_runs_one_cycle_per_new_mission_hour(monkeypatch):
    import asyncio

    from backend import app as app_mod

    seen: list[str] = []
    monkeypatch.setattr(app_mod, "cycle_once", lambda t: seen.append(t.isoformat()))
    now = [0.0]
    clock = MissionClock(as_of=datetime(2016, 1, 3, 12, 20, tzinfo=UTC), speed=3600, running=True, time_fn=lambda: now[0])
    task = asyncio.create_task(app_mod.background_loop(clock, poll_s=0.01))
    await asyncio.sleep(0.05)
    assert seen == ["2016-01-03T12:00:00+00:00"]  # first run: the current hour only
    now[0] += 3.0  # three more mission hours
    await asyncio.sleep(0.05)
    task.cancel()
    assert [s[11:13] for s in seen] == ["12", "13", "14", "15"]


def test_frame_pixels_honour_as_of(api):
    fid = f"B:{AS_OF}"
    early = "2016-01-06T11:00:00Z"
    for kind in ("image", "overlay"):
        url = f"/trays/B/frames/{fid}/{kind}"
        assert api.get(url).status_code == 200
        assert api.get(url, params={"as_of": early}).status_code == 404
        assert api.get(f"/trays/B/frames/B:{early}/{kind}", params={"as_of": early}).status_code == 200


@pytest.mark.parametrize("speed", [float("nan"), float("inf"), 1e308, -5, 0, 1e6 + 1])
def test_clock_rejects_bad_speed_and_keeps_working(api, speed):
    # NaN / inf are not valid JSON numbers: send the raw tokens the way a sloppy client would
    raw = {float("nan"): "NaN", float("inf"): "Infinity"}.get(speed, repr(speed))
    r = api.post("/clock", content=f'{{"speed": {raw}, "running": true}}', headers={"content-type": "application/json"})
    assert r.status_code == 422, (speed, r.text)
    assert get(api, "/clock")["as_of"] == AS_OF and not get(api, "/clock")["running"]
    assert get(api, "/trays")["items"]


def test_clock_unit_validation_and_overflow():
    c = MissionClock(speed=720, running=True)
    for bad in (float("nan"), float("inf"), 1e308, -1.0, 0.0):
        with pytest.raises(ValueError):
            c.set(speed=bad)
    with pytest.raises(ValueError):
        c.set(as_of="garbage")
    assert c.speed == 720.0
    now = [0.0]
    c = MissionClock(speed=720, running=True, time_fn=lambda: now[0])
    c._speed = 1e308  # even a corrupt speed cannot wedge _advance
    now[0] = 1e9
    assert c.now() == c.end and not c.running


def test_as_of_is_clamped_to_the_dataset_span(api):
    assert api.get("/trays", params={"as_of": "2099-01-01T00:00:00Z"}).status_code == 200
    assert api.get("/trays", params={"as_of": "0001-01-01T00:00:00"}).status_code == 200
    assert api.get("/trays", params={"as_of": "9999-12-31T23:59:59Z"}).status_code == 200
    assert api.get("/trays", params={"as_of": "not a date"}).status_code == 422
    assert api.post("/clock", json={"as_of": "2099-01-01T00:00:00Z"}).json()["as_of"] == "2016-01-18T23:59:59Z"


def test_canopy_provenance_follows_frame_source(api):
    import os
    import sqlite3

    db = sqlite3.connect(os.environ["DB_PATH"])
    db.execute("UPDATE frames SET source = 'fault_transform' WHERE tray_id = 'B' AND ts >= '2016-01-06T12:00:00Z'")
    db.commit()
    db.close()
    pts = {p["ts"]: p["source"] for p in get(api, "/trays/B/series", metric="area_cm2", hours=24)["items"]}
    assert pts["2016-01-06T11:00:00Z"] == "dataset" and pts["2016-01-06T12:00:00Z"] == "fault_transform"
    a = get(api, "/trays/A/series", metric="area_px", hours=24)["items"]
    assert {p["source"] for p in a} == {"dataset"}
    assert get(api, f"/evidence/B:area_cm2:{AS_OF}")["source"] == "fault_transform"
    assert get(api, "/evidence/B:area_cm2:2016-01-06T11:00:00Z")["source"] == "dataset"
    assert {p["source"] for p in get(api, "/trays/B/series", metric="vpd_kpa", hours=6)["items"]} == {"derived"}


def test_climate_series_fall_back_to_shared_air(api):
    for tray in ("A", "B", "C"):
        for m in ("temp_c", "rh_pct", "vpd_kpa"):
            body = get(api, f"/trays/{tray}/series", metric=m, hours=6)
            assert body["sensor_tray"] == "air" and body["items"], (tray, m)
    b = get(api, "/trays/B/series", metric="rh_pct", hours=6)["items"]
    assert {p["id"][:3] for p in b} == {"air"} and {p["source"] for p in b} == {"simulated"}
    assert [p["value"] for p in b] == [p["value"] for p in get(api, "/trays/air/series", metric="rh_pct", hours=6)["items"]]
    assert "sensor_tray" not in get(api, "/trays/air/series", metric="rh_pct", hours=6)
    assert "sensor_tray" not in get(api, "/trays/L/series", metric="rh_pct", hours=6)
    assert get(api, "/trays/L/series", metric="rh_pct", hours=6)["items"][0]["id"].startswith("L:")
    assert "sensor_tray" not in get(api, "/trays/B/series", metric="soil_vwc", hours=6)


def test_shared_alerts_on_every_tray(api):
    items = get(api, "/trays")["items"]
    for t in items:
        assert [a["rule"] for a in t["shared_alerts"]] == ["SENSOR_FAULT"] and t["shared_alerts"][0]["tray_id"] == "air"
    assert get(api, "/trays/A")["shared_alerts"] == get(api, "/alerts", tray_id="air")["items"]
    assert get(api, "/trays/B")["open_alerts"][0]["rule"] == "WILT_RISK"  # own alerts stay separate
    early = "2016-01-06T11:00:00Z"
    assert get(api, "/trays/A", as_of=early)["shared_alerts"] == []
    assert all(t["shared_alerts"] == [] for t in get(api, "/trays", as_of="2016-01-06T18:00:00Z")["items"])


async def test_background_loop_survives_a_failing_hour(monkeypatch):
    import asyncio

    from backend import app as app_mod

    seen: list[int] = []

    def flaky(t):
        seen.append(t.hour)
        if t.hour == 13:
            raise RuntimeError("boom")

    monkeypatch.setattr(app_mod, "cycle_once", flaky)
    now = [0.0]
    clock = MissionClock(as_of=datetime(2016, 1, 3, 12, 20, tzinfo=UTC), speed=3600, running=True, time_fn=lambda: now[0])
    task = asyncio.create_task(app_mod.background_loop(clock, poll_s=0.01))
    await asyncio.sleep(0.05)
    now[0] += 3.0
    await asyncio.sleep(0.1)
    task.cancel()
    assert seen == [12, 13, 14, 15]  # 13 failed once, was not retried, and did not stop 14 and 15


def test_catchup_cap_covers_forty_days():
    from backend import app as app_mod

    assert app_mod.MAX_CATCHUP_H == 24 * 40


def test_fault_start_is_hidden_before_it_happens(api):
    cfg = {t["id"]: t["fault_start"] for t in get(api, "/trays")["items"]}
    assert cfg["B"] is not None and cfg["A"] is None
    early = "2015-12-20T00:00:00Z"
    assert all(t["fault_start"] is None for t in get(api, "/trays", as_of=early)["items"])
    assert get(api, "/trays/B", as_of=early)["fault_start"] is None
    assert get(api, "/trays/B")["fault_start"] == cfg["B"]


def test_cors_allows_dashboard_origins(api):
    for origin in ("http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3100"):
        r = api.get("/health", headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") == origin
    r = api.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
