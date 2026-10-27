from backend.db import init_db

TABLES = {
    "trays", "frames", "plants", "masks", "embeddings", "sensor_readings", "alerts", "assessments",
    "tool_log", "crew_notes", "actions", "link_queue", "spend",
}


def test_schema(tmp_path):
    path = tmp_path / "x.db"
    conn = init_db(path)
    init_db(path)  # idempotent
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= names
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(frames)")}
    assert "source" in cols
    spend = {r["name"] for r in conn.execute("PRAGMA table_info(spend)")}
    assert {"role", "provider", "model", "input_tokens", "output_tokens", "images", "cost_usd"} <= spend
