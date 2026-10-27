from fastapi.testclient import TestClient

from backend.app import app


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["demo_mode"], bool)
    assert body["edge_mode"] in {"recorded", "remote_glimmer", "spark_contributor"}
