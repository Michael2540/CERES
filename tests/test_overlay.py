import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.api import deps
from backend.app import app
from backend.config import ROOT
from backend.db import connect, init_db
from backend.perception.metrics import rle_encode


def test_overlay_and_image(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    conn = init_db(db)
    img = np.full((60, 80, 3), 90, np.uint8)
    p = ROOT / "data" / "derived" / "_test_overlay.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    try:
        conn.execute(
            "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source) VALUES ('A:2016-01-03T01:00:00Z','A','2016-01-03T01:00:00Z',?,0.01,'nominal','dataset')",
            ("data/derived/_test_overlay.jpg",),
        )
        m = np.zeros((60, 80), bool)
        m[10:30, 10:40] = True
        conn.execute(
            "INSERT INTO masks(id, frame_id, tray_id, kind, rle) VALUES ('m1','A:2016-01-03T01:00:00Z','A','canopy',?)",
            (json.dumps(rle_encode(m)),),
        )
        conn.execute(
            "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source) VALUES ('A:2016-01-03T02:00:00Z','A','2016-01-03T02:00:00Z',?,0.01,'nominal','dataset')",
            ("data/derived/_test_overlay.jpg",),
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr(deps, "open_db", lambda: connect(db))
        c = TestClient(app)
        r = c.get("/trays/A/frames/A:2016-01-03T01:00:00Z/overlay")
        assert (
            r.status_code == 200
            and r.headers["content-type"] == "image/png"
            and r.content[:8] == b"\x89PNG\r\n\x1a\n"
        )
        out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert (
            out.shape == img.shape
            and (out[20, 20] != img[20, 20]).any()
            and (out[50, 70] == img[50, 70]).all()
        )
        assert c.get("/trays/A/frames/A:2016-01-03T01:00:00Z/image").headers["content-type"] == "image/jpeg"
        assert c.get("/trays/A/frames/A:2016-01-03T02:00:00Z/overlay").status_code == 404  # no mask yet
        assert c.get("/trays/A/frames/nope/image").status_code == 404
        assert c.get("/trays/B/frames/A:2016-01-03T01:00:00Z/image").status_code == 404
    finally:
        p.unlink(missing_ok=True)
