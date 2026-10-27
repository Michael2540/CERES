from backend.config import load_settings, load_trays


def test_trays():
    cfg = load_trays()
    assert [t.id for t in cfg.trays] == ["A", "B", "C", "L"]
    assert cfg.by_id("B").reference_tray_id == "A"
    assert cfg.by_id("C").reference_tray_id == "A"
    assert cfg.by_id("L").reference_tray_id is None
    assert cfg.by_id("A").fault_start is None
    assert cfg.by_id("B").fault.shrink_per_day and cfg.by_id("B").fault.desat_per_day
    assert cfg.by_id("C").fault.growth_factor == 0.4 and cfg.by_id("C").fault.brightness_factor == 0.6
    assert cfg.by_id("L").fault.rh_target_pct == 92
    assert all(t.crop == "arabidopsis_thaliana" for t in cfg.trays)


def test_settings_defaults(monkeypatch):
    for k in ("EDGE_MODE", "CERES_BUDGET_USD", "DEMO_MODE", "SAM_BACKEND", "DB_PATH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("backend.config.load_dotenv", lambda *a, **k: None)
    s = load_settings()
    assert s.edge_mode == "recorded" and s.budget_usd == 100 and not s.demo_mode
    assert s.sam_backend == "api" and s.db_path == "data/ceres.db"
