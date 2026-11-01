ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

.PHONY: dev test lint e2e demo demo-seed demo-backend
dev:
	@echo "Web: run 'cd web && npm run dev' in another terminal (http://localhost:3000)"
	$(PY) -m uvicorn backend.app:app --reload --port 8000
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
e2e:
	cd web && npx playwright test
# Offline demo: no network, no META_API_KEY, no paid calls. Every model result is replayed from data/demo and labelled RECORDED.
# (The very first run needs `npm install` once, which is the only network use.)
demo-seed:
	$(PY) scripts/seed_demo.py
demo-backend:
	DB_PATH=data/demo.db DEMO_MODE=1 EDGE_MODE=recorded IMAGES_ROOT=data/demo/frames $(PY) -m uvicorn backend.app:app --port 8000
demo: demo-seed
	@echo "CERES demo: backend http://localhost:8000, dashboard http://localhost:3000 (Ctrl+C stops both)"
	@DB_PATH=data/demo.db DEMO_MODE=1 EDGE_MODE=recorded IMAGES_ROOT=data/demo/frames $(PY) -m uvicorn backend.app:app --port 8000 & 	BACK=$$!; trap 'kill $$BACK 2>/dev/null' EXIT INT TERM; 	cd web && npm install --no-audit --no-fund && npm run dev
