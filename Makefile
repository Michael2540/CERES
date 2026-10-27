ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

.PHONY: dev test lint e2e demo
dev:
	@echo "Web: run 'cd web && npm run dev' in another terminal (http://localhost:3000)"
	$(PY) -m uvicorn backend.app:app --reload --port 8000
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
e2e:
	cd web && npx playwright test
demo:
	@echo "demo: not implemented yet (M9)"
