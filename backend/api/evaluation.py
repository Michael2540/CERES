"""Static eval reports under backend/eval/out/. Neither route is mission data: both ignore `as_of` and are not time-gated.

GET /eval/area-validation: the SAM-vs-ground-truth report (real, untransformed frames); 404 until
`scripts/validate_area.py` has written it.
GET /eval/latest: the latest agent eval run (`scripts/run_eval.py` writes latest.json); 404 until one has run.
`generated_at` is wall-clock UTC, not mission time."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.eval.validate_area import OUT_PATH

router = APIRouter()
LATEST_PATH = OUT_PATH.with_name("latest.json")  # tests point this at a temp file


@router.get("/eval/latest")
def eval_latest() -> dict:
    if not LATEST_PATH.is_file():
        raise HTTPException(404, "no agent eval run yet (scripts/run_eval.py)")
    return json.loads(LATEST_PATH.read_text(encoding="utf-8"))


@router.get("/eval/area-validation")
def area_validation() -> dict:
    if not OUT_PATH.is_file():
        raise HTTPException(404, "area validation has not been run (scripts/validate_area.py)")
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))
