"""Static eval reports under backend/eval/out/. Neither route is mission data: both ignore `as_of` and are not time-gated.

GET /eval/area-validation: the SAM-vs-ground-truth report (real, untransformed frames); 404 until
`scripts/validate_area.py` has written it.
GET /eval/latest?role=agronomist|edge: the latest agent eval run (edge: latest_edge.json) (`scripts/run_eval.py` writes latest.json); 404 until one has run.
`generated_at` is wall-clock UTC, not mission time."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException

from backend.eval.validate_area import OUT_PATH

router = APIRouter()
LATEST_PATH = OUT_PATH.with_name("latest.json")  # tests point this at a temp file
LATEST_EDGE_PATH = OUT_PATH.with_name("latest_edge.json")  # `scripts/run_eval.py --role edge`


@router.get("/eval/latest")
def eval_latest(role: Literal["agronomist", "edge"] = "agronomist") -> dict:
    path = LATEST_EDGE_PATH if role == "edge" else LATEST_PATH
    if not path.is_file():
        raise HTTPException(404, f"no {role} eval run yet (scripts/run_eval.py --role {role})")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/eval/area-validation")
def area_validation() -> dict:
    if not OUT_PATH.is_file():
        raise HTTPException(404, "area validation has not been run (scripts/validate_area.py)")
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))
