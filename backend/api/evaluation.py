"""GET /eval/area-validation: the static SAM-vs-ground-truth report (backend/eval/out/area_validation.json).

It is a validation artefact of real, untransformed frames, not mission data, so it ignores `as_of` and is not
time-gated. 404 until `scripts/validate_area.py` has written it."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.eval.validate_area import OUT_PATH

router = APIRouter()


@router.get("/eval/area-validation")
def area_validation() -> dict:
    if not OUT_PATH.is_file():
        raise HTTPException(404, "area validation has not been run (scripts/validate_area.py)")
    return json.loads(OUT_PATH.read_text(encoding="utf-8"))
