"""FastAPI entry point: uvicorn backend.app:app"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import load_settings

app = FastAPI(title="CERES")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    s = load_settings()
    return {"status": "ok", "demo_mode": s.demo_mode, "edge_mode": s.edge_mode}
