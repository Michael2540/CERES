"""Deterministic audit of an on-board edge triage against the Earth-side assessment (M8). Computed in code, no model.

Edge urgency maps onto the agronomist's status scale: now -> alert, next_check -> watch, log_only -> nominal.
`urgency_delta` is the signed difference on that ordinal scale, EDGE MINUS EARTH (positive: the edge was more urgent than
Earth, negative: it under-called). `cause_match`: the edge likely_cause equals the agronomist's top cause. `agree` is
cause_match and urgency_delta == 0. `notes` is a template over these fields only.
"""

from __future__ import annotations

from backend.agent.validate import STATUS_RANK
from backend.eval.score import top_cause

URGENCY_STATUS = {"now": "alert", "next_check": "watch", "log_only": "nominal"}


def edge_status(urgency: str | None) -> str | None:
    return URGENCY_STATUS.get(urgency) if isinstance(urgency, str) else None


def _delta_text(d: int) -> str:
    if d == 0:
        return "urgency equal"
    return f"edge {abs(d)} step{'s' if abs(d) > 1 else ''} {'more' if d > 0 else 'less'} urgent than Earth"


def invalid_audit(edge: dict | None, earth: dict | None, edge_errors: list[str], earth_errors: list[str]) -> dict:
    """An audit that is NOT meaningful: the edge triage and/or the Earth assessment failed validation, so agreeing or
    disagreeing would compare an answer that was never accepted. agree, urgency_delta and cause_match are null."""
    reasons = []
    if edge_errors:
        reasons.append("edge triage invalid: " + "; ".join(e[:160] for e in edge_errors[:3]))
    if earth_errors:
        reasons.append("Earth assessment invalid: " + "; ".join(e[:160] for e in earth_errors[:3]))
    return {
        "agree": None, "urgency_delta": None, "cause_match": None,
        "edge_cause": edge.get("likely_cause") if isinstance(edge, dict) else None,
        "earth_top_cause": top_cause(earth) if isinstance(earth, dict) else None,
        "edge_urgency": edge.get("urgency") if isinstance(edge, dict) else None,
        "earth_status": earth.get("status") if isinstance(earth, dict) else None,
        "notes": "; ".join(reasons) + "; audit not meaningful",
    }


def compute_audit(edge: dict | None, earth: dict | None) -> dict:
    """The audit dict {agree, urgency_delta, cause_match, edge_cause, earth_top_cause, edge_urgency, earth_status, notes}.

    Without an edge triage or an Earth assessment there is nothing to compare: agree, urgency_delta and cause_match are
    null and the notes say which side is missing."""
    e_urg = edge.get("urgency") if isinstance(edge, dict) else None
    e_cause = edge.get("likely_cause") if isinstance(edge, dict) else None
    e_stat = edge_status(e_urg)
    e_status = earth.get("status") if isinstance(earth, dict) else None
    e_top = top_cause(earth) if isinstance(earth, dict) else None
    base = {"edge_cause": e_cause, "earth_top_cause": e_top, "edge_urgency": e_urg, "earth_status": e_status}
    if e_stat is None or e_cause is None or e_status not in STATUS_RANK or e_top is None:
        missing = [n for n, ok in (("edge triage", e_stat and e_cause), ("Earth assessment", e_status in STATUS_RANK and e_top))
                   if not ok]
        return {"agree": None, "urgency_delta": None, "cause_match": None, **base,
                "notes": "Not compared: no usable " + " or ".join(missing) + "."}
    delta = STATUS_RANK[e_stat] - STATUS_RANK[e_status]
    match = e_cause == e_top
    return {
        "agree": match and delta == 0, "urgency_delta": delta, "cause_match": match, **base,
        "notes": (
            f"Edge: {e_urg} ({e_stat}) for {e_cause}. Earth: {e_status} for {e_top}. "
            f"Cause {'matches' if match else 'differs'}; {_delta_text(delta)}."
        ),
    }
