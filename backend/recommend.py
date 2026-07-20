"""Rule-based rollout strategy recommender.

Maps the risk score + dominant risk drivers to a concrete, defensible ship plan:
strategy, canary %, on-call requirement, and a safer suggested window. This is
the 'how to ship safely' layer that turns a score into an action.
"""
from __future__ import annotations

from datetime import datetime, timedelta


def _safer_window(deploy_hour: int, is_weekend: int) -> str:
    """Nearest low-risk window: a weekday mid-morning."""
    if not is_weekend and 9 <= deploy_hour <= 15:
        return "current window is already low-risk (weekday 9am-3pm)"
    return "Tue-Thu 10:00-14:00 (peak on-call, low traffic ramp)"


def recommend(scored: dict, feats: dict) -> dict:
    prob = scored["risk_probability"]
    tier = scored["risk_score"]
    top = {f["feature"]: f for f in scored["factors"]}
    drivers = [f["feature"] for f in scored["factors"][:3] if f["shap"] > 0]

    if tier == "High":
        strategy, canary = "canary", 5
    elif tier == "Medium":
        strategy, canary = "canary", 25
    else:
        strategy, canary = "blue-green", 100

    # Escalate protection when specific risk drivers dominate.
    notes = []
    if "lines_changed" in drivers and (feats.get("lines_changed") or 0) > 500:
        strategy = "canary"
        canary = min(canary, 10)
        notes.append("large changeset -> tighten canary and watch error budget for 30m")
    if "oncall_engineers_available" in drivers or (feats.get("oncall_engineers_available") or 2) <= 1:
        notes.append("thin on-call coverage -> require a second on-call engineer before shipping")
    if not feats.get("has_rollback_plan"):
        notes.append("no rollback plan attached -> block until a rollback runbook is linked")
    if "deploy_hour" in drivers:
        notes.append("off-hours risk is a top driver -> prefer the suggested window")

    require_approval = tier == "High"
    require_second_oncall = (feats.get("oncall_engineers_available") or 2) <= 1 and tier != "Low"

    return {
        "strategy": strategy,
        "canary_percent": canary,
        "suggested_window": _safer_window(int(feats.get("deploy_hour", 12)), int(feats.get("is_weekend", 0))),
        "require_senior_approval": require_approval,
        "require_second_oncall": require_second_oncall,
        "gate_decision": "BLOCK" if require_approval else ("WARN" if tier == "Medium" else "PASS"),
        "notes": notes,
    }
