"""Feature engineering + XGBoost inference + exact tree-SHAP explanations."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta

import numpy as np
import xgboost as xgb

from db import connect

HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, "..", "ml", "artifacts")

_booster: xgb.Booster | None = None
_meta: dict | None = None

# Friendly labels for the UI / NL summaries.
LABELS = {
    "service_criticality_tier": "service criticality",
    "deploy_hour": "deploy hour",
    "is_weekend": "weekend deploy",
    "lines_changed": "lines changed",
    "files_changed": "files changed",
    "incidents_last_30d": "recent incidents (30d)",
    "days_since_last_incident": "days since last incident",
    "oncall_engineers_available": "on-call engineers",
    "is_oncall_senior": "senior on-call",
    "has_rollback_plan": "rollback plan",
    "test_coverage_delta": "test coverage delta",
}


def load() -> None:
    global _booster, _meta
    if _booster is not None:
        return
    _booster = xgb.Booster()
    _booster.load_model(os.path.join(ART, "model.json"))
    with open(os.path.join(ART, "metadata.json")) as f:
        _meta = json.load(f)


def meta() -> dict:
    load()
    assert _meta is not None
    return _meta


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def tier_of(p: float) -> tuple[int, str]:
    t = meta()["thresholds"]
    if p < t["low_max"]:
        return 0, "Low"
    if p < t["high_min"]:
        return 1, "Medium"
    return 2, "High"


def service_tier(service_name: str) -> int:
    with connect() as con:
        row = con.execute(
            "SELECT criticality_tier FROM services WHERE service_name=?", (service_name,)
        ).fetchone()
    return int(row["criticality_tier"]) if row else 2


def _naive(s: str) -> datetime:
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d.replace(tzinfo=None) if d.tzinfo is not None else d


def _recent_incident_stats(service_name: str, ts: datetime) -> tuple[int, float]:
    """incidents_last_30d and days_since_last_incident from history before `ts`."""
    window_start = (ts - timedelta(days=30)).isoformat()
    with connect() as con:
        rows = con.execute(
            """SELECT deploy_timestamp FROM deployments
               WHERE service_name=? AND outcome=1 AND deploy_timestamp < ?
               ORDER BY deploy_timestamp DESC""",
            (service_name, ts.isoformat()),
        ).fetchall()
    if not rows:
        return 0, 90.0
    last = _naive(rows[0]["deploy_timestamp"])
    days_since = max(0.0, (ts - last).total_seconds() / 86400.0)
    count_30d = sum(1 for r in rows if r["deploy_timestamp"] >= window_start)
    return count_30d, round(days_since, 1)


def build_features(req: dict) -> dict:
    """Assemble the model feature vector from a scoring request + DB lookups.

    Any feature can be overridden directly in the request (used by the what-if
    simulator); otherwise it is derived from the timestamp and history.
    """
    ts = _naive(req["deploy_timestamp"])

    def pick(key, default):
        # Pydantic fills unset optionals with None, so `.get(k, default)` is not
        # enough — treat an explicit None as "not provided".
        v = req.get(key)
        return default if v is None else v

    inc30, days_since = _recent_incident_stats(req["service_name"], ts)
    feats = {
        "service_criticality_tier": pick("service_criticality_tier", service_tier(req["service_name"])),
        "deploy_hour": pick("deploy_hour", ts.hour),
        "is_weekend": pick("is_weekend", 1 if ts.weekday() >= 5 else 0),
        "lines_changed": req.get("lines_changed"),
        "files_changed": req.get("files_changed"),
        "incidents_last_30d": pick("incidents_last_30d", inc30),
        "days_since_last_incident": pick("days_since_last_incident", days_since),
        "oncall_engineers_available": pick("oncall_engineers_available", 2),
        "is_oncall_senior": pick("is_oncall_senior", 0),
        "has_rollback_plan": int(bool(pick("has_rollback_plan", False))),
        "test_coverage_delta": pick("test_coverage_delta", 0.0),
    }
    return feats


def score(feats: dict) -> dict:
    """Return probability, tier, and exact per-feature SHAP contributions."""
    load()
    assert _booster is not None and _meta is not None
    order = _meta["features"]
    row = np.array([[np.nan if feats.get(f) is None else float(feats[f]) for f in order]], dtype=float)
    dm = xgb.DMatrix(row, feature_names=order)
    it = (0, _meta["best_iteration"] + 1)

    prob = float(_booster.predict(dm, iteration_range=it)[0])
    contribs = _booster.predict(dm, pred_contribs=True, iteration_range=it)[0]
    base = float(contribs[-1])  # log-odds base value

    factors = []
    for i, f in enumerate(order):
        factors.append({
            "feature": f,
            "label": LABELS.get(f, f),
            "value": None if feats.get(f) is None else feats[f],
            "shap": round(float(contribs[i]), 4),  # log-odds contribution
        })
    factors.sort(key=lambda x: abs(x["shap"]), reverse=True)

    t_idx, t_name = tier_of(prob)
    return {
        "risk_probability": round(prob, 4),
        "risk_tier": t_idx,
        "risk_score": t_name,
        "base_value": round(_sigmoid(base), 4),
        "base_logodds": round(base, 4),
        "factors": factors,
    }
