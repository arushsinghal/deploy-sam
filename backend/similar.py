"""Historical deployment similarity — feature #6.

Finds the nearest past deployments (that have a known outcome) to a given
deploy, in standardized feature space, and reports how many of those neighbours
actually caused an incident. This is the "this deploy resembles 3 past deploys,
2 of which caused incidents" institutional-memory signal.

Uses z-scored Euclidean distance over the model feature vector — no extra
dependencies, computed on the fly from the deployments table.
"""
from __future__ import annotations

import math

from db import connect
from scoring import meta

# Features that describe the *change*, not the label — used for similarity.
SIM_FEATURES = [
    "service_criticality_tier", "deploy_hour", "is_weekend", "lines_changed",
    "files_changed", "incidents_last_30d", "days_since_last_incident",
    "oncall_engineers_available", "is_oncall_senior", "has_rollback_plan",
    "test_coverage_delta",
]


def _labelled_rows() -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM deployments WHERE outcome IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def _stats(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Per-feature (mean, std) for standardization; std floored to avoid /0."""
    stats: dict[str, tuple[float, float]] = {}
    n = len(rows)
    for f in SIM_FEATURES:
        vals = [float(r[f]) for r in rows if r.get(f) is not None]
        if not vals:
            stats[f] = (0.0, 1.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        stats[f] = (mean, max(math.sqrt(var), 1e-6))
    return stats


def _vec(row: dict, stats: dict) -> list[float]:
    out = []
    for f in SIM_FEATURES:
        v = row.get(f)
        mean, std = stats[f]
        out.append(0.0 if v is None else (float(v) - mean) / std)
    return out


def find_similar(deployment_id: str, k: int = 4) -> dict:
    with connect() as con:
        target = con.execute(
            "SELECT * FROM deployments WHERE deployment_id=?", (deployment_id,)
        ).fetchone()
    if target is None:
        return {"neighbors": [], "incident_rate": None, "n_pool": 0}
    target = dict(target)

    pool = [r for r in _labelled_rows() if r["deployment_id"] != deployment_id]
    if not pool:
        return {"neighbors": [], "incident_rate": None, "n_pool": 0}

    stats = _stats(pool)
    tvec = _vec(target, stats)

    scored = []
    for r in pool:
        rv = _vec(r, stats)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(tvec, rv)))
        scored.append((dist, r))
    scored.sort(key=lambda x: x[0])
    top = scored[:k]

    tiers = ["Low", "Medium", "High"]
    neighbors = [
        {
            "deployment_id": r["deployment_id"],
            "service_name": r["service_name"],
            "deploy_timestamp": r["deploy_timestamp"],
            "risk_score": tiers[r["risk_tier"]],
            "risk_probability": round(r["risk_probability"], 3),
            "outcome": "incident" if r["outcome"] == 1 else "clean",
            "similarity": round(1.0 / (1.0 + dist), 3),
        }
        for dist, r in top
    ]
    inc = sum(1 for n in neighbors if n["outcome"] == "incident")
    return {
        "neighbors": neighbors,
        "incident_count": inc,
        "incident_rate": round(inc / len(neighbors), 2) if neighbors else None,
        "n_pool": len(pool),
    }
