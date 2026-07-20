"""DeployIQ API — FastAPI.

Endpoints: /score /explain /recommend /simulate /history /trends /metrics
Serves the single-file dashboard at /.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ml"))  # for generate_data.SERVICES in seeding

import db  # noqa: E402
import nl  # noqa: E402
import recommend as rec  # noqa: E402
import scoring  # noqa: E402
import similar  # noqa: E402

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    scoring.load()
    yield


app = FastAPI(title="DeployIQ", version="1.0",
              description="Deployment Risk Intelligence", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ScoreRequest(BaseModel):
    service_name: str
    commit_sha: str | None = None
    author: str | None = None
    deploy_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    lines_changed: int
    files_changed: int
    has_rollback_plan: bool = False
    test_coverage_delta: float | None = None
    oncall_engineers_available: int | None = None
    is_oncall_senior: int | None = None
    # optional direct overrides (what-if)
    deploy_hour: int | None = None
    is_weekend: int | None = None
    incidents_last_30d: int | None = None
    days_since_last_incident: float | None = None
    persist: bool = True


class SimulateRequest(BaseModel):
    deployment_id: str
    overrides: dict


def _score_and_store(req: ScoreRequest) -> dict:
    payload = req.model_dump()
    feats = scoring.build_features(payload)
    scored = scoring.score(feats)
    recommendation = rec.recommend(scored, feats)
    summary = nl.explain(scored)

    dep_id = db.new_id()
    if req.persist:
        with db.connect() as con:
            con.execute(
                """INSERT INTO deployments(
                    deployment_id, service_name, commit_sha, author, deploy_timestamp,
                    deploy_hour, is_weekend, lines_changed, files_changed, incidents_last_30d,
                    days_since_last_incident, oncall_engineers_available, is_oncall_senior,
                    has_rollback_plan, test_coverage_delta, service_criticality_tier,
                    risk_probability, risk_tier, outcome
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    dep_id, req.service_name, req.commit_sha, req.author, req.deploy_timestamp,
                    feats["deploy_hour"], feats["is_weekend"], feats["lines_changed"],
                    feats["files_changed"], feats["incidents_last_30d"],
                    feats["days_since_last_incident"], feats["oncall_engineers_available"],
                    feats["is_oncall_senior"], feats["has_rollback_plan"],
                    feats["test_coverage_delta"], feats["service_criticality_tier"],
                    scored["risk_probability"], scored["risk_tier"], None,
                ),
            )
            for f in scored["factors"]:
                con.execute(
                    "INSERT INTO explanations(deployment_id, feature_name, shap_value, feature_value) VALUES (?,?,?,?)",
                    (dep_id, f["feature"], f["shap"], str(f["value"])),
                )

    return {
        "deployment_id": dep_id,
        "service_name": req.service_name,
        "features": feats,
        **scored,
        "nl_summary": summary,
        "recommendation": recommendation,
    }


@app.post("/api/v1/score")
def score_endpoint(req: ScoreRequest):
    return _score_and_store(req)


@app.get("/api/v1/explain/{deployment_id}")
def explain_endpoint(deployment_id: str):
    with db.connect() as con:
        dep = con.execute("SELECT * FROM deployments WHERE deployment_id=?", (deployment_id,)).fetchone()
        if not dep:
            raise HTTPException(404, "deployment not found")
        rows = con.execute(
            "SELECT feature_name, shap_value, feature_value FROM explanations WHERE deployment_id=? ORDER BY ABS(shap_value) DESC",
            (deployment_id,),
        ).fetchall()
    def num(s):
        try:
            f = float(s)
            return int(f) if f == int(f) else round(f, 2)
        except (TypeError, ValueError):
            return s

    if rows:
        factors = [
            {"feature": r["feature_name"], "label": scoring.LABELS.get(r["feature_name"], r["feature_name"]),
             "shap": r["shap_value"], "value": num(r["feature_value"])}
            for r in rows
        ]
        scored = {"risk_probability": dep["risk_probability"],
                  "risk_score": scoring.tier_of(dep["risk_probability"])[1], "factors": factors}
    else:
        # Seeded deployments have no stored SHAP — recompute exactly from features.
        feats = {f: dep[f] for f in scoring.meta()["features"]}
        scored = scoring.score(feats)
        factors = scored["factors"]

    return {"deployment_id": deployment_id, "factors": factors,
            "nl_summary": nl.explain(scored), "base_value": scoring.meta()["base_value_logodds"]}


@app.get("/api/v1/similar/{deployment_id}")
def similar_endpoint(deployment_id: str):
    return similar.find_similar(deployment_id)


@app.get("/api/v1/recommend/{deployment_id}")
def recommend_endpoint(deployment_id: str):
    with db.connect() as con:
        dep = con.execute("SELECT * FROM deployments WHERE deployment_id=?", (deployment_id,)).fetchone()
    if not dep:
        raise HTTPException(404, "deployment not found")
    feats = {k: dep[k] for k in dep.keys()}
    scored = scoring.score(feats)
    return rec.recommend(scored, feats)


@app.post("/api/v1/simulate")
def simulate_endpoint(req: SimulateRequest):
    with db.connect() as con:
        dep = con.execute("SELECT * FROM deployments WHERE deployment_id=?", (req.deployment_id,)).fetchone()
    if not dep:
        raise HTTPException(404, "deployment not found")
    base_feats = {f: dep[f] for f in scoring.meta()["features"]}
    original = scoring.score(base_feats)

    new_feats = dict(base_feats)
    new_feats.update(req.overrides)
    simulated = scoring.score(new_feats)

    delta = round(simulated["risk_probability"] - original["risk_probability"], 4)
    return {
        "original": {"risk_probability": original["risk_probability"], "risk_score": original["risk_score"]},
        "simulated": {"risk_probability": simulated["risk_probability"], "risk_score": simulated["risk_score"], "factors": simulated["factors"]},
        "delta": delta,
        "overrides": req.overrides,
        "nl_summary": nl.summarize(simulated),
        "recommendation": rec.recommend(simulated, new_feats),
    }


@app.get("/api/v1/history")
def history_endpoint(service_name: str | None = None, limit: int = 40):
    q = "SELECT deployment_id, service_name, deploy_timestamp, risk_probability, risk_tier, lines_changed, outcome FROM deployments"
    params: list = []
    if service_name:
        q += " WHERE service_name=?"
        params.append(service_name)
    q += " ORDER BY deploy_timestamp DESC LIMIT ?"
    params.append(limit)
    with db.connect() as con:
        rows = con.execute(q, params).fetchall()
    tiers = ["Low", "Medium", "High"]
    return {"deployments": [
        {**{k: r[k] for k in r.keys()}, "risk_score": tiers[r["risk_tier"]]}
        for r in rows
    ]}


@app.get("/api/v1/trends/{service_name}")
def trends_endpoint(service_name: str):
    with db.connect() as con:
        rows = con.execute(
            "SELECT substr(deploy_timestamp,1,10) AS d, AVG(risk_probability) AS avg_risk, COUNT(*) AS n FROM deployments WHERE service_name=? GROUP BY d ORDER BY d",
            (service_name,),
        ).fetchall()
    return {"service_name": service_name, "daily": [{"date": r["d"], "avg_risk": round(r["avg_risk"], 3), "n": r["n"]} for r in rows]}


@app.get("/api/v1/overview")
def overview_endpoint():
    tiers = ["Low", "Medium", "High"]
    with db.connect() as con:
        dist = con.execute("SELECT risk_tier, COUNT(*) AS c FROM deployments GROUP BY risk_tier").fetchall()
        top = con.execute(
            """SELECT service_name, AVG(risk_probability) AS avg_risk,
                      SUM(CASE WHEN risk_tier=2 THEN 1 ELSE 0 END) AS high_ct, COUNT(*) AS n
               FROM deployments GROUP BY service_name ORDER BY avg_risk DESC LIMIT 6""",
        ).fetchall()
        trend = con.execute(
            "SELECT substr(deploy_timestamp,1,10) AS d, AVG(risk_probability) AS avg_risk FROM deployments GROUP BY d ORDER BY d",
        ).fetchall()
    counts = {0: 0, 1: 0, 2: 0}
    for r in dist:
        counts[r["risk_tier"]] = r["c"]
    total = max(1, sum(counts.values()))
    return {
        "distribution": {tiers[k]: {"count": counts[k], "pct": round(100 * counts[k] / total)} for k in counts},
        "top_risky": [
            {"service_name": r["service_name"], "avg_risk": round(r["avg_risk"], 3), "high_count": r["high_ct"], "n": r["n"]}
            for r in top
        ],
        "org_trend": [{"date": r["d"], "avg_risk": round(r["avg_risk"], 3)} for r in trend],
    }


@app.get("/api/v1/metrics")
def metrics_endpoint():
    m = scoring.meta()
    return {
        "comparison": m["comparison"],
        "oracle_tier_recovery": m["oracle_tier_recovery"],
        "calibration": m["calibration"],
        "train_incident_rate": m["train_incident_rate"],
        "n_train": m["n_train"],
        "n_test": m["n_test"],
    }


@app.get("/api/v1/services")
def services_endpoint():
    with db.connect() as con:
        rows = con.execute("SELECT * FROM services ORDER BY criticality_tier DESC, service_name").fetchall()
    return {"services": [dict(r) for r in rows]}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "..", "frontend", "index.html"))
