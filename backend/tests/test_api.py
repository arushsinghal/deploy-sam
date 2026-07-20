"""Integration tests over the HTTP API via FastAPI TestClient."""
import pytest


@pytest.mark.integration
def test_score_risky_blocks(client, risky_payload):
    r = client.post("/api/v1/score", json=risky_payload)
    assert r.status_code == 200
    d = r.json()
    assert d["risk_score"] == "High"
    assert d["recommendation"]["gate_decision"] == "BLOCK"
    assert d["nl_summary"]
    assert len(d["factors"]) == 11


@pytest.mark.integration
def test_score_safe_passes(client, safe_payload):
    d = client.post("/api/v1/score", json=safe_payload).json()
    assert d["risk_score"] == "Low"
    assert d["recommendation"]["gate_decision"] == "PASS"


@pytest.mark.integration
def test_score_persists_and_is_explainable(client, risky_payload):
    dep_id = client.post("/api/v1/score", json=risky_payload).json()["deployment_id"]
    ex = client.get(f"/api/v1/explain/{dep_id}").json()
    assert len(ex["factors"]) == 11
    assert ex["nl_summary"]
    rec = client.get(f"/api/v1/recommend/{dep_id}").json()
    assert rec["strategy"] in {"canary", "blue-green"}


@pytest.mark.integration
def test_simulate_lowers_risk_when_conditions_improve(client, risky_payload):
    dep_id = client.post("/api/v1/score", json=risky_payload).json()["deployment_id"]
    sim = client.post("/api/v1/simulate", json={
        "deployment_id": dep_id,
        "overrides": {"deploy_hour": 10, "oncall_engineers_available": 5,
                      "is_oncall_senior": 1, "has_rollback_plan": 1, "lines_changed": 100},
    }).json()
    assert sim["delta"] < 0  # improving the drivers must reduce risk
    assert sim["simulated"]["risk_probability"] < sim["original"]["risk_probability"]


@pytest.mark.integration
def test_similar_returns_neighbors_with_outcomes(client, risky_payload):
    dep_id = client.post("/api/v1/score", json=risky_payload).json()["deployment_id"]
    s = client.get(f"/api/v1/similar/{dep_id}").json()
    assert s["n_pool"] > 0
    assert 1 <= len(s["neighbors"]) <= 4
    assert all(n["outcome"] in {"incident", "clean"} for n in s["neighbors"])
    assert 0.0 <= s["incident_rate"] <= 1.0


@pytest.mark.integration
def test_overview_and_metrics_shapes(client):
    ov = client.get("/api/v1/overview").json()
    assert set(ov["distribution"]) == {"Low", "Medium", "High"}
    assert ov["top_risky"] and ov["org_trend"]
    m = client.get("/api/v1/metrics").json()
    # the baseline-beating claim must be present and true.
    xgb = next(v for k, v in m["comparison"].items() if "XGB" in k)
    rules = next(v for k, v in m["comparison"].items() if "RULES" in k)
    assert xgb["f1"] > rules["f1"]
    assert m["oracle_tier_recovery"]["recall_high"] > 0.9


@pytest.mark.integration
def test_unknown_deployment_returns_404(client):
    assert client.get("/api/v1/explain/does-not-exist").status_code == 404
    assert client.get("/api/v1/recommend/does-not-exist").status_code == 404


@pytest.mark.integration
def test_index_and_history(client):
    assert client.get("/").status_code == 200
    h = client.get("/api/v1/history?limit=5").json()
    assert len(h["deployments"]) <= 5
