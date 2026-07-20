"""Unit tests for feature engineering, inference, tier mapping, and SHAP."""
import pytest

import scoring


@pytest.mark.unit
def test_tier_thresholds_match_metadata():
    t = scoring.meta()["thresholds"]
    assert scoring.tier_of(t["low_max"] - 0.01) == (0, "Low")
    assert scoring.tier_of((t["low_max"] + t["high_min"]) / 2) == (1, "Medium")
    assert scoring.tier_of(t["high_min"] + 0.01) == (2, "High")


@pytest.mark.unit
def test_build_features_fills_none_with_defaults():
    # Pydantic-style payload where optional fields are explicitly None.
    req = {
        "service_name": "payments-api", "deploy_timestamp": "2026-07-18T23:10:00",
        "lines_changed": 500, "files_changed": 10,
        "deploy_hour": None, "is_weekend": None, "oncall_engineers_available": None,
        "test_coverage_delta": None, "has_rollback_plan": None,
    }
    feats = scoring.build_features(req)
    # deploy_hour must fall back to the timestamp hour, not stay None.
    assert feats["deploy_hour"] == 23
    assert feats["oncall_engineers_available"] == 2
    assert feats["has_rollback_plan"] == 0
    assert feats["test_coverage_delta"] == 0.0
    assert all(feats[f] is not None for f in scoring.meta()["features"] if f != "test_coverage_delta")


@pytest.mark.unit
def test_score_returns_calibrated_prob_and_full_shap():
    feats = scoring.build_features({
        "service_name": "payments-api", "deploy_timestamp": "2026-07-18T23:10:00",
        "deploy_hour": 23, "lines_changed": 880, "files_changed": 16,
        "oncall_engineers_available": 1, "is_oncall_senior": 0,
        "test_coverage_delta": -6, "has_rollback_plan": False,
    })
    out = scoring.score(feats)
    assert 0.0 <= out["risk_probability"] <= 1.0
    assert out["risk_score"] in {"Low", "Medium", "High"}
    # one SHAP factor per model feature, sorted by |shap| descending.
    assert len(out["factors"]) == len(scoring.meta()["features"])
    mags = [abs(f["shap"]) for f in out["factors"]]
    assert mags == sorted(mags, reverse=True)


@pytest.mark.unit
def test_risky_scores_higher_than_safe():
    def p(**kw):
        base = {"service_name": "payments-api", "deploy_timestamp": "2026-07-18T12:00:00"}
        return scoring.score(scoring.build_features({**base, **kw}))["risk_probability"]

    risky = p(deploy_hour=23, lines_changed=900, files_changed=18,
              oncall_engineers_available=1, has_rollback_plan=False, test_coverage_delta=-8)
    safe = p(deploy_hour=10, lines_changed=40, files_changed=2,
             oncall_engineers_available=5, is_oncall_senior=1, has_rollback_plan=True, test_coverage_delta=4)
    assert risky > safe


@pytest.mark.unit
def test_missing_numeric_is_handled_not_crashing():
    # XGBoost sparsity-aware split should accept NaN for an omitted feature.
    feats = scoring.build_features({
        "service_name": "auth-svc", "deploy_timestamp": "2026-07-18T09:00:00",
        "lines_changed": None, "files_changed": None,
    })
    out = scoring.score(feats)
    assert 0.0 <= out["risk_probability"] <= 1.0
