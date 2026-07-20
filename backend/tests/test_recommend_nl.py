"""Unit tests for the rollout recommender and NL narrative."""
import pytest

import nl
import recommend as rec
import scoring


def _score(**kw):
    base = {"service_name": "payments-api", "deploy_timestamp": "2026-07-18T12:00:00"}
    feats = scoring.build_features({**base, **kw})
    return scoring.score(feats), feats


@pytest.mark.unit
def test_high_risk_blocks_and_requires_approval():
    scored, feats = _score(deploy_hour=23, lines_changed=950, files_changed=20,
                           oncall_engineers_available=1, has_rollback_plan=False, test_coverage_delta=-9)
    scored["risk_score"] = "High"  # force the branch under test
    r = rec.recommend(scored, feats)
    assert r["gate_decision"] == "BLOCK"
    assert r["require_senior_approval"] is True
    assert r["strategy"] == "canary"


@pytest.mark.unit
def test_low_risk_passes_blue_green():
    scored, feats = _score(deploy_hour=10, lines_changed=30, files_changed=2,
                           oncall_engineers_available=5, is_oncall_senior=1, has_rollback_plan=True)
    scored["risk_score"] = "Low"
    r = rec.recommend(scored, feats)
    assert r["gate_decision"] == "PASS"
    assert r["strategy"] == "blue-green"


@pytest.mark.unit
def test_missing_rollback_plan_adds_note():
    scored, feats = _score(has_rollback_plan=False, lines_changed=100, files_changed=3)
    r = rec.recommend(scored, feats)
    assert any("rollback" in n.lower() for n in r["notes"])


@pytest.mark.unit
def test_nl_summary_is_factual_and_mentions_tier():
    scored, _ = _score(deploy_hour=23, lines_changed=880, oncall_engineers_available=1,
                       files_changed=15, has_rollback_plan=False)
    s = nl.summarize(scored)
    assert scored["risk_score"] in s
    assert "%" in s  # states a probability
    assert len(s) > 40


@pytest.mark.unit
def test_large_changeset_phrasing_only_when_large():
    small, _ = _score(lines_changed=120, files_changed=3, deploy_hour=23,
                      oncall_engineers_available=1, has_rollback_plan=False)
    assert "large 120-line" not in nl.summarize(small)


@pytest.mark.unit
def test_nl_explain_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    scored, _ = _score(lines_changed=200, files_changed=4)
    # explain() must return the deterministic template, never raise.
    assert nl.explain(scored) == nl.summarize(scored)
