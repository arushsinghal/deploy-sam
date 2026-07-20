"""Turn SHAP factors into a plain-English risk narrative.

Two paths, same JSON contract:
  * Deterministic template (default) — reproducible for audit trails, no network.
  * Real Claude API — used automatically when ANTHROPIC_API_KEY is set, with a
    graceful fallback to the template on any error so the demo never breaks.
"""
from __future__ import annotations

import json
import os
import urllib.request

CLAUDE_MODEL = os.environ.get("DEPLOYIQ_CLAUDE_MODEL", "claude-haiku-4-5-20251001")


def _phrase(feature: str, value, positive: bool) -> str:
    v = value
    direction = "raising" if positive else "lowering"
    m = {
        "deploy_hour": f"deploying at {int(v):02d}:00" if v is not None else "the deploy hour",
        "lines_changed": (
            f"{'a large ' if (v is not None and v >= 400) else 'a '}{int(v)}-line changeset"
            if v is not None else "the changeset size"
        ),
        "files_changed": f"touching {int(v)} files" if v is not None else "the number of files",
        "oncall_engineers_available": (
            f"only {int(v)} on-call engineer{'s' if v != 1 else ''} available" if v is not None else "on-call coverage"
        ),
        "is_oncall_senior": ("a senior engineer on call" if v else "no senior on call"),
        "has_rollback_plan": ("a rollback plan attached" if v else "no rollback plan"),
        "incidents_last_30d": f"{int(v)} incident(s) in the last 30 days" if v is not None else "recent incident history",
        "days_since_last_incident": (
            f"a recent incident {int(v)} days ago" if (v is not None and v < 21) else "incident recency"
        ),
        "service_criticality_tier": f"a tier-{int(v)} critical service" if v is not None else "service criticality",
        "is_weekend": ("a weekend deploy" if v else "a weekday deploy"),
        "test_coverage_delta": (
            f"a {v:+.1f}% test-coverage change" if v is not None else "the test-coverage change"
        ),
    }
    label = m.get(feature, feature)
    return f"{label} ({direction} risk)"


def summarize(scored: dict, top_n: int = 4) -> str:
    prob = scored["risk_probability"]
    tier = scored["risk_score"]
    factors = scored["factors"][:top_n]
    up = [f for f in factors if f["shap"] > 0]
    down = [f for f in factors if f["shap"] < 0]

    parts = [f"This deploy scores {tier} risk ({prob:.0%} probability of an incident within 24h)."]
    if up:
        drivers = ", ".join(_phrase(f["feature"], f["value"], True).replace(" (raising risk)", "") for f in up[:3])
        parts.append(f"The main risk drivers are {drivers}.")
    if down:
        mit = ", ".join(_phrase(f["feature"], f["value"], False).replace(" (lowering risk)", "") for f in down[:2])
        parts.append(f"Partially offsetting this: {mit}.")
    if tier == "High":
        parts.append("Recommend shipping behind a small canary and requiring senior approval.")
    return " ".join(parts)


def explain(scored: dict, top_n: int = 4) -> str:
    """Preferred entry point: Claude if configured, else the template."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            return _claude_summary(scored, key)
        except Exception:
            pass  # never let a network/LLM hiccup break scoring
    return summarize(scored, top_n)


def _claude_summary(scored: dict, api_key: str, timeout: float = 8.0) -> str:
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 220,
        "messages": [{"role": "user", "content": build_llm_prompt(scored)}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


def build_llm_prompt(scored: dict) -> str:
    """The exact structured prompt a Claude API call would receive (upgrade hook)."""
    lines = [
        "You are a release-safety assistant. Given SHAP feature attributions for a",
        "deployment risk model, write ONE plain-English paragraph explaining the risk.",
        "Restate only the facts below; do not invent numbers.\n",
        f"Risk: {scored['risk_score']} ({scored['risk_probability']:.0%} incident probability)\n",
        "Top factors (feature = value, shap log-odds contribution):",
    ]
    for f in scored["factors"][:6]:
        lines.append(f"  - {f['label']} = {f['value']}  (shap {f['shap']:+.3f})")
    return "\n".join(lines)
