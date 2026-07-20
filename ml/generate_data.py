"""
Synthetic deployment dataset generator for DeployIQ.

Design goal: the data-generating process must contain GENUINE structure that a
naive rules baseline cannot capture, so that a gradient-boosted model earns its
keep instead of merely echoing the rules we wrote. To that end the latent
incident probability includes:

  * Nonlinear time-of-day risk (smooth, worst in the small hours).
  * Multiplicative INTERACTIONS (a large diff with thin on-call coverage is far
    worse than either factor alone).
  * Saturating effects (log on diff size).
  * A hidden CONFOUNDER (`team_maturity`) that raises test-coverage delta AND
    lowers base risk, so test_coverage_delta looks more protective than it
    causally is. This is what breaks single-feature "gut" heuristics.
  * Genuine LABEL NOISE (random incidents unrelated to features, plus risky
    deploys that happen to survive).
  * Red-herring columns (author, commit_sha) with no signal.

Everything is disclosed and reproducible via a fixed seed.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SERVICES = [
    # name, criticality_tier (1 low .. 3 critical), owning_team, base_maturity
    ("payments-api", 3, "payments", 0.35),
    ("auth-svc", 3, "identity", 0.55),
    ("billing-svc", 2, "payments", 0.45),
    ("checkout-web", 2, "storefront", 0.40),
    ("search-svc", 2, "discovery", 0.65),
    ("recommendations", 1, "discovery", 0.70),
    ("notifications", 1, "growth", 0.50),
    ("inventory-svc", 2, "supply", 0.48),
    ("catalog-api", 1, "storefront", 0.72),
    ("analytics-etl", 1, "data", 0.60),
]

AUTHORS = [f"eng{i:02d}" for i in range(1, 41)]  # red herring


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _hour_risk(hour: int) -> float:
    """Smooth nonlinear time-of-day risk, peaking ~3am, low midday."""
    # cosine bump centred on 3:00, plus a small Friday-evening-ish tail handled elsewhere
    return 0.5 * (1 + math.cos((hour - 3) / 24 * 2 * math.pi))


def generate(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    svc_meta = {s[0]: s for s in SERVICES}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    rows = []
    for i in range(n):
        name = rng.choice([s[0] for s in SERVICES])
        _, tier, team, base_maturity = svc_meta[name]

        # Hidden confounder: per-deploy team maturity around the service baseline.
        team_maturity = float(np.clip(rng.normal(base_maturity, 0.12), 0.05, 0.95))

        ts = start + timedelta(minutes=int(rng.integers(0, 60 * 24 * 180)))
        hour = ts.hour
        dow = ts.weekday()  # 0 Mon .. 6 Sun
        is_weekend = 1 if dow >= 5 else 0

        # Diff size: mature teams ship smaller, more frequent changes.
        lines_changed = int(np.clip(rng.lognormal(4.6 - 0.8 * team_maturity, 0.9), 1, 8000))
        files_changed = int(np.clip(rng.poisson(2 + lines_changed / 120), 1, 200))

        # Instability signals.
        incidents_last_30d = int(rng.poisson(1.5 * (1 - team_maturity) + 0.4 * (tier - 1)))
        days_since_last_incident = int(np.clip(rng.exponential(20 + 30 * team_maturity), 0, 365))

        # On-call coverage is thinner at night / weekends.
        night = 1 if (hour < 7 or hour >= 22) else 0
        base_oncall = 3 - night - (1 if is_weekend else 0)
        oncall_engineers_available = int(np.clip(base_oncall + rng.integers(-1, 2), 0, 6))
        is_oncall_senior = int(rng.random() < (0.3 + 0.4 * team_maturity))

        has_rollback_plan = int(rng.random() < (0.4 + 0.5 * team_maturity))

        # Coverage delta is boosted by maturity (the confounder) with noise.
        test_coverage_delta = float(np.clip(rng.normal(0.4 * (team_maturity - 0.4), 3.0), -12, 8))

        # ---- Latent incident logit -------------------------------------------------
        lc = math.log1p(lines_changed)
        thin_oncall = max(0, 2 - oncall_engineers_available)  # 0,1,2

        z = -3.1
        z += 0.85 * (tier - 1)                       # criticality
        z += 2.3 * _hour_risk(hour)                  # nonlinear time-of-day
        z += 0.42 * (lc - 4.0)                        # saturating diff size
        # KEY interaction: big diff AND thin on-call is multiplicatively worse
        z += 0.55 * max(0, lc - 4.5) * thin_oncall
        z += 0.30 * incidents_last_30d               # recent instability
        z += 0.9 * math.exp(-days_since_last_incident / 14.0)  # recency of last incident
        z += -1.15 * has_rollback_plan               # rollback plan protects
        z += -0.55 * is_oncall_senior
        z += 0.5 * is_weekend * (tier - 1)           # weekend x criticality interaction
        z += -0.10 * test_coverage_delta             # weak DIRECT effect...
        z += -1.6 * (team_maturity - 0.5)            # ...confounded by hidden maturity
        z += rng.normal(0, 0.45)                     # irreducible latent noise

        p = _sigmoid(z)
        outcome = int(rng.random() < p)
        # Genuine label noise: 4% random flips (surprise incidents / lucky escapes).
        if rng.random() < 0.04:
            outcome = 1 - outcome

        commit_sha = hashlib.sha1(f"{name}{i}{seed}".encode()).hexdigest()[:10]
        rows.append(
            dict(
                deploy_timestamp=ts.isoformat(),
                service_name=name,
                service_criticality_tier=tier,
                owning_team=team,
                author=str(rng.choice(AUTHORS)),      # red herring
                commit_sha=commit_sha,                # red herring
                deploy_hour=hour,
                day_of_week=dow,
                is_weekend=is_weekend,
                lines_changed=lines_changed,
                files_changed=files_changed,
                incidents_last_30d=incidents_last_30d,
                days_since_last_incident=days_since_last_incident,
                oncall_engineers_available=oncall_engineers_available,
                is_oncall_senior=is_oncall_senior,
                has_rollback_plan=has_rollback_plan,
                test_coverage_delta=round(test_coverage_delta, 2),
                _team_maturity=round(team_maturity, 3),  # hidden; NOT a model feature
                _p_true=round(p, 4),                      # oracle prob; NOT a feature
                outcome=outcome,
            )
        )

    df = pd.DataFrame(rows)

    # Inject realistic missingness (~6%) into two features to exercise XGBoost's
    # native sparsity-aware splits. We deliberately leave these as NaN.
    for col in ["test_coverage_delta", "days_since_last_incident"]:
        mask = rng.random(len(df)) < 0.06
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "deployments.csv"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    rate = df["outcome"].mean()
    print(f"Wrote {len(df):,} rows -> {args.out}")
    print(f"Incident rate: {rate:.1%}  (positives={int(df['outcome'].sum()):,})")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
