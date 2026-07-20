# DeployIQ — Deployment Risk Intelligence

> Scores every deployment **Low / Medium / High** *before* it ships, explains the
> score with exact **tree-SHAP**, shows which past deploys it resembles, and
> recommends how to ship it safely — then gates the PR on the result.

A working, tested slice of the HPE Synergy 2026 Round 1 concept. Not a mockup:
real model, real inference, real explanations, driven from a live dashboard.

![status](https://img.shields.io/badge/tests-19%20passing-2ec26b) ![python](https://img.shields.io/badge/python-3.12-4f8cff)

---

## Quick start

```bash
./run.sh              # launch API + dashboard  → http://127.0.0.1:8077
./run.sh --retrain    # regenerate synthetic data + retrain first
```

Run the tests:

```bash
.venv/bin/python -m pytest        # 19 tests: unit + integration
```

Optional — real Claude narratives instead of the deterministic template:

```bash
export ANTHROPIC_API_KEY=sk-...   # nl.explain() uses Claude, falls back on any error
```

---

## The demo loop (all live, all verified)

```
 CI / webhook ─▶ SCORE ─▶ EXPLAIN (SHAP + NL) ─▶ SIMILAR PAST DEPLOYS
                   │                                      │
                   ▼                                      ▼
              RECOMMEND (canary %, window)  ─────▶  GATE (PASS / WARN / BLOCK)
                   │
                   ▼
              WHAT-IF SIMULATOR (live re-score)
```

| Capability | Endpoint | UI |
|---|---|---|
| Risk score (XGBoost, calibrated) | `POST /api/v1/score` | Score a Deploy |
| Exact tree-SHAP + plain-English narrative | `GET /api/v1/explain/{id}` | Risk Detail |
| **Historical similarity** ("resembles N deploys, M caused incidents") | `GET /api/v1/similar/{id}` | Risk Detail |
| Rollout recommendation + gate decision | `GET /api/v1/recommend/{id}` | Risk Detail |
| What-if simulator (live re-scoring) | `POST /api/v1/simulate` | Simulator |
| Org overview / history / trends | `/overview` `/history` `/trends/{svc}` | Dashboard |
| Baseline comparison + calibration | `GET /api/v1/metrics` | Model & Metrics |
| CI gatekeeper (blocks High-risk PRs) | — | `.github/workflows/deploy-risk-gate.yml` |

---

## Why it isn't just echoing the data generator

This is the first hard question a judge asks. The **Model & Metrics** tab answers it:

- **Labels are stochastic.** Each deploy's incident outcome is drawn from a
  *latent* risk built from feature **interactions + noise** — there is no
  deterministic rule to memorize, so the model must recover a *probability*.
- **It beats the gut-feel baseline.** A hand-coded `RULES` heuristic (late-night +
  big diff + thin on-call) runs as a live competitor. XGBoost lifts incident-class
  **F1 ~120%** over it (0.59 vs 0.27) and **PR-AUC +0.16**, also beating logistic
  regression. That lift *is* the value the ML adds.
- **It's calibrated.** Predicted vs. observed incident rates track the diagonal
  (Brier reported) — what makes probability thresholds defensible in a CAB audit.
- **High-risk recall ≈ 98%** — the costly failure mode (missing a real High) is caught.

The `RULES` / `LOGREG` / `XGBOOST` table and the reliability curve are computed on
a held-out test split in `ml/train.py` and surfaced in the UI — nothing hand-waved.

---

## Model & features

**Model:** XGBoost (log-loss), calibrated, exact `pred_contribs` tree-SHAP for
explanations. Random Forest / LogReg / rules kept as documented baselines.

**Label:** P(incident within 24h) → 3-class risk via calibrated thresholds
(`<0.2` Low, `0.2–0.5` Medium, `>0.5` High).

**Features:** service criticality tier, deploy hour, weekend flag, lines/files
changed, incidents-last-30d, days-since-last-incident, on-call count, senior-on-call,
rollback plan, test-coverage delta. Missing numerics pass through XGBoost's
sparsity-aware split (no imputation needed) — tested.

---

## Layout

```
ml/            generate_data.py · train.py · artifacts/{model.json, metadata.json, holdout}
backend/
  scoring.py     feature engineering + inference + exact tree-SHAP
  recommend.py   rollout strategy + gate decision rule engine
  similar.py     nearest-neighbour institutional memory
  nl.py          plain-English narrative (Claude API or deterministic template)
  db.py          SQLite persistence (stands in for Postgres), self-seeding
  main.py        FastAPI app — 9 endpoints, serves the dashboard
  tests/         19 pytest cases (unit + integration via TestClient)
frontend/
  index.html     single-file dashboard: SHAP waterfall, simulator, model metrics
.github/workflows/deploy-risk-gate.yml   the CI status check that gates PRs
run.sh · requirements.txt · pytest.ini
```

---

## Tests

`19 passing` — run `.venv/bin/python -m pytest`.

- **Unit** — tier thresholds match metadata; `None`-vs-default feature handling;
  SHAP shape + ordering; risky > safe monotonicity; NaN tolerance; recommender gate
  logic; NL is factual and falls back without an API key.
- **Integration** (FastAPI TestClient, isolated throwaway DB) — score risky→BLOCK /
  safe→PASS; persist→explain→recommend round-trip; simulate lowers risk when drivers
  improve; similarity returns labelled neighbours; the baseline-beating claim in
  `/metrics` is asserted true; 404s; index + history.

---

## Stack

FastAPI · XGBoost · scikit-learn · exact tree-SHAP · SQLite (→ Postgres in prod) ·
vanilla-JS + inline-SVG dashboard (no build step) · optional Anthropic Claude for
narratives.

## Honest scope (what's Phase 2)

Synthetic data — disclosed, domain-informed, schema matches real
deployment/incident/roster tables for drop-in swap. Deliberately **not** built for
Round 1 (and documented as Phase 2 in the design doc): JWT/OAuth auth, managed
Postgres/Redis/Celery, MLflow registry, Slack/Jira bots, dependency-graph risk
propagation, drift-triggered retraining. The core intelligence loop —
**score → explain → similar → recommend → simulate → gate** — is real, tested, and runs.
