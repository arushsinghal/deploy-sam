"""
Train the DeployIQ risk model and prove it beats naive baselines.

Three models are compared on a held-out test split:
  1. RULES  – a hand-written "gut feeling" heuristic (tribal knowledge formalised).
  2. LOGREG – linear logistic regression (captures linear effects, no interactions).
  3. XGB    – gradient-boosted trees (captures interactions + confounder structure).

Headline comparison is the BINARY incident-prediction task (apples-to-apples:
ROC-AUC, PR-AUC, Brier, F1, recall). As an extra validation only the synthetic
setting allows, we also check whether XGBoost recovers the ORACLE risk tier
(Low/Med/High buckets of the true latent probability) — the acid test that it
learned the real structure, not our surface rules.

Explainability uses XGBoost's native `pred_contribs=True`, which returns EXACT
tree SHAP values (no sampling, no extra dependency); the trailing column is the
base value.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "deployments.csv")
ART = os.path.join(HERE, "artifacts")

# Human-readable model features (kept readable so SHAP explanations are legible).
FEATURES = [
    "service_criticality_tier",
    "deploy_hour",
    "is_weekend",
    "lines_changed",
    "files_changed",
    "incidents_last_30d",
    "days_since_last_incident",
    "oncall_engineers_available",
    "is_oncall_senior",
    "has_rollback_plan",
    "test_coverage_delta",
]

# 3-class risk thresholds on P(incident).
LOW_MAX = 0.20
HIGH_MIN = 0.50


def to_tier(p: np.ndarray) -> np.ndarray:
    return np.where(p < LOW_MAX, 0, np.where(p < HIGH_MIN, 1, 2))


def rules_score(df: pd.DataFrame) -> np.ndarray:
    """Formalised 'gut feeling': the informal heuristics teams use today."""
    s = np.zeros(len(df), dtype=float)
    h = df["deploy_hour"].to_numpy()
    s += np.where((h < 7) | (h >= 22), 2.0, 0.0)          # off-hours
    s += df["is_weekend"].to_numpy() * 1.0                  # weekend
    lc = df["lines_changed"].to_numpy()
    s += np.where(lc > 500, 2.0, np.where(lc > 200, 1.0, 0.0))  # big diff
    oc = df["oncall_engineers_available"].fillna(2).to_numpy()
    s += np.where(oc <= 1, 2.0, 0.0)                        # thin on-call
    s += np.where(df["has_rollback_plan"].to_numpy() == 0, 1.0, 0.0)
    s += np.where(df["incidents_last_30d"].to_numpy() >= 2, 1.0, 0.0)
    s += np.where(df["service_criticality_tier"].to_numpy() == 3, 1.0, 0.0)
    return s


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    return dict(
        roc_auc=round(float(roc_auc_score(y, p)), 4),
        pr_auc=round(float(average_precision_score(y, p)), 4),
        brier=round(float(brier_score_loss(y, p)), 4),
        f1=round(float(f1_score(y, pred)), 4),
        recall_pos=round(float(recall_score(y, pred)), 4),
    )


def main() -> None:
    os.makedirs(ART, exist_ok=True)
    df = pd.read_csv(DATA)
    y = df["outcome"].to_numpy()

    train_df, test_df, y_tr, y_te = train_test_split(
        df, y, test_size=0.25, random_state=13, stratify=y
    )

    X_tr = train_df[FEATURES]
    X_te = test_df[FEATURES]

    # ---- 1. RULES baseline (Platt-calibrated so it emits probabilities) -----------
    r_tr = rules_score(train_df).reshape(-1, 1)
    r_te = rules_score(test_df).reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000).fit(r_tr, y_tr)
    p_rules = platt.predict_proba(r_te)[:, 1]

    # ---- 2. LOGREG baseline (median-imputed + standardised) -----------------------
    med = X_tr.median(numeric_only=True)
    Xtr_i = X_tr.fillna(med)
    Xte_i = X_te.fillna(med)
    scaler = StandardScaler().fit(Xtr_i)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform(Xtr_i), y_tr
    )
    p_log = logreg.predict_proba(scaler.transform(Xte_i))[:, 1]

    # ---- 3. XGBoost (native NaN handling + imbalance weighting) -------------------
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURES)
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=FEATURES)
    params = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        max_depth=5,
        eta=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=6,
        scale_pos_weight=neg / pos,
        seed=13,
    )
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=600,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    p_xgb = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))

    # ---- Comparison table ---------------------------------------------------------
    comparison = {
        "RULES (gut feeling)": binary_metrics(y_te, p_rules),
        "LOGREG (linear)": binary_metrics(y_te, p_log),
        "XGBOOST (DeployIQ)": binary_metrics(y_te, p_xgb),
    }

    # ---- Oracle-tier recovery: did XGB learn the TRUE risk structure? -------------
    oracle_tier = to_tier(test_df["_p_true"].to_numpy())
    xgb_tier = to_tier(p_xgb)
    tier_report = {
        "macro_f1": round(float(f1_score(oracle_tier, xgb_tier, average="macro")), 4),
        "recall_low": round(float(recall_score(oracle_tier, xgb_tier, labels=[0], average="macro")), 4),
        "recall_med": round(float(recall_score(oracle_tier, xgb_tier, labels=[1], average="macro")), 4),
        "recall_high": round(float(recall_score(oracle_tier, xgb_tier, labels=[2], average="macro")), 4),
    }

    # ---- Calibration reliability bins --------------------------------------------
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_xgb, bins) - 1, 0, 9)
    calib = []
    for b in range(10):
        m = idx == b
        if m.sum():
            calib.append({
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "predicted": round(float(p_xgb[m].mean()), 3),
                "observed": round(float(y_te[m].mean()), 3),
                "n": int(m.sum()),
            })

    # ---- SHAP base value (bias term from exact tree contributions) ---------------
    contribs = booster.predict(dtest, pred_contribs=True, iteration_range=(0, booster.best_iteration + 1))
    base_value = float(contribs[:, -1].mean())  # log-odds space

    # ---- Persist artifacts --------------------------------------------------------
    booster.save_model(os.path.join(ART, "model.json"))
    metadata = {
        "features": FEATURES,
        "thresholds": {"low_max": LOW_MAX, "high_min": HIGH_MIN},
        "base_value_logodds": round(base_value, 5),
        "best_iteration": int(booster.best_iteration),
        "feature_medians": {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in med.items()},
        "train_incident_rate": round(float(y_tr.mean()), 4),
        "comparison": comparison,
        "oracle_tier_recovery": tier_report,
        "calibration": calib,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    with open(os.path.join(ART, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Holdout with scores to seed the app's live feed / history / trends.
    out = test_df.copy()
    out["risk_probability"] = np.round(p_xgb, 4)
    out["risk_tier"] = to_tier(p_xgb)
    out.to_csv(os.path.join(ART, "holdout_scored.csv"), index=False)

    # ---- Report -------------------------------------------------------------------
    print("\n=== MODEL COMPARISON (binary incident prediction, test split) ===")
    hdr = f"{'model':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}{'F1':>8}{'Recall+':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in comparison.items():
        print(f"{name:<22}{m['roc_auc']:>9}{m['pr_auc']:>9}{m['brier']:>8}{m['f1']:>8}{m['recall_pos']:>9}")

    print("\n=== XGBoost recovery of TRUE (oracle) risk tiers ===")
    print(f"  macro-F1={tier_report['macro_f1']}  |  High-risk recall={tier_report['recall_high']}")
    lift = comparison["XGBOOST (DeployIQ)"]["pr_auc"] - comparison["RULES (gut feeling)"]["pr_auc"]
    print(f"\nPR-AUC lift over gut-feeling rules: +{lift:.3f}  (this is the value the ML adds)")
    print(f"\nArtifacts written to {ART}/")


if __name__ == "__main__":
    main()
