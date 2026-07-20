"""SQLite persistence for DeployIQ (stands in for Postgres in the demo).

Seeds itself from the trained model's scored holdout so the dashboard has a
realistic live feed, history and trends on first boot.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager

import pandas as pd

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "deployiq.db")
ART = os.path.join(HERE, "..", "ml", "artifacts")

SCHEMA = """
CREATE TABLE IF NOT EXISTS services (
    service_name TEXT PRIMARY KEY,
    criticality_tier INTEGER,
    owning_team TEXT
);
CREATE TABLE IF NOT EXISTS deployments (
    deployment_id TEXT PRIMARY KEY,
    service_name TEXT,
    commit_sha TEXT,
    author TEXT,
    deploy_timestamp TEXT,
    deploy_hour INTEGER,
    is_weekend INTEGER,
    lines_changed INTEGER,
    files_changed INTEGER,
    incidents_last_30d INTEGER,
    days_since_last_incident REAL,
    oncall_engineers_available INTEGER,
    is_oncall_senior INTEGER,
    has_rollback_plan INTEGER,
    test_coverage_delta REAL,
    service_criticality_tier INTEGER,
    risk_probability REAL,
    risk_tier INTEGER,
    outcome INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS explanations (
    deployment_id TEXT,
    feature_name TEXT,
    shap_value REAL,
    feature_value TEXT
);
CREATE INDEX IF NOT EXISTS idx_dep_service_time ON deployments(service_name, deploy_timestamp);
CREATE INDEX IF NOT EXISTS idx_expl_dep ON explanations(deployment_id);
"""


@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def new_id() -> str:
    return str(uuid.uuid4())


def init_db(force: bool = False) -> None:
    if force and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with connect() as con:
        con.executescript(SCHEMA)
        already = con.execute("SELECT COUNT(*) AS c FROM deployments").fetchone()["c"]
        if already:
            return
        _seed(con)


def _seed(con: sqlite3.Connection) -> None:
    from generate_data import SERVICES  # type: ignore

    for name, tier, team, _ in SERVICES:
        con.execute(
            "INSERT OR REPLACE INTO services(service_name, criticality_tier, owning_team) VALUES (?,?,?)",
            (name, tier, team),
        )

    scored = os.path.join(ART, "holdout_scored.csv")
    if not os.path.exists(scored):
        return
    df = pd.read_csv(scored).sort_values("deploy_timestamp").tail(600)
    for _, r in df.iterrows():
        dep_id = new_id()
        con.execute(
            """INSERT INTO deployments(
                deployment_id, service_name, commit_sha, author, deploy_timestamp,
                deploy_hour, is_weekend, lines_changed, files_changed, incidents_last_30d,
                days_since_last_incident, oncall_engineers_available, is_oncall_senior,
                has_rollback_plan, test_coverage_delta, service_criticality_tier,
                risk_probability, risk_tier, outcome
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                dep_id, r["service_name"], r["commit_sha"], r["author"], r["deploy_timestamp"],
                int(r["deploy_hour"]), int(r["is_weekend"]), int(r["lines_changed"]),
                int(r["files_changed"]), int(r["incidents_last_30d"]),
                _f(r["days_since_last_incident"]), _i(r["oncall_engineers_available"]),
                int(r["is_oncall_senior"]), int(r["has_rollback_plan"]),
                _f(r["test_coverage_delta"]), int(r["service_criticality_tier"]),
                float(r["risk_probability"]), int(r["risk_tier"]), int(r["outcome"]),
            ),
        )


def _f(v):
    return None if pd.isna(v) else float(v)


def _i(v):
    return None if pd.isna(v) else int(v)
