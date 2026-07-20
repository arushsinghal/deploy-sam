"""Shared fixtures — an isolated DB + TestClient so tests never touch the demo db."""
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(__file__)
BACKEND = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.abspath(os.path.join(BACKEND, "..", "ml")))


@pytest.fixture(scope="session", autouse=True)
def isolated_db():
    """Point the app at a throwaway SQLite file seeded from the real artifacts."""
    import db
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db.init_db(force=True)
    yield
    os.unlink(tmp.name)


@pytest.fixture(scope="session")
def client(isolated_db):
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def risky_payload():
    return {
        "service_name": "payments-api",
        "deploy_timestamp": "2026-07-18T23:10:00",
        "deploy_hour": 23,
        "lines_changed": 880,
        "files_changed": 16,
        "oncall_engineers_available": 1,
        "is_oncall_senior": 0,
        "test_coverage_delta": -6,
        "has_rollback_plan": False,
    }


@pytest.fixture
def safe_payload():
    return {
        "service_name": "docs-site",
        "deploy_timestamp": "2026-07-14T10:10:00",
        "deploy_hour": 10,
        "lines_changed": 30,
        "files_changed": 2,
        "oncall_engineers_available": 4,
        "is_oncall_senior": 1,
        "test_coverage_delta": 3,
        "has_rollback_plan": True,
    }
