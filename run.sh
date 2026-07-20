#!/usr/bin/env bash
# DeployIQ — one-command demo launcher.
# Usage:
#   ./run.sh            # launch API + dashboard on :8077 (retrains if no model)
#   ./run.sh --retrain  # regenerate synthetic data + retrain, then launch
#
# Uses `uv` (https://docs.astral.sh/uv/) if available for fast, reproducible
# dependency resolution from pyproject.toml/uv.lock; falls back to a plain
# venv + requirements.txt otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  echo "→ syncing dependencies with uv"
  uv sync --quiet
  PY="$ROOT/.venv/bin/python"
else
  PY="$ROOT/.venv/bin/python"
  if [ ! -x "$PY" ]; then
    echo "→ uv not found, creating venv + installing deps with pip"
    python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/pip" install -q -r requirements.txt
  fi
fi

if [ "${1:-}" = "--retrain" ] || [ ! -f ml/artifacts/model.json ]; then
  echo "→ generating synthetic dataset"
  (cd ml && "$PY" generate_data.py)
  echo "→ training model + baselines"
  (cd ml && "$PY" train.py)
fi

echo "→ starting DeployIQ on http://127.0.0.1:8077"
rm -f backend/deployiq.db          # reseed fresh feed from the scored holdout
cd backend
exec "$PY" -m uvicorn main:app --host 127.0.0.1 --port 8077
