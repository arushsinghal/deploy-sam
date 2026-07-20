#!/usr/bin/env bash
# DeployIQ — one-command demo launcher.
# Usage:
#   ./run.sh            # launch API + dashboard on :8077 (retrains if no model)
#   ./run.sh --retrain  # regenerate synthetic data + retrain, then launch
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "→ creating venv + installing deps"
  python3 -m venv .venv
  ./.venv/bin/pip install -q -r requirements.txt
  PY=.venv/bin/python
fi

if [ "${1:-}" = "--retrain" ] || [ ! -f ml/artifacts/model.json ]; then
  echo "→ generating synthetic dataset"
  (cd ml && ../$PY generate_data.py)
  echo "→ training model + baselines"
  (cd ml && ../$PY train.py)
fi

echo "→ starting DeployIQ on http://127.0.0.1:8077"
rm -f backend/deployiq.db          # reseed fresh feed from the scored holdout
cd backend
exec ../$PY -m uvicorn main:app --host 127.0.0.1 --port 8077
