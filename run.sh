#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# --- OFFLINE env (MUST be set before Python imports HF libs) ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_ALLOW_CODE_DOWNLOAD=false
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ENV_NAME="rag_flask_env"

# Try to initialize conda hooks (non-fatal)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.zsh hook)" 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate "${ENV_NAME}" 2>/dev/null || true
fi

# Choose python
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="$(command -v python3 || echo python)"
fi

export PYTHONPATH="${ROOT}"
export FLASK_APP=app.app
export FLASK_ENV=development

mkdir -p "${ROOT}/logs" "${ROOT}/outputs"

echo "[run.sh] Using python: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m flask run --host=127.0.0.1 --port=5000