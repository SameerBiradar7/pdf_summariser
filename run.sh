#!/usr/bin/env bash
set -euo pipefail

# Always run from project root (where run.sh lives)
cd "$(dirname "$0")"
ROOT="$(pwd)"

# --- OFFLINE env (must be set before Python imports HF libs) ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_ALLOW_CODE_DOWNLOAD=false
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Use the rag_flask_env Python directly (no conda activate needed)
PYTHON_BIN="/opt/homebrew/Caskroom/miniforge/base/envs/rag_flask_env/bin/python"

# If that path breaks for some reason, fallback to whatever "python" is
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[run.sh] WARNING: $PYTHON_BIN not found or not executable. Falling back to 'python' on PATH."
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3 || echo python)"
  fi
fi

export PYTHONPATH="${ROOT}"
export FLASK_APP=app.app
export FLASK_ENV=development

mkdir -p "${ROOT}/logs" "${ROOT}/outputs"

echo "[run.sh] Using python: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m flask run --host=127.0.0.1 --port=5000