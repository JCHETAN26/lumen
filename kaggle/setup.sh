#!/usr/bin/env bash
# Install Lumen's pinned environment on Kaggle. Run from inside the cloned repo, AFTER the clone
# cell in kaggle/README.md:
#
#     bash kaggle/setup.sh
#
# Safe to re-run: uv sync is incremental, so a restarted session skips finished work. Nothing here
# reads or prints a token.
#
# Disk layout (Kaggle: /kaggle/working is ~20 GB and persistent; /kaggle/temp is larger, wiped):
#   - uv environment + cache    -> /kaggle/temp       (rebuilt quickly each session)
#   - VizWiz zips (~15 GB)      -> /kaggle/working/lumen-data     (persistent: no re-download)
#   - download folder           -> /kaggle/working/lumen-results  (the only thing you download)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEMP_ROOT="${LUMEN_TEMP_ROOT:-/kaggle/temp}"
[ -d "$TEMP_ROOT" ] || TEMP_ROOT="$(mktemp -d)"   # off-Kaggle fallback

export UV_CACHE_DIR="${UV_CACHE_DIR:-$TEMP_ROOT/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$TEMP_ROOT/lumen-venv}"
export LUMEN_DATA_DIR="${LUMEN_DATA_DIR:-/kaggle/working/lumen-data}"
export LUMEN_OUTPUT_DIR="${LUMEN_OUTPUT_DIR:-/kaggle/working/lumen-results}"

echo "== uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== dependencies (from uv.lock, exactly) =="
# Stage 1 needs only the core package. GPU groups are added in Stage 2.
uv sync --frozen --no-default-groups

echo "== verify =="
uv run python -c "import lumen, sklearn, PIL; print('lumen', lumen.__version__, 'ok')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "no GPU attached (correct for the data and mac_sample stages)"
fi

ENV_FILE="/kaggle/working/lumen-env.sh"
[ -d /kaggle/working ] || ENV_FILE="$REPO_ROOT/.kaggle-env.sh"
cat > "$ENV_FILE" <<ENV
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
export LUMEN_DATA_DIR="$LUMEN_DATA_DIR"
export LUMEN_OUTPUT_DIR="$LUMEN_OUTPUT_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
ENV
echo "== done. environment saved to $ENV_FILE =="
echo "Run stages with, e.g.:  source $ENV_FILE && uv run python kaggle/run_stages.py data"
