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

# Kaggle's image puts a sitecustomize.py on the Python path that imports `wrapt`. Our isolated uv
# environment doesn't have wrapt, so every `uv run` printed "Error in sitecustomize ... No module
# named 'wrapt'". It's harmless but noisy, and noise hides real warnings. Two clean fixes, no extra
# dependency:
#  1. Build the venv on a uv-managed Python, which ships no sitecustomize (vs. Kaggle's system one).
#  2. Drop PYTHONPATH entries that contain a sitecustomize.py; nothing in Lumen needs them.
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"
clean_pythonpath() {
  local out="" dir
  local IFS=:
  for dir in ${PYTHONPATH:-}; do
    [ -n "$dir" ] && [ ! -f "$dir/sitecustomize.py" ] && out="${out:+$out:}$dir"
  done
  printf '%s' "$out"
}
export PYTHONPATH="$(clean_pythonpath)"

echo "== uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== dependencies (from uv.lock, exactly) =="
# CPU sessions (data, mac_sample) need only the core package. With a GPU attached, also install
# the pinned CUDA 12.6 PyTorch stack (the `gpu` group) into /kaggle/temp.
if command -v nvidia-smi >/dev/null 2>&1; then
  uv sync --frozen --no-default-groups --group gpu
else
  uv sync --frozen --no-default-groups
fi

echo "== verify =="
uv run python -c "import lumen, sklearn, PIL; print('lumen', lumen.__version__, 'ok')"
if command -v nvidia-smi >/dev/null 2>&1; then
  uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
fi
# Show which sitecustomize (if any) the venv would import, so a returning warning is diagnosable.
uv run python -c "import importlib.util as u; s = u.find_spec('sitecustomize'); print('sitecustomize:', s.origin if s else 'none')"
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
export UV_PYTHON_PREFERENCE="$UV_PYTHON_PREFERENCE"
export PYTHONPATH="$PYTHONPATH"
export PATH="\$HOME/.local/bin:\$PATH"
ENV
echo "== done. environment saved to $ENV_FILE =="
echo "Run stages with, e.g.:  source $ENV_FILE && uv run python kaggle/run_stages.py data"
