#!/usr/bin/env bash
# Build the MaxText benchmark environment (macOS / Apple silicon).
#
#   scripts/model_bench/setup_maxtext.sh [VENV_PATH]
#
# Creates a venv (default: $METALJAX_BENCH_HOME/maxtext/venv), installs metaljax
# editable from this checkout, then MaxText and its accelerator-free dependency
# set. jax/jaxlib are constrained to 0.11.0 throughout -- MaxText's own
# requirements pin jax==0.7.1 and would otherwise downgrade the plugin's ABI.
#
# Idempotent: re-running reuses the checkout and the uv cache.
#
# Afterwards run convert_checkpoints.sh to produce the Orbax checkpoints.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
BENCH_HOME="${METALJAX_BENCH_HOME:-$HOME/.cache/metaljax-bench}"
MAXTEXT_DIR="$BENCH_HOME/maxtext"
MAXTEXT_REPO="${MAXTEXT_REPO:-$MAXTEXT_DIR/repo}"
VENV="${1:-$MAXTEXT_DIR/venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

mkdir -p "$MAXTEXT_DIR"

# ---------------------------------------------------------------- checkout
if [ ! -d "$MAXTEXT_REPO/src/maxtext" ]; then
  echo "==> fetching AI-Hypercomputer/maxtext into $MAXTEXT_REPO"
  # A tarball, not a clone: nothing here needs git history.
  curl -sL -o "$MAXTEXT_DIR/maxtext.tar.gz" \
    https://github.com/AI-Hypercomputer/maxtext/archive/refs/heads/main.tar.gz
  tar -xzf "$MAXTEXT_DIR/maxtext.tar.gz" -C "$MAXTEXT_DIR"
  mv "$MAXTEXT_DIR/maxtext-main" "$MAXTEXT_REPO"
  rm -f "$MAXTEXT_DIR/maxtext.tar.gz"
else
  echo "==> reusing MaxText checkout at $MAXTEXT_REPO"
fi

# ------------------------------------------------------------------- venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating venv $VENV (python $PYTHON_VERSION)"
  uv venv "$VENV" --python "$PYTHON_VERSION"
fi
export VIRTUAL_ENV="$VENV"

echo "==> installing metaljax (editable) from $REPO_ROOT"
uv pip install -e "$REPO_ROOT"

echo "==> installing MaxText dependencies"
uv pip install -r "$HERE/requirements_maxtext_macos.txt" -c "$HERE/constraints_maxtext.txt"

# tokamax pins its own jax; --no-deps keeps ours. Its transitive imports
# (immutabledict/einshape/typeguard) are in requirements_maxtext_macos.txt.
echo "==> installing tokamax (--no-deps)"
uv pip install --no-deps tokamax

echo "==> installing maxtext (--no-deps; its extras are TPU/CUDA only)"
uv pip install --no-deps -e "$MAXTEXT_REPO"

echo "==> verifying"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for p in ("jax", "jaxlib", "metaljax", "maxtext", "flax", "orbax-checkpoint", "qwix"):
    try:
        print(f"  {p:18s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:18s} MISSING")
assert md.version("jax") == "0.11.0", "jax was moved off 0.11.0"
assert md.version("jaxlib") == "0.11.0", "jaxlib was moved off 0.11.0"
PY

DECOUPLE_GCLOUD=TRUE JAX_PLATFORMS=cpu KMP_DUPLICATE_LIB_OK=TRUE \
  "$VENV/bin/python" -c "
from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
print('  maxtext imports OK')" >/dev/null 2>&1 \
  && echo "  maxtext imports OK" || { echo "  maxtext import FAILED"; exit 1; }

cat <<EOF

Done. Environment: $VENV
Next: $HERE/convert_checkpoints.sh "$VENV"
Then: JAX_PLATFORMS=metal $VENV/bin/python $HERE/adapter_maxtext.py qwen3-06b-maxtext
EOF
