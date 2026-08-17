#!/usr/bin/env bash
# The 8B bf16 command-buffer canary, run against a CHOSEN mlx build.
#
#   scripts/mlx_patch_canary.sh <label> [env=val ...]
#
# Stage 1 (the Python engine) on purpose: its MLX comes from the venv, so
# swapping libmlx is a venv swap and needs no plugin relink.  Everything is
# written under ~/.cache/metaljax-bench/logs/mlx-patch/ as it is produced.
#
# The asset is notes/data/qwen3_8b_prefill_36layer.mlir (real-shape 36-layer
# Qwen3-8B maxtext prefill, bf16 weights as arguments, no quantization).  The
# reference is the jax-CPU npz produced by row15_ladder.sh rung B; the harness
# has not changed since it was written, so it is reused rather than re-measured
# (16 GB of parameters on the CPU side).
#
# Caller holds the machine lock.  ~20 GB, ~1 min per arm.
set -uo pipefail
cd /Users/oleg/metaljax || exit 2

LABEL=${1:?label}; shift
V=${MLX_CANARY_VENV:-$HOME/.cache/metaljax-bench/venvs/mlxsrc}
L=$HOME/.cache/metaljax-bench/logs/mlx-patch
REF=${MLX_CANARY_REF:-$HOME/.cache/metaljax-bench/logs/row15-mechanism/B-cpu-ref.npz}
ASSET=notes/data/qwen3_8b_prefill_36layer.mlir
REPS=${MLX_CANARY_REPS:-3}
mkdir -p "$L"

[ -f "$REF" ] || { echo "missing reference $REF"; exit 2; }
[ -f "$ASSET" ] || { echo "missing asset $ASSET"; exit 2; }

LOG=$L/canary-$LABEL.log
echo "=== $LABEL $(date '+%F %T') ===" | tee "$LOG"
"$V/bin/python" -c "import mlx.core as mx; print('mlx', mx.__version__, mx.__file__)" 2>&1 | tee -a "$LOG"

# METALJAX_ENGINE=py: the pure-Python engine, so every MLX call comes from the
# venv's mlx and swapping libmlx is a venv swap.  The native accelerated engine
# (native/build) links libmlx directly and refuses a version skew at import.
env JAX_PLATFORMS=metal,cpu METALJAX_SYNC=1 METALJAX_ENGINE=py "$@" \
  scripts/model_bench/mem_guard.sh 70 "$L/canary-$LABEL-flight.log" \
  "$V/bin/python" scripts/run_stablehlo_bench.py "$ASSET" --platform metal \
    --reps "$REPS" --warmup 1 --save-out "$L/canary-$LABEL.npz" --check "$REF" \
  2>&1 | tee -a "$LOG"

echo "--- verdict ---" | tee -a "$LOG"
grep -E "^RESULT|check|FAIL|max_abs_err|max_norm_err" "$LOG" | tail -8
