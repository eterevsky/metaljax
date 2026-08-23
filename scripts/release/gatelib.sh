#!/usr/bin/env bash
# Shared helpers for the release-gate wrappers (scripts/release/*.sh).
# Sourced, never executed.  Every wrapper is safely re-runnable: output
# lands in a dated directory and per-step files are overwritten in place.
set -uo pipefail

MJ_ROOT=/Users/oleg/metaljax
MJ_PY=$MJ_ROOT/.venv/bin/python
MJ_LOCK=/tmp/metaljax-bench.lock

# The plugin every gate step measures.  Release rule 1 -- every number in a
# release table comes from THE release binary -- used to rest on whatever
# happened to sit in src/metaljax/lib/; naming it here makes the gated
# binary a fact of the run instead of an implication.  Point
# METALJAX_PLUGIN_PATH at the frozen dylib for a release gate (the default
# below is the last frozen path recorded by the vendoring battery); export
# METALJAX_PLUGIN_PATH yourself to override, or set it empty to accept
# whatever the environment resolves.
if [ -z "${METALJAX_PLUGIN_PATH+set}" ]; then
  _MJ_FROZEN=$(cat "$HOME/.cache/metaljax-bench/logs/mlx-vendoring/frozen-path.txt" 2>/dev/null)
  if [ -n "$_MJ_FROZEN" ] && [ -e "$_MJ_FROZEN" ]; then
    export METALJAX_PLUGIN_PATH=$_MJ_FROZEN
  fi
fi

# Dated gate directory.  Override the whole path with RELEASE_GATE_DIR, or
# just the date component with GATE_DATE (e.g. GATE_DATE=2026-08-04-rerun).
GATE_DATE=${GATE_DATE:-$(date +%F)}
GATE_DIR=${RELEASE_GATE_DIR:-$HOME/.cache/metaljax-bench/logs/release-gate/$GATE_DATE}

gate_init() {
  mkdir -p "$GATE_DIR" || exit 1
}

gate_ts() { date "+%Y-%m-%d %H:%M:%S"; }

gate_say() { echo "[$(gate_ts)] $*"; }

# ---- machine lock -------------------------------------------------------
# The convention for GPU-exclusive work on this machine (same protocol as
# scripts/model_bench/final_run.sh).  NOT used by model_gate.sh: final_run.sh
# takes the lock per cell itself, and an outer holder would deadlock it.
# Set METALJAX_GATE_LOCK=0 to skip (e.g. when the caller already holds it).
gate_lock() {
  [ "${METALJAX_GATE_LOCK:-1}" = "1" ] || { gate_say "lock: disabled"; return 0; }
  local waited=0
  until mkdir "$MJ_LOCK" 2>/dev/null; do
    if [ $waited -eq 0 ]; then gate_say "lock: waiting for $MJ_LOCK ..."; fi
    sleep 20
    waited=$((waited + 20))
  done
  MJ_LOCK_HELD=1
  gate_say "lock: acquired after ${waited}s"
}

gate_unlock() {
  if [ "${MJ_LOCK_HELD:-0}" = "1" ]; then
    rmdir "$MJ_LOCK" 2>/dev/null || true
    MJ_LOCK_HELD=0
  fi
}

# ---- misc ---------------------------------------------------------------
gate_now() { $MJ_PY -c 'import time; print(time.time())'; }

# elapsed seconds between two gate_now stamps
gate_elapsed() { $MJ_PY -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$1" "$2"; }
