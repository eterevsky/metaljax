#!/usr/bin/env bash
# Release gate step 2 — pinned jax-v0.11.0 test suite.
#
#   scripts/release/jax_suite.sh [--smoke]
#
# Runs scripts/run_jax_tests.py against the PINNED checkout in exactly the
# configuration that produced the authoritative 27,649-passed / 130-failed
# approval run (2026-07-31, tree 725bd84):
#
#     .venv/bin/python scripts/run_jax_tests.py <outdir> \
#         --jobs 1 --tests jax-v0.11.0/tests
#
# --jobs 1 is load-bearing: parallel (3/4-job) runs UNDER-report failures
# (campaign lesson, CLAUDE.md item 20 — fft_test showed 3 failures at 4 jobs
# vs 5 sequentially).  The --tests path must stay RELATIVE: pytest node ids
# inherit it, and the whitelist notes/data/pinned-0.11.0-failures.txt is
# keyed on "jax-v0.11.0/tests/<file>.py::...".
#
# Wall time: ~3.5 h for all 164 files (17:01 -> 20:37 on the reference run).
#
# Output (all under $GATE_DIR = ~/.cache/metaljax-bench/logs/release-gate/<date>):
#   jaxtests/            per-file .log, summary.csv, failures.txt (driver output)
#   jax_suite.log        raw driver stdout/stderr
#   jax_suite.md         markdown gate summary
#   jax_suite.json       machine-readable step record (read by summary.py)
#
# Exit code: 0 = no new failures, 1 = gate fail, 2 = harness error.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")
. "$HERE/gatelib.sh"

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

gate_init
OUTDIR=$GATE_DIR/jaxtests
LOG=$GATE_DIR/jax_suite.log
EXTRA=()
if [ $SMOKE -eq 1 ]; then
  OUTDIR=$GATE_DIR/jaxtests-smoke
  LOG=$GATE_DIR/jax_suite-smoke.log
  EXTRA=(--filter "${JAX_SMOKE_FILTER:-aot_test}")
  gate_say "SMOKE mode: single file filter '${JAX_SMOKE_FILTER:-aot_test}'"
fi

trap gate_unlock EXIT
gate_lock

# Fresh outdir: stale per-file logs would leak old FAILED lines into the diff.
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

gate_say "jax pinned suite -> $OUTDIR"
T0=$(gate_now)
# ${EXTRA[@]+...}: bash 3.2 + set -u errors on an empty array expansion
"$MJ_PY" "$MJ_ROOT/scripts/run_jax_tests.py" "$OUTDIR" \
    --jobs "${JAX_SUITE_JOBS:-1}" --tests jax-v0.11.0/tests \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    > "$LOG" 2>&1
RC=$?
T1=$(gate_now)
SECS=$(gate_elapsed "$T0" "$T1")
gate_unlock
gate_say "driver rc=$RC after ${SECS}s -> $LOG"

MD=$GATE_DIR/jax_suite.md
JSON=$GATE_DIR/jax_suite.json
[ $SMOKE -eq 1 ] && { MD=$GATE_DIR/jax_suite-smoke.md; JSON=$GATE_DIR/jax_suite-smoke.json; }

"$MJ_PY" "$MJ_ROOT/scripts/release/jax_suite_diff.py" "$OUTDIR" \
    --driver-log "$LOG" --driver-rc "$RC" --seconds "$SECS" \
    --md "$MD" --json "$JSON" $([ $SMOKE -eq 1 ] && echo --smoke)
DRC=$?

gate_say "JAX_SUITE_DONE rc=$DRC (${SECS}s)"
exit $DRC
