#!/usr/bin/env bash
# Release gate step 3 — texmo correctness + performance.
#
#   scripts/release/texmo_gate.sh [--smoke]
#
# Chains, strictly sequentially, holding the machine lock throughout:
#   (a) plugin-native/texmo_gate.py benchmarks/texmo-suite.csv
#       whole-model vs jax-CPU on every suite configuration, through the
#       PJRT plugin.  Baseline: 106 ok, 0 decline, 0 FAIL/ERROR.
#   (b) scripts/bench_texmo_pjrt.py top_confs.jsonl --out notes/data/<dated>
#       223-config perf sweep, also through the plugin (the jax route).
#   (c) scripts/texmo_topconfs_compare.py <anchor> <new>
#       geomean of per-config metal ms/step ratios vs the anchor.
#       Gate: geomean regression > TEXMO_GEOMEAN_TOL (default 3%) or any
#       single config > TEXMO_CONFIG_TOL x slower (default 1.3).
#
# ROUTE + ANCHOR RE-BASELINE (0.11.6, the Stage-1 retirement).  Both legs
# used to run Stage-1-only drivers -- scripts/texmo_check.py and
# scripts/texmo_topconfs.py drove `metaljax.engine` in-process, so they
# could not see a PJRT plugin at all and were deleted with the engine.
# Consequences, decided with Oleg before the switch:
#   * correctness moves to plugin-native/texmo_gate.py (same
#     sensitivity-scaled tolerance, now over the plugin), 106/106;
#   * perf moves to scripts/bench_texmo_pjrt.py, whose records are
#     per-platform (`ms_step`/`platform`), not the retired dual-leg
#     (`metal_ms_step`/`cpu_ms_step`) shape;
#   * the 0.11.2-era anchor notes/data/texmo-topconfs-final.jsonl is
#     RETIRED twice over: wrong route, and it covers the superseded
#     163-config set (top_confs.jsonl became the 223-config 16k set at
#     112ae10).  The anchor is now
#     notes/data/topconfs16k-metal-2026-08-22.jsonl -- same route, same
#     config set.
#   * the suite-CSV (106-config) perf anchor, for runs that bench the
#     suite rather than top_confs, is the 0.11.5 release native arm:
#     $TEXMO_SUITE_ANCHOR (see below).  It still lives in the bench cache;
#     commit it under notes/data/ with the next release's raw data.
#
# Wall time: (a) ~40-60 min, (b) ~1-1.5 h.
#
# Output under $GATE_DIR:
#   texmo_check.log, texmo_topconfs.log, texmo_compare.log,
#   texmo_gate.md, texmo_gate.json
#   perf raw data: notes/data/texmo-topconfs-<date>.jsonl (commit it)
#
# Exit: 0 pass, 1 gate fail, 2 harness error.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")
. "$HERE/gatelib.sh"

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

gate_init
SUITE_CSV=${TEXMO_SUITE_CSV:-$MJ_ROOT/benchmarks/texmo-suite.csv}
TOPCONFS=${TEXMO_TOPCONFS:-$MJ_ROOT/top_confs.jsonl}
ANCHOR=${TEXMO_ANCHOR:-$MJ_ROOT/notes/data/topconfs16k-metal-2026-08-22.jsonl}
# Not used by this gate's compare leg; recorded so the suite-perf anchor
# has one home.  (0.11.5 release native arm.)
TEXMO_SUITE_ANCHOR=${TEXMO_SUITE_ANCHOR:-$HOME/.cache/metaljax-bench/logs/release-0.11.5/suite106-native.jsonl}
NEW_JSONL=${TEXMO_OUT:-$MJ_ROOT/notes/data/texmo-topconfs-$GATE_DATE.jsonl}
CHECK_STEPS=${TEXMO_CHECK_STEPS:-8}
SUFFIX=""
CHECK_ARGS=()
BENCH_ARGS=()

if [ $SMOKE -eq 1 ]; then
  SUFFIX="-smoke"
  SUITE_CSV=${TEXMO_SUITE_CSV:-$HERE/smoke_texmo_configs.csv}
  NEW_JSONL=${TEXMO_OUT:-$GATE_DIR/texmo-topconfs-smoke.jsonl}
  CHECK_STEPS=${TEXMO_CHECK_STEPS:-2}
  BENCH_ARGS=(--only "${TEXMO_SMOKE_FILTER:-rnn.1.tanh-suffix.4-dense.1.tanh}")
  gate_say "SMOKE mode: csv=$SUITE_CSV steps=$CHECK_STEPS" \
           "filter=${TEXMO_SMOKE_FILTER:-rnn.1.tanh-suffix.4-dense.1.tanh}"
fi

CHECK_LOG=$GATE_DIR/texmo_check$SUFFIX.log
TOP_LOG=$GATE_DIR/texmo_topconfs$SUFFIX.log
CMP_LOG=$GATE_DIR/texmo_compare$SUFFIX.log
MD=$GATE_DIR/texmo_gate$SUFFIX.md
JSON=$GATE_DIR/texmo_gate$SUFFIX.json

trap gate_unlock EXIT
gate_lock
T0=$(gate_now)

gate_say "(a) plugin-native/texmo_gate.py $SUITE_CSV ($CHECK_STEPS steps/chunk) -> $CHECK_LOG"
(cd "$MJ_ROOT" && "$MJ_PY" plugin-native/texmo_gate.py "$SUITE_CSV" \
    "$CHECK_STEPS" ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}) > "$CHECK_LOG" 2>&1
CHECK_RC=$?
gate_say "    check rc=$CHECK_RC; $(grep -c '^ok' "$CHECK_LOG" 2>/dev/null) ok lines"

gate_say "(b) bench_texmo_pjrt.py -> $NEW_JSONL (log $TOP_LOG)"
# ${BENCH_ARGS[@]+...}: bash 3.2 + set -u errors on an empty array expansion
(cd "$MJ_ROOT" && "$MJ_PY" scripts/bench_texmo_pjrt.py "$TOPCONFS" \
    --out "$NEW_JSONL" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}) > "$TOP_LOG" 2>&1
TOP_RC=$?
gate_say "    topconfs rc=$TOP_RC"

gate_say "(c) texmo_topconfs_compare.py $ANCHOR $NEW_JSONL -> $CMP_LOG"
(cd "$MJ_ROOT" && "$MJ_PY" scripts/texmo_topconfs_compare.py \
    "$ANCHOR" "$NEW_JSONL") > "$CMP_LOG" 2>&1
CMP_RC=$?
gate_say "    compare rc=$CMP_RC"

T1=$(gate_now)
SECS=$(gate_elapsed "$T0" "$T1")
gate_unlock

"$MJ_PY" "$MJ_ROOT/scripts/release/texmo_gate_report.py" \
    --check-log "$CHECK_LOG" --check-rc "$CHECK_RC" \
    --topconfs-log "$TOP_LOG" --topconfs-rc "$TOP_RC" \
    --compare-log "$CMP_LOG" --compare-rc "$CMP_RC" \
    --anchor "$ANCHOR" --new "$NEW_JSONL" \
    --seconds "$SECS" --md "$MD" --json "$JSON" \
    $([ $SMOKE -eq 1 ] && echo --smoke)
RRC=$?

gate_say "TEXMO_GATE_DONE rc=$RRC (${SECS}s)"
exit $RRC
