#!/usr/bin/env bash
# Release gate step 3 — texmo correctness + performance.
#
#   scripts/release/texmo_gate.sh [--smoke]
#
# Chains, strictly sequentially, holding the machine lock throughout:
#   (a) scripts/texmo_check.py ~/texmo/benchmarks/m5-metal.csv
#       whole-model vs jax-CPU on all 104 suite configurations.
#       Baseline today: 96 ok + 8 ERROR on the tokens.32.raw_fold.* specs
#       (missing tokenset in ~/texmo/tokens — environmental).  The wrapper
#       classifies: raw_fold environment errors = warning; any other error,
#       or any FAIL, = gate fail.
#   (b) scripts/texmo_topconfs.py top_confs.jsonl --out notes/data/<dated>
#       163-config perf + per-config correctness sweep.
#   (c) scripts/texmo_topconfs_compare.py <anchor> <new>
#       geomean of per-config metal ms/step ratios vs the 0.11.2 anchor
#       notes/data/texmo-topconfs-final.jsonl.
#       Gate: geomean regression > TEXMO_GEOMEAN_TOL (default 3%) or any
#       single config > TEXMO_CONFIG_TOL x slower (default 1.3).
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
SUITE_CSV=${TEXMO_SUITE_CSV:-$HOME/texmo/benchmarks/m5-metal.csv}
TOPCONFS=${TEXMO_TOPCONFS:-$MJ_ROOT/top_confs.jsonl}
ANCHOR=${TEXMO_ANCHOR:-$MJ_ROOT/notes/data/texmo-topconfs-final.jsonl}
NEW_JSONL=${TEXMO_OUT:-$MJ_ROOT/notes/data/texmo-topconfs-$GATE_DATE.jsonl}
CHECK_STEPS=${TEXMO_CHECK_STEPS:-8}
SUFFIX=""
TOPCONF_ARGS=()

if [ $SMOKE -eq 1 ]; then
  SUFFIX="-smoke"
  SUITE_CSV=${TEXMO_SUITE_CSV:-$HERE/smoke_texmo_configs.csv}
  NEW_JSONL=${TEXMO_OUT:-$GATE_DIR/texmo-topconfs-smoke.jsonl}
  CHECK_STEPS=${TEXMO_CHECK_STEPS:-2}
  TOPCONF_ARGS=(--filter "${TEXMO_SMOKE_FILTER:-rnn.1.tanh-suffix.4-dense.1.tanh}")
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

gate_say "(a) texmo_check.py $SUITE_CSV ($CHECK_STEPS steps/chunk) -> $CHECK_LOG"
(cd "$MJ_ROOT" && "$MJ_PY" scripts/texmo_check.py "$SUITE_CSV" "$CHECK_STEPS") \
    > "$CHECK_LOG" 2>&1
CHECK_RC=$?
gate_say "    check rc=$CHECK_RC; $(grep -c '^ok' "$CHECK_LOG" 2>/dev/null) ok lines"

gate_say "(b) texmo_topconfs.py -> $NEW_JSONL (log $TOP_LOG)"
# ${TOPCONF_ARGS[@]+...}: bash 3.2 + set -u errors on an empty array expansion
(cd "$MJ_ROOT" && "$MJ_PY" scripts/texmo_topconfs.py "$TOPCONFS" \
    --out "$NEW_JSONL" ${TOPCONF_ARGS[@]+"${TOPCONF_ARGS[@]}"}) > "$TOP_LOG" 2>&1
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
