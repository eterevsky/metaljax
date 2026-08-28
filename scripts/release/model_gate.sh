#!/usr/bin/env bash
# Release gate step 4 — real-model benchmark suite.
#
#   scripts/release/model_gate.sh [--smoke]
#
# Wraps scripts/model_bench/final_run.sh (unmodified) and then audits its
# output:
#   (a) merge/normalize the '[maxtext] <id> / <backend>' + 'RESULT {json}'
#       lines from $OUT.maxtext into records shaped like final_run.jsonl
#   (b) scripts/model_bench/compare_tokens.py on the merged file
#       (bf16 rows must agree with CPU token-for-token; QUANT_ROWS are
#        informational per notes/int8-divergence-verdict.md)
#   (c) per-row timing table vs the newest column of models.md,
#       flagging rows regressed > MODEL_REGRESS_TOL (default 10%) and rows
#       that newly fail
#
# LOCKING: this wrapper deliberately takes NO lock.  final_run.sh grabs
# /tmp/metaljax-bench.lock around EVERY cell and releases between cells —
# that per-cell protocol is the design, and an outer holder would deadlock
# it.  final_run.sh already runs its cells strictly sequentially; we run it
# once, in the foreground, and never in parallel with anything else.
#
# Wall time: 2-4 h (gpt-oss-20b alone loads for ~8 min).
#
# --smoke SKIPS final_run.sh entirely and exercises the merge + compare +
# table code against the existing ~/.cache/metaljax-bench/logs/final_run.jsonl*
# from the previous gate.
#
# Output under $GATE_DIR: model_final_run.log, model_merged.jsonl,
# model_tokens.log, model_gate.md, model_gate.json
# (final_run.sh itself writes ~/.cache/metaljax-bench/logs/final_run.jsonl{,.maxtext}
#  and notes/data/texmo-topconfs-final.jsonl — see the note in the report.)
#
# Exit: 0 pass, 1 gate fail, 2 harness error.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")
. "$HERE/gatelib.sh"

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

gate_init
BENCH_LOGS=$HOME/.cache/metaljax-bench/logs
RAW=${MODEL_RUN_JSONL:-$BENCH_LOGS/final_run.jsonl}
SUFFIX=""; [ $SMOKE -eq 1 ] && SUFFIX="-smoke"
RUN_LOG=$GATE_DIR/model_final_run$SUFFIX.log
MERGED=$GATE_DIR/model_merged$SUFFIX.jsonl
TOK_LOG=$GATE_DIR/model_tokens$SUFFIX.log
MD=$GATE_DIR/model_gate$SUFFIX.md
JSON=$GATE_DIR/model_gate$SUFFIX.json

T0=$(gate_now)
if [ $SMOKE -eq 1 ]; then
  gate_say "SMOKE mode: skipping final_run.sh, auditing existing $RAW"
  echo "SMOKE: final_run.sh not executed; auditing pre-existing $RAW" > "$RUN_LOG"
  RUN_RC=0
else
  # final_run.sh APPENDS to $OUT.maxtext (only $OUT itself is truncated):
  # rotate the old side file away so the merge cannot pick up stale rows.
  if [ -s "$RAW.maxtext" ]; then
    mv -f "$RAW.maxtext" "$RAW.maxtext.prev-$(date +%Y%m%dT%H%M%S)"
    gate_say "rotated previous $RAW.maxtext out of the way"
  fi
  # final_run.sh's trailing texmo re-baseline writes the PERF ANCHOR
  # notes/data/texmo-topconfs-final.jsonl in place.  Step 3 compares against
  # that file, so preserve it: keep the fresh sweep under a dated name and
  # put the anchor back.  (Fixing this properly means editing final_run.sh,
  # which this wrapper must not do.)
  ANCHOR_FILE=$MJ_ROOT/notes/data/texmo-topconfs-final.jsonl
  ANCHOR_BAK=$GATE_DIR/texmo-topconfs-final.anchor.bak
  [ -f "$ANCHOR_FILE" ] && cp -f "$ANCHOR_FILE" "$ANCHOR_BAK"

  gate_say "final_run.sh (sequential, per-cell lock, hours) -> $RUN_LOG"
  (cd "$MJ_ROOT" && bash scripts/model_bench/final_run.sh) > "$RUN_LOG" 2>&1
  RUN_RC=$?
  gate_say "final_run.sh rc=$RUN_RC"

  if [ -f "$ANCHOR_BAK" ]; then
    if ! cmp -s "$ANCHOR_BAK" "$ANCHOR_FILE"; then
      cp -f "$ANCHOR_FILE" "$MJ_ROOT/notes/data/texmo-topconfs-$GATE_DATE-finalrun.jsonl"
      cp -f "$ANCHOR_BAK" "$ANCHOR_FILE"
      gate_say "final_run.sh overwrote the topconfs anchor; fresh sweep saved as" \
               "notes/data/texmo-topconfs-$GATE_DATE-finalrun.jsonl, anchor restored"
    fi
  fi
  # Snapshot the raw data next to the gate logs so a later re-run can't
  # overwrite the evidence (final_run.sh truncates final_run.jsonl).
  cp -f "$RAW" "$GATE_DIR/final_run.jsonl" 2>/dev/null || true
  cp -f "$RAW.maxtext" "$GATE_DIR/final_run.jsonl.maxtext" 2>/dev/null || true
fi

T1=$(gate_now)
SECS=$(gate_elapsed "$T0" "$T1")

"$MJ_PY" "$MJ_ROOT/scripts/release/model_gate_report.py" \
    --raw "$RAW" --merged "$MERGED" --tokens-log "$TOK_LOG" \
    --run-log "$RUN_LOG" --run-rc "$RUN_RC" \
    --ledger "$MJ_ROOT/models.md" \
    --seconds "$SECS" --md "$MD" --json "$JSON" \
    $([ $SMOKE -eq 1 ] && echo --smoke)
RRC=$?

gate_say "MODEL_GATE_DONE rc=$RRC (${SECS}s)"
exit $RRC
