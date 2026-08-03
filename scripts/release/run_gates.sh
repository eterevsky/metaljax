#!/usr/bin/env bash
# Release-gate orchestrator — steps 2, 3, 4 then the consolidated summary.
#
#   nohup scripts/release/run_gates.sh > ~/gates.log 2>&1 &
#   scripts/release/run_gates.sh --smoke        # cheap plumbing check
#   scripts/release/run_gates.sh --only texmo   # one step (jax|texmo|models)
#
# Runs the three gates STRICTLY SEQUENTIALLY (they all want the whole GPU)
# and CONTINUES ON GATE FAILURE, so one overnight run yields the complete
# picture rather than stopping at the first problem.  Per-step wall time and
# exit status are recorded in $GATE_DIR/steps.tsv; scripts/release/summary.py
# then consolidates every step into $GATE_DIR/summary.md — the step-5 report
# Oleg greenlights from.
#
# Everything lands in ~/.cache/metaljax-bench/logs/release-gate/<date>/
# (override: RELEASE_GATE_DIR, or GATE_DATE for the directory name only).
#
# Exit code = number of failed steps (0 = all clean).
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")
. "$HERE/gatelib.sh"

SMOKE=""
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke" ;;
    --only) shift; ONLY="${1:-}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

gate_init
STEPS=$GATE_DIR/steps.tsv
: > "$STEPS"
gate_say "release gates -> $GATE_DIR ${SMOKE:+(SMOKE)}"
gate_say "tree: $(cd "$MJ_ROOT" && git rev-parse --short HEAD) $(cd "$MJ_ROOT" && git status --porcelain | wc -l | tr -d ' ') dirty files"

FAILS=0

run_step() {  # name json-stem script...
  local name=$1 stem=$2; shift 2
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
    gate_say "skip $name (--only $ONLY)"
    printf '%s\tSKIP\t0\t0\n' "$name" >> "$STEPS"
    return 0
  fi
  local t0 t1 secs rc status json
  gate_say "==== START $name"
  t0=$(gate_now)
  "$@" ${SMOKE:+$SMOKE}
  rc=$?
  t1=$(gate_now)
  secs=$(gate_elapsed "$t0" "$t1")
  status=PASS
  [ $rc -eq 1 ] && status=FAIL
  [ $rc -gt 1 ] && status=ERROR
  # prefer the step's own verdict (it distinguishes WARN from PASS)
  json=$GATE_DIR/$stem${SMOKE:+-smoke}.json
  if [ -f "$json" ] && [ $rc -le 1 ]; then
    status=$("$MJ_PY" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' \
      "$json" 2>/dev/null) || status=PASS
    [ -z "$status" ] && status=PASS
  fi
  [ $rc -ne 0 ] && FAILS=$((FAILS + 1))
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$secs" "$rc" >> "$STEPS"
  gate_say "==== END   $name status=$status rc=$rc ${secs}s"
}

run_step jax    jax_suite  "$HERE/jax_suite.sh"
run_step texmo  texmo_gate "$HERE/texmo_gate.sh"
run_step models model_gate "$HERE/model_gate.sh"

gate_say "consolidating summary"
"$MJ_PY" "$HERE/summary.py" --gate-dir "$GATE_DIR" ${SMOKE:+--smoke} \
    --out "$GATE_DIR/summary${SMOKE:+-smoke}.md"

echo
echo "per-step wall time:"
awk -F'\t' '{printf "  %-8s %-6s %8.1f s (%.2f h) rc=%s\n", $1, $2, $3, $3/3600, $4}' "$STEPS"
echo
gate_say "summary: $GATE_DIR/summary${SMOKE:+-smoke}.md"
echo "RELEASE_GATES_DONE failed_steps=$FAILS dir=$GATE_DIR"
exit $FAILS
