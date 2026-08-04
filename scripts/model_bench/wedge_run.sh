#!/usr/bin/env bash
# One guarded, locked run of wedge_repro.py -- the ladder's run protocol.
#
#   scripts/model_bench/wedge_run.sh <rung> <budget_gb> <tag> [KEY=VAL ...]
#
# e.g.  wedge_run.sh small 12 soak1 BENCH_STREAM_MARK=100
#
# Same discipline as ~/.cache/metaljax-bench/logs/guarded_run.sh (which only
# knows manifest rows): take the machine lock FIRST, then require a
# recovered machine (claimed < 30 GB, swap < 2 GB) -- in that order, so the
# check sees the state after the previous run released, not during it --
# then run under mem_guard.sh with the budget the prediction justifies.
# Never chains: one rung, one run, exit.
#
# Logs go to ~/.cache/metaljax-bench/logs/wedge-ladder/<tag>-<stamp>{,-flight}
# .log (outside /private/tmp: a panic wipes the scratchpad).
set -uo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <rung> <budget_gb> <tag> [KEY=VAL ...]" >&2
                  exit 2; }
RUNG=$1; BUDGET=$2; TAG=$3; shift 3

REPO=$(cd "$(dirname "$0")/../.." && pwd)
L=/Users/oleg/.cache/metaljax-bench/logs/wedge-ladder
B=/Users/oleg/.cache/metaljax-bench/venvs/bench/bin/python
LOCK=/tmp/metaljax-bench.lock
WAIT_S=${LOCK_WAIT_S:-5400}

waited=0
until mkdir "$LOCK" 2>/dev/null; do
  sleep 20; waited=$((waited + 20))
  if [ $waited -ge $WAIT_S ]; then
    echo "lock still held after ${waited}s -- giving up"; exit 75
  fi
done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
[ $waited -gt 0 ] && echo "waited ${waited}s for the machine lock"

claimed=$(vm_stat | awk '/page size of/{ps=$8} /Pages wired down/{w=$4+0}
  /Anonymous pages/{a=$3+0} /Pages occupied by compressor/{c=$5+0}
  END{printf "%d", (w+a+c)*ps/1073741824}')
swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6/1024}')
echo "precheck: claimed=${claimed}G swap=${swap}G"
if [ "$claimed" -ge 30 ] || [ "$swap" -ge 2 ]; then
  echo "SYSTEM NOT RECOVERED -- refusing to start $RUNG/$TAG"; exit 78
fi

mkdir -p "$L"
STAMP=$(date +%m%d-%H%M%S)
echo "=== wedge $RUNG budget=${BUDGET}G tag=$TAG extra: $*"
env KERAS_BACKEND=jax KMP_DUPLICATE_LIB_OK=TRUE "$@" \
  bash "$REPO/scripts/model_bench/mem_guard.sh" "$BUDGET" \
  "$L/$TAG-$STAMP-flight.log" \
  "$B" "$REPO/scripts/model_bench/wedge_repro.py" load --rung "$RUNG" \
  > "$L/$TAG-$STAMP.log" 2>&1
rc=$?
echo "exit=$rc (137 = guard kill); logs: $L/$TAG-$STAMP{,-flight}.log"
grep -E '^\[load\] (PREDICTED|done)|^RESULT' "$L/$TAG-$STAMP.log" \
  || tail -5 "$L/$TAG-$STAMP.log"
exit $rc
