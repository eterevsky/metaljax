#!/usr/bin/env bash
# Memory watchdog + footprint profiler for the big-model benchmark rows.
#
#   scripts/model_bench/mem_guard.sh <budget_gb> <log> <cmd> [args...]
#
# Runs <cmd> as a child, samples ITS footprint about twice a second into
# <log>, and kills it (TERM, then KILL after 5 s) when EITHER
#
#   * its own footprint crosses <budget_gb>, or
#   * the whole machine crosses GUARD_SYS_GB (default 100 of 128), or
#   * its RSS crosses GUARD_RSS_GB (default 110; 0 disables) — RSS counts
#     the mmap'd checkpoint pages footprint ignores, and panic #7 wedged
#     the machine at RSS 101.9 with footprint 53 and every metric "ok", or
#   * its projected footprint one sample ahead — measured slope times the
#     sample period, doubled — would cross the budget.
#
# Exit status is the child's, or 137 if the guard fired.
#
# WHY.  A 128 GB machine gives no second chance: an LLM load that overshoots
# is a zero-locality sweep, so swap cannot help — the machine either
# jetsam-kills the process (best case) or wedges and PANICS (five panics on
# this project so far; the R1-32B keras load and a Qwen3.6 run that reached
# a 196 GB footprint are two of them).  The guard is the only thing that
# turns that into a data point.
#
# WHY THE SLOPE TERM.  Measured on this machine: a load can add 20-40 GB in
# five seconds.  A watchdog that only reacts after the threshold is already
# crossed is decoration at that ramp rate — by the time it fires the machine
# is 15 GB past the line.  So the guard also kills on trajectory.
#
# The log doubles as the load-time memory profiler: one line per sample,
#
#   <epoch> <footprint_gb> <rss_gb> <sys_used_gb> <swap_used_gb> <state> \
#       file=<file_backed_gb> free=<free_gb>
#
# The two trailing fields are APPENDED (nothing that parsed columns 2 and 3
# changes) and they are the ones the machine-wedge panics happened in: see the
# comment beside the vm_stat call below.
#
# Sampled with `top -l 1 -o mem -stats pid,mem,command` (MEM = the process's
# physical footprint, the number jetsam actually reads) plus `ps` RSS and
# `sysctl vm.swapusage`.  Merge it against the PHASE lines the loader
# prints to attribute a peak to a load phase.
#
# NB pick the budget to kill EARLY.  A load that needs more headroom than
# you budgeted is a result to report, not a threshold to raise.  Keep the
# log outside /private/tmp: a panic wipes the session scratchpad.
set -uo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 <budget_gb> <log> <cmd> [args...]" >&2
  exit 2
fi
BUDGET_GB=$1; LOG=$2; shift 2
SYS_GB=${GUARD_SYS_GB:-100}
RSS_CAP_GB=${GUARD_RSS_GB:-110}
PERIOD=${GUARD_PERIOD:-0.5}

"$@" &
CHILD=$!
trap 'kill -TERM $CHILD 2>/dev/null; exit 130' INT TERM

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
echo "# guard budget_gb=$BUDGET_GB sys_gb=$SYS_GB rss_gb=$RSS_CAP_GB period=$PERIOD pid=$CHILD cmd=$*" >> "$LOG"
KILLED=0
PREV=0
PREV_T=$(date +%s)
TRAJ_PENDING=0

fire() {  # reason
  KILLED=1
  echo "# GUARD FIRED: $1" >> "$LOG"
  echo "mem_guard: $1 -> killing $CHILD" >&2
  kill -TERM $CHILD 2>/dev/null
  ( sleep 5; kill -KILL $CHILD 2>/dev/null ) &
}

while kill -0 $CHILD 2>/dev/null; do
  SAMPLE=$(/usr/bin/top -l 1 -o mem -n 12 -stats pid,mem,command 2>/dev/null)
  FOOT=$(echo "$SAMPLE" | awk -v p=$CHILD '$1==p {print $2; exit}')
  # System pressure = CLAIMED memory (wired + anonymous + compressor), NOT
  # top's PhysMem "used": that figure includes the reclaimable file cache,
  # which a 60+ GB checkpoint read fills legitimately -- it false-tripped
  # the ceiling at zero actual pressure (rows 8/9, 2026-08-03).
  # ...and, in the same call, the two counters that were the ONLY unhealthy
  # thing at both machine wedges (panics #7 and #9): the file-backed page
  # count and the free list.  NOT a kill rule -- a checkpoint read fills the
  # page cache legitimately and killing on it would kill every big load.  They
  # are LOGGED because the flight logs of both wedges are missing exactly this
  # column: every metric the guard sampled read "ok" while physical memory was
  # full of file pages and the free list was on the floor.  The in-process
  # memory governor (runtime/memory.cc) is what acts on them.
  read -r SYS FILEB FREEB <<< "$(vm_stat 2>/dev/null | awk '
      /page size of/ {ps=$8}
      /Pages wired down/ {w=$4+0}
      /Anonymous pages/ {a=$3+0}
      /Pages occupied by compressor/ {c=$5+0}
      /Pages free/ {f=$3+0}
      /File-backed pages/ {fb=$3+0}
      END {printf "%.1fG %.1f %.1f", (w+a+c)*ps/1073741824,
                                     fb*ps/1073741824, f*ps/1073741824}')"
  RSSKB=$(ps -o rss= -p $CHILD 2>/dev/null | tr -d ' ')
  SWAP=$(/usr/sbin/sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}')
  NOW=$(date +%s)
  # top prints 12345K / 1234M / 61G (a trailing +/- means "growing")
  FOOT_GB=$(echo "${FOOT:-0}" | awk '{
      v=$0; sub(/[+-]$/,"",v); u=substr(v,length(v),1); n=substr(v,1,length(v)-1)+0;
      if (u=="G") printf "%.2f", n; else if (u=="M") printf "%.2f", n/1024;
      else if (u=="K") printf "%.3f", n/1048576; else printf "%.2f", $0/1073741824 }')
  RSS_GB=$(echo "${RSSKB:-0}" | awk '{printf "%.2f", $1/1048576}')
  SYS_USED=$(echo "${SYS:-0}" | awk '{
      u=substr($0,length($0),1); n=substr($0,1,length($0)-1)+0;
      if (u=="G") printf "%.1f", n; else if (u=="M") printf "%.1f", n/1024;
      else printf "0" }')
  printf '%s %s %s %s %s %s file=%s free=%s\n' "$NOW" "$FOOT_GB" "$RSS_GB" \
      "${SYS:-?}" "${SWAP:-?}" "$([ $KILLED = 1 ] && echo killing || echo ok)" \
      "${FILEB:-?}" "${FREEB:-?}" >> "$LOG"

  if [ $KILLED = 0 ]; then
    VERDICT=$(awk -v f="$FOOT_GB" -v p="$PREV" -v b="$BUDGET_GB" \
                  -v s="$SYS_USED" -v sb="$SYS_GB" -v per="$PERIOD" \
                  -v r="$RSS_GB" -v rb="$RSS_CAP_GB" 'BEGIN{
        if (f > b) { printf "footprint %.2f GB > budget %s GB", f, b; exit }
        if (s > sb) { printf "system %.1f GB used > %s GB", s, sb; exit }
        # RSS counts the mapped checkpoint pages footprint does not.
        # Kernel panic #7: a streamed load wedged the machine with
        # footprint 53 GB but RSS at 101.9 -- file-backed pages are
        # "reclaimable", so no memory metric looked unhealthy while the
        # VM fought a reclaim storm. RSS is the leading indicator for
        # that class; GUARD_RSS_GB (default 110) caps it.
        if (rb > 0 && r > rb) { printf "rss %.2f GB > GUARD_RSS_GB %s", r, rb; exit }
        d = f - p;                       # growth since the last sample
        if (d > 0 && f + 2*d > b)
          printf "projected %.2f GB (+%.2f GB/sample) crosses budget %s GB", f+2*d, d, b;
      }')
    # The trajectory rule needs TWO consecutive projected crossings: a
    # single big allocation landing between samples (an Orbax restore
    # placing a whole parameter set, a 1 GB expert-bank assign) reads as
    # an infinite ramp for exactly one sample, then d collapses to zero.
    # A real runaway (panic #5: +2-4 GB/sample SUSTAINED for seconds)
    # keeps projecting over and still dies within one extra period.
    # Absolute-threshold verdicts (footprint/system/rss) fire at once.
    case "$VERDICT" in
      projected*)
        if [ "$TRAJ_PENDING" = 1 ]; then fire "$VERDICT (2nd consecutive)"
        else TRAJ_PENDING=1; fi ;;
      "") TRAJ_PENDING=0 ;;
      *) fire "$VERDICT" ;;
    esac
  fi
  PREV=$FOOT_GB
  PREV_T=$NOW
  sleep "$PERIOD"
done

wait $CHILD
RC=$?
echo "# child exit=$RC killed=$KILLED" >> "$LOG"
[ $KILLED = 1 ] && exit 137
exit $RC
