#!/usr/bin/env bash
# Row-15 mechanism ladder, rungs A and B — the ones that need no checkpoint.
# Design and hypotheses: notes/row15-wrong-output-2026-08-17.md.
#
#   scripts/model_bench/row15_ladder.sh A        # int8 arithmetic, ~1 min, <2 GB
#   scripts/model_bench/row15_ladder.sh B        # the 8B bf16 canary, ~20 GB
#   scripts/model_bench/row15_ladder.sh A,B
#
# One machine-lock hold for the whole invocation, one guarded process per arm,
# artifacts written AS PRODUCED under ~/.cache/metaljax-bench/logs/row15-mechanism/.
# Never chains a big run onto an unsettled machine; never removes a lock it
# does not hold.
#
# ROW15_RAISED=1 additionally runs the MLX_MAX_MB_PER_BUFFER=2048 control on the
# STANDALONE replay (~20 GB, no checkpoint load).  The full-row equivalent is
# what panicked the machine twice (#4/#5) and needs Oleg's sign-off — TASKS.md:
# "DO NOT run 8B-class maxtext on metal at raised budgets without a memory
# watchdog and Oleg's sign-off."  The replay does not do the load that wedged
# it, and is guarded here, but the arm stays opt-in.
set -uo pipefail
STEPS=${1:?rungs, e.g. A or A,B}
cd /Users/oleg/metaljax || exit 2
L=$HOME/.cache/metaljax-bench/logs/row15-mechanism
LOCK=/tmp/metaljax-bench.lock
BENCH=$HOME/.cache/metaljax-bench/venvs/bench/bin/python
ASSET=notes/data/qwen3_8b_prefill_36layer.mlir
mkdir -p "$L"
D=$L/ladder.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$D"; }
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }

# The stacks under test: unset = Stage 1 (plugin/build/libmetal_pjrt.dylib),
# a frozen dylib = the native plugin.  Same selection g5_model.sh uses.
NATIVE=${ROW15_NATIVE_DYLIB:-$(cat "$HOME/.cache/metaljax-bench/logs/release-0.11.5/frozen-path.txt" 2>/dev/null)}

w=0
until mkdir "$LOCK" 2>/dev/null; do
  [ $w -eq 0 ] && say "lock: waiting for $LOCK ..."
  sleep 20; w=$((w+20)); [ $w -ge 43200 ] && { say "lock: gave up"; exit 75; }
done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
say "lock acquired after ${w}s; rungs=$STEPS; native=$NATIVE"

settle() {   # the machine has to be recovered before anything is measured
  local s=0 claimed swap
  while :; do
    claimed=$(vm_stat | awk '/page size of/{ps=$8} /Pages wired down/{w=$4+0}
      /Anonymous pages/{a=$3+0} /Pages occupied by compressor/{c=$5+0}
      END{printf "%d", (w+a+c)*ps/1073741824}')
    swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6/1024}')
    [ "$claimed" -lt 30 ] && [ "$swap" -lt 2 ] && break
    [ $s -ge 900 ] && { say "SYSTEM NOT RECOVERED (claimed=${claimed}G swap=${swap}G)"; return 78; }
    [ $s -eq 0 ] && say "settle: claimed=${claimed}G swap=${swap}G ..."
    sleep 30; s=$((s+30))
  done
  say "precheck: claimed=${claimed}G swap=${swap}G"
}

run() {  # run <tag> <budget_gb> <env...> -- <cmd...>
  local tag=$1 budget=$2; shift 2
  local envs=() cmd=()
  while [ $# -gt 0 ]; do
    case "$1" in --) shift; cmd=("$@"); break;; *) envs+=("$1"); shift;; esac
  done
  settle || return 78
  local out=$L/$tag-$(date +%m%d-%H%M%S)
  say "ARM $tag START (budget ${budget}G)"
  env ${envs[@]+"${envs[@]}"} \
    bash scripts/model_bench/mem_guard.sh "$budget" "$out-flight.log" \
    "${cmd[@]}" > "$out.log" 2>&1
  local rc=$?
  say "ARM $tag END rc=$rc (137=guard kill)  $(basename "$out").log"
  grep -h "^RESULT" "$out.log" | tee -a "$D"
  sleep 15
}

# ---------------------------------------------------------------- rung A
# int8 arithmetic at row-15 widths, against an exact numpy reference.  No
# checkpoint, no maxtext: if this fails, hypothesis H2 is the answer and no
# 79 GB run is needed to find it.
if has A; then
  for stack in native stage1; do
    envs=(JAX_PLATFORMS=metal,cpu METALJAX_DEBUG=1)
    [ "$stack" = native ] && envs+=(METALJAX_PLUGIN_PATH="$NATIVE")
    run "A-dots-$stack" 25 "${envs[@]}" -- \
      "$BENCH" scripts/model_bench/row15_probe.py dots
    # 4 layers at 8B widths is ~3.2 GB of weights per backend; climb the
    # layer count only after the first pass shows what the machine does with
    # it (each layer adds ~0.8 GB, and the CPU arm holds its own copy).
    run "A-qwix-$stack" 30 "${envs[@]}" -- \
      "$BENCH" scripts/model_bench/row15_probe.py qwix --widths 8b --layers 4 --structure unroll
  done
  # The one candidate that is an MLX command-buffer split AND deterministic
  # AND present in row 15 only: the untied logits weight is 622 M elements
  # against a ~537 M-element command buffer, so the cut lands inside that one
  # matmul every call.  Swept across the crossing in fresh processes (MLX
  # latches the budget at load); 2048 here is the STANDALONE dot, not the
  # 8B checkpoint load that panicked the machine.
  for mb in 512 768 1024 2048; do
    run "A-big-mb$mb" 30 JAX_PLATFORMS=metal,cpu MLX_MAX_MB_PER_BUFFER=$mb \
      METALJAX_PLUGIN_PATH="$NATIVE" -- \
      "$BENCH" scripts/model_bench/row15_probe.py big
  done
fi

# ---------------------------------------------------------------- rung B
# The 2026-08-03 canary: real-shape 36-layer 8B bf16 maxtext prefill, weights
# as arguments.  It corrupted in 0.3 s at MLX_MAX_MB_PER_BUFFER=512 when the
# shipped kernel budget was 400; the budget became 800 the next day (52b90a2)
# and NOBODY HAS RE-RUN IT SINCE.  Clean here = the inherited attribution is
# stale and row 15's collapse needs a new witness.
if has B; then
  [ -f "$ASSET" ] || { say "missing $ASSET"; exit 2; }
  # CPU reference first (bf16 in, f32 out), then the two metal replays: the
  # two must agree with each other bit-for-bit AND with the reference.
  run "B-cpu-ref" 40 JAX_PLATFORMS=cpu -- \
    "$BENCH" scripts/run_stablehlo_bench.py "$ASSET" --platform cpu \
      --reps 1 --warmup 1 --save-out "$L/B-cpu-ref.npz"
  for stack in native stage1; do
    envs=(JAX_PLATFORMS=metal,cpu METALJAX_SYNC=1)
    [ "$stack" = native ] && envs+=(METALJAX_PLUGIN_PATH="$NATIVE")
    run "B-metal-$stack" 40 "${envs[@]}" -- \
      "$BENCH" scripts/run_stablehlo_bench.py "$ASSET" --platform metal \
        --reps 3 --warmup 1 --save-out "$L/B-metal-$stack.npz" \
        --check "$L/B-cpu-ref.npz"
    # the op-by-op control: same program, no mx.compile anywhere
    run "B-metal-$stack-nocompile" 40 "${envs[@]}" METALJAX_COMPILE=0 -- \
      "$BENCH" scripts/run_stablehlo_bench.py "$ASSET" --platform metal \
        --reps 1 --warmup 1 --save-out "$L/B-metal-$stack-nocompile.npz" \
        --check "$L/B-cpu-ref.npz"
  done
  if [ "${ROW15_RAISED:-0}" = 1 ]; then
    run "B-metal-native-mb2048" 40 JAX_PLATFORMS=metal,cpu METALJAX_SYNC=1 \
      METALJAX_PLUGIN_PATH="$NATIVE" MLX_MAX_MB_PER_BUFFER=2048 -- \
      "$BENCH" scripts/run_stablehlo_bench.py "$ASSET" --platform metal \
        --reps 3 --warmup 1 --check "$L/B-cpu-ref.npz"
  fi
fi

say "ladder $STEPS complete"
