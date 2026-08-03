#!/usr/bin/env bash
# 0.11.2 release gate: sequential re-measurement of every runnable
# metaljax/CPU cell at shipped defaults, with token-agreement data
# recorded for compare_tokens.py. One process at a time, lock held.
set -uo pipefail

L=~/.cache/metaljax-bench/logs
V=~/.cache/metaljax-bench/venvs
M=~/.cache/metaljax-bench/maxtext/venv/bin/python
OUT=$L/final_run.jsonl
LOCK=/tmp/metaljax-bench.lock
cd /Users/oleg/metaljax

grab() { until mkdir $LOCK 2>/dev/null; do sleep 20; done; }
drop() { rmdir $LOCK 2>/dev/null || true; }
trap drop EXIT

run() {  # venv-python id backend
  grab
  echo "=== $2 $3"
  env KERAS_BACKEND=jax KMP_DUPLICATE_LIB_OK=TRUE "$1" \
    scripts/model_bench/run_bench.py "$2" "$3" --out "$OUT" || true
  drop
}

: > "$OUT"
run $V/gemma/bin/python gemma4-12b-bf16 metaljax
run $V/gemma/bin/python gemma4-12b-bf16 cpu
run $V/gemma/bin/python gemma4-31b-bf16 metaljax
run $V/bench/bin/python gemma4-e2b-bf16 metaljax
run $V/bench/bin/python gemma4-e2b-bf16 cpu
run $V/bench/bin/python qwen3-8b-bf16 metaljax
run $V/bench/bin/python qwen3-8b-bf16 cpu
run $V/bench/bin/python llama31-8b-bf16 metaljax
run $V/bench/bin/python llama31-8b-bf16 cpu
run $V/bench/bin/python gpt-oss-20b metaljax
run $V/bench/bin/python gemma4-26b-a4b metaljax
run $V/bench/bin/python gemma4-e2b-int4 metaljax
run $V/bench/bin/python gemma4-e2b-int4 cpu
run $V/bench/bin/python siglip2-so400m metaljax
run $V/bench/bin/python siglip2-so400m cpu
run $V/bench/bin/python lora-gemma4-e2b cpu
# lora metal: attempt once; the guard may kill the load transient
run $V/bench/bin/python lora-gemma4-e2b metaljax

# maxtext rows (decode + int8 + train WITH timing)
for spec in "qwen3-06b-maxtext metal" "qwen3-06b-maxtext cpu" \
            "maxtext-qwix-int8 metal" "maxtext-qwix-int8 cpu"; do
  set -- $spec
  grab; echo "=== $1 $2"
  env JAX_PLATFORMS=$2 MAXTEXT_PREFILL_LEN=64 KMP_DUPLICATE_LIB_OK=TRUE \
    $M scripts/model_bench/adapter_maxtext.py $1 --decode-tokens 64 \
    >> "$OUT.maxtext" 2>>$L/final_maxtext.err || true
  drop
done
for plat in metal cpu; do
  grab; echo "=== maxtext-train-06b $plat"
  env JAX_PLATFORMS=$plat MAXTEXT_TRAIN_STEPS=4 MAXTEXT_TRAIN_SEQ=256 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    $M scripts/model_bench/adapter_maxtext.py maxtext-train-06b --train \
    >> "$OUT.maxtext" 2>>$L/final_maxtext.err || true
  drop
done

# NB the texmo topconfs sweep moved to the release-gate step 3
# (scripts/release/texmo_gate.sh) — it must NOT run here: writing
# notes/data/texmo-topconfs-final.jsonl in place would clobber the perf
# anchor that step 3 compares against.

echo FINAL_RUN_DONE
