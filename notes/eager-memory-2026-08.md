# Eager-phase memory inflation: the interpreter kept every intermediate (2026-08)

## Symptom

Programs the engine runs **op by op** — anything impure or over the trace
budget, which in practice means every checkpoint-conversion and
parameter-load program — peaked at many times the data they produced. At 8B
this was lethal: the maxtext load ballooned to 67-109 GB for 16.4 GB of
weights and took the machine down twice (STATUS row 15,
notes/mlx-command-buffer-split.md). Tight `gc` + `mx.clear_cache` cadences
did not help, and the keras streaming-load path ramped R1-32B to 94 GB with
`BENCH_STREAM_CLEAR_GB=2`.

## What it is NOT

* **Not MLX's buffer cache.** `clear_cache` cannot free *referenced* buffers,
  and the ramps survived every clear cadence tried.
* **Not long lazy graphs.** MLX frees an intermediate the moment nothing
  references it, *during* the evaluation that produces it.

Both measured directly (`mlx_tape.py`, 20 dependent ops over 128 MB arrays):

| variant | peak (active) |
|---|---|
| chain, only the result referenced | 0.25 GB |
| chain, every intermediate held in a python list | **2.62 GB** |
| chain, eval every 4 ops, intermediates dropped | 0.75 GB |
| chain, eval every 4 ops, intermediates held | **2.62 GB** |
| chain, `mx.compile`d | 0.25 GB |

## What it is

`Interpreter.run_block`'s `env` mapped every SSA value to its array and kept
it until the block returned. So an eagerly interpreted block held *every*
intermediate simultaneously: peak footprint scaled with the length of the op
chain, not with the live set. Row 2 of that table is the bug.

Two consequences worth remembering:

* **Compiled programs never had it.** During an `mx.compile` trace the values
  in `env` are tracers with no data; the real intermediates live in MLX's
  tape, which frees them as it goes.
* **Forcing evaluation without fixing the retention makes it far worse** —
  a flush materializes everything retained at once instead of letting MLX
  free as it walks. Measured on the 0.6B maxtext load: 17 GB with neither
  mechanism, **75 GB and still climbing (guard-killed)** with flushing alone,
  3.5 GB with both. This is why every "clear more often" mitigation failed.

## Attribution at 0.6B (qwen3-0.6B via maxtext, 1.11 GB of bf16 weights)

The whole balloon is one program. `METALJAX_DEBUG=1` now also prints each
program's static result bytes:

| program | pure | cost | static MB | compiled | phase |
|---|---|---|---|---|---|
| `jit_create_sharded_state` | no | 127324 | 4573 | **no** | load |
| `jit__prefill_jit` | yes | 10074 | 2056 | yes | prefill |
| `jit__generate_jit` | yes | 13504 | 2013 | yes | decode |

`create_sharded_state` is impure *and* 6x over the trace budget, so it runs
eagerly in **every** configuration — which is why `METALJAX_COMPILE=0` and
the default measured identically (6.63 vs 6.70 GB peak, 6.0x the weights)
and why the compiled/uncompiled framing of the 8B ledger was a red herring.
Prefill and decode are flat (~1.2 GB) throughout.

## Fix (src/metaljax/interpreter.py)

1. **Liveness pruning** — `eager_plan(block)` computes, once per block, the
   last use of every value (consumers' operands plus the captures of any
   nested region, which is live for the whole of the op owning it) and the
   estimated bytes each op produces. The eager loop drops a value right after
   its last use. `METALJAX_ENV_PRUNE=0` restores the old behaviour.
2. **Byte-denominated flush** — after `METALJAX_EAGER_FLUSH_MB` (default
   1024) of estimated result data with no sync point, evaluate what is still
   live in `env`. Blocking by default, so the budget means what it says
   (with N async checkpoints in flight the peak is N x the budget: measured
   4.0 GB at N=4 versus 1.1 GB at N=1 on a 256 MB init).

Neither runs inside an `mx.compile` trace — a program that compiles does not
even build the plan, so it pays nothing.

## Results

0.6B maxtext, load+prefill+decode, MLX peak (weights 1.11 GB):

| configuration | peak GB | x weights | tokens |
|---|---|---|---|
| `METALJAX_COMPILE=0`, before | 6.63 | 6.0 | correct |
| default (compiled), before | 6.70 | 6.0 | pre-existing garbage |
| `METALJAX_COMPILE=0`, after | **3.49** | 3.1 | correct |
| default (compiled), after | **3.49** | 3.1 | pre-existing garbage |

Budget sweep (uncompiled, tokens correct at every point): 256 MB 3.49 GB /
512 3.49 / 1024 3.49 / 2048 4.07 / 4096 5.22 / off 6.63. The peak saturates
below 1024 MB, and tighter budgets only cost load time (6.30 s at 256 MB vs
5.41 at 1024), so 1024 MB is the cheapest budget that buys the whole win.

A single 256 MB jitted random init (the keras per-variable init shape),
peak active: eager 16.5 GB before / **1.75 GB** after. The same program
*compiled* peaks at 9.75 GB either way — see limitations.

Load weights are bit-identical to jax-CPU in every configuration tested
(sha1 over all 13 leaves, `weight_check.py`), before and after.

## Rejected: byte-denominating the LOOP flush cadence

`ops/control.py`'s eager loop flushes every `25000 // cost` iterations, which
says nothing about how much data an iteration produces — maxtext's load
writes the whole stacked parameter set per layer. Deriving the cadence from
carry bytes instead was implemented and **reverted**:

* it bought nothing (6.63 GB with it, 6.63 GB without — the block-level flush
  delivers the entire 3.49 GB win on its own);
* it produced a stable but **wrong** qwen3-0.6B token stream
  (`[12095, 12095, 5251, 8346, 11, 323]` versus CPU's
  `[12095, 13, 576, 6722, 315, 9625]`, reproduced twice), while the
  block-level flush alone is correct twice over.

That is the documented MLX command-buffer lottery: changing where a loop's
existing sync points fall reshuffles which producer/consumer pair a command
buffer boundary lands between. `ops/control.py` is therefore untouched by
this work, per TASKS.md ("eager-path scan flush cadence: values pinned by
tests; revisit only with the MLX fix").

## Limitations

* **Compiled programs are a separate wave and are NOT covered.** A compiled
  graph's intermediates live in MLX's tape; the 256 MB init above peaks at
  9.75 GB (39x) compiled. Bounding that means a *bytes* term in the compile
  decision (`MetalExecutable.runner`, next to the op-count trace budget) —
  legitimate as a memory gate (the addendum only refuted static byte
  estimates as a *correctness* gate), but it would turn compilation off for
  some programs, so it needs a perf sweep across the STATUS rows before it
  ships. `METALJAX_DEBUG=1`'s new `bytes=` field is the data to pick a
  threshold from. Warmup programs cross into the eager (covered) regime once
  they exceed the 20k op trace budget, which for maxtext-shaped models is
  somewhere above 40 layers.
* Pruning is plan-aware: recognizer matches declare emit_reads and liveness is computed against the rewritten schedule (originally pruning was disabled on rewrite-carrying blocks):
  it reads the operands of ops it *skips* when it emits the fused dot, so a
  static last use can fall on a skipped op. Those programs keep the old
  retention (and run compiled anyway).
* The estimate ignores fusion (~2x loose) and reports 0 for dynamic
  (shape-poly) dimensions, which only makes the net fire later there.

## Validation

* pytest 526 passed, identical with the mechanisms on and off.
* `texmo_check.py` whole-model gate: **104 ok, 0 FAIL, 0 error**.
* texmo perf A/B (paired, two passes, `bench_spec.py`): mgru.256 b64 4.969 vs
  4.934 ms/step, gru.1024 b32 57.79 vs 58.45, 4-block transformer b32 18.55
  vs 18.56 — all within noise. The flush does not fire on mgru.256 or the
  transformer; it fires 96 times on gru.1024 (whose chunk main is eager) and
  costs nothing measurable there.
* The gate harness itself runs the bare `Interpreter`, not the PJRT engine,
  so main blocks are eager by construction and the flush fires 88 times
  across the 104 configs — expected, and orthogonal to the perf numbers
  above, which use the real backend.
