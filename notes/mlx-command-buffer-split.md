# MLX corrupts compiled graphs split across Metal command buffers (2026-08)

## Symptom

maxtext Qwen3-0.6B decode on metal with bf16 weights emitted garbage tokens,
**different on every run at fixed inputs**:

| configuration | output |
|---|---|
| `JAX_PLATFORMS=cpu` | ` Paris. The capital of France is also the capital of the Republic of France.` |
| `JAX_PLATFORMS=metal METALJAX_COMPILE=0` | identical to CPU |
| `JAX_PLATFORMS=metal` (compiled, default) | `urancesuchos책international…`, then `burgh蹋 Bourbonmitted…`, … |

Repro: `scripts/model_bench/adapter_maxtext.py qwen3-06b-maxtext
--decode-tokens 16` (see `README_maxtext.md`).

## What it is NOT (each tested, not assumed)

* **Not an async race across executables.** `METALJAX_SYNC=1` (blocking
  `mx.eval` on every execute's outputs) still produced garbage. Adding a
  compiled-vs-eager comparison inside `engine.execute` showed the *first*
  executable — prefill — already wrong, before any chaining.
* **Not the bf16 bitcast upload** (`dtypes.to_mx`'s `mx.view`). Forcing the
  old f32-staging upload still produced garbage; and translating the whole
  program to f32 (`sed s/bf16/f32/`) reproduces it identically.
* **Not donation, not msl_scan, not the splat-constant change.** `METALJAX_MSL=0`
  no effect; reproduces on the pre-`d9d774e` tree; the compiled call does not
  mutate its own inputs (checked by snapshotting every argument around it).
* **Not the M5 matmul-precision arch pin.** `METALJAX_MATMUL_PRECISION=default`
  (native `applegpu_g17s`) reproduces.

## What it is

A standalone replay of the dumped prefill module (StableHLO text +
seeded random inputs, no maxtext) reproduces: the compiled graph disagrees
with op-by-op interpretation *and* with itself across calls. The single
knob that decides it is how much work MLX packs into one Metal command
buffer. MLX starts a new command buffer once the current one holds
`MLX_MAX_OPS_PER_BUFFER` kernels or `MLX_MAX_MB_PER_BUFFER` megabytes
(`mlx/backend/metal/device.h`, `CommandEncoder::needs_commit`).

Measured on the prefill module (8 of 28 layers, compiled main):

| kernels/buffer | bytes/buffer | result |
|---|---|---|
| 400 (metaljax) | 40 (MLX default) | **corrupt** |
| 400 | 80 | **corrupt** |
| 400 | 160, 320, …, 5120 | clean |
| 1, 2, 4 | unlimited | **corrupt** |
| 8, 16, 32, 128, 400 | unlimited | clean |

So corruption appears once command buffers are cut every ~4 kernels and
disappears by ~8. The byte budget is what fires in practice: an LLM layer's
tensors are megabytes, so 40 MB commits every few kernels.

Further properties, all consistent with buffer reuse across a split:

* **`mx.compile` is required.** The same module interpreted op-by-op, with
  the loop flush suppressed so it is still ONE huge `mx.eval`, is clean.
* **The first call of a compiled function is clean; calls 2+ are corrupt.**
  MLX retains the trace's arrays for call 1 and replaces them per call after.
* **Exposing every intermediate as a graph output makes it clean** — retained
  values cannot be recycled.
* Corrupted values are plausible-magnitude, not uninitialized garbage (0.18
  where 0.23 belonged), and the corruption enters at a layer boundary that
  moves from run to run, then propagates through the hidden state.
* `mx.set_cache_limit(0)` does not help, so it is not MLX's buffer *cache*.

It could not be reduced to a synthetic MLX program: hand-written chains of
matmul/reduce/argmax/softmax/`slice_update`, and a JAX transformer scan
(28 layers, GQA + RoPE + KV cache, bf16) written to imitate the maxtext
layer, all stay stable under maximal splitting. Only the real lowered
program triggers it, which is why the regression test ships that program.

## Fix

`src/metaljax/__init__.py` raises `MLX_MAX_MB_PER_BUFFER` to 512 (bounded above after the wired-memory panics; see src/metaljax/__init__.py) (MB)
before mlx loads, next to the existing `MLX_MAX_OPS_PER_BUFFER=400`. That
leaves the kernel count as the only splitter, ~50x inside the measured safe
region. Both remain overridable.

Cost, `scripts/bench_spec.py` (2 reps each, spread <0.2%):

| spec | 40 MB | 16384 MB |
|---|---|---|
| `bytes.emb.256\|mgru.256` 64 128 (mid08) | 4.740 ms | 4.80 ms (+1.3%) |
| `bytes.emb.1024\|gru.1024` 32 128 (big15) | 52.97 ms | 55.7 ms (+5.3%) |
| maxtext qwen3-0.6B decode | 16.4 ms/tok (WRONG) | 15.9 ms/tok |

The regression is entirely between 40 and 128 MB — 128/256/512/1024/4096/16384
all measure 55.5-56.1 ms on big15 — so no threshold buys correctness back
at 40 MB's speed. Committing early lets the GPU start while the CPU is still
encoding; big recurrent chunks lose that overlap. **Flagged for Oleg**: ~5%
on large recurrent texmo configs is the price of the fix; the alternative is
silent wrong results in any LLM-sized compiled program.

## Regression test

`tests/test_command_buffer.py` + `tests/data/qwen3_prefill_shrunk.mlir` (the
maxtext prefill program, shrunk to 8 layers / 6144 MLP / 2048 vocab — the
smallest variant that still corrupts). Runs in ~1 s; asserts three compiled
replays are bit-identical to each other and close to op-by-op evaluation.
Verified to fail 4/4 with `MLX_MAX_MB_PER_BUFFER=40` and pass with the
shipped default.

## Upstream

Worth reporting to MLX: "a single `eval` of an `mx.compile`d graph returns
different results per call when `MLX_MAX_MB_PER_BUFFER` / `_OPS__` cut the
command buffer every few kernels; correct at call 1 and when the graph's
intermediates are retained." Repro material is the test asset above.

## Same bug, other budget: the op-by-op path at 400 kernels (2026-08, fixed)

`maxtext-train-06b` (one training step, synthetic data) got the FIRST loss
wrong on the **uncompiled** path only — the opposite polarity of everything
above, which is why it was first written down as unrelated:

| path | loss step 1 | loss step 3 |
|---|---|---|
| jax CPU | 247.8117 | 119.9826 |
| metal, compiled (default) | 247.7775 | 120.0680 |
| metal, `METALJAX_COMPILE=0` (before) | **208.7800** | 124.6839 |
| metal, `METALJAX_COMPILE=0` (after) | 247.7775 | 130.4799 |

It is this same MLX bug, reached through `MLX_MAX_OPS_PER_BUFFER=400`.

### Where it goes wrong

A compiled-vs-eager diff of every PJRT execute (both runs logging per-buffer
sums) put the divergence in execute #10 — maxtext's **parameter init**, a
program with no inputs and 66 outputs that neither run compiles (cost
136k > trace budget). Its body is a 28-layer scan; the default run chunks it
(K=4 compiled replays), `METALJAX_COMPILE=0` runs single steps and flushes
every 5 iterations (`period = 25000 // cost`, cost 4877).

Dumping that module gives a standalone repro (no maxtext, no jax:
`Interpreter(...)`, run main to the `stablehlo.while`, iterate the body).
There, with the loop's own cost/cadence:

* flushing every 5 iterations disagrees with flushing every iteration —
  and with flushing *never*, and with the chunked path, which all agree;
* the whole difference is **one layer**: iteration 6 (the first in the
  second flush window) computes a wrong RNG key, so that layer's weights
  are drawn from a different stream. Layers 0-4 and 7-27 are bit-identical;
* 8 iterations are clean at any cadence, 10 are not — the corrupted work
  has to be followed by enough work in the same `eval`;
* it needs the real parameter shapes: the same program with tensors
  shrunk 4x does not reproduce.

Not msl_scan (`METALJAX_MSL=0` identical — the body has no eligible loop),
not splat constants (`METALJAX_SPLAT_CONST=0` identical), not donation of
the carries (retaining every intermediate carry does not help), not MLX's
buffer cache (`mx.set_cache_limit(0)` does not help).

What does decide it is where MLX cuts command buffers:

| `MLX_MAX_OPS_PER_BUFFER` | 200 | 400 | 800 | 1600 | 5000 | 10^9 |
|---|---|---|---|---|---|---|
| flush every 5 iterations | ok | **wrong** | ok | ok | ok | ok |

and at 400, cadences 1, 2, 3, 4, 6 and 8 are all clean — only 5 is not.
`MLX_BFS_MAX_WIDTH=8` (a different traversal order, same boundaries) is
also clean. So this is not "splits closer than N are unsafe": a particular
boundary lands between a producer and a consumer, and everything else about
the graph decides whether that costs you.

### Fix

`MLX_MAX_OPS_PER_BUFFER` 400 -> 800. Not "never split": the byte budget is
bounded above at 512 MB because unbounded command buffers panicked the
machine (see `__init__.py`), and removing the kernel budget too costs
~50% on decode (qwen3 16.1 -> 25.5 ms/tok, the GPU idles while the CPU
encodes the whole graph) — so both budgets stay finite and the value is
chosen by measurement.

Cost of 400 -> 800, paired runs:

| spec | 400 | 800 | 1600 |
|---|---|---|---|
| mid08 `bytes.emb.256\|mgru.256` 64 128 | 4.830 ms | 4.929 (+2.0%) | 4.912 |
| big15 `bytes.emb.1024\|gru.1024` 32 128 | 55.90 ms | 57.47 (+2.8%) | 58.42 |
| maxtext qwen3-0.6B decode | 15.58 ms/tok | 16.02 (+2.8%) | 16.59 |

1600 is clean too and buys no correctness we can demonstrate, so the
cheaper of the clean values wins.

### Regression test

`test_eager_scan_is_independent_of_flush_cadence` in
`tests/test_command_buffer.py`, over `tests/data/qwen3_init_scan.mlir` (that
init program, dumped as text). It runs 10 iterations of the scan body at the
cadence `ops.control._while` would use and again with an `mx.eval` per
iteration, and requires the carries to be bit-identical. ~3 s; fails with
`MLX_MAX_OPS_PER_BUFFER=400`, passes at the shipped 800. The asset keeps the
model's real shapes on purpose (shrunk, it stops reproducing).

**Standing risk**: every finite budget is a draw in the same lottery. The
two tests in that file are the only evidence that today's values are good
draws; any change to them, to the flush cadence in `ops/control.py`, or to
MLX itself needs both rerun.

---

## Addendum 2026-08-03 (post-0.11.2): accounting ground truth + the 8B bind

**What the budgets actually count** (decompiled from libmlx.dylib 0.32.0,
verified against live objects; the probe tools were lost to a tmp wipe —
offsets were build-specific — but the semantics are these):

    needs_commit() == buffer_ops_  > MLX_MAX_OPS_PER_BUFFER
                   || (buffer_sizes_ >> 20) > MLX_MAX_MB_PER_BUFFER
    set_input_array(a):  if first sighting of a.buffer().ptr() in this
                         command buffer: buffer_sizes_ += a.data_size()
    set_output_array(a): delegates to set_input_array first

So the "MB" budget accumulates `data_size()` — **ELEMENTS, not bytes**
(512 "MB" ≈ 1 GiB of bf16, 2 GiB of f32) — of every **distinct buffer**,
inputs and outputs alike, deduped per command buffer; a broadcast/stride-0
view charges its base region (1 element for a splat).

**Split count does NOT predict corruption.** Live-counter measurements on
tests/data/qwen3_prefill_shrunk.mlir: correct at 2 splits (budget 512) and
at 17 splits (80), wrong at 33 (40). Qwen3-8B: correct at >=39 splits
(2048), garbage at >=158 (512). What corrupts is *which* boundary lands
between a particular producer/consumer — not statically predictable, so a
static byte/element compile gate is NOT a correctness mechanism (that
design was refuted before implementation). Also: mx.compile fuses
elementwise chains, so static traffic estimates overshoot ~2x (1186 vs
592 Mi measured on the shrunk repro).

**Row-15 (qwix 8B) mitigation attempt ledger, all 2026-08-03, all at
otherwise-default settings:**

| configuration | outcome |
|---|---|
| compiled, budget 512 (shipped) | runs; OUTPUT GARBAGE (replay corruption) |
| compiled, budget 2048 | correct in probes; then KERNEL PANIC #4 at load (nondeterministic, wired-memory class) |
| METALJAX_COMPILE=0, 512 | load balloons 67 GB in ~27 s; watchdog-killed |
| METALJAX_BODY_COMPILE=0 (new flag), 512 | 100 GB balloon at load; killed |
| + LOOP_CLEAR_COST=2000, CLEAR_PERIOD=100 | 109 GB; killed — clear_cache cannot reclaim REFERENCED lazy-graph intermediates; uncompiled bodies leave the whole load DAG unevaluated and pinned |
| body-off, prefill 8, watchdog 110 GB | KERNEL PANIC #5 — memory slope 20-40 GB per 5 s outruns any userspace watchdog |

**Policy (hard):** 8B-class maxtext on metal is EMBARGOED on this
machine. Every configuration tried is wrong, ballooning, or lethal. The
one remaining our-side path is engine-side evaluation forcing for
uncompiled phases (bytes-denominated loop flush + periodic mx.eval of
execute outputs), developed and validated at SMALL scale first; a single
supervised 8B verification only after that lands, with Oleg's explicit
sign-off. Otherwise the row waits for the MLX upstream fix
(notes/mlx-command-buffer-upstream-issue.md is filing-ready).
