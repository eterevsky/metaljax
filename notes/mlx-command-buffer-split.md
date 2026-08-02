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

`src/metaljax/__init__.py` raises `MLX_MAX_MB_PER_BUFFER` to 16384 (MB)
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

## Unrelated finding, not fixed

`maxtext-train-06b` (one training step, synthetic data) gets the FIRST loss
wrong on the **uncompiled** path only:

| path | loss step 1 | loss step 3 |
|---|---|---|
| jax CPU | 247.8117 | 119.9826 |
| metal, compiled (default) | 247.7775 | 120.0680 |
| metal, `METALJAX_COMPILE=0` | **208.7800** | 124.6839 |

The default (compiled) path matches CPU, so this is not what users hit, but
something in op-by-op interpretation of that program is wrong. Not
msl_scan (`METALJAX_MSL=0` gives byte-identical numbers) and not the data
iterator (same synthetic batch). Not investigated further.
