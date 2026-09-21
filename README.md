# metaljax — a Metal backend for JAX


metaljax lets unmodified JAX code run on Apple-silicon GPUs:

```python
$ JAX_PLATFORMS=metal python -c \
    "import jax.numpy as jnp; a = jnp.array([1, 2, 3]); print(2 * a, (2*a).device)"
[2 4 6] MetalDevice(id=0)
```

From JAX's point of view it is a regular PJRT backend: `jax.devices()`
reports a `METAL` device, and `jit`, `grad`, `vmap`, `lax.scan`,
`jax.random` (threefry), optax training loops, etc. all work. Under the
hood a native plugin lowers the compiled StableHLO programs onto
[MLX](https://github.com/ml-explore/mlx) arrays, which execute on the GPU
via Metal. The wheel carries its own patched MLX runtime, so nothing else
needs installing.

**Status**: beta. Twenty-one real models — LLM decode (dense, MoE,
quantized), a vision encoder, diffusion, LoRA and full-parameter training
— run end-to-end through unmodified JAX code, and are re-measured on the
binary of every release (the table under *Benchmarks*, the ledgers
[`models.md`](models.md) and [`STATUS.md`](STATUS.md)). Training steps beat
PyTorch's MPS backend on both training rows; decode trails the
Metal-native inference stacks (mlx-lm, llama.cpp) by 1.1–1.9×. Every
release is gated by the pinned JAX test suite (99.54 % passing), a
whole-model correctness sweep against the CPU backend (106/106), and the
model battery. Coverage gaps remain — unsupported constructs are declined
at compile time, naming the op. If a Metal backend ever lands upstream in
the JAX ecosystem, this package will be deprecated in its favor.

## What's new in 0.11.8

- **Precision default.** `METALJAX_MATMUL_PRECISION=high` is now the
  shipped default: f32 GEMM/attention stay **off** the M5's neural
  accelerators (exact, the pinned ~1e-6-vs-f64 class), while bf16/f16 and
  quantized matmuls use them — what mlx-lm and torch-MPS do. `highest`
  pins the pre-accelerator arch for every dtype; `default` is MLX's own
  (accelerator f32, ~8e-4). f32 is never silently degraded.
- **Decode and MoE work.** A ragged decode form for the MoE expert
  dispatch (a direct `gather_mm` over sorted expert ids); stacked sibling
  packs (several stacked per-layer dots reading one activation become one
  `gather_mm` over their `[L, K, N]` views concatenated on the free axis);
  projection packs (sibling decode projections over one activation become
  one dot over concatenated weights); the rotate-half rope as one fused
  kernel over a reversed pair view; submit-ahead pipelining of dynamic
  `while` loops; the KV cache written in place; chunk boundaries donating
  their carries; loop-position specialization (one compiled body per chunk
  with the loop position folded into its dynamic slice starts, now on by
  default). In the vendored MLX fork: empty command buffers skipped in
  `gpu::finalize`, and a transient-only byte cadence (opt-in, off by
  default).
- **Recognizer coverage**: a fused gated-delta-net decode step; the norm
  recognizer covers keras-hub's Qwen3.5/gemma4 and flax-NNX spellings;
  fused attention on the keras MoE rows (sinks, chained masks) and on
  multi-span GQA decode.
- **keras-hub's Gemma4 int4 models are numerically broken upstream.** The
  Gemma4 decoder block's "HOTFIX" multiplies activations by the raw
  unpacked int4 codes, scale unapplied, so every backend decoded one
  repeated token; every row-13 cell through 0.11.7 was timing that broken
  graph. The benchmark harness now routes the int4 FFN through the
  quantized layers and gathers embedding rows before unpacking
  (`scripts/model_bench/int4_fix.py`). Write-up:
  [`notes/keras-hub-gemma4-int4-hotfix-issue.md`](notes/keras-hub-gemma4-int4-hotfix-issue.md).
- **Two msl_scan silent-wrongness fixes**: a value that is one scalar per
  lane is not a register vector (it was broadcast over a whole row of the
  buffer), and only the proven induction variable is a loop counter —
  every other scalar carry is now classified by what the body does with it
  (invariant carries pass through; stepped ones become kernel state).
  Both are described in `CLAUDE.md` and in their commits; no shipped
  number or token stream moved.
- **New comparator targets** in the ledgers: mlx-lm on a local bf16
  Mixtral 8×7B conversion (row 12), mlx-lm's own 4-bit quantization of
  gemma4-E2B (row 13), and a torch-MPS full AdamW train step at MaxText's
  defaults (row 19).

## Install

```bash
pip install metaljax
```

Requirements: Apple-silicon Mac, macOS 14+, Python 3.12+, jax 0.11.x
(installed automatically). No `mlx` install is needed — the wheel bundles
its own — and if you do have the public `mlx`, the two coexist. Then
select the backend per program:

```bash
JAX_PLATFORMS=metal python -c "import jax; print(jax.devices())"
```

CPU remains the default backend when `JAX_PLATFORMS` is unset, so
installing metaljax does not change existing workflows. metaljax ships as
a wheel only: the plugin is built against a pinned XLA workspace with
bazel, which is not something an sdist can compile at install time.

## How it works

```
jax.jit(f)(x)
  │  StableHLO (serialized portable artifact)
  ▼
libmetal_pjrt_native.dylib    ── the plugin: an xla::PjRtClient, with
  │  plugin-native/metal/        XLA's pjrt_c_api_wrapper_impl making the
  │                              PJRT C API around it. No Python, no GIL.
  │  compile: parse + lower StableHLO to a tape, decide what to fuse
  │  execute: replay the tape on device buffers
  ▼
plugin-native/runtime/        ── the executor: op emitters, MSL kernel
  │                              codegen, control flow, host LAPACK
  ▼
libmlx_metaljax.dylib         ── our vendored, privately install-named
                                 MLX: lazy Metal arrays, unified memory
```

- The plugin is self-contained: LLVM/MLIR/StableHLO/absl are linked in and
  private, and exactly two symbols are exported (`GetPjrtApi` and a
  callback bridge), so it coexists with TensorFlow-class carriers in one
  process.
- Registration happens through the `jax_plugins` namespace package
  (`src/jax_plugins/metal/`), at **priority −1**: CPU stays the default
  backend unless you opt in via `JAX_PLATFORMS`. That module is also where
  `jax.debug.print` / `pure_callback` callables live: the plugin calls back
  into them through one C function pointer.
- Python appears nowhere on the execute path.
- Before the first program runs, the plugin lowers what it can into fused
  forms: quantized matmuls, mixture-of-expert dispatches, softmax
  attention, norms, rope, sibling projection and per-layer stack packs,
  and — for recurrent scan bodies — generated Metal kernels. Each
  recognizer has an off switch (see *Environment variables*), and each is
  checked against the literal graph by the differential suites.

## Requirements

- Apple-silicon Mac (developed on an M5 Max, macOS 26.5, Xcode 26.6 —
  any arm64 Mac with a recent Xcode/CLT should work).
- [uv](https://docs.astral.sh/uv/) (only for creating the venv).
- Python **3.14** and jax/jaxlib **0.11.x** (what the venv setup below
  installs; the vendored PJRT header matches jaxlib 0.11.0).

## Developing from source

```bash
git clone https://github.com/eterevsky/metaljax && cd metaljax
uv venv --python 3.14 .venv
uv pip install -p .venv/bin/python jax numpy pytest
uv pip install -p .venv/bin/python -e .
./scripts/vendor_mlx.sh                        # build + stage the MLX runtime
cd plugin-native && bazel build //metal:libmetal_pjrt_native.dylib && cd ..
cp plugin-native/bazel-bin/metal/libmetal_pjrt_native.dylib src/metaljax/lib/
```

(The first bazel build is ~7 minutes; after that it is seconds. An
editable install loads the dylib out of `src/metaljax/lib/`, and
`METALJAX_PLUGIN_PATH` overrides it with any build you want to measure.)

Verify:

```bash
JAX_PLATFORMS=metal .venv/bin/python -c "import jax; print(jax.devices())"
```

should print `[MetalDevice(id=0)]`.

## Running the tests

The pytest suite runs everything through the real plugin — `jax.jit` on
the Metal device, or `compile_and_load` for hand-written StableHLO — and
compares against the JAX CPU backend:

```bash
.venv/bin/python -m pytest tests/ -q
```

Current suite: 512 tests (+1 xfail) across elementwise/transcendental ops,
shapes and broadcasting, `dot_general`/einsum, reductions and cumulative
ops, control flow (`while`/`cond`/`scan`), gather/scatter, sorting,
convolutions, linalg, complex, RNG, sub-byte and bf16/f16 dtypes, the
quantized-matmul / MoE / attention recognizers, the generated scan
kernels, donation, buffer pointers, concurrency and the Metal
command-buffer canaries.

The plugin has its own differential suites, which compare it against
jax-CPU expression by expression and on whole models:

```bash
.venv/bin/python plugin-native/execute_test.py     # vs jax-CPU
.venv/bin/python plugin-native/texmo_gate.py       # whole-model training
cd plugin-native && bazel test //...               # C++ unit tests
```

End-to-end smoke test (device buffers, compile, execute, PJRT events):

```bash
JAX_PLATFORMS=metal .venv/bin/python -c "
import jax, jax.numpy as jnp
g = jax.jit(jax.grad(lambda x: jnp.sum(jnp.tanh(x) ** 2)))(jnp.arange(4.0))
print(g, g.device)"
```

## Coverage and known gaps

Running the test suite of the exact jax release we pin (v0.11.0) executes
28,202 tests with **99.54 % passing** — 28,073 passed / 129 failed (plus
6,161 skipped), measured 2026-09-21 on the release binary. The failing set
is **id-for-id identical** to the previous two releases: zero new
failures, zero regressions. It concentrates in
`export_harnesses_multi_platform_test` (44), `lobpcg_test` (27),
`x64_context_test` (13 — the f64 policy below), `api_test` (5),
`export_test` (5), `shape_poly_test` (4), `xla_transform_test` (4) and
`async_collectives_test` (3). Every remaining failure has been
individually examined and classified with evidence
(`notes/jax-test-suite-2026-07.md`); they fall into three groups:

**Intentional (platform constraints, will not change):**

- **No float64.** Metal GPUs have no f64 ALUs. f64 values may pass
  *through* the device (stored as f32), but f64 *compute* fails at
  compile time naming the op; `METALJAX_F64=downcast` opts into f32
  emulation. Keep `jax_enable_x64` off. Same policy for complex128.
- **One physical device.** `pmap`/`shard_map`/collectives **work on a
  single device** (replica groups of size 1); actual multi-device
  sharding has no hardware to run on.
- **Denormals flush to zero** on the GPU (hardware behavior); tests
  asserting subnormal outputs (e.g. `jnp.spacing`) differ from CPU.
- **Complex special values at inf/NaN poles** for a handful of
  transcendentals (log/trig/hyperbolic family) follow MLX's kernel
  semantics rather than C99. Finite inputs match CPU; full C99 pole
  behavior would need per-element branches in hot paths (policy: not
  worth the slowdown). `sqrt`/`rsqrt`/`exp`/`expm1`/`tan`/`abs`/`sign`
  are rebuilt and exact.

**Remaining audited gaps** (every one re-examined during the 0.11.0
parity campaign and approved as-is; each carries evidence in
`notes/jax-test-suite-2026-07.md`):

- *Ordered-effect residue*: `buffer_callback` and `emit_python_callback`
  are rejected by jax-side platform allowlists (`callback.py`,
  `buffer_callback.py` hard-code cpu/cuda/rocm/tpu) — not reachable from
  a plugin; verified passing on CPU because cpu is inside those
  hard-coded lists. Ordered `debug.print`/`io_callback` work.
- *`testSincInfinities`, FD-reference gradient corners*: fail on the
  CPU backend too, or the test's finite-difference reference is
  numerically meaningless in f32 (documented with numbers).
- *Better-than-reference cases*: shape-polymorphic `jnp.insert` /
  `jnp.nonzero` — the harness asserts `NotImplementedError` because
  jax's CPU path cannot lower them; ours can, and values match CPU on
  concrete shapes. We fail these tests by succeeding.
- *`test_dce_sink_prevents_xla_dce`*: needs optimized-HLO text
  retrieval (`PJRT_Executable_OptimizedProgram`), a debugging surface
  we have not implemented.

**Supported** (each verified against the CPU backend): sorting
(`sort`/`argsort`/`top_k`/`approx_top_k`/`median`/`percentile`/
`unique`, key-value and **multi-key lexicographic** sorts —
`jnp.lexsort`, `unique(axis=)`, set operations — IEEE total-order NaN
handling, complex lexicographic order); convolutions (1/2/3-D float,
integer — exact, and complex; strided, dilated, grouped, transposed,
and their gradients); the full scatter family (windowed,
out-of-bounds-dropping, arbitrary elementwise bodies); general
`reduce`/`reduce_window` bodies and pooling with gradients
(`select_and_scatter`, `select_and_gather_add`); complex64 end-to-end
(arithmetic, FFT, linalg); linear algebra via LAPACK semantics on the
host (QR, eigh, eig, SVD, LU, Cholesky, triangular_solve, Schur,
Hessenberg — CPU-bound in every backend, free on unified memory) —
**including bfloat16/float16 inputs, which jax's CPU backend itself
rejects** (computed in f32, results in the requested dtype);
single-device `pmap`/`shard_map` with the full collective set;
`rng_bit_generator` (Philox and ThreeFry, **bit-exact vs CPU**, so the
`rbg`/`unsafe_rbg` PRNG implementations work); int4/uint4 and all
float8 dtypes (emulated: exact values in wider storage, grid-quantized
converts, 4-bit wraparound); host callbacks (`jax.debug.print`,
`pure_callback`, `io_callback`); shape-polymorphic `jax.export` of all
of the above; `popcnt`/`count_leading_zeros`; sparse (BCOO/BCSR)
workloads.

Unsupported constructs fail loudly at compile time with the op named —
nothing silently falls back to CPU or returns wrong dtypes.

## Using metaljax from another project

Add `metaljax` to your dependencies (it declares `jax` itself):

```toml
[project]
dependencies = ["metaljax"]
```

and set `JAX_PLATFORMS=metal` (or
`jax.config.update("jax_platforms", "metal")` before first use).

To develop against a local checkout instead, use a path source:

```toml
[tool.uv.sources]
metaljax = { path = "../metaljax", editable = true }
```

(with an editable install, build the plugin once in the checkout — see
*Developing from source* above).

## Environment variables

The knobs worth knowing. Every flag the code reads — including the
recognizer A/B switches and the debug-only ones — is listed with its
category in [`notes/env-flags.md`](notes/env-flags.md).

| Variable | Default | Meaning |
|---|---|---|
| `JAX_PLATFORMS` | *(unset)* | Set to `metal` (or `metal,cpu`) to select the backend; unset keeps CPU default. |
| `METALJAX_PLUGIN_PATH` | *(auto)* | Override the path to `libmetal_pjrt_native.dylib` — how a measurement pins one specific build. |
| `METALJAX_MATMUL_PRECISION` | `high` | Which MLX kernels the M5's GEMM / attention / quantized matmul may use. `high`: f32 stays off the neural accelerators (exact, ~1e-6 vs f64) while bf16/f16 and quantized matmuls use them. `highest`: the pre-accelerator arch pin, no accelerator kernels for any dtype. `default`: MLX's own default (accelerator f32, ~8e-4). |
| `METALJAX_F64` | `error` | Metal has no float64. Default: f64 values pass **through** the device (stored as f32, bit-identical to CPU), but any op that **computes** in f64 fails at compile time naming the op. `downcast`: emulate all f64 in f32 (one warning) — the opt-in for e.g. optax AdamW's `beta**step` under `jax_enable_x64`. |
| `METALJAX_MEM_GOVERNOR` | `1` | The memory governor: under host-memory pressure the plugin paces ingest, sweeps the page cache and trims its buffer pool rather than letting the machine wire itself to death, and raises a clean `RESOURCE_EXHAUSTED` if that is not enough. `0` disables it. |
| `METALJAX_MEM_BUDGET_MB` | ¾ of RAM | The governor's hard line on **this process's** footprint; past it a transfer or a program is refused. |
| `METALJAX_MEM_SYS_MB` | ¾ of RAM | The governor's hard line on the **machine's** unreclaimable memory (wired + anonymous + compressor). Big checkpoint restores may need this raised (see *Known limitations*). |
| `METALJAX_MEM_FREE_FLOOR_MB` | 1/16 of RAM | The soft line: the free list below which a load is paced and the page cache swept. |
| `METALJAX_COMPILE_BYTES_MB` | `65536` | Memory ceiling on a single fused trace (the op-count budget is `METALJAX_TRACE_BUDGET`). Over it, the program / while body / unrolled loop / chunked replay runs op by op instead. `0` disables the gate. |
| `METALJAX_TRACE_BUDGET` | `20000` | Max ops in one fused trace. |
| `METALJAX_EAGER_FLUSH_MB` | `1024` | Safety net for programs that run op by op: after this much estimated result data with no sync point, the engine settles what is live so the pending graph stays bounded. `0` disables it. |
| `METALJAX_INGEST_CLEAR_MB` | `8192` | Reclamation cadence of the host→device transfer path, in megabytes ingested — a model load reaches no other sync point. `0` disables it. |
| `METALJAX_CHUNK_MAX` | `16` | Max loop iterations replayed per compiled chunk. |
| `METALJAX_LOOP_SPECIALIZE` | `1` | Compile a counted loop's body once per chunk with the loop position folded into its dynamic slice starts (bit-identical, fewer dispatches). `0` replays one generic body. |
| `METALJAX_PROJ_PACK` | `auto` | Sibling decode projections over one activation packed into one dot over concatenated weights (`_MB` caps the device memory the packs may hold). `0` declines every group, `all` packs every eligible member. |
| `METALJAX_STACKED_PACK` | `auto` | The same for stacked per-layer dots reading one activation: one `gather_mm` over their concatenated views (`METALJAX_STACKED_RELAYOUT_MB` caps it). `0` declines, `all` packs everything. |
| `METALJAX_QMM_SCALES` | `auto` | Width of a quantized matmul's repacked scale/bias tables. `auto` keeps the model's own width when the folded bias is exactly representable and widens to f32 otherwise; `source` always keeps the narrow width; `f32` never narrows. |
| `METALJAX_QMM` / `_SDPA` / `_MOE` / `_ROPE_VIEW` / `_KV_INPLACE` / `_RECOGNIZE` | `1` | Recognizer switches: `0` runs the literal graph instead of the fused form (`_RECOGNIZE=0` turns off all of them). Diagnostic A/B arms — the fused forms are the measured path. |
| `METALJAX_DEBUG` | *(unset)* | `1` narrates every compile, loop, pack and recognizer decision on stderr. |

## Repository layout

```
CLAUDE.md                  project decisions/status (kept current)
models.md                  the 21-row model ledger (per-release columns)
STATUS.md                  current model cells + cross-framework comparators
pyproject.toml             python package + jax_plugins entry point
plugin-native/             THE ENGINE (bazel workspace)
  metal/                   the PJRT plugin: xla::PjRtClient, StableHLO
                           ingest, lowering, the recognizers and packs
  runtime/                 the executor: op emitters, MSL kernel codegen,
                           control flow, host LAPACK, memory governor
  third_party/mlx/         our vendored MLX, linked privately
  execute_test.py          differential suite vs jax-CPU
  texmo_gate.py            whole-model training gate vs jax-CPU
src/jax_plugins/metal/     backend registration (priority -1) + the
                           host-callback registry the plugin calls into
src/metaljax/              __version__, and lib/ where the plugin dylib
                           and the vendored MLX runtime land
tests/                     pytest suite (Metal vs CPU, through PJRT)
notes/                     investigation notes, env-flags, gate records
scripts/                   benchmark, gate & release drivers
  model_bench/             the 21-row model battery + its manifest
  vendor_mlx.sh            build + stage the vendored MLX runtime
  build_native_wheel.sh    build + verify the release wheel
```

## Benchmarks

### Real models

Twenty-one models through unmodified JAX code, all measured 2026-09-21 on
the 0.11.8 release binary (`frozen-m2main-7434ae86`), one GPU process at a
time with a cool-down between rows, timed through `np.asarray`
(`jax.block_until_ready` is a no-op on this backend). **goal** is the best
non-metaljax cell for that row at the **same precision and workload** —
same checkpoint, prompt window, generated token count, batch, resolution
and encoder set; custom kernels are fair game, different quantization is
not. Provenance, per-row protocols, memory footprints, jax-CPU cells and
every caveat live in [`STATUS.md`](STATUS.md) and [`models.md`](models.md).

Metric: LLM rows = warm decode ms/token; vision = forward ms; diffusion =
ms/diffusion-step; training = ms/step. Lower is better; ratio < 1 means
metaljax is ahead.

| # | model | metric | **metaljax 0.11.8** | goal (framework) | ratio |
|---|---|---|---:|---:|---:|
| 1 | gemma4-31B bf16 | ms/tok | **123.1** | 111.2 llama.cpp | 1.11× |
| 2 | gemma4-12B bf16 | ms/tok | **56.5** | 44.2 llama.cpp | 1.28× |
| 3 | gemma4-26B-A4B bf16 (MoE) | ms/tok | **31.1** | 16.9 llama.cpp | 1.84× |
| 4 | gemma4-E2B bf16 | ms/tok | **16.9** | 10.5 mlx-lm | 1.61× |
| 5 | Qwen3-8B bf16 | ms/tok | **36.0** | 29.6 llama.cpp | 1.22× |
| 6 | Llama-3.1-8B bf16 | ms/tok | **37.5** | 29.2 llama.cpp | 1.28× |
| 7 | gpt-oss-20b (native MXFP4) | ms/tok | **15.6** | 8.8 mlx-lm | 1.77× |
| 8 | Qwen3.6-35B-A3B (MoE) | ms/tok | **23.3** | 13.7 mlx-lm | 1.70× |
| 9 | R1-Distill-32B bf16 | ms/tok | **196.0** | 114.9 llama.cpp | 1.71× |
| 10 | DeepSeek-V2-Lite (maxtext) | ms/tok | **19.8** | 10.5 mlx-lm | 1.89× |
| 11 | Qwen3-0.6B (keras-hub) | ms/tok | **5.1** | 3.2 mlx-lm | 1.59× |
| 12 | Mixtral 8×7B bf16 | ms/tok | **72.0** | 53.5 mlx-lm | 1.35× |
| 13 | gemma4-E2B keras-int4 | ms/tok | **6.0** | 4.5 mlx-lm 4-bit | 1.33× |
| 14 | Qwen3-0.6B qwix-int8 | ms/tok | **26.4** | — | — |
| 15 | Qwen3-8B qwix-int8 | ms/tok | **268.4** | — | — |
| 16 | SigLIP 2 (b1 forward) | ms | **41.7** | 29.8 torch-MPS | 1.40× |
| 17 | SD 3.5 Large @512² | ms/step | **456.5** | 553 torch-MPS | **0.83×** |
| 17 | SD 3.5 Large @1024² | ms/step | **2076** | 3078 torch-MPS | **0.70×** |
| 18 | LoRA gemma4-E2B train | ms/step | **112.1** | 135.6 torch-MPS | **0.83×** |
| 19 | Qwen3-0.6B maxtext train | ms/step | **362.6** | 818 torch-MPS | **0.44×** |
| 20 | Qwen3-235B-A22B 3-bit | ms/tok | **43.3** | 28.0 mlx-lm | 1.55× |
| 21 | Qwen3.8-27B bf16 | ms/tok | **142.7** | 98.2 llama.cpp | 1.45× |

Reading the table: metaljax is ahead of PyTorch-MPS on every training and
diffusion row, and behind the dedicated Metal inference stacks on decode —
mlx-lm runs on the same Metal library underneath, so that band is the
optimization target, and llama.cpp's hand-written kernels lead even
mlx-lm on bf16. Rows without a goal cell have no like-for-like
non-metaljax implementation (rows 14/15 are qwix-quantized JAX models).
Several rows do not run on the JAX CPU backend at all at these sizes;
where they do, the CPU cells are in `STATUS.md` (e.g. row 19: 1402 vs
362.6 ms/step, row 16: 533 vs 41.7 ms, row 2: 316.1 vs 56.5 ms/token).

Correctness for these rows is gated the same night: greedy token streams
are compared against the jax-CPU backend, and every divergence is
accounted for (at this gate: one certified-benign 1-bf16-ULP logit tie on
gemma4-E2B, the accepted accumulation-order class; all other
CPU-comparable rows agree exactly over 64 tokens).

### Training and recurrent workloads

The acceptance workload is a 106-config language-model training suite
(dense, GRU/LSTM-family and linear-RNN cells, from tens of weights to
several million) plus a 223-config performance sweep. At this release:
**106/106 correct** — one jitted training chunk per config executed on
both backends from identical inputs, every output leaf compared against
jax-CPU at a 1-ULP sensitivity-scaled tolerance — and the perf sweep is
**1.09× faster** than the standing anchor over 223 matched configs (82
configs improved >5 %, 4 regressed >5 %).

How it gets there: pure programs and counted-loop (`scan`/`fori_loop`)
bodies are traced once into a fused Metal graph and replayed; small
statically-counted loops are unrolled into the enclosing trace, so a whole
recurrent-model training step (forward scan + backward + AdamW) becomes a
single graph replay. On top of that, recurrent scan bodies that
pattern-match as elementwise/matvec cells (rnn/gru/mgru/lrnn/rglru family
— forward *and* the AD-generated backward loop) compile to a single
generated persistent Metal kernel: the whole scan is one kernel launch,
with state in registers (small cells), register-block lanes (small block
matvecs, in-lane reductions and narrow rectangular readouts), or one
threadgroup per batch element with the feature dim as the thread axis
(full-width cells like `gru.256`, including rectangular fused-gate dots).
Very wide cells (`gru.1024`-class) deliberately stay on the compiled-graph
path, where batched matmul wins. Weight-gradient accumulations are handled
by loop fission: the kernel stacks per-step operands and the einsum runs
as one batched matmul after it.

### openxla/xla benchmark suite

The single-device benchmarks from
[xla/tools/benchmarks](https://github.com/openxla/xla/tree/main/xla/tools/benchmarks)
(HLO converted to StableHLO with `xla-translate`, run via
`scripts/run_stablehlo_bench.py`; ms per call, identical seeded inputs,
outputs cross-checked against the CPU results). **These cells were
measured on the 0.11.6 binary and have not been re-run since** — they are
kept for the cross-device comparison, not as current-release numbers:

| benchmark | jax CPU (M5 Max) | metaljax (0.11.6) | RTX 4090 |
|---|---:|---:|---:|
| gemma3_1b_flax_call | 84.6 | 35.3² | 4.0 |
| gemma3_4b_flax_call | 586.5 | 68.8² | 11.2 |
| gemma3_12b_flax_call | 2178.3 | 153.1² | —¹ |
| gemma2_2b_keras_jax | 156.9³ | 2.7³ | 10.9 |
| gemma4_2b_bf16 | 505.8³ | 2.8³ | 2.5 |
| maxtext 2.5B train step | 101066 | 11606⁴ | —¹ |

¹ exceeds the 4090's 24 GB VRAM; the M5's 128 GB unified memory runs
gemma3_12b (23.5 GB of bf16 weights) where the discrete GPU cannot.
² the imported modules contain one plain `stablehlo.dot` (the logits
matmul), which the native plugin declines by design (jax never emits
it); measured with that one op rewritten to the equivalent
`dot_general`, validated end-to-end against CPU references from the
pristine modules.
³ VACUOUS under the suite's seeded inputs: both are generate-loop
programs whose while-loop runs zero iterations, so no forward pass
executes on any backend — the cells measure loop-condition + state-copy
overhead only. Kept for completeness.
⁴ compiled whole-graph; ~9× CPU.

Correctness vs CPU on identical inputs: the gemma3 family diverges
≤3.8 % in bf16 KV-cache tensors (a few bf16 ULPs across 26+ layers) —
the 4090 shows the same divergence class vs CPU (≤4.2 %), so that's
cross-backend bf16 numerics, not a backend bug. maxtext NaN placement
matches CPU exactly on all 11 NaN-carrying outputs.

## Known limitations

Three platform constraints are permanent (detailed under *Coverage and
known gaps* above): no float64 or complex128 **compute** (pass-through
is fine; `METALJAX_F64=downcast` emulates in f32), one physical device
(single-device `pmap`/`shard_map`/collectives work; real multi-device
sharding has no hardware), and denormals flushing to zero on the GPU.

Performance and operational, not correctness:

- **Decode trails the dedicated Metal inference stacks** by 1.1–1.9× on
  the model table above; **prefill and model load trail by more** (see
  STATUS.md's gap band). Decode is the optimized path.
- **Large checkpoint restores can need the governor's machine line
  raised.** The restore transient of the biggest rows sits above the
  shipped `METALJAX_MEM_SYS_MB` default — row 10 (DeepSeek-V2-Lite) runs
  at `METALJAX_MEM_SYS_MB=107520` on a 128 GB machine. Under the default
  the governor refuses cleanly (`RESOURCE_EXHAUSTED`) rather than wedging
  the machine, which is the contract, but the run does not start.
- **A big quantized model pays a one-time pack wave.** Row 20
  (Qwen3-235B-A22B 3-bit) spends ~58 minutes packing quantized weights at
  a ~103 GB peak before steady decode at 43.3 ms/token. It is once per
  process, not per step.
- **keras-hub's Gemma4 int4/int8 quantization is broken upstream** (still
  present as of 2026-09-19): the decoder block's "HOTFIX" multiplies by
  unscaled codes, producing a numerically broken model on *every*
  backend. Our benchmark harness works around it
  (`scripts/model_bench/int4_fix.py`); a user's own quantized Gemma4 will
  be wrong until keras-hub fixes it.
- **Scan bodies that don't fit the kernel-codegen patterns** (gather/
  scatter in the loop, non-affine indexing, bodies exceeding the trace or
  binding budgets) fall back to per-timestep compiled-graph replay, which
  pays per-step dispatch.
- **Buffer donation is honoured** (`donate_argnums` invalidates the
  donated inputs, matching other backends), but MLX cannot write outputs
  into the donated memory in place — the win is prompt buffer release
  rather than CUDA-style aliasing.

## License / provenance

Experimental personal project; builds against public JAX/OpenXLA (PJRT
header vendored from openxla/xla) and Apple's MLX. Apache-2.0.
