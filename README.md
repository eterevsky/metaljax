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

**Status**: beta. Real training runs work end-to-end (transformer and
recurrent language models with optax, including long `lax.scan`
training loops), transformer training steps run within a few percent of
PyTorch's MPS backend, and every release is gated by a whole-model
correctness sweep against the CPU backend. Coverage gaps remain —
unsupported constructs are declined at compile time, naming the op. If a Metal
backend ever lands upstream in the JAX ecosystem, this package will be
deprecated in its favor.

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
- Python appears nowhere on the execute path. (Through 0.11.5 there was a
  second, Stage 1 implementation — a trampoline dylib driving a StableHLO
  interpreter written in Python; it was retired in 0.11.6.)

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

Current suite: 484 tests across elementwise/transcendental ops, shapes and
broadcasting, `dot_general`/einsum, reductions and cumulative ops, control
flow (`while`/`cond`/`scan`), gather/scatter, sorting, convolutions,
linalg, complex, RNG, sub-byte and bf16/f16 dtypes, the quantized-matmul /
MoE / attention recognizers, donation, buffer pointers, concurrency and
the Metal command-buffer canaries.

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

Running the test suite of the exact jax release we pin (v0.11.0)
executes ~27,800 tests with **99.53% passing** (27,649 passed / 130
failed). Every remaining failure has been individually examined and
classified with evidence (`notes/jax-test-suite-2026-07.md`); they
fall into three groups:

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

- *Ordered-effect residue* (~3): `buffer_callback` and
  `emit_python_callback` are rejected by jax-side platform allowlists
  (`callback.py`, `buffer_callback.py` hard-code cpu/cuda/rocm/tpu) —
  not reachable from a plugin; verified passing on CPU because cpu is
  inside those hard-coded lists. Ordered `debug.print`/`io_callback`
  work.
- *`testSincInfinities`, FD-reference gradient corners*: fail on the
  CPU backend too, or the test's finite-difference reference is
  numerically meaningless in f32 (documented with numbers).
- *Better-than-reference cases* (4): shape-polymorphic `jnp.insert` /
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

**Behavioral differences under investigation** are tracked in the notes
file above. Unsupported constructs fail loudly at compile time with the
op named — nothing silently falls back to CPU or returns wrong dtypes.

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

| Variable | Default | Meaning |
|---|---|---|
| `JAX_PLATFORMS` | *(unset)* | Set to `metal` (or `metal,cpu`) to select the backend; unset keeps CPU default. |
| `METALJAX_MATMUL_PRECISION` | `highest` | On M5-class GPUs MLX routes f32 GEMM through the neural accelerators at ~bf16 input precision (~4e-3 error). `highest` pins MLX kernels to the previous GPU generation for exact f32; set `default` to allow the fast path. |
| `METALJAX_F64` | `error` | Metal has no float64. Default (`error`): f64 values may pass **through** the device (x64 mode wraps Python scalars as f64 buffers that programs immediately convert to f32 — stored as f32, which rounds exactly once and stays bit-identical to CPU), but any op that **computes** in f64 fails at compile time, naming the op. `downcast`: emulate all f64 in f32 (one warning). Example: under `jax_enable_x64`, optax AdamW's `beta**step` bias correction is real f64 arithmetic — strict mode rejects it, and `downcast` is the opt-in for such workloads. |
| `METALJAX_QMM` | `1` | Recognize weight-only quantized matmuls (integer codes plus a scale/zero-point map, dequantized and fed to a dot — what keras `quantize("int4")` and `jnp.int4` weights emit) and run them as one `mx.quantized_matmul` on a weight repacked once, instead of materializing the dequantized weight per call. Set `0` to execute such graphs literally. |
| `METALJAX_QMM_SCALES` | `auto` | Width of the repacked scale/bias tables. `auto` keeps the model's own (bf16/f16) width whenever the folded bias is exactly representable in it, and widens to f32 otherwise so the reconstructed weight stays bit-exact — which costs 3–12% of the matmul at batch 1, since the tables are then 12.5% of the 4-bit weight instead of 6.25%. `source` always keeps the narrow width (faster; the bias rounds to ≤0.5 ULP); `f32` never narrows. |
| `METALJAX_SDPA` | `1` | Recognize softmax attention (`softmax(Q@Kᵀ·s + mask) @ V`, in any of the layouts jax emits, including grouped-query attention and the deferred normalization real LLM lowerings use) and run it as one `mx.fast.scaled_dot_product_attention` instead of materializing the `[batch, heads, q, k]` logits five times over. Set `0` to execute such graphs literally. The fused kernel accumulates the softmax in f32 whatever the input dtype, so it is *more* accurate than the chain it replaces at f16/bf16 and ties at f32. |
| `METALJAX_COMPILE_BYTES_MB` | `65536` | Memory ceiling on a single fused trace, alongside the op-count budget (`METALJAX_TRACE_BUDGET`). The two are independent: op count bounds how many Metal buffers a trace holds, this bounds how much they hold. A program can sit at 2% of the op budget and still make the compiled path hold gigabytes — a jitted parameter initializer is 365 ops and turns 256 MB of weights into 58 GB of traffic — and unlike the eager path (whose peak is capped by `METALJAX_EAGER_FLUSH_MB`), the compiled path's peak grows with the program: measured 0.18 / 0.72 / 2.87 / 10.0 GB for that initializer at 4 / 16 / 64 / 256 MB of output, against 3.25 GB for the largest of them once it is refused. Over this budget the whole program, the while body, the unrolled loop or the chunked replay in question runs op by op instead. The default is measured: ~1.5x above the largest thing metaljax compiles today (a 16-iteration texmo chunked replay, 41.9 GB estimated / 14.8 GB peak) and ~2x below the smallest it must refuse (a Qwen3-8B prefill, 139.8 GB). The estimate counts traffic, not peak, so it reads 3–6x high. `0` disables the gate; `METALJAX_DEBUG=1` prints every program's `bytes=` and every fired gate. |
| `METALJAX_EAGER_FLUSH_MB` | `1024` | Memory safety net for programs that run op by op (anything impure or over the trace budget — checkpoint conversion and parameter-load programs are the usual ones). After this much estimated result data has been produced with no sync point, the engine settles what is still live, so the pending graph and the Metal buffers it pins stay bounded. Costs one command-buffer roundtrip per budget's worth of data; never fires on small workloads (a texmo train step produces kilobytes to megabytes per block). `0` disables it. |
| `METALJAX_MOE` | `1` | Recognize a dense mixture-of-experts dispatch — a top-k router whose one-hot-weighted scores multiply the outputs of **every** expert before being summed over the expert axis, which is how `jax.numpy` MoE layers are written and how XLA runs them — and evaluate only the `k` selected experts per token, through `mx.gather_mm` (float weights) or `mx.gather_qmm` (weights packed by `METALJAX_QMM`). The routing tensor must be provably zero off the selection; anything else (capacity-factor routing, expert outputs read outside the dispatch) falls back to the dense form. Set `0` to always run it densely. |
| `METALJAX_MOE_VERIFY` | `1` | Before a recognized dispatch is used, evaluate the router tail on random logits and check that the scores really are the top-k weights scattered at the matched indices. Costs microseconds, once per program. Set `0` to trust the structural match alone. |
| `METALJAX_MEM_GOVERNOR` | `1` | The memory governor: under host-memory pressure the plugin throttles ingest and trims its buffer pool rather than letting the machine wire itself to death, and raises a clean `RESOURCE_EXHAUSTED` if that is not enough. `METALJAX_MEM_BUDGET_MB` / `METALJAX_MEM_FREE_FLOOR_MB` move the thresholds; `0` disables it. |
| `METALJAX_PLUGIN_PATH` | *(auto)* | Override the path to `libmetal_pjrt_native.dylib` — how a measurement pins one specific build. |

## Repository layout

```
CLAUDE.md                  project decisions/status (kept current)
pyproject.toml             python package + jax_plugins entry point
plugin-native/             THE ENGINE (bazel workspace)
  metal/                   the PJRT plugin: xla::PjRtClient, StableHLO
                           ingest, lowering, the qmm/moe/sdpa recognizers
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
scripts/                   benchmark, gate & release drivers
  vendor_mlx.sh            build + stage the vendored MLX runtime
  build_native_wheel.sh    build + verify the release wheel
```

## Benchmarks

Full training steps (fwd + bwd + AdamW), f32, M5 Max, via
`scripts/bench_compare.py` (16 timed steps after warmup):

| workload | jax CPU | **metaljax** | torch MPS | torch CPU |
|---|---:|---:|---:|---:|
| transformer d256 L4 T256 b32 | 156.6 | **30.6** | 30.0 | 209.3 |
| transformer d512 L4 T256 b64 | 1059.9 | **159.4** | 151.7 | — |
| GRU.256 T256 b256 (scan) | 255.9 | **53.5** | 48.2¹ | — |

¹ torch uses its hand-fused `nn.GRU` kernel; metaljax generates its
kernel from the StableHLO loop body and lands within 10%.

How: pure programs and counted-loop (`scan`/`fori_loop`) bodies are traced
once into a fused Metal graph and replayed; small
statically-counted loops are unrolled into the enclosing trace, so e.g. a
whole recurrent-model training step (forward scan + backward + AdamW)
becomes a single graph replay. On top of that, recurrent scan bodies that
pattern-match as elementwise/matvec cells (rnn/gru/mgru/lrnn/rglru
family — forward *and* the AD-generated backward loop) compile to a
single generated persistent Metal kernel: the whole scan is one kernel
launch, with state in registers (small cells), register-block lanes
(small block matvecs, in-lane reductions, and narrow rectangular
readouts — the lrnn family), or one threadgroup per batch element with
the feature dim as the thread axis (full-width cells like `gru.256`,
including rectangular fused-gate dots like `mullstm.32`). Very wide
cells (`gru.1024`-class) deliberately stay on the compiled-graph path,
where batched matmul wins.
Weight-gradient accumulations are handled by loop fission: the kernel
stacks per-step operands and the einsum runs as one batched matmul
after it. `METALJAX_COMPILE=0` disables compilation, `METALJAX_MSL=0`
disables kernel codegen, `METALJAX_TRACE_BUDGET` (default 20000 ops)
caps trace sizes, and `METALJAX_DEBUG=1` logs loop/compile decisions.

On the 106-config language-model training suite (dense, GRU/LSTM-family,
and linear-RNN cells from tens of weights to several million; 0.11.6
release gate), 86 configs train faster on metal than on the M5's CPU
cores: **every** production-size and mid-size config wins (34/34 and
30/30, median 3x and 6.1x faster) — only the sub-millisecond
kernel-codegen microbenchmarks remain partly CPU territory
(kernel-dispatch floor). An earlier 104-config sweep also measured 41
configs outpacing an RTX 4090 running jax-CUDA. Every optimization is
gated by a whole-model correctness sweep: one jitted training chunk per
suite config executed on both backends from identical inputs, every
output leaf compared (106/106 at this release).

### openxla/xla benchmark suite

The single-device benchmarks from
[xla/tools/benchmarks](https://github.com/openxla/xla/tree/main/xla/tools/benchmarks)
(HLO converted to StableHLO with `xla-translate`, run via
`scripts/run_stablehlo_bench.py`; ms per call, identical seeded inputs,
outputs cross-checked against the CPU results):

| benchmark | jax CPU (M5 Max) | **metaljax** | RTX 4090 |
|---|---:|---:|---:|
| gemma3_1b_flax_call | 84.6 | **35.3**² | 4.0 |
| gemma3_4b_flax_call | 586.5 | **68.8**² | 11.2 |
| gemma3_12b_flax_call | 2178.3 | **153.1**² | —¹ |
| gemma2_2b_keras_jax | 156.9³ | 2.7³ | 10.9 |
| gemma4_2b_bf16 | 505.8³ | 2.8³ | 2.5 |
| maxtext 2.5B train step | 101066 | **11606**⁴ | —¹ |

¹ exceeds the 4090's 24 GB VRAM; the M5's 128 GB unified memory runs
gemma3_12b (23.5 GB of bf16 weights) where the discrete GPU cannot.
² the imported modules contain one plain `stablehlo.dot` (the logits
matmul), which the native plugin declines by design (jax never emits
it); measured with that one op rewritten to the equivalent
`dot_general`, validated end-to-end against CPU references from the
pristine modules.
³ VACUOUS under the suite's seeded inputs (found at the 0.11.6
re-measure): both are generate-loop programs whose while-loop runs zero
iterations, so no forward pass executes on any backend — the cells
measure loop-condition + state-copy overhead only, and one output is a
bit-exact passthrough of the input ids. Kept for completeness; earlier
releases' 17.5/16.9 ms cells were Stage-1 dispatch on the same empty
program.
⁴ compiled whole-graph; ~9× CPU. +9 % vs the July-era 10618 cell, which
predates the memory governor and the 0.11.x engine work — the drift is
named in the 0.11.6 gate record (the governor's pacing is what buys the
no-panic contract).

Correctness vs CPU on identical inputs: the gemma3 family diverges
≤3.8 % in bf16 KV-cache tensors (a few bf16 ULPs across 26+ layers) —
the 4090 shows the same divergence class vs CPU (≤4.2 %), so that's
cross-backend bf16 numerics, not a backend bug. maxtext NaN placement
matches CPU exactly on all 11 NaN-carrying outputs.

### Gemma 4 end-to-end inference

Real LLM inference through unmodified JAX code: `google/gemma-4-12B-it`
and `google/gemma-4-31B-it` (HF safetensors mapped into DeepMind's
[gemma](https://github.com/google-deepmind/gemma) library, greedy
`ChatSampler`, batch 1, ~40-token prompt, 218–275 generated tokens).
*decode* is the steady-state warm rate; *warmup* is the one-time
first-generation overhead (jax tracing + metaljax compile + Metal
kernel builds), measured as cold minus warm generation time. Memory is
device-active for metaljax, weight footprint for CPU.

| model | dtype / backend | decode ms/tok | tok/s | warmup | memory |
|---|---|---:|---:|---:|---:|
| gemma-4-31B-it | bf16 **metaljax** | **235.2** | 4.25 | 6.2 s | 66 GB |
| gemma-4-31B-it | f32 metaljax | —¹ | | | 123 GB |
| gemma-4-31B-it | f32 jax CPU | —¹ | | | 123 GB |
| gemma-4-12B-it | bf16 **metaljax** | **92.1** | 10.9 | 6.3 s | 25 GB |
| gemma-4-12B-it | f32 metaljax² | 254 | 3.93 | 6 s | 50 GB |
| gemma-4-12B-it | f32 jax CPU² | 938 | 1.07 | 11 s | 48 GB |

¹ f32 weights alone are 122.8 GB: metaljax loads them but decode —
which streams every weight byte per token — pages a 128 GB machine into
the ground (the CPU attempt took the whole OS with it). bf16 is the
only way to run the 31B locally; bf16 on the CPU backend is omitted
because XLA:CPU upcasts bf16 matmuls to f32 internally.
² measured at 0.11.0 (the Stage-1 engine); kept for the dtype/backend
comparison. The bf16 rows are the 0.11.6 release-gate cells (native
engine, greedy decode, tokens exact vs CPU on the 12B-class rows) —
the full 20-model release table lives in `STATUS.md` and
`benchmarks/models.md`.

The Stage 1 (Python) engine these tables originally showcased carried
~120 ms/token of dtype-independent host dispatch; the native engine
that replaced it in 0.11.5 removed that overhead (31B: 374 → 235.2,
12B: 189 → 92.1).

## Known limitations

Three platform constraints are permanent (detailed under *Coverage and
known gaps* above): no float64 or complex128 **compute** (pass-through
is fine; `METALJAX_F64=downcast` emulates in f32), one physical device
(single-device `pmap`/`shard_map`/collectives work; real multi-device
sharding has no hardware), and denormals flushing to zero on the GPU.
Everything else still open is itemized in the "Under review" list
above.

Performance, not correctness:

- Scan bodies that don't fit the kernel-codegen patterns (gather/scatter
  in the loop, non-affine indexing, bodies exceeding the trace or binding
  budgets) fall back to per-timestep compiled-graph replay, which pays
  per-step dispatch.
- Buffer donation is honoured (`donate_argnums` invalidates the donated
  inputs, matching other backends), but MLX cannot write outputs into
  the donated memory in place — the win is prompt buffer release rather
  than CUDA-style aliasing.

## License / provenance

Experimental personal project; builds against public JAX/OpenXLA (PJRT
header vendored from openxla/xla) and Apple's MLX.
