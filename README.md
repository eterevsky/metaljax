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
hood the compiled StableHLO programs are interpreted onto
[MLX](https://github.com/ml-explore/mlx) arrays, which execute on the GPU
via Metal.

**Status**: beta. Real training runs work end-to-end (transformer and
recurrent language models with optax, including long `lax.scan`
training loops), transformer training steps run within a few percent of
PyTorch's MPS backend, and every release is gated by a whole-model
correctness sweep against the CPU backend. Coverage gaps remain —
unsupported ops fail with a clear `UnsupportedOpError`. If a Metal
backend ever lands upstream in the JAX ecosystem, this package will be
deprecated in its favor.

## Install

```bash
pip install metaljax
```

Requirements: Apple-silicon Mac, macOS 14+, Python 3.12+, jax 0.11.x
(installed automatically). Then select the backend per program:

```bash
JAX_PLATFORMS=metal python -c "import jax; print(jax.devices())"
```

CPU remains the default backend when `JAX_PLATFORMS` is unset, so
installing metaljax does not change existing workflows. Installing from
the source distribution (rather than the wheel) additionally requires the
Xcode command-line tools, since the PJRT plugin compiles at build time.

## How it works

```
jax.jit(f)(x)
  │  StableHLO (serialized portable artifact)
  ▼
plugin/metal_pjrt.cc          ── PJRT C-API dylib loaded by jaxlib.
  │                              No dependencies; trampolines every call
  ▼                              back into Python (same process, GIL).
src/metaljax/engine.py        ── compile: deserialize + wrap Interpreter
  │                              execute: run on device buffers
  ▼
src/metaljax/interpreter.py   ── walks the StableHLO module op by op
  │  + src/metaljax/ops/*     ── one handler per op family
  ▼
mlx.core                      ── lazy Metal arrays; unified memory
```

- The dylib implements PJRT API v0.114 (`plugin/vendor/pjrt_c_api.h`,
  vendored at jaxlib's exact openxla/xla pin).
- Registration happens through the `jax_plugins` namespace package
  (`src/jax_plugins/metal/`), at **priority −1**: CPU stays the default
  backend unless you opt in via `JAX_PLATFORMS`.

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
uv pip install -p .venv/bin/python jax mlx numpy pytest
uv pip install -p .venv/bin/python -e .
./plugin/build.sh          # builds plugin/build/libmetal_pjrt.dylib (clang)
```

Verify:

```bash
JAX_PLATFORMS=metal .venv/bin/python -c "import jax; print(jax.devices())"
```

should print `[MetalDevice(id=0)]`.

## Running the tests

The pytest suite lowers each construct with `jax.jit(...).lower()`, runs
the StableHLO module through the interpreter on the GPU, and compares
against the JAX CPU backend (this exercises the interpreter directly and
does not need the plugin dylib):

```bash
.venv/bin/python -m pytest tests/ -q
```

Current suite: 129 tests across elementwise/transcendental ops, shapes and
broadcasting, `dot_general`/einsum, reductions and cumulative ops, control
flow (`while`/`cond`/`scan`), gather/scatter, RNG, and bf16/f16/x64 dtype
handling.

End-to-end smoke test through the real plugin (device buffers, compile,
execute, PJRT events):

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

**Under review — remaining audited gaps** (every one re-examined
during the 0.4.5 parity campaign; each carries evidence in
`notes/jax-test-suite-2026-07.md`):

- *Ordered-effect residue* (~3): `buffer_callback` and
  `emit_python_callback` are rejected by jax-side platform allowlists
  (`callback.py`, `buffer_callback.py` hard-code cpu/cuda/rocm/tpu) —
  not reachable from a plugin. Ordered `debug.print`/`io_callback`
  work.
- *Complex special values at inf/NaN poles*: MLX's complex kernels
  differ from C99 at infinities for a handful of transcendentals we
  have not rebuilt (finite inputs match; sqrt/rsqrt/exp/expm1/tan/abs/
  sign are rebuilt and exact).
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

(with an editable install, build the plugin once in the checkout via
`./plugin/build.sh`). A git source
(`metaljax = { git = "https://github.com/eterevsky/metaljax" }`) works
too; like sdist installs it compiles the plugin during the build, which
needs the Xcode command-line tools.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `JAX_PLATFORMS` | *(unset)* | Set to `metal` (or `metal,cpu`) to select the backend; unset keeps CPU default. |
| `METALJAX_MATMUL_PRECISION` | `highest` | On M5-class GPUs MLX routes f32 GEMM through the neural accelerators at ~bf16 input precision (~4e-3 error). `highest` pins MLX kernels to the previous GPU generation for exact f32; set `default` to allow the fast path. |
| `METALJAX_F64` | `error` | Metal has no float64. Default (`error`): f64 values may pass **through** the device (x64 mode wraps Python scalars as f64 buffers that programs immediately convert to f32 — stored as f32, which rounds exactly once and stays bit-identical to CPU), but any op that **computes** in f64 fails at compile time, naming the op. `downcast`: emulate all f64 in f32 (one warning). Example: under `jax_enable_x64`, optax AdamW's `beta**step` bias correction is real f64 arithmetic — strict mode rejects it, and `downcast` is the opt-in for such workloads. |
| `METALJAX_COMPILE_OPTIONS` | *(unset)* | `jit(..., compiler_options={...})` entries are validated like XLA validates them (unknown name → `No such compile option`, wrong type → `is not a valid <type> value`) and then ignored, since metaljax has no XLA flag surface. Set `ignore` to skip the check and accept anything. |
| `METALJAX_PLUGIN_PATH` | *(auto)* | Override the path to `libmetal_pjrt.dylib`. |

## Repository layout

```
CLAUDE.md                  project decisions/status (kept current)
pyproject.toml             python package + jax_plugins entry point
src/metaljax/
  interpreter.py           StableHLO walker (SSA env, blocks, funcs)
  ops/                     op handlers: elementwise, shape, linalg,
                           reduction, control, gather
  engine.py                PJRT-facing compile/execute/buffer layer
  dtypes.py, _ir.py        dtype tables, MLIR context & attr decoding
src/jax_plugins/metal/     backend registration (priority -1)
plugin/
  metal_pjrt.cc            the PJRT C-API dylib (no deps, ~1100 lines)
  vendor/pjrt_c_api.h      vendored PJRT header (API 0.114)
  build.sh                 clang build → plugin/build/libmetal_pjrt.dylib
tests/                     pytest suite (Metal vs CPU)
scripts/                   benchmark & training drivers
```

## Benchmarks

Full training steps (fwd + bwd + AdamW), f32, M5 Max, via
`scripts/bench_compare.py` (16 timed steps after warmup):

| workload | jax CPU | **metaljax** | torch MPS | torch CPU |
|---|---:|---:|---:|---:|
| transformer d256 L4 T256 b32 | 174.3 | **30.2** | 30.0 | 209.3 |
| transformer d512 L4 T256 b64 | — | **153.9** | 151.7 | — |
| GRU.256 T256 b256 (scan) | 256.6 | **53.5** | 48.2¹ | — |

¹ torch uses its hand-fused `nn.GRU` kernel; metaljax generates its
kernel from the StableHLO loop body and lands within 10%.

How: pure programs and counted-loop (`scan`/`fori_loop`) bodies are traced
once through `mx.compile` and replayed as fused Metal graphs; small
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

On a 104-config language-model training suite (dense, GRU/LSTM-family,
and linear-RNN cells from tens of weights to several million), 84
configs train faster on metal than on the M5's CPU cores; **every**
config above 10k weights wins (median 3–6.6x faster), and 41 of 104
outpace an RTX 4090 running jax-CUDA. Only sub-10k-weight models remain
CPU territory (kernel-dispatch floor). Every optimization is gated by a
whole-model correctness sweep: one jitted training chunk per suite
config executed on both backends from identical inputs, every output
leaf compared.

### openxla/xla benchmark suite

The single-device benchmarks from
[xla/tools/benchmarks](https://github.com/openxla/xla/tree/main/xla/tools/benchmarks)
(HLO converted to StableHLO with `xla-translate`, run via
`scripts/run_stablehlo_bench.py`; ms per call, identical seeded inputs,
outputs cross-checked against the CPU results):

| benchmark | jax CPU (M5 Max) | **metaljax** | RTX 4090 |
|---|---:|---:|---:|
| gemma3_1b_flax_call | 80.1 | **42.5** | 4.0 |
| gemma3_4b_flax_call | 666.9 | **81.5** | 11.2 |
| gemma3_12b_flax_call | 2187.9 | **172.7** | —¹ |
| gemma2_2b_keras_jax | 158.0 | **17.5** | 10.9 |
| gemma4_2b_bf16 | 512.0 | **16.9** | 2.5 |
| maxtext 2.5B train step | 101066 | **10618**² | —¹ |

¹ exceeds the 4090's 24 GB VRAM; the M5's 128 GB unified memory runs
gemma3_12b (23.5 GB of bf16 weights) where the discrete GPU cannot.
² compiled whole-graph after working around an MLX limitation (equal
constant-valued outputs break `mx.compile`); ~10× CPU.

Correctness vs CPU on identical inputs: gemma2/gemma4 outputs bit-exact;
the gemma3 family diverges ≤3.6% in bf16 KV-cache tensors (a few bf16
ULPs across 26+ layers) — the 4090 shows the same divergence class vs
CPU (≤4.2%), so that's cross-backend bf16 numerics, not a backend bug.

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
| gemma-4-31B-it | bf16 **metaljax** | **374** | 2.68 | 9 s | 65 GB |
| gemma-4-31B-it | f32 metaljax | —¹ | | | 123 GB |
| gemma-4-31B-it | f32 jax CPU | —¹ | | | 123 GB |
| gemma-4-12B-it | bf16 **metaljax** | **189** | 5.28 | 6 s | 25 GB |
| gemma-4-12B-it | f32 **metaljax** | **254** | 3.93 | 6 s | 50 GB |
| gemma-4-12B-it | f32 jax CPU | 938 | 1.07 | 11 s | 48 GB |

¹ f32 weights alone are 122.8 GB: metaljax loads them but decode —
which streams every weight byte per token — pages a 128 GB machine into
the ground (the CPU attempt took the whole OS with it). bf16 is the
only way to run the 31B locally; bf16 on the CPU backend is omitted
because XLA:CPU upcasts bf16 matmuls to f32 internally.

Single-token decode is the worst case for a Python interpreter: ~120
ms/token of the metal rows is dtype-independent host-side dispatch
(measured via process-CPU vs wall time), which is why f32 costs only
1.34× bf16 rather than the 2× that pure bandwidth would predict. That
overhead is the target of the planned native replay engine.

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
