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

Running the entire upstream `jax/tests` suite against metaljax executes
~28,200 tests with **98.4% passing** (27,779 passed / 418 failed as of
v0.4.2). Roughly half the remainder is version skew against the test
checkout and test infrastructure that doesn't know the platform; the
rest is audited, documented best-effort residue (complex special
values at poles, ordered-effect tokens, a few PJRT surface APIs) —
see the notes for the per-item audit. The gaps fall into three groups:

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

**Under review — real gaps, individually audited, support not yet
decided** (~190 test failures; these all pass on the CPU backend; see
`notes/jax-test-suite-2026-07.md` for the full audit):

- *Ordered-effect token threading* (~50): jax threads
  `!stablehlo.token` values through function signatures to sequence
  ordered side effects (`jax.debug.print(..., ordered=True)`,
  `io_callback(..., ordered=True)`). The interpreter treats tokens as
  sentinels; full threading through main signatures is unimplemented.
  Unordered callbacks work. Moderate plumbing.
- *complex64 special values at inf/NaN poles* (~24): MLX's complex
  `sin`/`cos`/`tan`/`sqrt`/`log`/... disagree with XLA's
  C99-conformant kernels at infinities and NaN (finite inputs match to
  f32 precision). Fixing means reimplementing the special-value
  branches op by op.
- *PJRT surface APIs* (~40): `Array.unsafe_buffer_pointer` (blocked —
  MLX exposes no raw device pointers), buffer donation (implementable;
  a memory optimization, not a correctness issue), the `pinned_host`
  memory space, optimized-executable text retrieval, strict
  compile-options validation.
- *Window-dilation numeric corners* (~7): specific
  dilation-plus-padding parameter combinations under `vmap` produce
  values that differ from XLA's windowing semantics.
- *>3-D convolutions*: MLX's conv kernels are 1/2/3-D.
- *Singletons*: singular triangular/tridiagonal solves raise instead
  of returning inf/NaN (5), `approx_top_k` NaN-padding placement in
  one exported-vs-native comparison (1), a holomorphic-grad case at
  1e-6 tolerance where MLX's complex sin is 4e-6 off (1), one
  scan-gradient fixed-point corner (1).

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
