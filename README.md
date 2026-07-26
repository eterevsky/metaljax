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

**Status**: experimental, published as a demo. Real training runs work
(the [texmo](https://github.com/eterevsky/texmo) project trains
end-to-end) and transformer training steps run within a few percent of
PyTorch's MPS backend, but expect gaps — unsupported ops fail with a
clear `UnsupportedOpError`. If a Metal backend ever lands upstream in the
JAX ecosystem, this package will be deprecated in its favor.

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

## Running texmo on Metal

texmo pins its platform in its own `config.py`, so use the bundled driver,
which imports texmo's `ManagerJax` directly (set `TEXMO_DIR` if your
checkout is not `~/texmo`):

```bash
# extra deps texmo imports at module level (torch is never executed by JAX)
uv pip install -p .venv/bin/python optax safetensors regex scipy scikit-learn torch

# platform spec steps batch length precision
.venv/bin/python scripts/texmo_train.py metal,cpu 'bits.1+bp|mgru.4-dense.4.gelu' 64 16 64 fp32
```

texmo's own GRU benchmark accepts a platform flag:

```bash
cd ~/texmo
~/metaljax/.venv/bin/python scripts/bench_jax.py --platform metal --mode ram --steps 8 --batch 64
```

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
| `METALJAX_F64` | `error` | Metal has no float64. Default (`error`): f64 values may pass **through** the device (x64 mode wraps Python scalars as f64 buffers that programs immediately convert to f32 — stored as f32, which rounds exactly once and stays bit-identical to CPU), but any op that **computes** in f64 fails at compile time, naming the op. `downcast`: emulate all f64 in f32 (one warning). Example: under `jax_enable_x64`, optax AdamW's `beta**step` bias correction is real f64 arithmetic — strict mode rejects it, so `scripts/texmo_train.py` sets `downcast` explicitly. |
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
scripts/texmo_train.py     texmo-on-Metal driver
```

## Benchmarks

Full training steps (fwd + bwd + AdamW), f32, M5 Max, via
`scripts/bench_compare.py` (16 timed steps after warmup):

| workload | jax CPU | **metaljax** | torch MPS | torch CPU |
|---|---:|---:|---:|---:|
| transformer d256 L4 T256 b32 | 174.3 | **30.2** | 30.0 | 209.3 |
| transformer d512 L4 T256 b64 | — | **153.9** | 151.7 | — |
| GRU.256 T256 b256 (scan) | — | **53.5** | 48.2¹ | — |
| texmo `bench_jax.py` GRU b256 | 273.3 | **59.2** | — | — |

¹ torch uses its hand-fused `nn.GRU` kernel; metaljax generates its
kernel from the StableHLO loop body and lands within 10%.

How: pure programs and counted-loop (`scan`/`fori_loop`) bodies are traced
once through `mx.compile` and replayed as fused Metal graphs; small
statically-counted loops are unrolled into the enclosing trace, so e.g. a
whole texmo training step (forward scan + backward + AdamW) becomes a
single graph replay. On top of that, recurrent scan bodies that
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

On texmo's own 104-config benchmark suite (0.3): 84 configs train
faster on metal than on the M5's CPU cores; **every** config above 10k
weights wins (median 3–6.6x faster), and 41 of 104 outpace an RTX 4090
running jax-CUDA. Only sub-10k-weight models remain CPU territory
(kernel-dispatch floor). Every optimization is gated by a whole-model
correctness sweep: one jitted training chunk per suite config executed
on both backends from identical inputs, every output leaf compared.

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

- Scan bodies that don't fit the kernel-codegen patterns (in-cell
  reductions, gather/scatter in the loop, non-affine indexing) fall back
  to per-timestep compiled-graph replay, which pays per-step dispatch.
- float64 is emulated in f32 (see `METALJAX_F64`); complex dtypes are
  unsupported.
- Not yet implemented (fail with a clear `UnsupportedOpError`):
  `sort`, general `reduce_window` (only cumsum/cumprod/cummax/cummin
  patterns), partial-window scatter, send/recv, multi-device anything.
- Buffer donation is ignored (correct, but no memory savings).

## License / provenance

Experimental personal project; builds against public JAX/OpenXLA (PJRT
header vendored from openxla/xla) and Apple's MLX.
