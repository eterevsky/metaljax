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

**Status**: correctness-first Stage 1. Real training runs work (the
[texmo](https://github.com/eterevsky/texmo) project trains end-to-end);
performance work (fusion via `mx.compile`, removing interpreter dispatch
overhead) is the current focus, so small models are still slower than the
CPU backend.

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

## Setup

```bash
cd metaljax
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

Short of publishing on PyPI, the easiest way is a **path dependency on
this checkout**. metaljax declares `jax` as a dependency, so your project
just needs:

```toml
# your project's pyproject.toml
[project]
dependencies = ["metaljax"]

[tool.uv.sources]
metaljax = { path = "/Users/oleg/metaljax", editable = true }
```

then `uv sync` (or with plain pip: `pip install -e /Users/oleg/metaljax`).
Build the plugin once in this checkout (`./plugin/build.sh`) — the dylib is
found automatically. Non-editable installs (`uv pip install
/Users/oleg/metaljax`, no `[tool.uv.sources]` `editable` flag) work as
well: the wheel bundles the dylib as long as `plugin/build.sh` was run
before installing. After that, any JAX program in your project runs on
Metal with `JAX_PLATFORMS=metal` (or
`jax.config.update("jax_platforms", "metal")` before first use).

Once the repo is pushed to GitHub, a git dependency also works:

```toml
[project]
dependencies = ["metaljax"]

[tool.uv.sources]
metaljax = { git = "https://github.com/eterevsky/metaljax" }
```

with the caveat that git/PyPI installs don't include a prebuilt dylib —
run `plugin/build.sh` from a checkout and point `METALJAX_PLUGIN_PATH` at
the result (packaging the dylib into a wheel is supported: `build.sh`
copies it into the package, and wheels built afterwards bundle it).

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
| transformer d256 L4 T256 b32 | 174.3 | **31.2** | 30.1 | 209.3 |
| transformer d512 L4 T256 b64 | — | **157.6** | 154.6 | — |
| GRU.256 T256 b256 (scan) | — | **112.0** | 50.7¹ | — |
| texmo `bench_jax.py` GRU b256 | 273.3 | **147.7** | — | — |

¹ torch uses its fused `nn.GRU` kernel; metaljax runs the scan as a
compiled-body loop — sequential models still pay per-timestep dispatch.

How: pure programs and counted-loop (`scan`/`fori_loop`) bodies are traced
once through `mx.compile` and replayed as fused Metal graphs; small
statically-counted loops are unrolled into the enclosing trace, so e.g. a
whole texmo training step (forward scan + backward + AdamW) becomes a
single graph replay. `METALJAX_COMPILE=0` disables compilation,
`METALJAX_TRACE_BUDGET` (default 20000 ops) caps trace sizes, and
`METALJAX_DEBUG=1` logs loop/compile decisions.

On texmo itself: models of texmo's typical tiny size (a handful of units)
remain CPU territory — thousands of microscopic kernels can't beat XLA:CPU
compiling the whole scan natively — while at `mgru.256` metal already
edges out CPU (5.1s vs 5.3s for 64 steps, batch 64, length 256).

## Known limitations

- Sequential (scan-heavy) models run each timestep as a compiled-graph
  replay from Python; fused-RNN-kernel frameworks are ~2× faster there.
- float64 is emulated in f32 (see `METALJAX_F64`); complex dtypes are
  unsupported.
- Not yet implemented (fail with a clear `UnsupportedOpError`):
  argmax/argmin-style multi-result `reduce`, `sort`, general
  `reduce_window` (only cumsum/cumprod/cummax/cummin patterns),
  partial-window scatter, send/recv, multi-device anything.
- Buffer donation is ignored (correct, but no memory savings).

## License / provenance

Experimental personal project; builds against public JAX/OpenXLA (PJRT
header vendored from openxla/xla) and Apple's MLX.
