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

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `JAX_PLATFORMS` | *(unset)* | Set to `metal` (or `metal,cpu`) to select the backend; unset keeps CPU default. |
| `METALJAX_MATMUL_PRECISION` | `highest` | On M5-class GPUs MLX routes f32 GEMM through the neural accelerators at ~bf16 input precision (~4e-3 error). `highest` pins MLX kernels to the previous GPU generation for exact f32; set `default` to allow the fast path. |
| `METALJAX_F64` | `downcast` | Metal has no float64. `downcast`: compute in f32 while reporting f64 shapes/dtypes to JAX (one warning). `error`: fail on any f64. |
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

## Known limitations

- **Performance**: the interpreter currently dispatches ops eagerly from
  Python (~500 ms/step vs ~110 ms/step CPU on texmo's GRU.256 bench).
  The planned fix: pre-compiled per-block closures, counted-loop
  detection for `scan`, and `mx.compile` fusion.
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
