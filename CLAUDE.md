# metaljax — a Metal backend for JAX

## Goal
Build a JAX backend that runs on Apple-silicon GPUs via Metal, on this machine
(M5 Max, macOS 26.5, Xcode 26.6). North star: **everything the `~/texmo`
project runs on CPU must run on Metal, preferably faster** (texmo trains many
small language models: dense/GRU-style layers, norms, softmax/CE loss, Adam,
`lax.scan`, RNG). Milestone zero: `a = jnp.array([1, 2, 3]); print(2 * a)`
executes on the Metal device through plain `jax.numpy`.

## Architecture decisions (agreed with Oleg)
- **Staged integration.**
  - *Stage 1 (current):* a real PJRT plugin — a thin native dylib implementing
    the PJRT C API that **trampolines back into Python** (same process) for
    compile & execute. JAX sees a genuine `METAL` device; unmodified
    `jax.numpy` code works.
  - *Stage 2 (after Stage 1 proves out):* migrate the engine to fully native
    code (C++/Obj-C++, StableHLO parsed natively).
- **Execution engine (Stage 1): MLX** (`mlx.core`). Chosen per Oleg's
  criteria (compat with arbitrary jit'd ops + vmap/etc → performance → build
  ease): MLX is Apple's maintained Metal array library, lazy by default, and
  `mx.compile` lets us trace our StableHLO interpreter into a fused, cached
  Metal graph per executable. The op layer is kept behind an internal
  interface so hand-written Metal shaders / MPS can replace pieces later.
- **Compile path:** PJRT hands us serialized StableHLO (MLIR bytecode). Parse
  with `jaxlib.mlir` bindings (already shipped in jaxlib), interpret module
  op-by-op mapping StableHLO ops → MLX ops; wrap whole executables in
  `mx.compile` where control flow allows.
- **Dtypes:** Metal/MLX have no float64. f32/f16/bf16/int/bool only;
  `jax_enable_x64` stays off. f64 in programs is an expected failure, not a
  target.

## Ground rules
- All changes live in `/Users/oleg/metaljax`. **Never modify `metaljax/jax/`**
  or `metaljax/llvm-project/` — read-only reference clones (gitignored).
  JAX clone HEAD ≈ 2026-07-23, matches installed jax 0.11.0.
- Git: repo remote is git@github.com:eterevsky/metaljax.git. **Commit locally;
  never push** — Oleg pushes himself. No PRs.
- Python via **uv**: venv at `metaljax/.venv` (CPython 3.13.5), installed:
  `jax 0.11.0`, `jaxlib 0.11.0`, `mlx 0.32.0`, numpy, pytest.
  Run things with `.venv/bin/python`.
- Correctness bar: every implemented op/feature gets a pytest comparing Metal
  results against the CPU backend (tolerances appropriate for f32).
- `~/texmo` is the acceptance workload (read-only; don't modify it either).

## Layout (planned)
- `CLAUDE.md` — this file.
- `pyproject.toml`, `src/metaljax/` — Python package: StableHLO→MLX
  interpreter, runtime glue, `jax_plugins` registration entry point.
- `plugin/` — native PJRT plugin (C/C++/Obj-C++), built with clang from
  Xcode; links nothing heavy, resolves Python symbols via
  `-undefined dynamic_lookup`, vendors `pjrt_c_api.h`.
- `tests/` — pytest suite (Metal vs CPU).
- `notes/` — investigation notes worth keeping.

## Roadmap / status
1. ✅ Decisions above; env set up (uv venv, jax 0.11.0, mlx 0.32.0).
2. ✅ StableHLO→MLX interpreter (`src/metaljax/`): 122 pytest cases pass vs
   CPU — elementwise, shapes, dot_general/einsum, reductions, cumops,
   while/if/scan, threefry RNG, gelu/composites, bf16/f16.
3. ✅ PJRT plugin (`plugin/metal_pjrt.cc` → `plugin/build/libmetal_pjrt.dylib`,
   build via `plugin/build.sh`): `2 * jnp.array([1,2,3])` runs on
   MetalDevice(id=0); jit(grad) matches CPU ~1e-8; RNG bit-exact vs CPU;
   scan/matmul correct through the real backend.
4. ✅ **texmo trains on Metal.** gather/scatter (incl. windowed + batching
   dims), sdy identity ops. 131 tests. f64 policy (per Oleg): STRICT by
   default — f64 pass-through OK (stored f32, bit-identical), f64 *compute*
   fails at compile naming the op; METALJAX_F64=downcast opts into f32
   emulation (scripts/texmo_train.py sets it explicitly: optax AdamW's
   beta**step is real f64 arithmetic under x64).
   Driver: scratchpad texmo_metal_train.py (imports ManagerJax directly,
   avoids texmo.py's hardcoded platforms; torch installed just for imports).
   `bench_jax.py --platform metal` works (flag committed to texmo repo).
   Verified: `bits.1+bp|rnn.1.tanh` and `bits.1+bp|mgru.4-dense.4.gelu` train
   end-to-end with sane losses; train_and_eval incl. recurrent eval path.
5. ✅ **Performance pass 1** (counted-loop detection + mx.compile): pure
   mains and counted while bodies (scan/fori: cond `i < N`, body `i+1`)
   trace once through mx.compile, then replay fused graphs; caches keyed by
   ir.Block (pointer-stable across traversals). METALJAX_COMPILE=0 disables.
   Numbers (full train steps, f32, scripts/bench_compare.py):
   transformer d256/b32: metaljax 31.2 ms vs torch-MPS 30.1 vs jax-cpu 174;
   d512/b64: 157.6 vs torch 154.6 (within 2% — Oleg's aspirational goal met
   for transformers). GRU.256/b256: 112 vs torch fused nn.GRU 50.7 (2.2×
   gap = per-timestep python replay); texmo bench_jax b256: 147.7 vs cpu
   273.3; b64: 133.6 vs cpu 113.7. Was 502 ms/step before this pass.
   Pass 2 (loop unrolling): small statically-counted loops unroll into the
   enclosing mx.compile trace (interp._in_trace flag decides per context;
   never nest mx.compile). METALJAX_TRACE_BUDGET (20000) caps any single
   trace: MLX retains every intermediate while tracing AND each pending
   replay pins its buffers → Metal's ~500k live-buffer limit; eager counted
   loops flush (mx.eval) every ~25000/cost iterations. _block_cost must
   traverse func.call/composite callees (undercounting made the engine
   compile texmo's whole 256-step chunk → buffer exhaustion). Result: texmo
   train step = ONE graph replay (~34ms for mgru.4 — kernel-launch-bound;
   CPU wins tiny models on physics; parity at mgru.256: 5.1s vs cpu 5.3s).
   METALJAX_DEBUG=1 logs loop/compile decisions.
   Future perf ideas: prepared-closure interpreter rewrite (kills
   MLIR-wrapper overhead on eager paths + trace time); if/case branch
   compile; hoisting loop-invariant input projections (torch's fused-GRU
   trick). Remaining coverage gaps: argmax/argmin multi-result reduce
   (sampling path), sort, partial-window scatter.
6. ✅ PyPI release prep (v0.1.0): plugin ported to CPython **limited API**
   (>=3.12) → single wheel py3-none-macosx_14_0_arm64 for all Pythons;
   hatch_build.py compiles the dylib at wheel-build time. Gotcha: never use
   '#' formats in PyObject_CallMethod (ABI differs across versions — broke
   on 3.12). Wheel verified on fresh 3.12 venv; twine check passes.
   LICENSE = Apache-2.0 (flagged for Oleg's confirmation). RELEASING.md has
   the upload steps; **Oleg publishes himself** (like git pushes).
7. ⬜ Stage 2: migrate engine to native code (llvm-project clone available).

## Environment note
- venv is **Python 3.14.4** (texmo needs PEP-649 lazy annotations; jaxlib
  0.11.0 / mlx 0.32 ship cp314 wheels). torch (CPU wheel) installed only so
  texmo modules import; the JAX path never calls it.
- Consumers depend on metaljax via a path dependency (uv sources, editable
  or not) — build.sh copies the dylib into src/metaljax/lib/ so wheels
  bundle it; verified with a fresh-venv non-editable install. jax is a
  declared dependency (>=0.11,<0.12 — the PJRT header pin).

## Implementation notes (hard-won)
- Select the backend with `JAX_PLATFORMS=metal` (or `metal,cpu` to keep CPU
  available for comparisons). Plugin registered at priority -1 so CPU stays
  default otherwise. Registration: `src/jax_plugins/metal/__init__.py`
  (namespace pkg + entry point); env override `METALJAX_PLUGIN_PATH`.
- PJRT programs arrive as **StableHLO portable artifacts** (VHLO bytecode).
  `ir.Module.parse` "succeeds" but yields vhlo.* ops — must
  `stablehlo.deserialize_portable_artifact(ctx, code)` first (engine.py does).
- The MLIR context must register **sdy** (+ mpmd) — jax 0.11 emits sdy attrs
  and `sdy.sharding_constraint` ops even single-device (identity for us).
- pjrt_c_api.h 0.114 defines PJRT_Error/PJRT_Memory as vtable-carrying
  structs (new C-ABI style); we subclass/instantiate them. jaxlib **fatally
  requires** (CHECK-fails on error): Device_GetAttributes with non-null
  attributes_deleter, LoadedExecutable_AddressableDeviceLogicalIds, and
  LoadedExecutable_GetDeviceAssignment (returns hand-encoded
  DeviceAssignmentProto bytes — see metal_pjrt.cc).
- **M5 GPU (applegpu_g17s) MLX f32 matmul is low-precision** (~4e-3, neural
  accelerators). metaljax defaults to accuracy: sets
  MLX_METAL_GPU_ARCH=applegpu_g16g before mlx loads; opt out with
  METALJAX_MATMUL_PRECISION=default (see src/metaljax/__init__.py).
- bf16 constants can't cross the MLIR python bindings as numpy — decoded via
  attribute text (incl. hex-blob form) in `_ir.dense_to_np`.
- Everything in the plugin is synchronous; all PJRT events are born ready.
  Trampoline: C shim (no deps, `-undefined dynamic_lookup`) → GIL →
  `metaljax.engine` (compile_program/execute/buffer_from_host/to_host).
  engine.py + interpreter must import only jaxlib/mlx/numpy, never jax.
