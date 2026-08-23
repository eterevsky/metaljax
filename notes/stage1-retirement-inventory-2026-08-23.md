# Stage-1 retirement — Stage A inventory (for 0.11.6)

Date: 2026-08-23. Tree: main @ 741a160, clean. No deletions performed; this is
the plan for the main agent's review.

Sources: two Explore sweeps (repo-wide Stage-1 reference map; tests/
classification), first-hand reads of pyproject.toml / hatch_build.py /
scripts/build_native_wheel.sh / src/jax_plugins/metal/__init__.py /
src/metaljax/__init__.py / plugin-native/metal/metal_client.cc / RELEASING.md /
README.md, and the 0.11.5 gate logs
(~/.cache/metaljax-bench/logs/release-0.11.5/).

---

## 0. Executive summary

- The Stage-1 surface is ~26k lines: src/metaljax Python engine 18,828 lines
  (25 tracked files), native/ pre-PJRT C++ engine 5,637 lines (21 files),
  plugin/ trampoline 1,483 lines (3 files). All deletable; plugin-native has
  **zero build dependencies** on any of it (PJRT headers come from @xla in
  bazel; WORKSPACE:161-164 records that native/ was forked into //runtime at
  6c2bb5e and left frozen).
- **The dev tree still runs Stage 1 by default today**: src/metaljax/lib/
  holds only the trampoline dylib, so every `JAX_PLATFORMS=metal` script with
  no `METALJAX_PLUGIN_PATH` resolves to it. Dropping
  libmetal_pjrt_native.dylib into src/metaljax/lib/ retargets run_jax_tests,
  run_export_harnesses, model_gate/final_run and every default-resolution
  consumer at once — the cheapest lever in the whole retirement.
- **The tests "native leg" never tested the shipped plugin.** tests/helpers.py
  calls metaljax.engine in-process, and METALJAX_ENGINE=native selects the OLD
  native/ nanobind extension. The 1053-passed baseline therefore overstates
  plugin coverage; the retained suite must be re-pointed (helpers.py rewrite)
  and re-counted, not derived from 1053. Details + reconciliation in §3.
- Nothing in the shipped wheel relies on src/metaljax/__init__.py side
  effects: plugin-native/metal/metal_client.cc:131-154 pins
  MLX_MAX_OPS_PER_BUFFER=800, MLX_MAX_MB_PER_BUFFER=512 and the
  MLX_METAL_GPU_ARCH precision default itself (setenv overwrite=0). §4.
- Blocking cross-dependency found: plugin-native/texmo_gate.py:108-109 imports
  scripts/texmo_check.py (Stage-1) just for `tc.train_set` — must be inlined
  before texmo_check.py can be deleted.
- Judgment calls needing a decision are collected in §9 (texmo perf-anchor
  re-baseline, sdist story, METALJAX_SYNC, mlx_patch_canary capability loss,
  command-buffer canary port, and others).

---

## 1. The Stage-1 surface (deletion set)

### 1a. src/metaljax/ — 18,828 lines Python (git-tracked)

| file | lines | note |
|---|---:|---|
| engine.py | 903 | PJRT-facing compile/execute; reads METALJAX_ENGINE (:104), METALJAX_SYNC (:596) |
| interpreter.py | 939 | StableHLO walker |
| tape.py | 2,641 | Stage-1 native-tape Python side |
| msl_scan.py | 3,558 | MSL codegen (native re-implementation lives in plugin-native/metal/metal_msl.cc) |
| qmm.py | 2,501 | quantized-matmul recognizer (native: re-implemented) |
| moe.py | 1,461 | MoE recognizer (native: re-implemented) |
| sdpa.py | 1,439 | SDPA recognizer (native: re-implemented) |
| dtypes.py | 374 | mlx<->np dtype tables |
| compile_options.py | 236 | superseded: native validates via xla `ApplyAllOptionOverrides()` (metal_client.cc:598-604) |
| _ir.py | 224 | MLIR ctx/attr decoding |
| diagnostics.py | 37 | live_buffer_floor; **zero live consumers outside src/** (grep: prose-only hits) |
| ops/ (13 files) | 4,445 | control 1073, elementwise 662, lapack 532, gather 478, reduction 457, conv 302, sort 247, shape 205, rng 185, linalg 154, collectives 85, callbacks 49, __init__ 16 |
| __init__.py | 71 | **NOT deleted — reduced** to docstring + __version__ + __all__ (§4) |

### 1b. native/ — the pre-PJRT C++ engine (21 tracked files, 5,637 lines .cc/.h + build.sh)

bindings.cc, compile.cc, config.cc, control.cc, dtypes.cc, emits.cc, host.cc,
metaljax_native.cc, msl.cc/.h, ops_*.cc (7), program.cc/.h, runtime.cc,
build.sh. Frozen since 6c2bb5e; live copy is plugin-native/runtime/. The only
live reference outside notes/ is scripts/mlx_patch_canary.sh:38 (itself
Stage-1-only, §2a) and the stale claim in plugin-native/README.md:38-40
(fix — it asserts a `../native` build dependency that no longer exists).

### 1c. plugin/ — the trampoline PJRT shim (3 tracked files)

metal_pjrt.cc (1,483), build.sh, vendor/pjrt_c_api.h. plugin-native does NOT
use plugin/vendor/ (its PJRT surface is `@xla//xla/pjrt/c:*` in
plugin-native/metal/BUILD:226-229). Referenced by: hatch_build.py
(_build_trampoline), pyproject sdist include, README, RELEASING.

### 1d. Stage-1 branches / machinery in kept files

- src/jax_plugins/metal/__init__.py: `_TRAMPOLINE_DYLIB`, `_library_path()`
  (imports metaljax to find the trampoline), the Stage-1 arm of
  `initialize()`, `_is_native_dylib()` (exists only to disambiguate the two
  plugins), and the `register=None` → `metaljax.ops.callbacks` fallback in
  `_register_callback_lowerings` (line 398). All go; details §4.
- hatch_build.py: `_build_trampoline`, `_PARKED` parking, `finalize()`
  restore, the `METALJAX_WHEEL_PLUGIN` two-leg switch. §5.
- pyproject.toml: `mlx>=0.32` dependency; sdist entries `plugin/metal_pjrt.cc`,
  `plugin/build.sh`, `plugin/vendor`. §5.

### 1e. Stage-1-only environment variables (die with the engine)

- `METALJAX_ENGINE` (engine.py:104 only; set only by mlx_patch_canary.sh:38)
- `METALJAX_SYNC` (engine.py:596 only; **gates run_stablehlo_bench.py:170** — §9.4)
- `METALJAX_COMPILE_OPTIONS` (compile_options.py ignore knob; native uses XLA's
  real validation, no knob)
- NOT Stage-1-only (native re-reads them, survive): METALJAX_COMPILE, _MSL,
  _DEBUG, _MEMDBG, _COMPILE_BYTES_MB, _QMM*, _MOE*, _SDPA, _MSL_*,
  _MATMUL_PRECISION, _TRACE_BUDGET, _RECOGNIZE, plus the native-only
  memory-governor family (63 vars total read by plugin-native; list gathered
  for the README env-table rewrite).
- `METALJAX_PLUGIN_PATH` — loader-level, survives.
- `METALJAX_WHEEL_PLUGIN` — becomes dead once hatch_build is native-only; drop.

---

## 2. Reference map and consumer classification

### (a) Stage-1-only — dies with Stage 1

| file | evidence | action |
|---|---|---|
| scripts/texmo_check.py | imports mlx + metaljax {dtypes, engine, Interpreter} (:44-47); pins jax_platforms=cpu at import (:28) — can never reach a plugin | DELETE. Its one consumer is plugin-native/texmo_gate.py:108 (inline `tc.train_set`, ~4 lines: set_tokens_dir + DataSet(path=TEXMO/"data"/"pride.txt")) |
| scripts/texmo_topconfs.py | imports engine (:54-56,:196); two-branch run_metal (:197-210) | DELETE. scripts/bench_texmo_pjrt.py is the stated drop-in replacement |
| scripts/mlx_patch_canary.sh | METALJAX_ENGINE=py (:38); whole design = swap libmlx by swapping the venv — only possible for the Python engine | RETIRE (capability loss flagged, §9.3) |
| tests/helpers.py | all 6 helpers are engine.* calls | REWRITE onto PJRT (§3) |
| RELEASING.md §5.5-7 | `uv build` with METALJAX_WHEEL_PLUGIN unset = trampoline wheel by definition; sdist story is trampoline-only | REWRITE (§6) |
| pyproject mlx dep + plugin/ sdist entries | — | REMOVE (§5) |

### (b) Two-leg — needs retargeting

| file | evidence | action |
|---|---|---|
| hatch_build.py | the canonical two-leg file (:57 env switch, :74-76 trampoline default, :81-93 finalize/park) | invert default to native; delete trampoline leg + parking + finalize (§5) |
| scripts/release/texmo_gate.sh | runs texmo_check (:66), texmo_topconfs (:73), compare (:79-80); never consults a plugin — in-process engine | repoint: correctness → plugin-native/texmo_gate.py; perf → bench_texmo_pjrt.py. **Perf anchor re-baseline required** (§9.1) |
| scripts/release/texmo_gate_report.py | parses texmo_check/topconfs output shapes (:9-19, :152-165) | retarget onto texmo_gate.py's ok/ok~/decline/FAIL vocabulary + bench_texmo_pjrt JSONL |
| scripts/release/run_gates.sh + gatelib.sh | route-agnostic except step 3 (texmo); selects no engine — inherits default resolution | step 3 repoint; ADD explicit METALJAX_PLUGIN_PATH pin to gatelib.sh so release rule 1 rests on a named binary, not on src/metaljax/lib contents |
| plugin-native/texmo_gate.py | imports scripts/texmo_check for tc.train_set (:108-109,:124) | inline the dataset construction — BLOCKING for (a) row 1 |
| scripts/bench_recognizers.py | `--stage1` arm (:193-208), native_over_stage1 ratio (:241-242) | drop the stage1 arm; core native+emits/-emits A/B untouched |
| scripts/model_bench/row15_ladder.sh | explicit stack loop stage1-vs-native (:38,:92-93,...) | collapse to native-only or freeze as closed-investigation artifact; TASKS.md:294 still names it (§9.8) |
| scripts/model_bench/row15_probe.py | doc/prose only ("unset = Stage 1", :15-17,:316) | 3-string doc fix |
| README.md | §How-it-works = Stage-1 pipeline; §layout; plugin/build.sh; env table | REWRITE (§6) |

### (c) Pure-native / route-agnostic — keeps working

scripts/bench_texmo_pjrt.py (promote to gate), run_jax_tests.py (env
JAX_PLATFORMS=metal,cpu; picks up whatever src/metaljax/lib carries),
run_export_harnesses.py, scripts/release/{jax_suite.sh, jax_suite_diff.py,
model_gate.sh, model_gate_report.py, summary.py, gatelib.sh},
bench_compare.py, bench_spec.py, cpu_parity.py, texmo_train.py (NOT Stage-1 —
plain jax.config platforms), texmo_topconfs_compare.py (pure JSONL math),
plugin-native/{execute_test, ingest_test, decline_census, coexist_test}.py
(pin METALJAX_PLUGIN_PATH to bazel-bin), scripts/vendor_mlx.sh (KEEP — MLX
staging tool), plugin-native/third_party/mlx/workspace.bzl (reads
src/metaljax/lib/mlx — the lib/ dir must survive), benchmarks/*.md and
notes/*.md (historical evidence — leave, do not scrub; 31 of 48 notes mention
Stage 1), scripts/model_bench/ rest (final_run.sh, adapters, mem_guard —
survivor env vars only).

### (d) Judgment calls — see §9

run_stablehlo_bench.py (METALJAX_SYNC guard), plugin-native/smoke_test.py +
wheel_poc_test.py (Stage-1-absence assertions go vacuous),
check_mxfp4_truncated.py (qmm.stats() silently degrades to {}),
row15_forensics.py (needs metaljax.__version__ — satisfied by the kept
minimal __init__), plugin-native/README.md:38-40 (stale native/ claim — fix
now), CLAUDE.md (main agent's; proposed patch in §6c), STATUS/TASKS
(historical; TASKS.md:294 flag), metaljax.diagnostics (no consumers → plain
delete, no dev-tooling carve-out needed).

---

## 3. tests/ classification

### Cross-cutting finding (decides the counts)

helpers.py drives metaljax.engine IN-PROCESS; METALJAX_ENGINE=native selects
the old native/ nanobind extension. Even "pure PJRT" tests (jax.devices)
resolved to the trampoline dylib → Stage-1 engine. So the 0.11.5 "native leg"
(1053 passed / 205 deselected, g2_suite_tests.sh:53-56 — an ad-hoc command
line, also recorded at notes/release-gates-0.11.5.md:661; no conftest, no
markers) measured mostly Stage-1. Reconciliation (notes/release-gates-0.11.5.md:665-672):

| | collected | pass natively* | fail natively* |
|---|---:|---:|---:|
| 4 deselected files (moe, qmm, qmm_mxfp4, engine_gc) | 205 | 134 | 71 |
| everything else | 1053 | 1053 | 0 |
| total | 1258 | 1187 | 71 |

*"natively" = METALJAX_ENGINE=native = the old extension, NOT the plugin.
The 71 = Python recognizer-counter assertions (moe 28, qmm 26, mxfp4 16,
engine_gc 1). The 134 passing deselected tests are real parity coverage NOT
in the 1053. Retained-set counts must be re-measured after the helpers
re-point.

### The hinge: rewrite tests/helpers.py

`check`/`run_metal` → jax.jit under jax.devices("metal")[0] (reference stays
pinned to CPU as today); `run_module`/`execute_module` → raw-StableHLO route
`dev.client.compile_and_load(text, [dev], xc.CompileOptions())` (already used
by plugin-native/execute_test.py:2641-2647); `run_metal_device` (device-storage
read) has no plugin equivalent — drop (1 caller). After this, verdict-1 files
below become genuine plugin tests with zero edits.

### Verdict 1 — runs through PJRT as-is (after the helpers re-point)

test_complex.py (9 defs), test_conv.py (13), test_elementwise.py (27, one raw
MLIR module), test_gather.py (18), test_linalg.py (26, two raw modules +
jax.export), test_random.py (7), test_reduction.py (16), test_shape.py (23),
test_sort.py (11), test_donation.py (4 — already pure PJRT, no helpers),
test_pjrt_surface.py (9 — pure PJRT except the `from metaljax import
compile_options` unit assertions at :100/:110, which die with the module; the
through-PJRT validation tests stay, error-message expectations may need
adjusting to XLA's own text).

### Verdict 2 — portable with small edits

- test_constants.py (7): drop the 2 mx.compile-retention tests (engine import
  :64, mx.get_active_memory); keep 5 splat-value checks.
- test_control.py (14): drop 2 introspection tests (engine.NATIVE.stats,
  ex.interpreter._body_cache, ex.runner); keep 12 parity tests.
- test_msl_scan.py (28): drop ~10 plan-inspection/monkeypatch tests
  (_msl_cache, _COOP_PREF, _PACK_TRIGGER, _reg_src, METALJAX_MSL_FORCE_BUILD_FAIL);
  keep ~18 pure scan-numerics checks — real coverage of the native MSL path
  once re-pointed.
- test_subbyte_float.py (15): drop/rework 1 run_metal_device use; rest ports.
- test_f64_policy.py (5): rewrite 2 compile-rejection tests to expect the
  plugin's compile-time decline (XlaRuntimeError) instead of a Python
  exception; c128 round-trip already pure PJRT.
- test_sdpa.py (18): keep the two qwen3_prefill_shrunk.mlir asset correctness
  tests re-implemented over compile_and_load; drop recognizer-count
  introspection (sdpa.analyze, _block_cost, _native_prog).
- test_concurrency.py (7): drop 2 engine-stream tests + the
  py/native engine parametrize; the 2 compile-option-table tests target the
  deleted module — delete unless trivially re-pointed at the plugin surface.
- test_dtypes.py (5): unit tests of metaljax.dtypes — port the bf16
  NaN-payload/sign bitcast contract to a device_put/np.asarray round-trip
  (the shipped property that fixed the 31B cold-load), delete the rest.
- test_qmm.py (36) / test_qmm_mxfp4.py (27) / test_moe.py (18): keep the
  numeric halves (they drive jax.jit on the metal device and pass natively
  today — the 134); drop qmm.stats()/moe.stats() counter assertions and
  internals unit tests (the 71) unless the plugin exposes equivalent counters.
- tests/data/ KEEP: qwen3_prefill_shrunk.mlir + qwen3_init_scan.mlir are the
  shipping MLX command-buffer upstream repro assets (TASKS.md:30).

### Verdict 3 — engine-internal, delete

test_engine_gc.py (4), test_eager_prune.py (16), test_compile_bytes.py (21),
and all five test_native*.py: test_native.py (4), test_native_buffers.py (11),
test_native_tape.py (109 defs, 2,053 lines), test_native_control.py (41),
test_native_gather.py (47), test_native_msl.py (24), test_native_tail.py (55)
— every one drives the OLD native/ nanobind extension via
sys.path.insert(native/build) + importorskip("metaljax_native").
~276 defs combined (more collected via parametrization) — the single largest
deletion.

### test_command_buffer.py — correction to the task premise

Its "native detectors" are ALSO Stage-1: `_native_mismatches` asserts
`engine.NATIVE is not None` under METALJAX_ENGINE=native (the old extension),
`_pipeline_mismatches` patches metaljax.ops.control internals. Salvageable
as-is: only `test_command_buffer_budgets_are_bounded` (:415, pure env check —
still meaningful, the budgets are pinned in metal_client.cc). RECOMMENDATION
(§9.5): port the two workload canaries (compiled-LLM-step determinism on
qwen3_prefill_shrunk.mlir; eager init-scan flush-cadence independence on
qwen3_init_scan.mlir) onto the PJRT/compile_and_load route rather than losing
them — the MLX corruption class they guard is still open upstream and
TASKS.md:38-41 demands they be rerunnable on any budget/MLX change.

---

## 4. Post-cleanup src/metaljax + runtime confirmation

What the real (non-staged) tree must keep:

- `src/metaljax/__init__.py` — reduced to docstring + `__version__` +
  `__all__`, exactly the shape build_native_wheel.sh:97-113 generates today,
  but hand-maintained (delete the generator). Consumers: wheel surface,
  row15_forensics.py:233, RELEASING version-bump checklist, hatch/version
  asserts.
- `src/metaljax/lib/` — gitignored build-products landing dir. Carries
  libmetal_pjrt_native.dylib (copied from frozen gated binary or bazel-bin)
  and mlx/{lib,include,VENDOR_STAMP} (vendor_mlx.sh output). It is ALSO the
  bazel MLX source: plugin-native/third_party/mlx/workspace.bzl:29 defaults to
  /Users/oleg/metaljax/src/metaljax/lib/mlx. Must survive. NOTE: include/ is a
  build input, stays in the tree, is NOT shipped in the wheel.
- `src/jax_plugins/metal/__init__.py` — simplified loader (below).

Runtime env-side-effect audit (task §A.4): CONFIRMED nothing relies on
src/metaljax/__init__.py. plugin-native/metal/metal_client.cc:131-154 pins,
in a static initializer that runs before MLX builds its Metal device:
MLX_MAX_OPS_PER_BUFFER=800, MLX_MAX_MB_PER_BUFFER=512, and (gated on
METALJAX_MATMUL_PRECISION=highest default) MLX_METAL_GPU_ARCH=applegpu_g16g —
all setenv(overwrite=0), so user exports win. The loader additionally sets the
GPU_ARCH default before dlopen (jax_plugins/metal/__init__.py:125-126) — keep,
harmless and earlier. Stage-B follow-up: metal_client.cc's comment says "KEEP
IN SYNC with src/metaljax/__init__.py, which OWNS the numbers and the
measurements" — ownership must move to metal_client.cc (the measurement
history is preserved in CLAUDE.md/notes; repoint the comment).

Loader simplification plan:
- Delete `_TRAMPOLINE_DYLIB`, `_library_path()`, `_is_native_dylib()`, and the
  Stage-1 arm of `initialize()`; `METALJAX_PLUGIN_PATH` override registers
  what it names, unconditionally (the disambiguation existed only because two
  plugins existed).
- `_register_callback_lowerings`: `register` becomes required (the native
  registrar from `_install_native_callbacks`); delete the
  `from metaljax.ops import callbacks` fallback (line 398). If the bridge is
  missing, callback lowerings stay unregistered (current graceful behavior).
- Everything else (linalg lowerings, donation, export-stability set, the
  ctypes bridge) is native-path code and stays verbatim.
- Loader continues to never import metaljax (find_spec) — unchanged contract.

---

## 5. Build-system rewrite (kills the staged build)

pyproject.toml:
- dependencies: DROP `mlx>=0.32` (nothing packaged imports it; vendored
  runtime is in the wheel; coexist test covers a user-installed mlx). KEEP
  jax>=0.11,<0.12, numpy, ml_dtypes (the loader's callback bridge imports
  both).
- description → "Metal backend for JAX on Apple silicon: native PJRT plugin
  executing StableHLO on a vendored MLX runtime" (or similar); classifiers
  unchanged (still Beta).
- sdist: at minimum drop the plugin/ entries; recommendation §9.2 = go
  wheel-only (an installable-from-source sdist is impossible: the native
  build needs the bazel workspace + ~160MB static deps + vendored MLX, which
  an sdist cannot reasonably ship).
- [tool.hatch.build.targets.wheel] packages/artifacts unchanged
  (src/metaljax now contains only __init__ + lib/).

hatch_build.py (native-only rewrite):
- initialize(): dylib from METALJAX_NATIVE_DYLIB (prebuilt/frozen — the
  release path, keeps rule 1 a build-time fact) else bazel build in
  plugin-native/; force_include the vendored MLX runtime from
  src/metaljax/lib/mlx (METALJAX_VENDORED_MLX override kept for out-of-tree
  staging); tag py3-none-macosx_14_0_arm64.
- DELETE `_build_trampoline`, `_PARKED`, `finalize()`, the
  METALJAX_WHEEL_PLUGIN switch.
- Fold in verification from build_native_wheel.sh: assert
  `__version__ == pyproject version` at build start (the 0.11.4 cosmetic-drift
  bug); after-build checks stay in the wrapper (below).

scripts/build_native_wheel.sh → thin wrapper: no staging tree, no generated
__init__, no mlx-dep stripping (the three workarounds its header documents
all die). Keeps: default dylib = frozen-path.txt, `uv build --wheel`, and the
verification block (no-Stage-1-modules listing check — now a regression guard
against resurrection; wheel-vs-gated-binary sha256; contents listing).

scripts/vendor_mlx.sh: UNCHANGED (still the MLX staging tool).
plugin-native/third_party/mlx/workspace.bzl: unchanged.

---

## 6. Docs

### 6a. README.md rewrite (Stage B)

- §How it works: new pipeline — jax → jax_plugins/metal (loader) →
  libmetal_pjrt_native.dylib (a self-contained xla::PjRtClient; StableHLO
  ingested natively, executed through the plugin's tape/MSL engine) → vendored
  private-install-name MLX runtime (metaljax/lib/mlx/) → Metal. Note: no
  Python in the hot path; callbacks/linalg host ops via the ctypes bridge.
- §Install: wheel carries its own Metal runtime; pip mlx not required and
  coexists if present. Source installs = git checkout + bazel (sdist story per
  §9.2 decision).
- §Developing from source: bazel build //metal:libmetal_pjrt_native.dylib;
  scripts/vendor_mlx.sh; execute_test/texmo_gate/ingest_test drivers.
- §Running the tests: pytest through the real plugin (JAX_PLATFORMS=metal),
  new counts from Stage C.
- §Env table: ~90% survives (native re-reads the same knobs); drop
  METALJAX_COMPILE_OPTIONS (native uses XLA's real validation), fix the
  METALJAX_PLUGIN_PATH row to name libmetal_pjrt_native.dylib; consider adding
  the memory-governor family (METALJAX_MEM_*) rows.
- §Repository layout: plugin-native/ + runtime/, src/ reduced to loader +
  version + lib/.
- Benchmark tables: refresh from the 0.11.5 release column (STATUS.md fn36,
  benchmarks/models.md) — release rule 1: numbers from the release binary;
  the current Gemma table is Stage-1-era.

### 6b. RELEASING.md rewrite (Stage B)

- Intro + §5.5: the wheel is native-only, built by build_native_wheel.sh (or
  plain `uv build` + METALJAX_NATIVE_DYLIB once hatch is native-only); the
  "sdist compiles at install" sentence dies; wheel-tag note updated.
- §0 preflight: version-two-places check unchanged (both files still exist).
- §3 texmo gate: rewritten around plugin-native/texmo_gate.py (106/106
  baseline) + bench_texmo_pjrt.py + the re-baselined anchor (§9.1).
- Gate knobs table: TEXMO_* knobs re-keyed to the new report script.

### 6c. Proposed CLAUDE.md patch (main agent's to apply — NOT edited by me)

1. "Architecture decisions" — replace the two-stage framing with: Stage 1
   (Python engine) RETIRED in 0.11.6; the native plugin (plugin-native/,
   xla::PjRtClient, bazel) is the only engine; vendored-MLX ground rule
   unchanged.
2. "Layout" — src/metaljax/ = "__init__.py (version only) + lib/ (native
   dylib + vendored MLX land here; also the bazel MLX source dir)"; delete
   plugin/ and native/ rows; add plugin-native/ + runtime/ as the engine.
3. Roadmap — add item 23: "Stage-1 retirement (0.11.6): src/metaljax engine
   (18.8k lines), native/ (5.6k), plugin/ trampoline (1.5k) deleted;
   wheel builds native-only from the real tree; tests re-pointed through
   PJRT; texmo gate = plugin-native/texmo_gate.py + bench_texmo_pjrt.py."
4. "Implementation notes" — mark the trampoline notes (sync events,
   engine.py import rules, '#'-format gotcha) historical; keep the MLX
   precision/budget notes but state metal_client.cc now owns them; update
   the METALJAX_PLUGIN_PATH note.
5. "Environment note" — consumers get the dylib via the wheel/editable
   install; build.sh reference updated to the native flow.

---

## 7. Stage B execution order (dependency-safe)

1. plugin-native/texmo_gate.py: inline dataset construction (unblocks 2).
2. Fix plugin-native/README.md:38-40 (stale native/ dependency claim).
3. tests/: rewrite helpers.py onto PJRT; apply verdict-2 edits; delete
   verdict-3 files; port the two command-buffer canaries (per §9.5 decision).
4. Loader simplification (src/jax_plugins/metal/__init__.py).
5. pyproject + hatch_build native-only rewrite; reduce
   src/metaljax/__init__.py; place the native dylib in src/metaljax/lib/
   (THE default-resolution flip — from this commit the dev tree runs the
   native plugin by default).
6. Delete: src/metaljax engine modules + ops/, native/, plugin/;
   scripts/texmo_check.py, scripts/texmo_topconfs.py, mlx_patch_canary.sh
   (per §9.3).
7. Retarget: release texmo gate (texmo_gate.sh + report), bench_recognizers
   --stage1 removal, row15_* doc strings, run_stablehlo_bench sync guard
   (per §9.4), check_mxfp4_truncated explicit-degradation, gatelib
   METALJAX_PLUGIN_PATH pin.
8. build_native_wheel.sh → thin wrapper; metal_client.cc ownership comment.
9. README + RELEASING rewrite.
10. Grep-proof pass (zero refs to deleted modules outside notes/).

## 8. Stage C validation plan

1. Wheel: `uv build` from the CLEAN tree (no staging); carries the frozen
   gated dylib (sha256 == frozen-vendor-d651add3) + vendored MLX;
   12-file-class contents; fresh-venv (3.12/3.14) install drives Metal with
   NO pip mlx; coexist check with pip mlx present.
2. tests/: full suite through PJRT (JAX_PLATFORMS=metal / default
   resolution). Report retained counts reconciled against 1258 collected /
   1053+205 (expected: ~verdict-1+2 files, minus ~350-400 deleted defs, plus
   the 134 recovered recognizer-parity tests; exact numbers from the run —
   the old 1053 is NOT the comparison bar, see §3).
3. plugin-native: execute_test.py, ingest_test.py, `bazel test //...`.
4. plugin-native/texmo_gate.py: 106/106 (machine lock held, durable log in
   py-retire/).
5. Grep-proof: zero references to engine/interpreter/tape/msl_scan/qmm/moe/
   sdpa/dtypes/_ir/compile_options/diagnostics/ops/, native/, plugin/,
   METALJAX_ENGINE, METALJAX_SYNC, libmetal_pjrt.dylib outside notes/ (+
   benchmarks/ historical tables, STATUS/TASKS history — reported, not
   scrubbed).
Rule-2 discipline: any regression in any of these = named in the report,
never PASSed over.

## 9. Decisions needed (main agent / Oleg)

1. **texmo perf anchor**: notes/data/texmo-topconfs-final.jsonl was measured
   on the Stage-1 route; retargeting the gate to bench_texmo_pjrt.py
   invalidates it. Candidate ready-made anchor: the 0.11.5 native topconfs
   run (release-0.11.5/topconfs-native-b.jsonl / the regate texmo-tests
   sweep). Needs an explicit re-baseline sign-off (gate rules).
2. **sdist**: recommend wheel-only distribution (native build can't ship in
   an sdist). Alternative: keep a non-installable source archive.
3. **mlx_patch_canary.sh**: venv-swap A/B of MLX patches only worked against
   the Python engine; the native dylib links vendored MLX statically by
   install name. Retire the script and accept the capability loss, or
   redesign (e.g. rebuild plugin against a candidate MLX). Genuine loss —
   Oleg should see this one.
4. **METALJAX_SYNC / run_stablehlo_bench.py:170**: the guard hard-refuses
   metal runs and the var dies with engine.py. Decide the native
   timing-barrier story (is block_until_ready honest on the native plugin?
   If yes, drop the guard; if no, time through np.asarray and reword). The
   script drives the xla suite AND the 8B command-buffer canary — do not
   delete blind.
5. **Command-buffer canaries**: port the two workload detectors to the PJRT
   route (recommended — the MLX bug class is open and TASKS.md requires
   rerunnable canaries) vs. keep only the budget-bounds test.
6. **Recognizer telemetry in tests**: test_qmm/moe counter assertions died
   with qmm.stats(); if the native plugin exposes equivalent counters
   (decline_census / METALJAX_DEBUG output?), some could be re-pointed —
   worth it only if cheap.
7. **smoke_test/wheel_poc_test Stage-1-absence assertions**: keep one release
   as resurrection guards (recommended), then drop.
8. **row15_ladder.sh**: TASKS.md:294 names it for an OPEN investigation that
   assumes both stacks. Freeze vs. collapse — Oleg's call on the
   investigation's status.
