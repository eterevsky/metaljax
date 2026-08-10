# Stage 2 P0: the C++-PjRtClient route, measured

Experiment (2026-08-10): can we build a *fully native* metaljax PJRT plugin by
subclassing `xla::PjRtClient` and letting XLA's `pjrt_c_api_wrapper_impl`
manufacture the whole PJRT C API, instead of hand-rolling the C surface the way
`plugin/metal_pjrt.cc` does today?

**Verdict: yes, comfortably.** All four checkpoints passed. Numbers, gotchas and
the reasoning are below.

Everything lives in `plugin-native/`. Nothing under `src/`, `plugin/`,
`native/`, `xla/` or `jax-v0.11.0/` was touched (only `.gitignore` gained
`plugin-native/bazel-*`).

## What was built

| file | lines | what |
|---|---|---|
| `plugin-native/WORKSPACE` | 159 | xla via `local_repository` + xla's own workspace cascade |
| `plugin-native/.bazelrc` | 41 | xla's `tensorflow.bazelrc` by absolute path + the flags that live in `xla/.bazelrc` itself |
| `plugin-native/third_party/mlx/workspace.bzl` | 45 | reusable repo rule for the MLX wheel's C++ SDK |
| `plugin-native/build_defs/xla.bzl` | 48 | `XLA_SHARED_OBJECT_SENSITIVE_DEPS` + `metaljax_cc_binary` (shared with the future runtime package) |
| `plugin-native/metal/BUILD` | 81 | 3 cc_libraries + the loadable dylib |
| `plugin-native/metal/metal_client.{h,cc}` | 430 | `MetalClient` / `MetalDevice` / `MetalDeviceDescription` / `MetalMemorySpace`, `BufferFromHostBuffer`, `CompileAndLoad` |
| `plugin-native/metal/metal_buffer.{h,cc}` | 252 | `PjRtBuffer` over `mlx::core::array` |
| `plugin-native/metal/metal_c_pjrt.{h,cc}` | 76 | `GetPjrtApi()` + the three create-hooks |
| `plugin-native/smoke_test.py` | 92 | the checkpoint harness |
| `plugin-native/README.md` | 20 | build + test commands |

~1,370 lines total, of which **758 are actual plugin C++**. For comparison,
`plugin/metal_pjrt.cc` is 1,483 lines of hand-written PJRT C API — and that is
only the C surface; all the behaviour behind it is Python.

## Versions and build configuration

- bazel **8.7.0** via bazelisk, pinned in `plugin-native/.bazelversion`.
  Chosen because that is `xla/.bazelversion` at this pin; jax's 7.7.0 is the
  other candidate but XLA's BUILD files at this commit are the ones we have to
  analyse, and they are validated against 8.7.0.
- **WORKSPACE mode**, not bzlmod: `common --noenable_bzlmod --enable_workspace`.
  XLA at this pin defaults to exactly that (`xla/.bazelrc` line 2); Bazel 8
  defaults the other way, hence both flags.
- XLA: `local_repository(name = "xla", path = "/Users/oleg/metaljax/xla")`,
  commit `131bf41acb4650e4391a640c3f1859c1c86ad74b` — the commit
  `jax-v0.11.0/third_party/xla/revision.bzl` pins. Nothing about XLA is
  downloaded. (jax's own `WORKSPACE` documents this substitution as the
  supported local-development arrangement.)
- MLX: `third_party/mlx/workspace.bzl` symlinks
  `<venv>/lib/python3.14/site-packages/mlx/{include,lib}` into an `@mlx` repo
  (`METALJAX_MLX_DIR` overrides) and exposes `cc_import` + `cc_library`. Same
  copy the Python side loads, so there can be no ABI skew.
- Build command:
  `cd plugin-native && bazel build //metal:libmetal_pjrt_native.dylib`
- Test command:
  `JAX_PLATFORMS=metal METALJAX_PLUGIN_PATH=$PWD/bazel-bin/metal/libmetal_pjrt_native.dylib .venv/bin/python plugin-native/smoke_test.py`

## Measurements (M5 Max, macOS 26.5)

| what | time |
|---|---|
| workspace fetch + load + analysis, very first run | 51 s (then failed at load, gotcha 1) |
| cold build of XLA's `example_plugin`, empty disk cache | 376 s (5,852 actions, critical path 65.8 s) |
| **cold build of our plugin** after `bazel clean --expunge`, `--disk_cache=` off | **399 s** (5,849 actions, critical path 70.5 s, 13,638 targets configured) |
| `bazel clean` + rebuild, warm disk cache | **3.0 s** (5,217 of 5,849 actions = disk-cache hits) |
| edit one of our `.cc` files, recompile only | 4.3 s |
| edit + relink the dylib | 6.2 s |
| no-op build | 0.11 s |

(The expunged run re-extracts external repos from bazel's repository cache but
does not re-download them; a truly first-ever machine adds the download of
~3.4 GB of archives on top.)

| artifact | size |
|---|---|
| `libmetal_pjrt_native.dylib` | 165,127,736 B (157 MB) |
| external repositories on disk | 3.4 GB |
| `~/.cache/metaljax-bazel` (disk cache) | 1.6 GB |
| exported symbols in the dylib | 155,034 (`_GetPjrtApi` among them) |

The 3.0 s figure is the important one: the LLVM/MLIR cost is paid **once per
machine**, and `--disk_cache` makes even a wiped `bazel-out` nearly free. Day to
day the loop is a 4–6 s edit-build.

`otool -L` on the dylib: `libc++`, `libSystem`, `CoreFoundation`, `Foundation`,
`libobjc`, and `@rpath/libmlx.dylib`. **No CPython, no nanobind** — the goal of
the experiment.

## Checkpoints

### 1. Workspace builds — PASS

`bazel build @xla//xla/pjrt/plugin/example_plugin:pjrt_c_api_myplugin_plugin.so`
succeeded from our out-of-tree workspace in 376 s, proving the WORKSPACE
cascade, the macOS toolchain and the whole `pjrt_client` +
`pjrt_c_api_wrapper_impl` dep cone.

### 2. jaxlib loads our plugin — PASS

```
jax.devices() -> [MetalDevice(id=0)]
  platform          : metal
  device_kind       : Apple GPU
  client.platform_version: PJRT C API\nmetaljax-native-p0
  memory spaces     : ['device']
```

`metaljax-native-p0` is our sentinel, so this is unambiguously *our* dylib and
not `plugin/build/libmetal_pjrt.dylib`. Every C-API entry point jaxlib
CHECK-fails without (`Device_GetAttributes` + non-null deleter,
`LoadedExecutable_AddressableDeviceLogicalIds`,
`LoadedExecutable_GetDeviceAssignment`) is synthesised by the wrapper from
`MetalDevice` / `MetalDeviceDescription` — we wrote none of them, and none of
the hand-encoded `DeviceAssignmentProto` bytes `metal_pjrt.cc` carries.

### 3. MLX links in and round-trips f32 — PASS

`jax.device_put(np.arange(12, dtype=np.float32).reshape(3,4)*0.5-1.0)` then
`np.asarray(...)`: shape, dtype and every value bit-exact.

Path: `PJRT_Client_BufferFromHostBuffer` → `MetalClient::BufferFromHostBuffer`
(copies into an `mlx::core::array`, `eval()`s it, releases the host buffer) →
`PJRT_Buffer_ToHostBuffer` → `MetalBuffer::ToLiteral` (memcpy out of unified
memory). Scope is deliberately narrow and *honest*: non-f32, non-row-major and
strided host buffers return `Unimplemented` naming the reason rather than
transferring the wrong bytes.

**Linker friction: none.** XLA statically links its own LLVM/MLIR/absl into the
dylib; `libmlx.dylib` stays a separate dynamic library and exports 2,626
symbols, **zero** of which contain `absl` or `llvm`. Nothing to interpose, no
`-force_load` games, no duplicate-symbol warnings. `bazel` records three
`LC_RPATH` entries, one of which is the absolute venv path we inject via
`linkopts`, so the dylib resolves `@rpath/libmlx.dylib` even when loaded by a
process that has not already imported `mlx`.

### 4. StableHLO arrives natively parsed — PASS

The wrapper's `PJRT_Client_Compile` calls `ParsePjrtProgram` (which runs
`xla::ParseMlirModuleString` → `mlir::parseSourceString` +
`UpgradeVersionedStablehlo`) and then
`client->CompileAndLoad(xla::MaybeOwningMlirModule, CompileOptions)`. We
override that method and log the module's ops. For `jax.jit(lambda x: x * 2)`
on an f32[2,3]:

```
[metaljax-native] CompileAndLoad(mlir): 6 ops:
  stablehlo.constant stablehlo.broadcast_in_dim stablehlo.multiply
  func.return func.func builtin.module
```

`stablehlo.*`, not `vhlo.*`. **The VHLO portable-artifact deserialisation and
version upgrade that `src/metaljax/engine.py` does by hand
(`stablehlo.deserialize_portable_artifact`) is free on this route** — the C++
engine receives a live `mlir::ModuleOp` in an `MLIRContext` with all HLO
dialects already registered. jax then surfaced our `Unimplemented` cleanly:
`UNIMPLEMENTED: metaljax-native P0: received a parsed MLIR module with 6 ops;
no executor yet.`

## Gotchas, with fixes

1. **`@local_config_cuda` is mandatory even for a Metal-only macOS build.**
   `@xla//xla/tsl:tsl.bzl` does an unconditional
   `load("@local_config_cuda//cuda:build_defs.bzl", ...)` at *load* time, so
   every XLA package is unanalysable without it:
   `ERROR: ... The repository '@@local_config_cuda' is not defined.`
   Fix: run the whole `cuda_json_init_repository` → `cuda_redist_init_repositories`
   → `cuda_configure` chain (plus nccl and nvshmem, same story) from
   `rules_ml_toolchain`. They only download small redistribution manifests when
   CUDA is absent; no CUDA payload is fetched. This is not optional and cannot
   be stubbed cheaply.

2. **`xla_cc_binary()` cannot be used out of tree.** Its
   `_XLA_SHARED_OBJECT_SENSITIVE_DEPS` list mixes `Label("//xla:...")` objects
   (repo-anchored, fine) with *bare strings* like
   `"//xla/tsl/profiler/backends/cpu:traceme_recorder_impl"`, and Starlark
   resolves bare label strings in a macro against the **calling** package's
   repository. Symptom:
   `ERROR: no such package 'xla/tsl/profiler/backends/cpu'` — pointing at our
   own workspace. Fix: plain `cc_binary(linkshared = True)` plus the identical
   list re-spelled with `@xla//` / `@tsl//` prefixes. It lives in
   `build_defs/xla.bzl` as `XLA_SHARED_OBJECT_SENSITIVE_DEPS` (wrapped by a
   `metaljax_cc_binary` macro) so every cc target in the workspace shares one
   copy, and so it can be diffed against XLA's when we bump the pin.

3. **bazelrc `import` and `%workspace%`.** `xla/.bazelrc` ends with
   `import %workspace%/tensorflow.bazelrc`; importing `xla/.bazelrc` from our
   workspace would resolve that against *our* root and fail. Fix: import
   `/Users/oleg/metaljax/xla/tensorflow.bazelrc` by absolute path and copy the
   handful of flags that live in `xla/.bazelrc` proper (bzlmod/workspace mode,
   `--incompatible_disallow_empty_glob=false`,
   `--legacy_external_runfiles=true`, the `clang_local` config definition).

4. **macOS needs jax's toolchain switch, not XLA's.**
   `tensorflow.bazelrc` sets `--apple_crosstool_top=@local_config_apple_cc//:toolchain`
   for `:macos` but leaves hermetic CC on. `jax/.bazelrc` adds
   `common:macos --config=clang_local`; without it the build tries the hermetic
   Linux toolchain. Copied verbatim. Nothing else from jax's rc was needed — in
   particular we do **not** want its
   `--linkopt=-Wl,-undefined,dynamic_lookup` (that exists so jaxlib can leave
   Python symbols unresolved; our plugin must have none, and keeping the
   default makes a missing symbol a build error instead of a runtime crash).

5. **`PjRtMemorySpace::ToCApiPtr()` is pure virtual at this XLA revision.**
   The example plugin does not show you this because it reports
   `device_count() == 0` and owns no memory spaces at all. Implement with the
   provided helper: `PjRtMemorySpaceCApiDelegator capi_delegator_{this};` and
   `ToCApiPtr() { return capi_delegator_.ToCApiPtr(); }`. Everything else about
   the device/memory-space surface follows
   `xla/pjrt/interpreter/interpreter_client.h`, which is a far better template
   than `example_plugin` — the example is a compile-only skeleton.

6. **`platform_version` is prefixed by jax.** `dev.client.platform_version`
   comes back as `"PJRT C API\nmetaljax-native-p0"`, so identity assertions must
   use substring matching.

7. **Bazel does not rebuild on `touch`.** Content digests, not mtimes — an
   apparent "0.1 s incremental build" after `touch` means nothing was stale.
   Measure incremental cost by actually changing bytes.

## Pure virtuals you must implement (this XLA revision)

Small and stable, which is much of why this route is cheap:

- `PjRtClient` (9): `process_index`, `device_count`,
  `addressable_device_count`, `devices`, `addressable_devices`,
  `memory_spaces`, `platform_id`, `platform_name`, `platform_version`.
  Everything else — compile, execute, transfers, layouts, topology — has a
  default that returns `Unimplemented`, so a plugin grows one method at a time.
- `PjRtDevice` (8): `client`, `IsAddressable`, `local_hardware_id`,
  `CreateAsyncTrackingEvent`, `TransferToInfeed`, `TransferFromOutfeed`,
  `memory_spaces`, `default_memory_space`.
- `PjRtMemorySpace` (8) and `PjRtDeviceDescription` (6): pure metadata.
- `PjRtBuffer` (17, of which 6 are the ones that do work: `on_device_shape`,
  `ToLiteral`, `LazyToLiteral`, `CopyRawToHost`, `GetOnDeviceSizeInBytes`,
  `GetReadyFuture`; the rest are accessors or can honestly return
  `Unimplemented`). This is the only class P0 had to think about.

## Integrating the executor runtime (`native/`) later

Oleg's directive: the runtime behind `native/program.h` will eventually be a
bazel `cc_library` the plugin links directly, replacing the prebuilt-dylib
arrangement. It is not in P0 scope (11 of `native/`'s 21 translation units
`#include` nanobind or `Python.h`, so a de-Python refactor has to come first),
but the workspace is already shaped for it: `@mlx` is a standalone reusable
repo rule that a runtime target can depend on exactly as `//metal:metal_buffer`
does, and `//metal` is a package rather than the workspace root, so a sibling
`//runtime` (or an external `@metaljax_runtime`) drops in without moving
anything.

Three ways to get those sources into this build:

**(a) `new_local_repository` at `../native`, build file supplied by us.**
Sources stay where they are, shared with the nanobind extension and
`native/build.sh`; no fork, no divergence. Mechanically proven already — this
is precisely what `third_party/mlx/workspace.bzl` does, and repo rules with
`local = True` re-resolve every build, so edits in `native/` are picked up
immediately. The BUILD file should be a checked-in
`plugin-native/third_party/metaljax_runtime/BUILD.runtime` (referenced via
`build_file`) rather than a string inside a `.bzl`, so a 20-file `srcs` list
stays readable. It also lets the migration be *incremental*: the `srcs` list
names exactly the TUs that have been de-Pythonised, and the nanobind ones stay
out of bazel until they are converted.
Cons: the source of truth sits outside the bazel workspace, so bazel cannot
enforce layering on it; `native/build/` has to be glob-excluded; and IDE/tooling
sees the files at two paths.

**(b) Move the workspace root up to `/Users/oleg/metaljax` with a
`.bazelignore`.** Conceptually the tidiest — one workspace, `//native`,
`//plugin-native/metal`, real bazel deps between them. But the `.bazelignore`
would have to cover `xla/`, `llvm-project/`, `jax/`, `jax-v0.11.0/`, `.venv/`,
`dist/`, `results/`, `native/build/`, `plugin/build/`; `xla/` in particular
*must* be ignored because a `local_repository` may not be nested inside the main
repo otherwise. That puts bazel's directory scanning across a tree that also
holds a constantly-churning `.venv` and multi-GB reference clones, and it makes
every metaljax contributor's `bazel` invocation the repo-wide one. Large blast
radius for a build we currently want to keep contained.

**(c) Fork the runtime sources into `plugin-native/`.** Zero build coupling and
the cleanest bazel story, but it duplicates ~200 KB of C++ that is under active
development, and the two copies would diverge the first time a bug is fixed in
one of them.

**Recommendation: (a).** It keeps `native/` as the single source of truth (the
nanobind extension and `METALJAX_ENGINE=native` keep working unchanged through
the transition), it uses a mechanism this workspace already proves, it supports
a file-at-a-time de-Python migration, and it leaves the option of graduating to
(b) later — moving the root is a strictly easier change once the sources
already build under bazel. Reserve (c) for the case where the runtime needs
bazel-specific source changes (generated headers, `select()`s) that would be
awkward to carry in a tree that `build.sh` also compiles.

## Verdict

The C++-PjRtClient route is viable and is the one I would take for phase 2. The
handshake cost is a one-time 6-minute build and a 160-line WORKSPACE; after
that the edit-build loop is 4–6 seconds, and `--disk_cache` makes a wiped
`bazel-out` a 3-second restore. In exchange XLA hands us, for free, the entire
PJRT C API — including the three entry points jaxlib CHECK-fails without and
the hand-encoded `DeviceAssignmentProto` that `plugin/metal_pjrt.cc` carries by
hand — plus natively-parsed StableHLO with the VHLO downgrade/upgrade dance
already done, which deletes a real chunk of `engine.py`. The pure-virtual
surface is small (9 methods for a client that loads) and everything else
defaults to `Unimplemented`, so the plugin can be grown one PJRT feature at a
time with jax reporting clean errors for the rest — exactly the incremental path
Stage 2 needs.

The ongoing costs are real but bounded, and worth naming:

- **A 3.4 GB external-repo tree and a 157 MB dylib.** The dylib is ~50× the
  hand-rolled plugin because it statically carries LLVM/MLIR/absl. That is fine
  for development but is a genuine question for wheel distribution — either we
  ship it (large wheels), or we prune the dep cone, or the released plugin keeps
  a hand-rolled C surface while development happens here. This needs a decision
  before release, not before phase 2.
- **Bazel becomes a build prerequisite** for anyone touching the native plugin,
  alongside the existing artisanal-clang `plugin/build.sh` and
  `native/build.sh`. Two build systems until Stage 2 lands.
- **Coupling to XLA internals that are not a stable API.** `PjRtClient`'s
  pure-virtual set, `MaybeOwningMlirModule`, `PjRtMemorySpaceCApiDelegator` and
  the `_XLA_SHARED_OBJECT_SENSITIVE_DEPS` copy all move between XLA releases.
  Since we pin XLA to the jax revision anyway, a bump is a scheduled, visible
  chore — the compiler tells us exactly what changed — rather than a silent
  break. The hand-rolled plugin is coupled to `pjrt_c_api.h` instead, which is
  more stable but which we then have to implement in full by hand.
- **`local_repository` hardcodes an absolute path.** Fine for this machine;
  CI/another checkout needs either the `tf_http_archive` fallback (jax's
  `revision.bzl` + sha256 are ready to copy) or a `--override_repository` flag.
