# Vendoring MLX: build plan and decision brief (2026-08-17)

Written after the command-buffer defect was located and fixed
(`notes/mlx-patch-diagnosis.md`). This is the *how*, with measured numbers,
for Oleg's decision. Nothing here is shipped: the plugin's linkage is
unchanged in this milestone.

## 0. The headline

| question | measured answer |
|---|---|
| does our fork build from source on this machine? | yes — 1 m 50 s clean, ~1 min incremental (`-j12`, M5 Max) |
| does the build need anything new? | **Xcode's Metal Toolchain** (`xcodebuild -downloadComponent MetalToolchain`, 688 MB, one-time) — the wheels ship a prebuilt metallib, source builds compile it |
| is the output a drop-in for the wheel's C++ SDK? | yes — same layout (`mlx/lib/libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`; `mlx/include/{mlx,jaccl,metal_cpp}`), same file sizes |
| how much does the payload weigh? | 183 MB on disk, **64.2 MB compressed** (`mlx.metallib` is 162 MB of it) |
| how many wiring changes does the plugin need? | **one**: `METALJAX_MLX_DIR` (or the default in `plugin-native/third_party/mlx/workspace.bzl`) |
| what does it buy today | a Qwen3-8B prefill that is correct and deterministic instead of silently wrong 2 draws in 4 (§ diagnosis), and the retirement of the `MLX_MAX_*_PER_BUFFER` pins |

## 1. Fork shape (in place, `mlx-src/`, gitignored, never pushed by agents)

```
main                          v0.32.0 (7a1d4f5c), untouched
fix/command-buffer-split      cherry-pick of upstream 7e8b4ccc (#4099) + its test
fix/temporary-fence-tracking  generic end_encoding() hardening (ours, upstream-worthy)
diag/split-instrumentation    MLX_SPLIT_DEBUG / MLX_SPLIT_FIX scaffolding (never ship)
vendor/0.32.0                 main + the two fix branches, merged -- the build candidate
```

Remote is `https://github.com/ml-explore/mlx` (fetch only). When Oleg pushes the
fork, `origin` becomes his fork and `upstream` the Apple repo; the branches are
already shaped for `gh pr create` (one topic each, PR bodies drafted in
`notes/patches/`).

## 2. Build mechanics

```bash
# one-time, if `xcrun -sdk macosx metal --version` fails
xcodebuild -downloadComponent MetalToolchain

cd mlx-src && git checkout vendor/0.32.0
CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_METAL_CPP=$PWD/build/.../metal_cpp-src" \
CMAKE_BUILD_PARALLEL_LEVEL=12 \
  uv pip install --python <venv>/bin/python --no-build-isolation -e .
```

Notes that cost time to learn:

* **Pin metal-cpp.** MLX's CMake fetches
  `https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip` at configure
  time; Apple's server returned **502** twice tonight and the build fails hard.
  Vendoring must keep a local copy and point `FETCHCONTENT_SOURCE_DIR_METAL_CPP`
  at it (or vendor metal-cpp into the fork).
* `--no-build-isolation` + `cmake` and `setuptools>=80` in the build venv gives
  an *in-tree* `build/` that makes patch iteration ~1 min instead of ~2.
* The editable install writes `python/mlx/{lib,include}` — i.e. `cmake
  --install` output — which is exactly what `third_party/mlx/workspace.bzl`
  consumes.

## 3. Linkage plan (NOT executed in this milestone)

### 3.1 The native plugin (bazel)

`plugin-native/third_party/mlx/workspace.bzl` already resolves MLX from a
directory that `METALJAX_MLX_DIR` can override, and already emits a
`cc_import` + rpaths. Vendoring is therefore:

1. point the default at the vendored tree (`src/metaljax/lib/mlx/`) instead of
   `.venv/.../site-packages/mlx`;
2. keep the two rpaths as they are — `@loader_path/../../mlx/lib` becomes
   `@loader_path/mlx/lib` once the tree lives beside the plugin dylib;
3. copy `lib/` into the wheel in `hatch_build.py` next to
   `libmetal_pjrt.dylib`.

### 3.2 Private install name — the reason to bother

`libmlx.dylib`'s install name is `@rpath/libmlx.dylib`. If a consumer's process
also imports the public `mlx` wheel, dyld can serve **two** MLX runtimes, each
with its own `Device`, allocator, residency set and command encoders — the same
class of hazard the workspace file's comment already records (a 3.13 venv
silently loading a 3.14 venv's libmlx). So the vendored copy gets

```bash
install_name_tool -id @rpath/libmlx_metaljax.dylib libmlx.dylib
install_name_tool -change @rpath/libmlx.dylib @rpath/libmlx_metaljax.dylib core.*.so
```

and the plugin links the renamed one. Then "user also has mlx installed" is
merely two independent libraries rather than a symbol-interposition lottery.

### 3.3 The Python side (Stage 1) — the one real decision

`src/metaljax/engine.py` imports `mlx.core` at module import, so the shipped
wheel needs *an* mlx python module. Three options:

| option | what ships | cost |
|---|---|---|
| **A. vendor the whole python package** as `metaljax/_vendor/mlx/`, import it privately, fall back to public `mlx` | +64 MB compressed; drop the `mlx>=` dependency | one import shim; guarantees ABI identity between plugin and python side |
| B. vendor only the C++ SDK for the plugin, keep depending on public `mlx` for python | +22 MB (dylib only, no metallib? no — the metallib is needed by our dylib too) | **two libmlx in one process**: rejected |
| C. vendor C++ SDK, make the python engine optional (native engine only) | +64 MB, and Stage 1 stops working in wheels | loses the reference engine that today's parity harness uses |

**Recommendation: A.** It is the only one that preserves today's invariant
(*exactly one* libmlx per process, the same one on both sides), and it is what
makes the ABI handshake in `engine.py` (`compiled/linked/python` version
triple) collapse from a runtime check into a build-time fact — we control both
sides, so a skew is impossible by construction and the check becomes an
assertion that never fires.

### 3.4 Wheel size

| payload | on disk | compressed |
|---|---|---|
| `mlx.metallib` | 162.4 MB | ~63 MB |
| `libmlx.dylib` | 21.7 MB | ~1 MB |
| `libjaccl.dylib` | 0.93 MB | — |
| `include/` (C++ SDK, only needed to *build* against) | 3.9 MB | ~1 MB |
| **total lib+include** | **183 MB** | **64.2 MB** (zip -1) |

Today's metaljax wheel is ~110 KB and pulls `mlx` (a wheel of the same size) as
a dependency, so option A leaves the *installed* footprint roughly unchanged and
moves ~64 MB from PyPI's mlx wheel into ours. `include/` can be dropped from
the wheel (it is a build input, not a runtime one), saving ~1 MB.

If size ever matters: `-DMLX_METAL_JIT=ON` shrinks the metallib dramatically by
compiling kernels from source at first use — at the cost of first-call latency
and of the runtime shader compiler being in the loop. Not recommended for a
backend whose selling point is warm-graph latency.

## 4. Maintenance loop

1. Upstream tags `vX.Y.Z`; fetch it into `mlx-src`.
2. `git rebase --onto vX.Y.Z v0.32.0 fix/<branch>` for each fix branch. Branches
   that upstream has merged (e.g. `fix/command-buffer-split` once #4099 ships)
   **disappear** — rebasing drops them, which is the desired outcome and the
   signal to delete the branch.
3. Rebuild `vendor/vX.Y.Z`; run OUR acceptance, in this order:
   * `notes/data/mlx-cbuf-repro/repro_c.py` (2 s, must print 0 wrong),
   * `tests/test_command_buffer.py` (~30 s; the two corruption canaries must
     now *fail to find* a corrupting budget — that is the patched-build
     signature, and it is how we detect a regression in the other direction),
   * `scripts/mlx_patch_canary.sh` on the 8B asset, 3 draws at 40 / 512 / 2048
     MB (must be identical to the digit),
   * `scripts/texmo_check.py` + the release perf battery (the patch changes
     command-buffer *waits*, so measure, do not assume, the perf delta).
4. Version-stamp: the fork's build already reports
   `0.32.0.dev20260817+<sha>`; the wheel should carry the fork sha so a bug
   report names the exact MLX.

## 5. Which of the 8 tallied MLX bugs get a branch now

CLAUDE.md item 20's tally, with today's evidence:

| # | bug | reproducer in hand? | branch |
|---|---|---|---|
| — | **command-buffer split corruption** (all three faces) | yes: `repro_c.py`, 20 lines, pure MLX | **`fix/command-buffer-split` (done)** + `fix/temporary-fence-tracking` (done) |
| 8 | compiled scatter-add drops updates (fusion) | partial: the two-jax-test repro + `METALJAX_VERIFY_COMPILE`; no pure-MLX repro yet | **next**: needs an MLX-level scatter repro (MLX's python API has no general scatter — a C++ `tests/` case or `mx.put_along_axis` may serve) |
| 1 | strided-view reductions | fixed upstream (`a1e0e0b5`, post-0.32.0) | rebase onto next tag; no branch needed |
| 2 | strided argsort | fixed upstream (`052d4281`) | same |
| 3 | conv zero-dim short buffer | ours to re-check against main | issue-in-notes |
| 4 | rfftn unit-last-length | ours to re-check | issue-in-notes |
| 5 | FFT-vs-async_eval race | plausibly the same fence family — **re-test on the patched build** | issue-in-notes, then branch if it survives |
| 6 | `%.7g` rank-0 constant baking (1 ULP) | trivial repro, cosmetic-but-numeric | branch when convenient (`fix/compiled-constant-precision`) |
| 7 | complex sqrt cancellation | numerical, not a race | issue-in-notes |

Three of the eight are already fixed upstream after v0.32.0, which is itself an
argument for the vendored fork: *we get them by rebasing rather than by waiting
for a release*.
