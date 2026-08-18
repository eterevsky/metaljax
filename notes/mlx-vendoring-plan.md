# Vendoring MLX: build plan, and what was built (2026-08-17 / 18)

Written after the command-buffer defect was located and fixed
(`notes/mlx-patch-diagnosis.md`). §§1–5 are the original plan and decision
brief; **§6 is the as-built record** of the milestone that executed it on
2026-08-18. Where the two disagree, §6 is what exists.

**Status: BUILT AND ACCEPTED.** The plugin links our fork, the release wheel
carries it, and the acceptance battery is in §6.4. The sentence that used to
stand here — "nothing here is shipped" — is no longer true.

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

> **Done, 2026-08-18: Oleg pushed the branches to `eterevsky/mlx`.** The
> vendored source is therefore durable rather than living only in this
> machine's gitignored checkout — which matters, because `VENDOR_STAMP`
> identifies the build by commit (§6.2) and that commit is now fetchable.
> What remains upstream is the release request on ml-explore/mlx#4099 and
> our own hardening PR.

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

---

# 6. As built (2026-08-18)

The milestone that executed §§2–3, plus the two things the plan did not
foresee. Artifacts: `~/.cache/metaljax-bench/logs/mlx-vendoring/`
(`v1r*` build, `v2*` contracts, `v3*` command buffer, `v4r*` texmo,
`v5*` model rows, `v6r*` wheels).

## 6.1 What the build produces

`scripts/vendor_mlx.sh` stages the fork's install into
`src/metaljax/lib/mlx/` (gitignored — it is a build output, not source):

| file | size | what it is |
|---|---:|---|
| `lib/libmlx_metaljax.dylib` | 21.6 MB | our fork, install name `@rpath/libmlx_metaljax.dylib` |
| `lib/libjaccl_metaljax.dylib` | 0.94 MB | its companion, renamed to match |
| `lib/mlx.metallib` | 162.4 MB | kept under its own name — MLX finds it by dladdr'ing itself and looking in its own directory, never by install name |
| `include/` | 3.9 MB | build input only; deliberately not shipped |
| `VENDOR_STAMP` | — | fork branch @ sha, dirty count, sha256 of every staged file |

The rename is the whole point: a consumer process may also import the public
`mlx` wheel, and two images claiming `@rpath/libmlx.dylib` in one process is
a symbol-interposition lottery over one GPU. Renamed, they are two
independent libraries — measured in §6.4 (`v2h`, `v6r`), where `vmmap` and
`DYLD_PRINT_LIBRARIES` show both loaded and both working.

The vendored pair is self-locating: `libmlx_metaljax.dylib` carries its own
`LC_RPATH` of `@loader_path`, so it finds its libjaccl in the wheel, in
`bazel-bin`, and in the staging tree alike. Both are re-signed after
`install_name_tool` (which invalidates the adhoc signature on arm64).

## 6.2 The ABI handshake — the plan's one real miss

§3.3 predicted the handshake would "collapse from a runtime check into a
build-time fact". It did, but not for free: the first vendored build made it
**fail**.

`native/metaljax_native.cc` reports `compiled_mlx_version()` from
`MLX_VERSION_{MAJOR,MINOR,PATCH}` — *numbers* — while `linked_mlx_version()`
and `mlx.core.__version__` are both `mx::version()`, the string CMake was
given. `setup.py` appends `.dev<date>+<sha>` to that string unless
`PYPI_RELEASE` is set, so a fork build is **skew by construction**, and
`engine.py` raises `ImportError` on skew. The first build produced

    ('0.32.0', '0.32.0.dev20260817+651c39cd', '0.32.0.dev20260817+651c39cd')  SKEW

which would have blocked every native run in the battery.

`src/` and `native/` are frozen this era, so the fix is on the build side:
the vendored build sets **`PYPI_RELEASE=1`**, whose version is exactly
`0.32.0` for both the C++ and the Python side, and the triple agrees. It
changes nothing else — the two-stage wheel split (`mlx` / `mlx-metal`) is
gated on `MLX_BUILD_STAGE`, not on this.

**The cost, and it is a real one: our patched MLX now reports the same
version string as upstream's unpatched release.** Provenance moves entirely
into `VENDOR_STAMP` (staged beside the library, shipped inside the wheel at
`metaljax/lib/mlx/VENDOR_STAMP`), which records the fork branch, the commit
and the sha256 of every file. *Read the stamp, not `pip list`, to answer
"which MLX is this?".* The better long-term fix is to compare only
`major.minor.patch` in `engine.py`'s handshake, which would let the fork keep
its `+sha` — a one-line change in a frozen file, for Oleg.

## 6.3 Linkage and the wheel

`plugin-native/third_party/mlx/workspace.bzl` now defaults to
`src/metaljax/lib/mlx` and picks its rpath from what it finds there: the
vendored tree gets `@loader_path/mlx/lib` (the library lives *inside* the
wheel, next to the plugin), a pip wheel's `mlx` still gets the old
`@loader_path/../../mlx/lib`, so a pre-vendoring A/B can still be built with
`METALJAX_MLX_DIR`. Order is load-bearing and unchanged: the relative rpath
precedes this machine's absolute one, or a wheel installed elsewhere on this
machine would silently load the build tree's library.

The release wheel is **native-only** (Oleg, 2026-08-18) and is built by
`scripts/build_native_wheel.sh` from a *staged* source tree rather than from
the repo:

* hatchling's file selection is static config, so a build hook cannot
  subtract `src/metaljax`'s Stage 1 modules from the wheel; and moving them
  aside during the build — the trick the hook already uses for the
  trampoline — would mutate a tree other agents run against. Staging
  touches nothing.
* `src/metaljax/__init__.py` imports the interpreter and the whole ops tree
  at package import, so shipping it over an excluded engine would make
  `import metaljax` an ImportError. The staged tree carries a generated
  minimal `__init__` holding `__version__` (parsed from the real file, so it
  cannot drift) and nothing else.
* Two new build-hook env hooks make that possible and also make the wheel
  honest: `METALJAX_NATIVE_DYLIB` packages a **prebuilt** plugin — the
  gated binary, byte-identical, instead of whatever bazel produces at
  packaging time (release rule 1) — and `METALJAX_VENDORED_MLX` locates the
  183 MB runtime outside the tree being packaged.
* `mlx` is dropped from the staged dependency list: no packaged module
  imports it, and the Metal runtime is inside the wheel.

Result: 12 files, 65 MB, no Stage 1 module, and the plugin dylib in the
wheel sha256-identical to the binary the battery measured.

## 6.4 The acceptance battery

Everything below is the **release binary** `frozen-vendor-d651add3`
(sha256 `d651add3…6abeb8`, 47.4 MB) — `plugin-native` at HEAD linked against
`libmlx_metaljax.dylib` — with the machine lock held per measured phase.
`plugin-native/` source is unchanged between these runs and the recorded
pre-vendoring ones, so **libmlx is the only variable**.

### Contracts and canaries

| item | result |
|---|---|
| `smoke_test` | pass (4 checkpoints) |
| `execute_test` | **pass** — see the note below |
| `ingest_test` | pass, 0 failed |
| `decline_census` | 35 of 35 programs lower |
| `coexist_test` (bench venv, gemma venv) | pass |
| `bazel test //...` | 1/1 pass |
| pip-mlx + private libmlx in ONE process | pass — `vmmap` shows all four images, both libraries compute |
| a python with **no mlx at all** | pass — `MetalDevice(id=0)`, matmul correct |
| `texmo_gate` (whole-model vs jax-CPU) | **106 ok, 0 decline, 0 FAIL, 0 error, of 106** |

*The `execute_test` note.* Its first run reported one failure, in the
benefit-gate milestone's new "earn rule" test. That result attests nothing:
the file was **edited at 11:36:02, inside the 11:35:35–11:36:17 run**, and
the message it printed exists in no version of the file since. Re-run on the
settled file (sha256 `3ff58598…`, modified 11:45:19) it **passes, rc=0** —
"all cases match the CPU backend". Contract suites are green; the moving
file, not the vendored library, produced the failure.

### `tests/test_command_buffer.py` — all 11, and the canaries invert

The diagnosis milestone could only run the four Python-engine cases (its
patched mlx lived in a 3.13 venv while `native/build`'s extension is a 3.14
artifact whose handshake refused the skew). §6.2 removed that, so **every
detector ran for the first time**:

| | result |
|---|---|
| 6 correctness tests (eager, compiled, engine-vs-op-by-op, native, **pipelined**) | **PASS** |
| 5 corruption canaries (kernel + byte, Python / native / pipelined) | **FAIL — which is the patch's signature** |

The canaries assert that *some* budget still corrupts; they exist so a clean
run cannot be mistaken for a fixed bug. On the vendored build none of them
can find one, across **28 budget settings** (kernel 50–2000, bytes 8–256 MB)
and three sync-point layouts:

> `no kernel budget in (400, 200, 100, 2000, 50, 1600) corrupts the init scan
> any more … Either MLX fixed the command-buffer split bug … or our lowering
> moved`

It is the first. The **eager kernel-budget face** (the face that gave maxtext
a wrong first-step loss) and the **compiled byte-budget face** are both gone,
as is the pipelined-vs-serial disagreement — one one-line patch, three faces.

`notes/data/mlx-cbuf-repro/repro_c.py`, 20 lines of pure MLX, under the lock:

| build | fresh processes × 20 evaluations | wrong |
|---|---|---|
| public pip wheel 0.32.0 (`1876795e…`) | 3 | **1/20 every time** — the 7 lands at `[0,0]` |
| vendored fork (`06da3cfd…`) | 1 | **0/20** |

That also closes the provenance gap `mlx-patch-diagnosis.md` §3.5 left open:
the packaged reproducer has now been run wheel-side and **does fail there**,
so it is safe to attach to ml-explore/mlx#4099.

### Performance — the fence fix costs nothing

The patch adds at most one `waitForFence` per encoder boundary, so it was
measured, not assumed. Native-only per Oleg's 2026-08-18 scope change; the
control is the same binary's own **recorded** native suite on the
pre-vendoring library (`today/recorded`, >1 = slower):

| comparison | geomean | median | within 1.2× | worst row |
|---|---:|---:|---|---|
| vs 0.11.5 gate native (`ebe56e71`, 14:23) | **0.9989** | 1.0020 | 105/106 | big01-b8l256 1.050 |
| vs p28 native (22:57) | **1.0026** | 1.0027 | 106/106 | mid12-b64l128 1.037 |
| *(control)* those two recorded runs against each other | 0.9963 | 0.9997 | 105/106 | mid11-b64l128 1.094 |

Both vendored comparisons sit **inside the run-to-run noise of two
pre-vendoring runs of the same binary**, and the control's worst row (1.094)
is worse than either vendored worst. No measurable cost, confirming the
diagnosis' timing arms.

### Model rows

| row | result | vs its cell |
|---|---|---|
| **15** qwix-int8 Qwen3-8B, native | **10/10 same first token** (12095 `" Paris"`), 0 collapses, decode `" Paris. The capital"`, 76 GB peak | ✗ WRONG OUTPUT → **✅ FIXED** |
| **19** maxtext train 0.6B | 462.2 ms/step, `loss` 87.0428237915039 and `loss_first` 228.39447021484375 **bit-identical to all eight prior runs** | inside the required 456–470 |
| **5** Qwen3-8B bf16 | 58.4 / 58.5 / 58.4 / 58.9 ms/tok | at its 58.5 cell |
| **7** gpt-oss-20b | 21.9 / 18.0 / 21.9 ms/tok | at its 21.9 cell |

### Rows 5 and 7 — the nondeterminism **survives the fence fix**

The open question (release-status: "accepted as a release-note item *if it
survives the fence fix*") is answered: it survives.

| row | draws | outcome | divergence index |
|---|---:|---|---|
| 5 Qwen3-8B | **6** | 5 identical, **1 diverges from all five** | 51 |
| 7 gpt-oss-20b | 3 | **every pair diverges** | 50 / 51 |

**Row 5's first three draws agreed** — and the gate document had already
recorded that a two-sample agreement on this exact row "was luck, not
determinism". Three more draws were run for that reason; the fourth broke
it. Stopping at three would have published a false retirement. The *rate*
changed, the property did not, and both rows keep their token-50/51
signature — consistent with §4 of the diagnosis: the fence drop is
command-buffer synchronisation, this is fused-attention emit ordering, and
they are different mechanisms.

### Wheel

One artifact now — the native-only release wheel (Oleg, 2026-08-18):
**65 MB** (67,580,186 B), sha256 `95d6f1f0…86087be`, twine check PASSED,
**12 files**, `Requires-Dist: jax, ml-dtypes, numpy` (no `mlx`).

| venv | result |
|---|---|
| 3.12 / 3.13 / 3.14, **no mlx installed** | plugin loads, `wheel_poc` passes, eigh + msl scan+grad smoke passes, `__version__` and dist both 0.11.5 |
| 3.13 **with** pip `mlx==0.32.0` | both libraries load and compute; plugin correct |
| purity, all four | **no Stage 1 module shipped, none imported** |

`DYLD_PRINT_LIBRARIES` settles the rpath question both ways: the no-mlx venv
maps **only** its own `site-packages/metaljax/lib/mlx/lib/libmlx_metaljax.dylib`
— never this machine's build tree — and the pip-mlx venv maps **all four**
images, the public pair and ours.

The native path's import purity is exact: a meta-path trace shows
`jax_plugins.metal.initialize()` only ever *consults* `find_spec("metaljax")`
(a top-level lookup, which does not execute the module). The `metaljax` entry
that appears in `sys.modules` during the purity probe is the probe's own
dotted `find_spec("metaljax.engine")` importing the parent package — not the
loader. So the wheel's generated `__init__` is a courtesy to `import
metaljax; metaljax.__version__`, not a load-bearing part of the plugin path.

### Row 1 — measured, unattributed to the vendoring, and named

Row 1 (gemma4-31B bf16) drew **258.3 ms/tok** in the battery against its
published **237.3** cell (+8.8 %). It did not survive first contact with the
standalone rule (CLAUDE.md item 12) — re-run alone after a hard settle it
read **256.8**, then **275.6** on a second standalone draw (both inside the
day's drift span), so it is not a sweep artefact — and the published cell was
measured on 2026-08-17 12:18, since when `plugin-native` source has moved
(P27, P28). "Vendored MLX" and "newer source" were therefore confounded.

Unconfounded with one source tree and two libraries, three arms back to back
in one hold (control plugin built with `METALJAX_MLX_DIR` at the public
wheel's mlx; `bazel-bin` restored to the vendored link afterwards so nothing
downstream inherits a control build):

| arm | libmlx | decode ms/tok |
|---|---|---:|
| B | public pip 0.32.0 (`1876795e…`) | 292.3 |
| **A** | **vendored fork** (`06da3cfd…`) | **291.0** |
| B2 | public pip 0.32.0 | 285.7 |

**The substitution is flat**: the vendored arm lands between the two public
arms. What the same three hours also show is that this row's absolute value
drifts hard with machine state — 256.8 → 292.3 within 90 minutes **on both
libraries**, against a historical span of 237–302 across campaigns.

So: row 1 **holds as a vendoring matter**, and the gap between today's
readings and the 237.3 cell is a separate, still-open question about that
cell's reproducibility on the current source — named here rather than
absorbed, per release rule 2. It is not evidence against the fence fix.
