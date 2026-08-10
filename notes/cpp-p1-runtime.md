# Stage 2 P1: the executor runtime as a Python-free bazel library

Follows `notes/pjrt-native-p0.md` (which proved the `xla::PjRtClient` route and
recommended sharing `native/` with the bazel workspace rather than forking it).
P1 does three things:

* **A.** evicts Python from the core of `native/` — `program.h` and every
  translation unit under it;
* **B.** builds those sources as a bazel `cc_library` and links it into
  `//metal:libmetal_pjrt_native.dylib`;
* **C.** proves the runtime executes on the GPU from a process with no
  interpreter in it.

All three landed. Numbers, the shape of the split, and the traps are below.

## A. de-Pythonising the core

Five touch points, no more — the runtime was already close.

| was | is |
|---|---|
| `program.h` included `<nanobind/nanobind.h>` + four stl casters | includes `<mlx/mlx.h>` and the standard library |
| `Entry::host` was `nb::object` | `HostFn = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>` |
| `Program::run` did `nb::gil_scoped_release` around the walk | the binding releases the GIL; `run` never mentions it |
| `host.cc` did `nb::module_::import_("gc").attr("collect")()` | `g_gc_hook`, a `std::function<void()>` the adapter sets |
| `op_histogram` / `opcodes` / `dtype_codes` / `stats` returned `nb::dict` | core returns `std::map` / `std::vector<std::pair<std::string,int>>` / the `Stats` struct |

`config.cc` mixed the two halves and was split: it keeps the registry, the
`configure` cadences, the counters and the new hook (158 lines, no Python), and
the nanobind registration moved to a new **`native/bindings.cc`** (191 lines).
So the boundary is now exactly two files — `bindings.cc` and
`metaljax_native.cc` — and `native/build.sh` names them first in its source
list for that reason.

**The Python-visible API is byte-identical.** No file under `src/` or `tests/`
was touched. `bindings.cc` absorbs every change:

* `Program.add` is a lambda that takes the same eleven keyword arguments with
  the same defaults and calls `wrap_host()` on the last one;
* `Program.run` is a lambda that releases the GIL around `Program::run` —
  arguments are cast before the release and results after it, in nanobind's own
  code, which is where they were cast before;
* `opcodes` / `dtype_codes` / `stats` / `op_histogram` build their dicts here.

### Things worth knowing if you touch this again

* **`wrap_host` acquires the GIL inside the wrapper**, and formats a
  `nb::python_error` into a `std::runtime_error` *there* — that discipline is
  unchanged, only relocated. It has to stay inside: the recovery ladder in
  `program.cc` calls `what()` on whatever it catches, from a stack that has
  released the GIL, and nanobind formats a `python_error`'s message through the
  Python C API.
* **`g_gc_hook` is set to a *captureless* lambda.** A global `std::function`
  holding an `nb::object` would decref it at process teardown, after the
  interpreter is gone. The lambda imports `gc` on each call (Python caches it in
  `sys.modules`) and swallows exceptions *while holding the GIL*, so the
  exception object dies where its destructor is legal. `host.cc`'s `gc_collect`
  keeps an outer `catch (...)` for a non-Python embedder's hook.
* **Naming collision, and it is a compile error, not a silent one.** With
  `using namespace metaljax;` inside `register_tape`, an adapter function named
  `opcodes()` is ambiguous against `metaljax::opcodes()`. The adapter's are
  `py_opcodes` / `py_dtype_codes` / `py_stats`.
* The comment on `Program::lock_` changed meaning: the lock used to be
  documented as "taken INSIDE the GIL release". It still must be, but the
  release is now the caller's — so the header states the *contract* (an
  embedder holding an interpreter lock drops it before calling `run`) rather
  than describing what `run` does.

### Verification

* `bash native/build.sh` — clean (only MLX's own C++20/deprecated-copy
  warnings, as before). The repo venv is the only one `build.sh` documents;
  `native/build/` also holds a stale cp313 artifact from an older session that
  nothing rebuilds.
* Every core TU compiles **with no `-I` for nanobind or CPython at all**:

  ```sh
  for f in program.cc config.cc dtypes.cc runtime.cc compile.cc \
           ops_*.cc emits.cc control.cc msl.cc host.cc; do
    clang++ -std=c++17 -fsyntax-only -I "$SP/mlx/include" -DNDEBUG "$f"
  done
  ```

  15/15 OK. This is the check to re-run before believing a future refactor —
  it fails loudly the moment someone reaches for `nb::` in a handler.
* `pytest tests/ -q`: **1258 passed**, identical to HEAD (1258).
* `METALJAX_ENGINE=native JAX_PLATFORMS=metal,cpu scripts/texmo_check.py`:
  **106 ok, 0 FAIL, 0 error** (tol 0.002).

## B. the runtime as a bazel library

Route **(a)** from the P0 note, as recommended — `new_local_repository` over
`../native` with a checked-in build file. It works, including in Bazel 8's
WORKSPACE mode, with no `load()` (the native repo rule is still there).

```python
# plugin-native/WORKSPACE
new_local_repository(
    name = "metaljax_runtime",
    build_file = "//third_party/metaljax_runtime:BUILD.runtime",
    path = "/Users/oleg/metaljax/native",
)
```

`plugin-native/third_party/metaljax_runtime/BUILD.runtime` (59 lines) declares
one `cc_library` with **15 `srcs`** and 2 `hdrs`, `deps = ["@mlx"]`. Notes on
it:

* `srcs` is spelled out, never globbed. A glob would sweep in `bindings.cc` and
  `metaljax_native.cc` (the whole point is that they are absent) and it would
  sweep in `native/build/`, the extension's output directory, which lives
  inside the source tree.
* `copts` are only `-Wno-deprecated-copy -Wno-c++20-extensions`, both for MLX's
  own headers. **No `-std=` needed**: `xla/tensorflow.bazelrc` already sets
  `common:macos --cxxopt=-std=c++17`.
* `alwayslink = 1`. The plugin does not call the tape yet, so without it the
  linker would drop every object and the "does it link" claim would be empty.
* Nothing nanobind-adjacent is needed — no robin_map, no `NB_DOMAIN`. Those are
  `build.sh`'s, for the two boundary TUs only.
* A dependent gets `#include "program.h"` working through bazel's automatic
  `-iquote external/metaljax_runtime`; no `includes = ["."]` required.

**The shared route really is shared, and I checked rather than assumed:**
appending `static_assert(false, ...)` to `native/runtime.cc` fails the *bazel*
build immediately (`external/metaljax_runtime/runtime.cc:138:15`), because the
repo is symlinks onto the live tree. Adding a *new* file works too, as long as
you list it in `BUILD.runtime` — editing that file is itself what triggers the
repo refetch, so the two steps are one step. There is no sync command to
remember.

Linked into the plugin:

| | before | after |
|---|---:|---:|
| `libmetal_pjrt_native.dylib` | 165,127,736 B | 165,481,672 B (+353,936, **+0.21 %**) |
| exported symbols | 155,034 | 155,116 |
| `otool -L` | libc++, libSystem, CoreFoundation, Foundation, libobjc, `@rpath/libmlx.dylib` | unchanged |
| native wheel | 41,736,182 B | 41,880,421 B |

**No linker friction whatsoever** — the thing this step existed to find out.
No duplicate symbols, no `-force_load` games, no interposition between the
tape, `libmlx.dylib` and XLA's statically-linked LLVM/MLIR/absl. That is
consistent with P0's finding that libmlx exports zero `absl`/`llvm` symbols.

Build times: the runtime library from cold is **4.1 s** (17 actions); relinking
the dylib after adding it, **2.7 s**; the native wheel still rebuilds in
seconds.

Re-verified after linking: `plugin-native/smoke_test.py` — all checkpoints
pass; native wheel installed into a **fresh 3.13 venv**, `wheel_poc_test.py` —
all four checkpoints pass, `metaljax.engine` still absent from `sys.modules`,
and `src/metaljax/lib/` back to holding only the trampoline dylib afterwards.

## C. the GIL-free execute proof

`//metal:runtime_gil_free_test` (`plugin-native/metal/runtime_gil_free_test.cc`,
207 lines) — a `cc_test` that depends on `@metaljax_runtime//:runtime` and
`@mlx` and on *nothing else*: no XLA, no plugin, because the claim is that the
tape stands on its own. `bazel test //...` runs it (0.3 s).

It builds a four-entry tape through the C++ API — `stablehlo.multiply`,
`stablehlo.add`, a `stablehlo.constant` payload the Program owns for its
lifetime, and a `stablehlo.reduce` with the attribute vector `[0, 2, 0, 1]`
(sum over both dims) — computing `a*b + a` and its sum over f32[2,3], and:

* looks the opcodes up **by name through `metaljax::opcodes()`**, the same
  lookup `tape.py` does, minus the dict;
* runs it interpreted and checks every element and the scalar exactly;
* runs a second copy with `set_compile(true, {}, 1)` twice, and asserts through
  `g_stats` that exactly one `mx::compile` trace was built and both calls went
  through it, and that `compiled_dropped()` is false. **That is the load-bearing
  half**: `compile.cc` keys MLX's compile cache by an engine-owned id, where
  MLX's own convention is the address of a *Python function object*.
* asserts the process is Python-free three ways: no CPython/nanobind image in
  dyld's list, `dlsym(RTLD_DEFAULT, ...)` finds none of `Py_Initialize` /
  `Py_IsInitialized` / `PyGILState_Ensure`, and `g_gc_hook` is empty.

```
metaljax runtime, no interpreter in the process
  device: gpu
  ok   MLX's default device is the GPU
  ok   no Python/nanobind image is loaded
  ok   CPython's entry points are not resolvable
  ok   the gc hook is empty (nothing to collect without an interpreter)
  ok   tape holds four entries
  ok   liveness pruning caps the environment at five
  ok   interpreted: shapes and dtypes / a*b+a elementwise / sum of all six (66.5)
  ok   the program says it may compile
  ok   compiled (first call: traced) ... (second call: replayed) ...
  ok   the compiled path was never retired
  ok   exactly one mx::compile trace was built
  ok   both calls went through the compiled graph
native runtime executed GIL-free: ok
```

### Traps this test hit

1. **`tags = ["local"]` is required.** The test talks to the Metal device and
   MLX writes compiled kernels into the user's cache directory; neither
   survives bazel's darwin sandbox.
2. **Do not substring-match image *paths* for "python".** `libmlx.dylib` is
   loaded out of the build venv's `site-packages` (that is where the wheel keeps
   it, and what the `@mlx` rpath points at), so the first version of the check
   flagged `.../python3.14/site-packages/mlx/lib/libmlx.dylib` and proved
   nothing. Match the image's **basename** (`libpython*`, `Python`,
   `*cpython-*`, `*nanobind*`) plus `Python.framework` anywhere in the path.

## What a reviewer should look hardest at

* **`program.h`'s `Entry::host` and `Program::add` signature.** Everything else
  in the header is comment movement; those two are the ABI of the tape.
* **`Program::run`'s comment block.** The GIL release left the function; what
  replaced it is a *contract* on the caller, and the deadlock it prevents (a
  waiter on `lock_` holding the GIL vs. a holder recovering through
  `g_gc_hook`) is real and untested by anything automatic.
* **`bindings.cc`'s `Program.add` lambda** — eleven arguments retyped by hand.
  A wrong order there would be a silent mis-lowering, not a compile error, for
  the pairs of adjacent same-typed parameters (`operands`/`results`/`drops` are
  all `std::vector<int>`). The suite covers it (`tests/test_native_tape.py`
  calls `add` with keywords), which is why the 1258 count matters more than
  usual here.

## Not done, and deliberately

* The plugin still does not *call* the runtime — `CompileAndLoad` returns
  `Unimplemented`. Linking it in ahead of that is what makes any symbol
  collision a build failure now rather than a mystery in P2.
* `native/build.sh` still compiles all seventeen TUs in one clang invocation,
  including the two boundary ones. Two build systems over one source tree is
  the cost the P0 note already flagged; the `srcs` list in `BUILD.runtime` and
  the source list in `build.sh` are the two places that must agree, and each
  says so in a comment pointing at the other.
* `new_local_repository` hardcodes `/Users/oleg/metaljax/native`, exactly as the
  XLA `local_repository` hardcodes its path. Same fix when it matters
  (`--override_repository`).

---

**Followed by P2** ([`cpp-p2-lowering.md`](cpp-p2-lowering.md)): the plugin now
*calls* this runtime. `CompileAndLoad` lowers the parsed StableHLO into a
`Program` and `MetalLoadedExecutable::Execute` replays it, so the "not done,
and deliberately" item above is closed. Nothing under `native/` had to change
for it — `program.h` was the whole interface, as intended.
