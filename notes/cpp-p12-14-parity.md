# Stage 2 P12-P14: the parity tail, and the measurement

Follows [`cpp-p11-dtypes.md`](cpp-p11-dtypes.md).  The last six families of
P8's census — single-device collectives, effect tokens, host callbacks,
donation, the PJRT surface and shape polymorphism — and then the whole pinned
suite through the native stack, which is what the phase-2 ledger has been
counting towards since P8.

Everything lands in the forked runtime (`plugin-native/runtime/`), the plugin
and `execute_test.py`, plus the NATIVE branch of
`src/jax_plugins/metal/__init__.py` (the registrations jax needs before a
module can reach a backend at all — P9's shape, repeated for callbacks and
donation).  `native/` and the rest of `src/` are untouched.

| file | what |
|---|---|
| `plugin-native/runtime/host_callback.{h,cc}` (new, 263 lines) | the callback ABI: a C trampoline, buffers in and out, no Python anywhere |
| `plugin-native/metal/metal_lowering.cc` | the collectives, tokens, `metaljax_callback`, donated parameters, the layout guard |
| `plugin-native/metal/metal_c_pjrt.cc` | the dylib's second exported symbol, which installs the trampoline |
| `plugin-native/metal/metal_buffer.{h,cc}` | a settled VIEW is kept (buffer identity), `CopyToMemorySpace` |
| `plugin-native/metal/metal_client.{h,cc}` | `GetDefaultLayout`, compile-option validation |
| `plugin-native/metal/metal_executable.{h,cc}` | the donation contract, `GetCostAnalysis` |
| `plugin-native/metal/metal_stream.{h,cc}` | the submission lock, now recursive |
| `src/jax_plugins/metal/__init__.py` | the native callback registry + ctypes trampoline; `donation=True` |
| `plugin-native/execute_test.py` | 2 module rows, 9 contract rows |

## 1. Single-device collectives (P12)

`src/metaljax/ops/collectives.py`, op for op: this plugin has ONE device, so
every replica group has size 1 and each collective degenerates — a reduction
over one replica is the value itself, a gather concatenates one shard, a
permute has nowhere to go, and `replica_id`/`partition_id` are zero.  That is
what makes `jax.pmap` and `shard_map` programs run.

Two things the Python handler does not do, and both are guards rather than
behaviour: a **group-size check** (`replica_groups`' last dimension) and a
**shape check** per operand/result pair.  Aliasing is how the identity arms are
lowered — the result IS the operand's slot, so no entry reaches the tape and
the aliasing taints ride along by construction — and aliasing a result whose
shape differs from its operand would hand back the wrong array instead of
declining.  `all_reduce` and `reduce_scatter` are handled AHEAD of the region
gate, since their reduction is a region that on one device has nothing to
reduce.

`collective_permute` is the one arm that is not an identity: an EMPTY pair list
means this replica receives nothing, which XLA fills with zeros.

## 2. Effect tokens (P12)

An ordered-effect token is not a ranked tensor, which is exactly what the
lowering declined on.  It is now what it is on the Stage 1 engine
(`src/metaljax/dtypes.py::token_value`): the **empty bool array**, in three
places — `CheckValue` accepts a `!stablehlo.token`, `Dims` answers `[0]`, and
main's parameter/result specs report `PRED[0]`.  That last one is not a
convention we chose: `dispatch.RuntimeTokenSet` hands the runtime
`np.zeros(0, np.bool_)` for a token argument and expects the same back.

`create_token` and `after_all` lower to the runtime's existing `kToken`
opcode.  Nothing else was needed — a token in a while carry or a cond branch is
an ordinary value once it has a shape, which is the whole reason the
bool[0] representation was chosen on the Python engine.

## 3. Host callbacks (P13)

`jax.debug.print`, `debug.callback`, `pure_callback` and `io_callback` lower on
platform `metal` to a `metaljax_callback` custom call whose `backend_config` is
an index into a registry of Python callables.  Stage 1 keeps that registry in
`metaljax.ops.callbacks` and the trampoline plugin simply calls it; the native
plugin holds no interpreter, so **the registry moved to where the lowerings are
registered** (`src/jax_plugins/metal/__init__.py`) and the dylib reaches it
through one C function pointer.

    dylib                                   python
    ----------------------------------      --------------------------------
    kHostCall entry (impure, never in       _NATIVE_CALLBACKS[index]
    an mx::compile trace)                        ^
      -> MakeHostCallback(index, specs)          |
      -> g_trampoline(index, ins, outs) --> ctypes CFUNCTYPE callback
         (host_callback.h's C ABI)               (acquires the GIL here,
                                                  and nowhere else)

`metaljax_native_set_callback_trampoline` is the dylib's second exported symbol
(after `GetPjrtApi`).  ctypes is what makes the arrangement safe: a `CFUNCTYPE`
callback acquires the GIL for the duration of the call and releases it after,
which is the same contract `native/bindings.cc` gives the Stage 1 extension —
and it means the runtime still names no Python symbol.

Details worth keeping:

* **The dtype codes are their own enum** (`MetaljaxHostDtype`), not the tape's:
  the tape's come from a runtime registry keyed by MLIR names and may be
  renumbered by any milestone that adds a type, where these cross a process
  boundary.  Appended to, never reordered.
* **Outputs are staged zero-filled** and handed to MLX as the array's own
  storage (`host_lapack.cc`'s `Out::Finish` arrangement), so a callable that
  writes nothing leaves zeros rather than whatever the allocator held.
* **An emulated element type declines**: the ABI carries VALUES, and an f8's
  value lives in a wider storage dtype, so a callback would see the storage
  where the Python engine hands the user `ml_dtypes`.
* **Errors cross as a message**: the trampoline writes into a 512-byte buffer
  and returns nonzero, which becomes a `std::runtime_error` and then a PJRT
  error naming the Python exception.
* With **no trampoline installed** (an older dylib, or ctypes unable to open
  it) the lowerings are not registered at all, so jax refuses at TRACE time
  with its own message instead of the plugin failing at execute.

Ordering is the tape's order, which is the block's order.  A callback in a loop
BODY makes the body impure, so the loop runs iteration by iteration and the
callbacks come out in loop order — which is what the `execute_test` row checks,
count included (the Stage 1 bug P8.5 found was a pipelined loop running its
condition's effect one time too many).

## 4. Donation (P13)

Two halves, as P9's registrations were.  jax only sets up input-output aliasing
for platforms in `mlir._platforms_with_donation`, so `metal` is appended there
(the native branch only); the module then carries `tf.aliasing_output` (an
aliased output) or `jax.buffer_donor` (a plain donor) on the donated arguments.

The plugin collects the promise: `Lowering::Run` records which parameters are
donated, and `RunOnce` deletes those buffers after a run that SUCCEEDED.  This
engine never writes into an input — the aliasing an output would need is
exactly what the copy rule refuses — so the buffer is simply released, which is
the whole of what jax observes: a reuse raises, and the memory goes back to
MLX's pool one execute earlier than the caller's own reference would have freed
it.  `ExecuteOptions::non_donatable_input_indices` is the caller taking the
promise back for one call, and it wins.

## 5. The PJRT surface (P13)

Five things the tests actually exercise, each small:

* **`unsafe_buffer_pointer` is the buffer's identity.**  `MetalBuffer::Settled`
  gathered a non-contiguous array afresh on every read and threw the copy
  away — so a value held as a broadcast VIEW (which every `jnp.ones` is) handed
  out a NEW address per call, and `testArrayCopy*` asserted on exactly that.
  The gathered array is now KEPT (`array_` is mutable; every path that reaches
  it holds the submission lock).  The values never change, only which storage
  holds them; from PJRT's point of view the buffer always had its own dense
  storage.
* **`GetDefaultLayout`** answers dense row-major, the only layout this backend
  has.  Left Unimplemented it turned three files red on an attribute read that
  has nothing to do with computing (`heap_profiler`, `profiler`, `random`).
* **`CopyToMemorySpace`** on the one memory space is a real copy (`mx::copy`;
  `contiguous`/`astype` short-circuit on an array that is already both).  jax
  asks for it whenever it wants a buffer that provably shares nothing with the
  source — `device_put(x, may_alias=False)`, and the donating `device_put`.
* **Compile options are validated** (`CompileOptions::ApplyAllOptionOverrides`)
  so an unknown `compiler_options` key is the caller's error, by name, instead
  of being accepted in silence.
* **`GetCostAnalysis`** reports the tape's own facts (entries, argument and
  result bytes, output copies, whether the whole tape compiles) rather than
  XLA's property names: there is no cost model here and no flop count to give,
  and jax documents the structure as arbitrary.

### the submission lock had to become recursive

A host callback runs the USER's Python in the middle of an execute, and
`RunOnce` holds `SubmissionLock` for the whole of it.  Nothing stops that
Python from touching a metal array — a `device_put`, another jitted call — and
a second `SubmissionLock()` on the same thread through a plain `std::mutex` is
a deadlock.  Measured, not reasoned about: a `pure_callback` whose body runs
`jax.jit(lambda a: a * 3)` on a metal array hangs the process against the
pre-fix dylib and returns the right answer against this one.

Same-thread re-entry is not the concurrency the lock exists to prevent (MLX
0.32's process-wide command-encoder map, P4): it is one submission after
another, which is what a single-threaded process does anyway.
`std::recursive_mutex`, and the `METALJAX_CONCURRENT_EXECUTE=1` escape hatch is
unchanged.

## 6. Shape polymorphism (P14): nothing to do

Re-measured: `shape_poly_test` is **2342 passed / 4 failed**, and those four
(`jnp_insert`, `jnp_nonzero`) are the SAME four Stage 1 fails on this suite —
the "assertions that a platform SHOULD have failed" class, where this backend
computes a case the harness expects to be refused.  P8's 17 became 4 through
P9-P11 (LAPACK, the constant decode, `reduce_precision`), and the remainder is
shared, not native-only.  The family closed without a line of code.

## The layout guard, and the one row it costs

`GetDefaultLayout` let two `layout_test` cases run further than before, and one
of them then failed: it asks for a COLUMN-major parameter
(`Layout(major_to_minor=(1,0))` at rank 2, `mhlo.layout_mode = "{0,1}"`), which
MLX's storage cannot be.  jax caught it by comparing what the executable
reports against what it asked for — an assertion naming neither the backend nor
the reason.

The lowering now reads `mhlo.layout_mode` on main's parameters and results and
declines anything but the default, by name.  The test still fails (this backend
really has one layout), but it fails loudly, and the check is a correctness
guard as much as a diagnostic: a path where jax did NOT compare would have
computed on row-major bytes the caller believes are transposed.

Net on that file: 1 failure before, 2 after — against 10 tests that
`GetDefaultLayout` unblocked in the other three files.

## Two traps met while measuring

1. **`METALJAX_PLUGIN_PATH` selected the plugin by FILENAME.**  A measurement
   pins a build by copying the dylib somewhere stable — and a copy called
   `p1214.dylib` fell through `_native_library_path`'s `p.name == _NATIVE_DYLIB`
   test into the Stage 1 branch, which registered the native dylib as the
   plugin but wired jax to the Stage 1 callback registry (pulling the Python
   engine into the process) and left the trampoline uninstalled.  Every
   callback program then declined with "a host callback with no trampoline
   installed" — a WRONG measurement, not an error.  The override now asks the
   file itself (`_is_native_dylib`: does it export the bridge symbol), so a
   copy under any name registers the same way the original does.
2. **The C API carries a cost property as a FLOAT and CHECKs it.**
   `PJRT_Executable_GetCostAnalysis` does
   `CHECK(std::holds_alternative<float>(...))`, so an `int64_t` in the returned
   map aborts the process inside jaxlib — `Fatal Python error: Aborted`, no
   exception, and in a suite run it would take the whole file's results with
   it.  Every property this plugin reports is a float.

## Validation

| | result |
|---|---|
| `plugin-native/execute_test.py` | **493 checks**, 0 failures (482 at P11) |
| `plugin-native/texmo_gate.py` | **106/106** ok (19 via sensitivity scaling), twice — before and after the final build, 0 decline, 0 FAIL |
| `plugin-native/decline_census.py` | **35 of 35 lower** (34 at P11; `debug_print` was the one left) |
| `plugin-native/smoke_test.py` | 4/4 checkpoints |
| `plugin-native/wheel_poc_test.py` | 4/4 from a fresh 3.13 venv with the native wheel, and a `jax.debug.print` through the wheel's own bridge |
| `bazel test //...` | PASSED (`//metal:runtime_gil_free_test`) |
| the pinned jax suite | **99.50 %**, 12 native-only failures — see THE measurement |
| `tests/` on the native plugin | 84 -> **71** failures |
| dylib | 165,997,624 -> **166,041,960 B** (+44,336, **+0.027 %**) |

### the family slice

The 17 files P8's reasons name for the six families, before and after on this
tree, one process per file, sequentially.

| file | before (pass/fail) | after | delta |
|---|---:|---:|---:|
| `pmap_test` | 147/60 | 207/**0** | −60 |
| `debugging_primitives_test` | 24/32 | 56/**0** | −32 |
| `api_test` | 701/30 | 724/**7** | −23 |
| `shard_map_test` | 51/15 | 65/**1** | −14 |
| `jaxpr_effects_test` | 48/15 | 62/**1** | −14 |
| `lax_numpy_test` | 3199/11 | 3209/**1** | −10 |
| `profiler_test` | 6/8 | 14/**0** | −8 |
| `export_test` | 111/11 | 115/**7** | −4 |
| `pjit_test` | 133/3 | 136/**0** | −3 |
| `checkify_test` | 97/4 | 99/**2** | −2 |
| `lax_control_flow_test` | 539/3 | 541/**1** | −2 |
| `heap_profiler_test` | 0/1 | 1/**0** | −1 |
| `random_test` | 237/1 | 238/**0** | −1 |
| `custom_linear_solve_test` | 13/1 | 14/**0** | −1 |
| `shape_poly_test` | 2342/4 | 2342/**4** | 0 |
| `python_callback_test` | (all skipped) | (all skipped) | 0 |
| `layout_test` | 7/1 | 6/**2** | **+1** |
| **TOTAL** | **7,655/200** | **7,829/26** | **−174** |

The "before" column is HEAD (0da49c1) and the "after" column is these files'
rows in THE measurement below, so the two halves of this milestone are counted
the same way.

Of the 26 that remain, **24 are the shared whitelist** — they fail on Stage 1
too, on this same suite: `test_double_donation`, `test_print_token_buffer_error`,
`test_inline_optimized_hlo`, the f64/autodidax rows, the multi-platform export
rows, `test_can_execute_python_callback`, the `checkify` assert-primitive pair,
the four shape-poly "should have failed" rows, and
`shard_map::test_custom_linear_solve_rep_rules`.  The two native-only ones are
`layout_test::test_in_layouts_jit_jnp_input` (above) and
`lax_numpy_test::testArrayExplicitDtypes` (`element type f64`, intentional).

## THE measurement

The whole pinned suite through the native plugin, **all 164 `*_test.py` files,
one process per file, strictly sequential** — exactly as P8 ran it (CLAUDE.md
item 20: parallel runs UNDER-report failures), same tree, same environment, and
the plugin pinned by copying the dylib and pointing `METALJAX_PLUGIN_PATH` at
it.

```
METALJAX_PLUGIN_PATH=<native dylib> JAX_PLATFORMS=metal,cpu JAX_ENABLE_X64=0 \
  .venv/bin/python scripts/run_jax_tests.py <out> --jobs 1 --tests jax-v0.11.0/tests
```

| | passed | failed | skipped | errors | pass rate | wall |
|---|---:|---:|---:|---:|---:|---:|
| native, P8 (no code changes yet) | 26,133 | 2,059 | 6,173 | 35 | 92.70 % | 23.1 min |
| **native, now (P12-P14)** | **28,057** | **142** | 6,158 | 35 | **99.50 %** | **25.4 min** |
| Stage 1, P8, same tree | 28,062 | 137 | 6,158 | 35 | 99.51 % | 47.4 min |
| Stage 1, 0.11.3 release artifact | 28,067 | 132 | — | — | 99.53 % | 45.1 min |

The 35 errors are identical to Stage 1's, file for file (33
`compilation_cache_test` + the two files `hypothesis` is missing for).

Set arithmetic on the failing ids, against Stage 1 on the same tree:

```
native fails      : 142
stage1 fails      : 137
shared            : 130   <- the whitelist
NATIVE-ONLY (gap) :  12
STAGE1-ONLY (up)  :   7
```

**Every one of the twelve is an intentional decline, by name.**  There is no
numeric row, no unexplained row, and no family left to port:

| # | reason | class | where |
|---:|---|---|---|
| 9 | `element type f64` | intentional (the f64 policy) | `x64_context` 6, `pickle` 1, `dtypes` 1, `lax_numpy` 1 |
| 1 | `complex scatter multiply without unique indices` | intentional (P10) | `lax_numpy_indexing::testStaticIndexing5` |
| 1 | `conv: complex with no spatial dimensions` | intentional (P7) | `lax_test::testConvGeneralDilated0D2` |
| 1 | `a parameter layout of {0,1}` | intentional (one layout) | `layout_test::test_in_layouts_jit_jnp_input` |

So the native-only gap is **1,918 -> 12** across P9-P14, and what is left is the
f64 policy (9 of 12) plus three declines this phase deliberately kept.

### the seven the native stack fixes

```
core_test::test_reference_cycles                       # no Python engine to hold refs
core_test::test_reference_cycles_jit
debugging_primitives_test::test_can_print_inside_while_loop_cond{0,1}
debugging_primitives_test::test_can_print_in_batched_while_cond{0,1}
lax_test::testScatter1                                 # the position-dependent race
```

The four `debug_print`-in-a-while-cond rows are the live Stage 1 bug P8 found
(M5a's pipelined loop runs an impure condition one extra time).  P8.5 fixed
`Program::reads_host` on both sides of the fork, so this is the native stack
demonstrating the fix on the rows that exposed it — the same four failed on
Stage 1 in the P8 baseline this table compares against.

### `tests/` through the native plugin

The Stage 1 suite with `METALJAX_PLUGIN_PATH` at the native dylib, the second
standing leg: **84 -> 71 failures**, and both files that moved are this
milestone's: `test_pjrt_surface` **10 -> 0** and `test_donation` **3 -> 0**
(`test_pjrt_surface` was also the FLAKY one, and it is now green because the
buffer-pointer fix is what it was flaking on).  What remains is the
recognizer-emit families, whose Python-side pack building the plugin has no
path to — `test_moe` 28, `test_qmm` 26, `test_qmm_mxfp4` 16 = 70 — plus
`test_engine_gc` 1, which imports `metaljax.engine` and measures the Python
engine's reclaim policy.  Those 71 are the performance phases' business, not
parity's.

### artifacts

| file | what |
|---|---|
| `notes/data/p12-14-native-failures.txt` | the 142 failing ids, whole pinned suite, native |
| `notes/data/p12-14-native-summary.csv` | per-file pass/fail/skip/seconds |
| `notes/data/p12-14-native-only.txt` | the 12 |
| `notes/data/p12-14-stage1-only.txt` | the 7 the native stack fixes |
| `notes/data/p12-14-family-before.txt`, `-after.txt` | the 17-file slice, HEAD vs now |

## Reviewer-scrutiny list

1. **The collectives are ALIASES, and the shape check is the only thing
   between that and a wrong answer.**  If a future StableHLO lets an
   `all_gather` keep its operand's shape while meaning something else, this
   lowering would hand back the operand.  The group-size and shape guards are
   both cheap and both load-bearing.
2. **`Entry::regrid`'s sibling problem, for tokens**: `Dims` answers `[0]` for
   a token, so a token that reached an op expecting a real tensor would be
   computed on as an empty bool array rather than declining.  Nothing in reach
   does (jax only threads tokens through `after_all`, control flow and the
   function boundary), and `LowerCallback` refuses one explicitly.
3. **The callback ABI's dtype codes are a wire contract** with
   `src/jax_plugins/metal/__init__.py`'s `_host_dtypes()`.  The two lists are
   in the same order and must stay that way; a reordering is silent
   reinterpretation of every callback operand.
4. **The trampoline must outlive the plugin.**  `_NATIVE_TRAMPOLINE` is a
   module global for that reason: a ctypes callback object that is garbage
   collected leaves the dylib holding a dangling function pointer, and the
   failure would be a crash inside a user's `debug.print`.
5. **Donation deletes the buffer even when the caller passed it twice** (one
   position donated, one not).  XLA raises on that; here the donated position
   wins and the other reference is invalidated.  `test_double_donation` is the
   test, it is on the shared whitelist (Stage 1 does not raise either), and the
   caller has broken the contract in any case.
6. **The recursive submission lock is a policy change, not a bug fix.**  It
   permits nested submission from one thread; it does NOT permit concurrent
   submission from several, which is what MLX 0.32's encoder map cannot take.
   A reviewer replacing it with a plain mutex will get a deadlock in a
   callback, and one who removes the lock will get P4's segfault.
7. **`GetCostAnalysis` reports metaljax-specific keys.**  Anything that expects
   XLA's `flops` / `bytes accessed` will find neither, deliberately; and every
   value must stay a FLOAT (the C API CHECKs the variant).
