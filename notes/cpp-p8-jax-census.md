# Stage 2 P8: the native plugin against the pinned jax suite

Follows [`cpp-p7-conv.md`](cpp-p7-conv.md). P7 left the decline census at 32 of 35
probes with "everything left is LAPACK-on-Accelerate or JAX's own". That census is
35 hand-written probes; **this milestone measures the real thing** — the whole
pinned `jax-v0.11.0` suite through the native plugin, against the same suite
through Stage 1, on the same tree, back to back.

No code was changed. The deliverable is the failure-delta census below and the
phase ordering it implies.

## Headline: no silent wrongness

**Every one of the 1,918 native-only failures is a LOUD failure** — a decline
naming its op, a JAX-side "no MLIR translation rule", an `INTERNAL` guard, or a
warning-turned-error. Grepping the reason strings for numeric-comparison
failures (`Mismatched elements`, `Not equal to tolerance`, ...) returns **one**
test, and it is a precision divergence with a mechanism, not a wrong answer:

```
jax-v0.11.0/tests/scipy_stats_test.py::LaxBackedScipyStatsTests::testCauchyIsf1
E   Not equal to tolerance rtol=0.0003, atol=1e-06        <- jtu._CompileAndCheck
E   Mismatch at index [0, 0, 2]: 0.43540000915527344 (ACTUAL) 0.4356412887573242 (DESIRED)
```

`_CompileAndCheck` compares the **jit'd** result against the **eager** result of
the same function on the same backend, so this is a self-consistency check, not a
comparison against CPU. Minimal repro (`cauchy.isf(q, loc, scale)` =
`tan(pi/2 - pi*q) * scale + loc`, the test's own inputs, captured by spying on
`_CompileAndCheck`):

| path | rel err vs scipy-f64 | jit-vs-eager |
|---|---:|---:|
| native plugin, compiled | **1.47e-06** | 1.78e-06 |
| native plugin, eager | 3.16e-07 | — |
| native plugin, `METALJAX_COMPILE=0` | 3.16e-07 | **0** |
| Stage 1 (compile on, default) | 3.16e-07 | **0** |
| jax-CPU | 1.61e-07 | — |

So: the native plugin's `mx::compile`d kernel for this program is ~5 ULP less
accurate than its own eager path, Stage 1's compiled kernel is bit-identical to
eager, and `METALJAX_COMPILE=0` removes the difference. In the test's unlucky
draw the final `tan(...)*scale + loc` cancels to ~0.435 out of terms of ~1e2, and
5 ULP of the product becomes 5.5e-4 of the answer — which is what broke the 3e-4
tolerance. The two stacks receive the same module (dumped and compared;
`METALJAX_DUMP_MODULE=1` vs Stage 1's, identical but for constant-hoist order), so
the difference is in what MLX does inside the fused kernel, and the shape of it is
P5's open finding ("MLX does not fuse the FIRST compiled executable in a process
— a few ULP on transcendental chains; Stage 1 does not show it"). It is NOT that
finding literally: four warm-up compiles do not move it.

Not fixed, per the mission. It is the only numeric row in 28k tests and the
values are within 1.5e-6 of CPU; the exposure is programs that cancel after a
transcendental. Worth an hour in a later phase to learn which rank-0 constant or
fast-math flag differs.

## The slice, and how to reproduce it

Both stacks ran **the whole pinned suite, all 164 `*_test.py` files, one process
per file, strictly sequential** (CLAUDE.md item 20: parallel runs UNDER-report
failures; the 0.11.3 artifact was a sequential run too). Same tree (`c69060b`),
same env, native first, then Stage 1 — never concurrently.

```
# native plugin
METALJAX_PLUGIN_PATH=$PWD/plugin-native/bazel-bin/metal/libmetal_pjrt_native.dylib \
JAX_PLATFORMS=metal,cpu JAX_ENABLE_X64=0 \
  .venv/bin/python scripts/run_jax_tests.py <out> --jobs 1 --tests jax-v0.11.0/tests

# Stage 1: the same command with METALJAX_PLUGIN_PATH unset
```

Plugin identity was verified before the runs (`plugin-native/smoke_test.py`:
`platform_version == metaljax-native-p0`, `metaljax.engine imported: False`), and
every native-only decline message below carries the `metaljax-native:` prefix, so
there is no doubt about which dylib served.

`hypothesis` is not installed (2 files error at collection on BOTH stacks);
`flatbuffers` is. Optional-dep state is therefore identical between the stacks and
matches the 0.11.3 artifact's conditions.

A **core slice** of 35 files (the mission's "breadth first" list — `lax_test`,
`lax_numpy_*`, `lax_control_flow`, `lax_vmap`, `random_*`, `nn`, `stax`, `image`,
`dtypes`, `fft`, `scipy_fft`, `lax_scipy_special_functions`, `scipy_ndimage`,
`linalg`, `debugging_primitives`, `python_callback`, `jaxpr_effects`, `api`,
`array`, `aot`, `jax_jit`, `layout`, `pjit`, `pmap`, `batching`, `core`) is
reported separately below; it is a subset of the same runs, so no extra GPU time
was spent on it.

## Totals

| | passed | failed | skipped | pass rate | wall |
|---|---:|---:|---:|---:|---:|
| core slice, native | 15,036 | 917 | 2,079 | 94.25 % | 8.9 min |
| core slice, Stage 1 | 15,945 | 23 | 2,064 | 99.86 % | 17.5 min |
| whole suite, native | 26,133 | 2,059 | 6,173 | **92.70 %** | **23.1 min** |
| whole suite, Stage 1 | 28,062 | 137 | 6,158 | **99.51 %** | **47.4 min** |

Stage 1's 137 is the release artifact's 132
(`notes/data/pinned-0.11.3-failures.txt`) plus five — see the Stage-1 regression
section below, which is this milestone's second finding. Zero whitelist entries
were fixed, so the whitelist is still the right reference for the shared set.

Set arithmetic on the failing test ids:

```
native fails      : 2051   (the counts say 2059: qdwh_test's short summary lists
                            38 of its 46 failures, an 8-id undercount in one file)
stage1 fails      :  137
shared            :  133   <- the known whitelist
NATIVE-ONLY (gap) : 1918
STAGE1-ONLY (up)  :    4
```

**Skips did not hide anything**: native skipped 15 MORE tests than Stage 1 over
the whole suite (`api_test` +12, `debugging_primitives_test` +3) — the gap is in
failures, not in silently-skipped families.

The native run is 2x faster in wall time, which is a measurement artifact, not a
perf result: a declined program fails in milliseconds where Stage 1 computes it.

## The gap table

Every one of the 1,918 native-only ids was re-run with `--tb=line` to capture its
reason string (`notes/data/p8-native-only-reasons.txt`, id + reason per line), then
bucketed. **The classification is exhaustive — 1,918 of 1,918 rows land in a
family**, and the three that first looked unclassified are `GetDefaultLayout is
not supported` (folded into PJRT surface).

| # | family | class | where it concentrates | phase |
|---:|---|---|---|---|
| **823** | LAPACK / host linalg — `eigh` 319 (JAX-side, see below), `triangular_solve` 234, `cholesky` 61, `eig` 32, `tridiagonal` 10, `perturb_singular` 18, plus 139 of the `custom_call with 2 results` declines (qr/lu/svd/orthogonal-initializer) | missing-lowering **+ no host-op path at all** | linalg 296, shape_poly 139, eigh 78, svd 65, qdwh 38, lobpcg 27, lax_scipy_sparse 26, custom_linear_solve 11 | **P9** (already planned: Accelerate) |
| **44** | ApproxTopK — the other `custom_call with 2 results` | missing-lowering (Stage 1 answers it with exact top-k) | ann 32, nn 7, random_lax 5 | P9 (same lowering site) |
| **329** | `scatter on complex` | missing-lowering (Python handler scatters by parts) | sparse_bcoo_bcsr 108, sparse 103, linalg 51, sparsify 24 | P10 |
| **158** | sort with a KEY chain / lexicographic comparator (`comparator ends in stablehlo.or`, `complex lexicographic`) | missing-lowering (needs the successive-stable-argsort shape, i.e. a `take_along_axis` entry) | sparsify 68, sparse_bcoo_bcsr 42, lax_numpy_setops 18, lax_numpy 10 | P10 |
| **126** | dtypes: `element type <unknown>` (extended/key dtypes) 111, `element type f64` 9, `no host transfer for S4` 6 | emulated-dtypes | dtypes 68, lax 30, lax_numpy 9 | P11 |
| **86** | single-device collectives (`all_reduce`, `all_gather`, `all_to_all`, `collective_permute`, `reduce_scatter`, `partition_id`, `replica_id`) | missing-lowering (identities; Stage 1 has them since 0.4.1) | pmap 60, shard_map 14, profiler 7 | P12 |
| **66** | `stablehlo.reduce_precision` | missing-**executor** (no `kReducePrecision` opcode in `native/` at all) | api 35, lax 11, lax_vmap 10, shape_poly 6 | P11 |
| **56** | `Too many inputs/outputs fused in the Metal Compiled primitive` | **robustness**: Stage 1 falls back to eager on an `mx::compile` failure, the plugin does not | scipy_stats 34, lax_scipy_special_functions 15, linalg 6 | **P9-parallel (bug)** |
| **55** | scatter tail: `scatter combiner apply` 29, `select_and_scatter` 21, `scatter on a rank-0 operand` 5 | missing-lowering/executor | lax_vmap 20, lax 10, lax_numpy_indexing 9 | P10 |
| **52** | `a constant whose raw data is the wrong size` | **BUG** in the native lowering's constant decode | lax_numpy_operators 32 (mixed f32/f16 `nextafter`, `ldexp`, ...), shape_poly 10, linalg 6 | **P9-parallel (bug)** |
| **26** | host callbacks — `debug_print` 25, `debug_callback` 1 (JAX-side: no rule registered for platform `metal`) | callbacks | debugging_primitives 25 | P13 |
| **23** | values that are not ranked tensors (effect tokens) | effects/tokens | jaxpr_effects 14, api 4, export 4 | P12 |
| **24** | PJRT surface: `GetDefaultLayout` 3, cross-memory-space copy 9, compile-options validation 5, `unsafe_buffer_pointer` identity 10 (`testArrayCopy*`), cost analysis 1 | PJRT-surface | lax_numpy 10, api 6 | P13 |
| **18** | `result 0 came back as [...], the module declares s8/s16/u8[...]` | **BUG** in the executor: a reduce over a small int type produces the wrong element type | lax_numpy 9, lax_numpy_reducers 7, lax_control_flow 2 | **P9-parallel (bug)** |
| **17** | shape polymorphism (`InconclusiveDimensionOperation`, `_DimExpr` as int, tracer-in-shape) | shape-poly | shape_poly 17 | P14 |
| **12** | donation: `Some donated buffers were not usable` (warning → error under the suite's filters) | PJRT-surface (donation unimplemented) | api 8, export 3, layout 1 | P13 |
| 1 | `conv: complex with no spatial dimensions` | intentional decline (P7) | lax 1 | — |
| 1 | `tridiagonal_solve` grad: **GPU command-buffer TIMEOUT** | robustness | linalg 1 | P9 |
| 1 | `testCauchyIsf1` numeric (above) | precision | scipy_stats 1 | investigate |

### Three bugs to fix regardless of phase order

1. **Small-int reduce result dtype** (18 tests). Repro:
   `jax.jit(lambda a: jnp.sum(a, dtype=jnp.int8))(jnp.array([1,2,3], jnp.int8))`
   → `INTERNAL: metaljax-native: jit__lambda result 0 came back as [], the module
   declares s8[]`. The result-shape guard in `MetalLoadedExecutable` catches it, so
   it is loud, but the dtype the reduce produces is not the declared one (and the
   printed "came back as `[]`" means `ShapeString` could not even name it).
   `jnp.cumsum(..., dtype=int8)` on the same array is fine.
2. **Constant decode: "raw data is the wrong size"** (52 tests). Concentrated in
   `lax_numpy_operators_test` mixed-width ops (`nextafter_float32_float16` and
   friends), so the suspicion is a sub-32-bit or mixed-dtype `DenseElementsAttr`
   whose splat/raw-data length the lowering computes from the wrong element width.
3. **No eager fallback when `mx::compile` refuses** (56 tests). MLX's
   "Too many inputs/outputs fused ... exhausted the available argument buffers"
   is a limit on big fused graphs. Stage 1 has a handler for exactly this —
   `src/metaljax/engine.py`'s `except (RuntimeError, IndexError, ValueError)`
   arm, whose comment names "fused-kernel argument-buffer exhaustion", clears
   `_can_compile` and re-runs the pure program through the interpreter — and the
   plugin has none, so the same graph becomes an `INTERNAL` error. This is the
   robustness hole in the table a real model could hit (today it is 34
   `scipy_stats` distributions and 15 special functions). The C++ side already
   owns the machinery: `Program::drop_compiled` plus the recovery ladder in
   `native/program.cc`; what is missing is the plugin catching the throw.

## Upside: what the native stack fixes

Four tests fail on Stage 1 and pass natively; three are structural and one is a
Stage-1 flake:

```
core_test.py::CoreTest::test_reference_cycles          # no Python engine to hold refs
core_test.py::CoreTest::test_reference_cycles_jit
export_back_compat_test.py::CompatTest::test_custom_call_coverage
lax_test.py::LaxTest::testScatter1                     # position-dependent, see below
```

The reference-cycle pair is the interpreter's absence showing up as a
user-visible property, and `test_custom_call_coverage` passes because the plugin
does not register Stage 1's metal-platform custom-call lowerings.

`testScatter1` is **not** an upside: on Stage 1 it passes 3/3 standalone and
passes in a `-k Scatter` run of its own file (82 tests), and only fails inside the
full 2,573-test file run. That is the position-dependent class the tracked-open
sparse pair belongs to, and it is a Stage-1-only symptom (the native run passes it
in-file).

## Second finding: five Stage-1 regressions since the 0.11.3 release

The Stage-1 leg is also the first re-run of the pinned suite on this tree, and it
found five failures the release artifact does not have (and zero fixes):

```
debugging_primitives_test.py::DebugPrintControlFlowTest::test_can_print_inside_while_loop_cond0
                                                     ::test_can_print_inside_while_loop_cond1
                                                     ::test_can_print_in_batched_while_cond0
                                                     ::test_can_print_in_batched_while_cond1
lax_test.py::LaxTest::testScatter1                   # the position-dependent one above
```

**The four are M5a's while pipelining.** `METALJAX_WHILE_PIPELINE=0` makes them
pass, and so does `METALJAX_ENGINE=python`; the shipped default fails them. The
mechanism is exactly what the milestone note describes: the pipelined loop builds
iteration t+1's condition BEFORE reading t's, so a `debug_print` *inside the
condition* runs one extra time and the captured output has an extra line. Only
prints in a cond are affected: every other `debugging_primitives_test` case
(prints in bodies, in `cond`, in `switch`, unrolled, under remat) still passes on
Stage 1. (On the native plugin all 25 of them fail earlier, at trace time — no
`debug_print` rule is registered for platform `metal` — so the native run cannot
see this bug at all.)

This is a live correctness-of-effects bug in the shipped Stage 1 engine, not a
phase-2 issue. It is cheap to fix in the direction the pipeline already allows
(speculate on building the body only when the cond has no effects), and it is
worth doing before the next release regardless of where the migration stands.

## Recommended phase ordering (by measured test count)

1. **P9 — LAPACK on Accelerate: 823 tests** (+44 ApproxTopK riding on the same
   `custom_call` lowering site), the largest family by a factor of 2.5, and the one
   already scheduled. Note the shape: 319 of them do not even
   reach the plugin — they die at trace time with *"MLIR translation rule for
   primitive 'eigh' not found for platform metal"*, because Stage 1 registers those
   lowerings from `src/metaljax` and the plugin registers nothing. **P9 is two
   jobs**: the JAX-side lowering registrations (which need a home that is not the
   Python package) and the host-op execution path (`kHostCall` exists in the
   runtime; the plugin has no way to reach it). Fixing only the second leaves 319
   tests failing.
2. **P9-parallel — the three bugs above: 126 tests**, none of which is a family
   port; each is a defect with a one-line repro.
3. **P10 — the scatter/sort tail: 542 tests** (complex scatter 329, key/lexicographic
   sort 158, scatter combiner + select_and_scatter + rank-0 scatter 55). All three
   have Python executors to transliterate, all three were already on the phase-2
   decline disposition list, and they unblock the entire `sparse` family
   (`sparse_bcoo_bcsr` 155, `sparse` 104, `sparsify` 92 = 351 tests in three files).
4. **P11 — dtypes + reduce_precision: 192 tests**. `reduce_precision` (66) is a new
   op family in the runtime, cheap. The extended/key dtypes (111) are the "element
   type `<unknown>`" wall, and the sub-byte grids were deliberately declined in M5c
   — the phase-2 disposition says PORT.
5. **P12 — collectives + effect tokens: 109 tests**. Single-device identities plus
   `partition_id`/`replica_id`; the tokens are the `bool[0]` aval shape Stage 1
   already uses.
6. **P13 — callbacks, PJRT surface, donation: 62 tests**. The C-trampoline question
   for `debug_print` (26), `GetDefaultLayout`/memory-space copies/compile-option
   validation/buffer identity (24), donation (12).
7. **P14 — shape polymorphism: 17 tests**, the tail.

Closing 1-4 removes 1,727 of the 1,918 native-only failures and takes the native
plugin from 92.70 % to an estimated **98.8 %** (332 failures left of 28,192),
i.e. to within ~190 tests of Stage 1's 137.

## Artifacts

| file | what |
|---|---|
| `notes/data/p8-native-failures.txt` | 2,051 failing ids, native plugin, whole pinned suite |
| `notes/data/p8-stage1-failures.txt` | 137 failing ids, Stage 1, same suite same tree |
| `notes/data/p8-native-only-reasons.txt` | the 1,918-row gap list, `id<TAB>reason` |
| `notes/data/p8-stage1-only.txt` | the 4 upside tests |
| `notes/data/p8-native-summary.csv`, `p8-stage1-summary.csv` | per-file pass/fail/skip/seconds, both stacks |

## Traps met while measuring

1. **`_CompileAndCheck` is a self-consistency check, not a CPU comparison.** The one
   numeric row would have been misread as "native disagrees with CPU" if the
   tolerance line (`rtol=0.0003`) had not been traced back to which assert it came
   from.
2. **`--tb=no` truncates the reason to ~60 characters**, and a census needs the op
   name. The runs use `--tb=no` for speed; the reason capture is a second, cheap
   pass over the native-only ids with `--tb=line`.
3. **Pairing reasons to ids by ORDER works only if both lists come from the same
   re-run.** `IndexedUpdateTest::testStaticIndexing5` and
   `IndexingTest::testStaticIndexing5` are different tests in the same file — the
   first is a complex `.at[].set`, the second passes — so a class-blind id match
   would have produced a wrong bucket.
4. **A file whose whole run is skipped reports no counts** in the harness's regex
   (`182 skipped in 0.29s` has no "passed"/"failed"), which reads as zeros.
   `python_callback_test.py` is that file, on both stacks equally.
