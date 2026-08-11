# Stage 2 P9: the linalg family, on Accelerate

Follows [`cpp-p85-fixes.md`](cpp-p85-fixes.md).  P8's census put this family at
the top of the ladder by a factor of 2.5 — **823 tests**, plus the 44
ApproxTopK rows that ride on the same `custom_call` lowering site — and said
the shape it would have to take:

> P9 is two jobs: the JAX-side lowering registrations (which need a home that
> is not the Python package) and the host-op execution path (`kHostCall` exists
> in the runtime; the plugin has no way to reach it).  Fixing only the second
> leaves 319 tests failing.

Both are here.  It is also the first milestone that lands in the FORKED
runtime: `plugin-native/runtime/` is where the executor changes go now, and
`native/` is untouched.

## The three parts

| file | lines | what |
|---|---:|---|
| `plugin-native/runtime/host_lapack.h` | **+81** (new) | the line between the halves: which factorization, what result types, which flags |
| `plugin-native/runtime/host_lapack.cc` | **+1,158** (new) | `src/metaljax/ops/lapack.py` on Accelerate's LAPACK — twelve factorizations, batched, both element types |
| `plugin-native/runtime/BUILD` | +13 | `-framework Accelerate`, `-DACCELERATE_NEW_LAPACK` |
| `plugin-native/runtime/program.h`, `config.cc`, `ops_index.cc` | +40 | `kApproxTopK`: XLA's approximate top-k, answered exactly |
| `plugin-native/metal/metal_lowering.cc` | +290/-9 | `LowerCustomCall` / `LowerHostLinalg` / `LowerApproxTopK`, the `HostFn` on the tape entry, and the purity arm |
| `src/jax_plugins/metal/__init__.py` | +22/-8 | the native branch registers the linalg lowerings (callbacks and donation stay behind) |
| `execute_test.py` | +355/-8 | 70 linalg cases, a new decline, the two half-precision contracts |
| `smoke_test.py`, `wheel_poc_test.py` | +36/-13 | cholesky COMPUTES at both checkpoints now; `reduce_precision` is the stand-in decline |

## Half 1: what jax lowers on platform `metal`, measured

The mission's instruction was to discover the target set empirically rather
than from Stage 1's table, and the two do not agree.  Lowering every linalg
entry point through the native plugin with no rules registered gives:

| jax API | what reaches the plugin |
|---|---|
| `jnp.linalg.cholesky`, `jax.scipy.linalg.cholesky` | `stablehlo.cholesky` |
| `solve_triangular`, `solve`, `inv`, and every QR/LU VJP | `stablehlo.triangular_solve` |
| `jnp.linalg.qr` (and `lstsq`, `jax.nn.initializers.orthogonal`) | `custom_call @Qr` + `@ProductOfElementaryHouseholderReflectors` |
| `jax.lax.approx_max_k` / `approx_min_k` | `custom_call @ApproxTopK` |
| `lu_factor`, `det`, `slogdet`, `tridiagonal_solve` | **nothing** — jax's generic device lowerings serve them |
| `eigh`, `eigvalsh`, `svd`, `eig`, `schur`, `hessenberg`, `tridiagonal` | **no rule at all**: `NotImplementedError: MLIR translation rule for primitive 'eigh' not found for platform metal`, at TRACE time |

That last row is P8's 319, and it is why `pinv`, `matrix_rank`, `cond`,
`norm(x, 2)`, the whole of `svd_test`/`eigh_test`/`lobpcg`/`qdwh` and a third
of `linalg_test` never even reached the plugin.  The `lapack_*_ffi` targets
Stage 1's table also carries are jax's **CPU** lowerings; they are served here
too (their result conventions differ — an FFI eigh hands back a trailing
`info`) but nothing on this platform emits them outside a deserialized
artifact.

`_initialize_native` now calls the same `_register_linalg_lowerings` the
trampoline branch calls, with two things switched off:

* **callbacks** — `_register_callback_lowerings` stashes the user's Python
  callable in `metaljax.ops.callbacks`, which is the Stage 1 interpreter this
  plugin exists to replace.  P13's job.
* **donation** — `mlir._platforms_with_donation.append("metal")` makes jax
  emit input/output aliasing the plugin does not honour.  Also P13's.

The trampoline branch is unchanged (`_register_linalg_lowerings()` with both
defaults true).

## Half 2: the Accelerate handlers

`host_lapack.h` is the whole interface.  The PLUGIN reads the IR — which
target, which flags, what result types — and asks `MakeHostLinalg` for a
`HostFn`; the implementation knows only arrays.  So the executor's side of the
boundary never sees an attribute, and the lowering never sees a matrix.

Twelve factorizations, each a transliteration of the handler in
`src/metaljax/ops/lapack.py`:

| kind | routine | notes |
|---|---|---|
| cholesky | `potrf` | the untouched triangle is zeroed (numpy does it too); a failed factorization is all NaN, not an error |
| qr | `geqrf` | results are the packed form and the reflector scalars |
| orgqr | `orgqr` / `ungqr` | the **zero-tau completion**: `full_matrices` wants more columns of Q than there are reflectors, and a zero tau is an identity reflector |
| eigh | `syevd` / `heevd` | `metaljax_eigh` guards a non-finite operand with NaN; the FFI form zero-fills its trailing `info` |
| svd | `gesdd` | `jobz` picked from the declared U/Vt widths, exactly as the Python picks `full_matrices` |
| eig | `geev` | the real arm unpacks LAPACK's conjugate-pair columns into complex vectors |
| lu | `getrf` | pivots converted from LAPACK's 1-based to jax's 0-based, plus the permutation sweep |
| schur | `gees` | real Schur form for a real operand, complex otherwise |
| hessenberg | `gehrd` | `ilo = 1, ihi = n` |
| tridiagonal | `sytrd` / `hetrd` | |
| triangular solve | `cblas_?trsm` | ...unless a pivot is exactly zero |
| tridiagonal solve | `getrf` + `getrs` | ...falling back to the Thomas sweep |

Three semantics carried over verbatim because they are XLA's rather than
LAPACK's, and a test notices each:

* **A singular triangular solve divides THROUGH the zero pivot** to ±inf/nan
  instead of failing.  `jnp.linalg.det`'s JVP depends on it (it runs the solve
  unconditionally and filters the non-finite results with a `where`).  LAPACK's
  `trtrs` refuses such a system and BLAS's `trsm` is free to scale by a
  reciprocal, so the singular case runs a substitution written here — which is
  exactly where the Python handler falls out of scipy.
* **A non-positive-definite cholesky is all NaN.**
* **`perturb_singular`** nudges a tiny diagonal so the solve stays finite.

...and one that is this backend's own: **halves compute in f32 and cast back**
(`_np_in`).  That is the property CLAUDE.md item 16 calls "bf16/f16 linalg
EXCEEDS CPU", and it survives the port — `jnp.linalg.eigh` of a bf16 matrix
runs here and raises on jax-CPU, whose LAPACK tables have no half entry.

### LP64, not ILP64

Accelerate ships both: the default interface takes 32-bit integers
(`__LAPACK_int` is `int`), and `ACCELERATE_LAPACK_ILP64` selects the 64-bit one
under `$ILP64`-suffixed symbols.  This file takes **LP64**, with
`ACCELERATE_NEW_LAPACK` for the modern (LAPACK 3.9.1) prototypes rather than
the frozen 3.2.1 legacy set.

The reason is that ILP64 buys exactly one thing — matrix dimensions past 2^31 —
which no program that reaches here can have: a single f32 matrix of that order
is 17 EB of operand.  Every dimension is checked on the way in (`Fit`), so an
impossible one is a loud throw and never a truncated `int`, and the batch
dimensions — which really can be large — are loop counters here and never cross
into LAPACK.  Taking ILP64 would also mean an `int64_t` pivot array where jax
declares `s32`, i.e. a conversion pass over every result that LP64 gets for
free.

### Layout

LAPACK is column-major and everything above this line is row-major, so every
matrix crosses through `ToCol`/`ToRow`.  Explicitly, rather than by flipping
`uplo`/`trans` flags to make the transpose free: the flag tricks are correct
only per routine (an `uplo` flip is right for `syevd` and wrong for `gesdd`),
and the cost is a memcpy-sized transpose of a matrix about to be factorized in
O(n^3).

One place where the flags DO carry it: `eigh` needs no explicit
symmetrization.  After `ToCol` the column-major buffer *is* the operand, so
`uplo = "L"` reads the operand's lower triangle — which is precisely what
`np.tril(x) + np.tril(x, -1).conj().T` built for numpy.

### One deliberate divergence from Stage 1

`metaljax_eig` with `compute_left_eigenvectors=True`: the Python handler could
not ask numpy for left eigenvectors, so it took an eigendecomposition of the
adjoint and matched its columns to `conj(w)` by nearest eigenvalue.  `geev`
computes them directly under `jobvl`, in the same order as `w` and with the
same normalization jax's CPU backend hands back.  Same vectors, less machinery,
closer to the reference.

## ApproxTopK

The 44 rows the census bracketed with this family are not a host call at all:
`@ApproxTopK` is ordinary device work, and an EXACT top-k satisfies its
contract (recall 1.0 ≥ any `recall_target`), which is how Stage 1 answers it.
So it is a new opcode — `metaljax.approx_top_k`, a pseudo-name like the conv's
— reading `top_k` and `reduction_dim` out of the `mhlo.backend_config`
dictionary and `_gt_`/`_lt_` out of the comparator's symbol name.

Two details from `ops/sort.py` that a plain `top_k` does not need:
`aggregate_to_topk = false` asks for a result WIDER than `top_k` (slicing to
`top_k` would under-fill the buffer and leave the tail as uninitialised device
memory), and the sort key is CANONICALIZED first (−0 ties with +0, every NaN
with every other NaN) before the total-order key is taken.

## Purity: the arm that was "nothing to call"

P5's transliteration of `interpreter.block_is_pure` noted that the Python's
`custom_call_host_hook` arm could not fire, because a custom call declined the
lowering long before anything asked about purity.  It fires now: a block
holding a LAPACK target computes off the device, so it can no more be traced
through `mx::compile` than a while's host read can.  `IsHostOp` is the
structural test — the two StableHLO ops whose handler is a host one, plus any
custom call whose target is in the table — and it is what
`BlockIsPure` consults, which in turn gates the whole-main compile, the
while-body compile and `WhileTraceable`'s unroll.

`Program::reads_host` already counted `kHostCall` (P8.5), so a dynamic while
holding one goes down the serial path rather than the pipelined one.  Three
execute_test rows exercise the combination directly: a cholesky inside a
`fori_loop` body, a solve inside a `scan`, an eigh inside a `cond`.

## Validation

### execute_test

284 → **357 checks**, of which 70 are the new linalg rows.  A factorization is
not determined by its inputs the way a matmul is — an eigenvector may be
negated, a singular vector rotated inside a degenerate subspace, a Q column's
sign flipped — so most rows hand back an INVARIANT (a reconstruction, a
residual, an orthogonality product), which is what jax's own `linalg_test`
asserts on, and only the quantities that really are unique (a cholesky factor,
the eigenvalues, the singular values, an LU permutation) compare elementwise.

Covered: cholesky (f32/c64/batched/vmapped/upper/singular), QR (tall, wide,
square, complete vs reduced, the zero-tau completion, c64, batched, the
orthogonal initializer), eigh (symmetric, hermitian, batched, degenerate
spectrum, identity, grad, UPLO="U"), svd (values, full, thin, wide, c64, rank
deficient, batched, pinv, matrix_rank, cond), eig (real → complex, complex, a
conjugate pair, batched), LU (factor, reconstruction, permutation), triangular
solve (all four side/lower combinations, transposed, unit-diagonal, adjoint
c64, batched, vmapped, a zero pivot → infinities in the same places as CPU),
the solvers on top (solve, inv, det, slogdet, det grad, cho_solve, lstsq, solve
grad, matrix_power), schur/hessenberg/tridiagonal/tridiagonal_solve, the four
ApproxTopK shapes, and three host-op-inside-control-flow rows.

**62 of the 70 linalg rows are BIT-identical to jax-CPU**, which is the
strongest thing this milestone can say and follows from what it is: jax's CPU
backend calls LAPACK, and so does this, so on the same operand the two run the
same algorithm.  Of the eight that are not, six are at 1e-7..1e-8 (a QR
completion, a complex eigendecomposition, `pinv`, `lu`), one is `lu factor`
(6e-8) and one is `linalg.det` at 2.4e-4 relative 1e-6 — det has no host op at
all, it is jax's blocked DEVICE LU against CPU's `getrf`, and it is the same
gap the pre-P9 plugin had.

312 of 314 cases are bit-identical between the compiled and the eager plugin
(the two that are not are the pre-existing transcendental rows).

The two half-precision rows have **no CPU answer to compare with**, so they are
contracts rather than cases: bf16 and f16 eigh/svd/cholesky must RUN, and what
comes back must be a factorization of the (half-rounded) operand.  Worst
invariant 4.8e-03 at bf16 (8 mantissa bits) and 5.6e-04 at f16.

### the census slice

Every pinned-suite file with a LAPACK-family failure in
`notes/data/p8-native-only-reasons.txt` — 20 files — re-run through the native
plugin, one process per file, sequentially, on this tree, before and after.

| file | before (pass/fail) | after | delta |
|---|---:|---:|---:|
| `linalg_test` | 374/349 | 669/**54** | −295 |
| `shape_poly_test` | 2165/181 | 2319/**27** | −154 |
| `eigh_test` | 5/78 | 83/**0** | −78 |
| `svd_test` | 0/65 | 65/**0** | −65 |
| `qdwh_test` | 8/46 | 44/**2** | −44 |
| `scipy_stats_test` | 942/40 | 981/**1** | −39 |
| `ann_test` | 0/32 | 32/**0** | −32 |
| `random_lax_test` | 512/32 | 544/**0** | −32 |
| `lobpcg_test` | 1/54 | 28/**27** | −27 |
| `lax_scipy_sparse_test` | 29/26 | 48/**7** | −19 |
| `nn_test` | 363/15 | 377/**1** | −14 |
| `polynomial_test` | 2/18 | 13/**7** | −11 |
| `lax_scipy_test` | 93/10 | 103/**0** | −10 |
| `custom_linear_solve_test` | 3/11 | 12/**2** | −9 |
| `scipy_signal_test` | 48/28 | 56/**20** | −8 |
| `lax_scipy_spectral_dac_test` | 0/5 | 5/**0** | −5 |
| `custom_root_test` | 3/5 | 8/**0** | −5 |
| `array_extensibility_test` | 606/8 | 610/**4** | −4 |
| `cholesky_update_test` | 0/2 | 2/**0** | −2 |
| `scipy_spatial_test` | 69/2 | 71/**0** | −2 |
| **TOTAL** | **5223/1007** | **6070/152** | **−855** |

**Zero regressions**: the set difference `after − before` over the failing
test ids is empty.

**No numeric mismatch on a lowered path.**  All 152 remaining failures were
re-run with `--tb=line` and every one is either a LOUD decline or a
better-than-CPU assertion:

| # | reason | phase |
|---:|---|---|
| 92 | `scatter on complex` | P10 |
| 28 | `element type f64` (all of `lobpcg`'s F64 classes) | intentional |
| 10 | `sort: comparator ends in stablehlo.or` / `complex lexicographic` | P10 |
| 6 | `op stablehlo.select_and_scatter` | P10 |
| 6 | `op stablehlo.reduce_precision` | P11 |
| 3 | `cross-memory-space copies` | P13 |
| 1 | no `debug_callback` rule for platform metal | P13 |
| 5 | assertions that a platform SHOULD have failed | shared whitelist |
| 1 | `testCauchyIsf1` | the pre-existing P8 precision row |

The last two rows are the ones worth naming.  All five "should have failed"
tests (`testQrInvalidDtypeCPU`, four `shape_poly` harnesses) fail on **Stage 1
too** — they are in `notes/data/p8-stage1-failures.txt` and in the 0.11.0
release artifact — and `testQrInvalidDtypeCPU` is precisely the
exceeds-CPU property: it asserts that a **float16** QR raises, and here it
computes.  `testCauchyIsf1` is P8's single numeric row (0.43540 vs 0.43564,
`mx::compile`'s fused kernel being ~5 ULP less accurate than its own eager
path); it failed before this milestone with the same two values.

### the rest

`texmo_gate.py` **106/106** (20 via sensitivity scaling, 0 decline, 0 FAIL) —
texmo has no linalg, so this is a no-regression check.  `smoke_test.py` passes,
and `wheel_poc_test.py` passes from a fresh 3.13 venv with the wheel installed
— which is what says `-framework Accelerate` really is linked into the dylib a
wheel carries, since that test's cholesky checkpoint now COMPUTES instead of
declining.  `bazel test //...` green (the GIL-free runtime test included).

Dylib: 165,826,952 → **165,943,816 B** (+116,864, **+0.070 %**).

## Reviewer-scrutiny list

1. **`RunOrgqr`'s two branches** (`host_lapack.cc`).  The completion case hands
   LAPACK `ncols` reflectors of which the last `ncols - cols` have a zero tau;
   the reduced case hands it `k` and slices.  `_householder_product` is the
   spec.  The guard `want <= m` and `src = min(cols, acols)` are mine, not the
   Python's: `x[:, :cols]` on a matrix with FEWER columns than `cols` silently
   yields a narrower array in numpy and then pads to the wrong width, which
   this shape cannot reach from jax but the Python would mis-handle if it did.
2. **`LeftSolve`'s singularity test** is `m[i][i] == 0` exactly, which is
   LAPACK's own `trtrs` condition, and the fallback is the substitution written
   here.  Everything else goes to `cblas_?trsm`.  A reviewer should ask whether
   a *nearly* zero pivot can make `trsm` and the substitution disagree beyond
   tolerance; the answer this milestone gives is that all four side/trans rows,
   the zero-pivot row, and every `linalg_test` solve/inv/det/lstsq case are
   bit-identical to jax-CPU, which runs `trtrs` (i.e. `trsm`) for the same
   inputs.
3. **The FFI result conventions** (`kEighFfi`, `kSvdFfi`, `kEigFfi`) are
   implemented from the Python handlers' comments and are **not exercised by
   anything in this suite** — jax lowers to those targets on platform `cpu`
   only, so they can be reached here only through a deserialized artifact.
   They are cheap and they are Stage 1's semantics, but they are untested code.
   The alternative was to decline them by name, which is also defensible.
4. **`geev`'s left eigenvectors** replace `_metal_eig`'s adjoint-and-match
   hack (see above).  Deliberate, and closer to jax-CPU, but it IS a semantic
   change from Stage 1 — a program comparing the two engines' `vl` element by
   element would see different normalizations.  jax's `jnp.linalg.eig` never
   asks for them (`compute_left_eigenvectors=False`), so no test covers it.
5. **`ApproxTopK`'s `k` widening.**  `k = max(k, min(out_n, n))` when
   `aggregate_to_topk = false`; getting this wrong under-fills the result and
   leaves uninitialised device memory, which no comparison would catch
   reliably.  One execute_test row covers it; the Python has the same line.
6. **The purity arm.**  `IsHostOp` is consulted by `BlockIsPure`, which gates
   the whole-main compile, the while-body compile and `WhileTraceable`'s
   unroll.  If a host target were ever added to `HostTargets()` without that
   being true, the tape would try to run LAPACK on tracers.  The three
   host-op-in-control-flow execute_test rows are the guard.
7. **`Alias` for `Sharding` / `annotate_device_placement`.**  New in this
   milestone (they used to decline), and an alias carries the taints that keep
   XLA's no-alias contract.  It follows `optimization_barrier`'s arm exactly.
8. **What the native branch does NOT register**: callbacks and donation.  Both
   are deliberate, both are P13, and both would REGRESS the suite if switched
   on today (donation especially — jax would emit aliasing the plugin ignores).
9. **`_register_linalg_lowerings` gained two parameters with defaults**, so the
   trampoline branch's call site is unchanged and behaves as before.
10. **`indices_of_shape_operands` declines**, loudly.  A custom call that still
    carries its result-shape operands has not been through jax's
    refine-polymorphic-shapes pass, so its result types are dynamic and there
    is nothing to size a host buffer from.  In practice the pass always runs
    before the plugin sees the module (shape_poly's LU harnesses pass), but a
    reviewer should confirm that is guaranteed rather than lucky.
11. **The three decline checkpoints moved again** — `smoke_test`,
    `wheel_poc_test` and one `execute_test` row now use
    `stablehlo.reduce_precision` (P11's family) where they used
    `stablehlo.cholesky`, which used to be `stablehlo.convolution`.  A new
    execute_test decline covers the other half of the custom-call arm: an
    unknown target must decline BY NAME.
