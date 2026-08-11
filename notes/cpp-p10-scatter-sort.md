# Stage 2 P10: the compiled-constant defect, complex scatter, lexicographic sort

Follows [`cpp-p9-linalg.md`](cpp-p9-linalg.md).  Three items: a live
correctness defect the native plugin re-exposed, and the two families P8's
census put next on the ladder (**329** complex-scatter tests, **158** sort
tests).  All three land in the forked runtime (`plugin-native/runtime/`) and
the plugin's lowering; `native/` and `src/` are untouched.

| file | change | what |
|---|---:|---|
| `plugin-native/metal/metal_lowering.cc` | +204/-20 | the rank-0 constant rule, complex scatter's arms, the two select-tree comparators |
| `plugin-native/runtime/ops_shape.cc` | +15/-4 | `kConstant` reshapes the one-element buffer |
| `plugin-native/runtime/ops_index.cc` | +140/-54 | `sort_key`/`stable_argsort`, `kLexSort`, complex scatter by parts |
| `plugin-native/runtime/config.cc` | +7 | `metaljax.lex_sort` |
| `plugin-native/runtime/program.h` | +6/-2 | `kLexSort`, the `kConstant` attr, scatter method 6 |
| `plugin-native/execute_test.py` | +248/-19 | 27 new cases (+27 checks); three declines retired, two added, one re-aimed |

## 1. The compiled-constant precision defect

`tests/test_elementwise.py` has two tests that PASS through Stage 1 and FAILED
through this plugin — Stage 1's own regression tests for an MLX bug:

> `mx::compile` inlines a **rank-0** constant into generated Metal source as a
> decimal literal printed with **7 significant digits**, one short of
> float32's round-trip requirement of 9.  Measured over 4,000 random f32
> constants, 2,675 came back from a fused kernel 1 ULP away from the correctly
> rounded product, and `float32("%.7g" % c)` predicted the compiled result in
> 4,000 of 4,000 cases.

The failure, natively: `pi * 0.995 * 0.995` came back `3.110255718231201`
where the CPU (and any IEEE backend) says `3.110255479812622` — 4 of 10
constants wrong in the first test, and the second test's ill-conditioned
consumer (`tan(pi/2 - pi*q)`, which is `scipy.stats.cauchy.isf`) turned that
ULP into 4.7e-3 relative error near the pole.

**Stage 1's mechanism** is `ops/elementwise.py`'s `_roundtrips_as_literal` plus
one line of `_constant`: a rank-0 float constant whose value survives `%.7g`
stays a literal (nothing is lost, and a literal costs no buffer binding); one
that does not is built as a ONE-ELEMENT array and reshaped to rank 0, which
makes it a computed node rather than a leaf, and MLX bakes only leaves.

**Ported, with the reshape moved.**  `LowerConstant` runs the same round-trip
test and stores the payload at shape `{1}` with an attribute flag; the
`kConstant` entry reshapes it to rank 0 on every read.  Stage 1 does the
reshape once, when it decodes the constant, and that is a latent hazard this
port does not inherit:

```python
>>> a = mx.reshape(mx.array(np.array([np.float32(np.pi)])), ())
>>> probe(a)              # exact
>>> mx.eval(a); probe(a)  # DIFFERS -- 1 ULP
```

`eval` **detaches** the reshape into a leaf, and a rank-0 leaf is bakeable
again — so a program that runs eagerly before it is compiled (a body the
runner interprets for one chunk size and compiles for another; a program that
takes the recovery ladder's eager arm and is later traced) would put the
literal back.  Rebuilding the node per read costs one graph node per constant
per trace and cannot regress.

**What it cleared**, verified one by one:

| row | before | after |
|---|---|---|
| `tests/test_elementwise.py::test_rank0_constant_is_not_a_lossy_literal` | FAIL (4/10 products) | pass |
| `tests/test_elementwise.py::test_ill_conditioned_constant_expression_matches_cpu` | FAIL | pass |
| `scipy_stats_test::testCauchyIsf1` (P8's single numeric row, P9's last) | FAIL 0.43540 vs 0.43564 | **pass** |

**P5's "the first compiled executable in a process is unfused" finding is NOT
this, and it did not move.**  Two probes — P5's own shape (three distinct
executables of one program in one process) and the `execute_test` row that
reports the 4.8e-7 (`logistic/erf/sqrt f32`) — were run against the HEAD dylib
and the fixed one:

* neither probe reproduces the unfused-first difference on this tree at all
  (three executables agree bit for bit, before and after);
* `execute_test`'s eager-vs-compiled arm reports the same **4.8e-07** on
  `logistic/erf/sqrt f32` before and after, and standalone that program's
  compiled and eager answers are bit-identical (`0` differing elements) — so
  the arm's difference is a whole-process property, exactly as P5 said, and
  not the rank-0 constant rule.

### the finding: MLX bakes SPLAT constants too, and both engines lose that ULP

`is_scalar` in MLX's compiled-kernel builder is a **size-1** test, not a rank-0
one.  Measured (pure MLX, no metaljax in the process):

| constant, as it reaches a fused kernel | result |
|---|---|
| rank-0 leaf | **DIFFERS** (1 ULP) |
| `broadcast_to(leaf{1}, (4,))` — our splat path | **DIFFERS** |
| `broadcast_to(leaf{1,1}, (4,4))` | **DIFFERS** |
| `reshape(leaf{1}, ())` — the rank-0 fix | exact |
| `broadcast_to(reshape(expand_dims(leaf{1},0), (1,)), (4,))` | exact |
| a materialized full-shape constant | exact |

So every SPLAT constant that does not round-trip through `%.7g` — which is
what `dense<3.14159274> : tensor<17xf32>` becomes on both engines, a broadcast
from a one-element buffer — is still baked, in Stage 1 and here alike.  A
reshape node in front of the broadcast fixes it, in the same way and for the
same reason as the rank-0 rule.

**Not done, deliberately** (CLAUDE.md: perf-costing correctness fixes need
Oleg's sign-off).  The cost is not zero: a baked literal binds no buffer, and
a fused kernel that gains one argument per lossy splat constant runs into
Metal's 31-buffer limit — which is a live failure mode with a name in this
project (`Too many inputs/outputs fused in the Metal Compiled primitive`, 56
census tests before P8.5 gave it a fallback).  erf's polynomial alone would add
about ten.  The patch is the same shape as the one above (round-trip test at
lowering; `reshape(expand_dims(payload, 0), unit)` in the splat arm), the
accuracy gain is 1 ULP per affected constant against the CPU reference, and
the decision is Oleg's.

## 2. Complex scatter, by parts

MLX has no complex scatter kernels at all.  `ops/gather.py`'s `_scatter`
recurses on `mx.real`/`mx.imag` and recombines with `make_complex`; the entry
now does the same, and the lowering's blanket `scatter on complex` decline is
replaced by a decline of the combiners the decomposition is not exact for.

* **set / add / subtract** are componentwise: writing the parts separately IS
  the complex write.
* **multiply** is not.  `ops/gather.py` rewrites it (`method = "apply"`):
  gather the current values, combine, and SET the result — which equals the
  combiner only while no two updates land on the same slot.  The Python
  handler assumes that; here it is **checked**, against the op's own
  `unique_indices` flag.  A promise-carrying scatter (`scatter_apply`, the
  static indexing forms, `jax.scipy.signal.csd`'s spectrum doubling) takes
  the rewrite as method 6; a plain `.at[i].multiply(u)` declines
  (`complex scatter multiply without unique indices`).  Method 6 also takes
  the drop rule a **set** takes — it writes rather than combines, and
  neutralizing an update cannot express "leave the slot alone" for an
  assignment — and its gather reads the operand BEFORE the dummy pad, over
  the same clamped indices, so a dropped update's product is discarded with
  the pad.
* **max/min** cannot arise: complex has no order.  They decline naming the
  combiner.

Two measured notes on the multiply arm.  It is not bit-exact against
jax-CPU — one complex multiply contracts to an FMA on this GPU where the
CPU's does not (**7e-8** relative, the same arithmetic the Python engine
runs) — and where the promise is BROKEN it disagrees by design: with
`unique_indices=True` and two updates on one slot (which is what
`x.at[jnp.array([-1, 5, 99])]` becomes, since jax wraps the negative index
before the scatter), XLA:CPU applies both and this applies one.  That is
undefined behaviour on both sides of the promise, and it is why the
`execute_test` rows keep their indices honest.

Everything P4 built stays where it was — the index plan, the clamp, XLA's
OOB **drop** through either strategy — because none of it depends on the
element type.  Two details do:

* the **neutral** value (drop strategy 1) is now the PART's, `0.0f` rather
  than a complex zero, which is what the Python handler computes too: its
  recursion reaches `_combiner_neutral` with `mx.real(operand)`'s dtype;
* the **dummy pad** (strategy 2) grows each part, and the index redirection
  that feeds it is computed ONCE — it is an index question, and computing it
  twice would be two `where`s over the same mask.

## 3. Lexicographic sort, and complex sort

P6 ported the sort recognizer's compare arm and declined the two SELECT TREES
by name.  Both are now recognized, structurally, exactly as `ops/sort.py` does
it — by reading which operand pairs the tree decides on, never by evaluating
the tree:

* **complex** (one key pair, complex operand) → the existing `kSort` entry
  with a new key kind: `_sort_key`'s complex arm, the (re, im) pair of
  canonicalized totalOrder keys packed into one u64.
* **multi-key** (`jnp.lexsort`, `lax.sort(num_keys>1)`, `jnp.unique` over
  rows, and every sparse index canonicalization) → a new opcode, **`kLexSort`**
  under the pseudo-name `metaljax.lex_sort`: successive STABLE argsorts from
  the last key to the first, each threaded through the permutation the
  previous ones built, then every operand gathered by it.  That is
  `_lex_sorted` line for line, and it needs no "argsort by a permutation the
  tape computed" mechanism — the whole loop is inside one entry, where the
  permutation is a local.

The key kinds are now a table (`sort_key`, beside the handler), because the
canonicalization belongs to different places on the two arms: the compare arm
evaluates jax's own key chain (which canonicalizes) and takes a bare
totalOrder key; the tree arms never evaluate anything, so `-0 -> +0` and
`NaN -> canonical NaN` happen in the handler.  `chlo.top_k` (bare key) and
ApproxTopK (`_sort_key`) sit on either side of that line and now say which
they want rather than each carrying its own copy of the code.

**A guard the Python does not have.**  The lexicographic reading is inferred
rather than evaluated, so its vocabulary is checked: every op in the tree must
be a compare (direction LT, EQ or NE), a boolean combinator, a `real`/`imag`,
or a constant.  A tree holding a GT decides the other way somewhere and would
be silently mis-ordered by an ascending execution; it declines instead
(`sort: comparator tree compares GT`).  Two hand-written modules in
`execute_test.py` pin both refusals.

**One Stage 1 quirk is reproduced on purpose**: a tree over exactly ONE
non-complex key declines.  `ops/sort.py` reaches that state through
`ks.pop()`, which empties the set its lexicographic test then reads — and it
is unreachable from jax, whose single-key sorts get a bare compare.

## Validation

| | result |
|---|---|
| `plugin-native/texmo_gate.py` (x2) | **106/106** ok (22 and 25 via sensitivity scaling), 0 decline, 0 FAIL |
| `plugin-native/execute_test.py` | **384 checks**, 0 failures (357 before) |
| `plugin-native/smoke_test.py` | 4/4 checkpoints |
| `plugin-native/decline_census.py` | **34 of 35** lower (32 at P7); the one left is `debug_print`, a JAX-side registration gap (P13) |
| `plugin-native/wheel_poc_test.py` | 4/4 from a fresh 3.13 venv with the native wheel |
| `bazel test //...` | PASSED (`//metal:runtime_gil_free_test`) |
| `pytest tests/test_elementwise.py test_sort.py test_gather.py test_complex.py test_constants.py` (native) | 134 passed |
| dylib | 165,926,056 → **165,943,720 B** (+17,664, **+0.011 %**) |

### the census slice

The 17 files P8's reasons file names for the two families, plus
`scipy_stats_test` for the constant row — before and after on this tree, one
process per file, sequentially.

| file | before (pass/fail) | after | delta |
|---|---:|---:|---:|
| `sparse_bcoo_bcsr_test` | 263/157 | 413/**7** | −150 |
| `sparse_test` | 306/104 | 410/**0** | −104 |
| `sparsify_test` | 124/92 | 216/**0** | −92 |
| `linalg_test` | 669/54 | 722/**1** | −53 |
| `scipy_signal_test` | 56/20 | 76/**0** | −20 |
| `lax_numpy_test` | 3172/38 | 3192/**18** | −20 |
| `lax_numpy_setops_test` | 134/18 | 152/**0** | −18 |
| `shape_poly_test` | 2319/27 | 2330/**16** | −11 |
| `lax_test` | 2458/63 | 2468/**53** | −10 |
| `lax_scipy_sparse_test` | 48/7 | 52/**3** | −4 |
| `array_extensibility_test` | 610/4 | 614/**0** | −4 |
| `lax_numpy_indexing_test` | 359/13 | 362/**10** | −3 |
| `lax_numpy_ufuncs_test` | 202/3 | 205/**0** | −3 |
| `scipy_optimize_test` | 11/2 | 13/**0** | −2 |
| `qdwh_test` | 44/2 | 46/**0** | −2 |
| `ode_test` | 13/1 | 14/**0** | −1 |
| `custom_linear_solve_test` | 12/2 | 13/**1** | −1 |
| `scipy_stats_test` | 981/1 | 982/**0** | −1 |
| **TOTAL** | **11,781/608** | **12,280/109** | **−499** |

**Zero regressions**: the set difference `after − before` over the failing
test ids is empty.

**No numeric mismatch on a lowered path.**  Every one of the 109 that remain
was re-run with `--tb=line`; they are loud declines or known non-P10 rows:

| # | reason | phase |
|---:|---|---|
| 38 | `element type <unknown>` (extended/key dtypes) | P11 |
| 19 | `scatter combiner apply` | the scatter tail |
| 17 | `op stablehlo.reduce_precision` | P11 |
| 9 | `unsafe_buffer_pointer` identity (`testArrayCopy*`) | P13 |
| 6 | `op stablehlo.select_and_scatter` | the scatter tail |
| 5 | `scatter on a rank-0 operand with no indices` | the scatter tail |
| 5 | assertions that a platform SHOULD have failed (the shared whitelist) | — |
| 3 | cross-memory-space copies | P13 |
| 2 | `test_bcoo_spdot_general{0,6}` — position-dependent, pass standalone | tracked-open |
| 1 | `complex scatter multiply without unique indices` | intentional (stricter than Stage 1) |
| 1 | `debug_callback` has no rule for platform metal | P13 |
| 1 | `custom call target 'dce_sink'` | P13 |
| 1 | `element type f64` | intentional |
| 1 | `conv: complex with no spatial dimensions` | intentional (P7) |

**The scatter tail is what P10 leaves behind**: 30 tests over three declines
(`scatter combiner apply`, `select_and_scatter`, a rank-0 operand with no
indices), which P8's census bracketed with this phase and this mission did
not carry.  They are the next thing in this family.

### tests/ through the native plugin

The Stage 1 suite, run with `METALJAX_PLUGIN_PATH` at the native dylib, is a
second standing leg: **88 -> 87 failures**, and the movement is exactly this
milestone's three acceptance rows —
`test_elementwise::test_rank0_constant_is_not_a_lossy_literal`,
`test_elementwise::test_ill_conditioned_constant_expression_matches_cpu`,
`test_sort::test_unique_complex_nans` — against noise of ±2 in
`test_pjrt_surface::test_buffer_pointer_of_broadcast`, which is FLAKY on both
dylibs (5/2/3 failures on three runs of the old one, 6/2/2 on the new): it
asserts that two reads of a broadcast's transient pointer agree, which is the
address-reuse trap of CLAUDE.md item 20 rather than anything a lowering
decides.  What remains, by file: `test_moe` 28, `test_qmm` 26,
`test_qmm_mxfp4` 16, `test_pjrt_surface` 7, `test_subbyte_float` 6,
`test_donation` 3, `test_engine_gc` 1 — the recognizer-emit families (whose
Python-side pack building the plugin has no path to), P11's sub-byte floats,
and P13's PJRT surface.

**A harness trap worth recording**: `src/jax_plugins/metal/__init__.py` picks
the native branch only when `METALJAX_PLUGIN_PATH`'s **file name** is
`libmetal_pjrt_native.dylib`.  A dylib copied aside for an A/B run under any
other name still LOADS (the Stage 1 branch honours the same variable), but
jax then registers the trampoline's lowerings — callbacks and buffer donation
included — so `test_donation` reads differently for a reason that has nothing
to do with the plugin.  Both census passes were re-run with correctly named
copies; the totals are identical to the first pair (11,781/608), which is
also the evidence that the registration difference changes none of these
tests.



## Reviewer-scrutiny list

1. **The constant rule's attribute.** `kConstant` grew an optional attrs
   vector (`[1]` when the payload is a one-element stand-in, empty otherwise
   — so a constant that round-trips lowers byte-identically to before).  The
   entry's reshape runs on every read, INCLUDING inside an `mx::compile`
   trace, which is the point; if a future reader moves it back to lowering
   the defect returns silently the first time a program runs eagerly before
   it is traced.
2. **The splat finding is unfixed on purpose** (section 1).  It is a live
   1-ULP gap against jax-CPU on both engines, and the reason it is not fixed
   here is a buffer-count cost, not a doubt about the mechanism.
3. **Complex scatter's combiner set.**  set/add/subtract go by parts;
   multiply takes the gather-multiply-set rewrite ONLY under the op's
   `unique_indices`; max/min decline.  A reviewer should check the neutral
   dtype (f32, the part's) and that the pad strategy is forced for method 6.
4. **The lexicographic arm is inferred, not evaluated.**  It reads the
   comparator's argument dependencies and trusts that the tree is ascending —
   Stage 1's contract, with a vocabulary guard added.  The guard is what
   stands between a hand-written descending comparator and a silently
   mis-ordered result; the two hand-written `execute_test` modules are its
   only coverage, since jax cannot emit either shape.
5. **`_sort_key`'s two float kinds.**  Kind 1 (bare totalOrder) is for the
   arm that evaluated jax's own canonicalizing key chain; kind 4
   (canonicalize, then totalOrder) is for the arms that never evaluate the
   comparator.  Getting them the wrong way round is invisible on values and
   visible only on argsort ties over -0 and NaN — which is what the
   `argsort complex over signed zeros and NaNs` and
   `unique over complex with NaN and -0 ties` rows exist for.  ApproxTopK
   moved from 1 to 4 with this change: `ops/sort.py` keys it through
   `_sort_key`, so 1 was a latent (untested) divergence.
6. **The opcode enum shifted.**  `kLexSort` was inserted after `kSort`, which
   renumbers every opcode after it.  Nothing serializes an opcode number —
   the plugin and the runtime are one build, and the registry is keyed by
   name — but a reviewer should satisfy themselves of that.
7. **Duplicate indices are still the hazard.**  The arithmetic
   `execute_test` rows use unique indices deliberately: a complex scatter-add
   with duplicates is order-nondeterministic on this GPU (it differs between
   the eager and the compiled path, which is what the eager-vs-compiled arm
   caught while these rows were being written), and a duplicate SET is the
   implementation-defined race P8.5 classified.
