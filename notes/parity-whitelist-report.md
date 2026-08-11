# The 142: a per-test review of every failing row in the pinned jax suite

> **STATUS after Oleg's review verdicts (2026-08-11): 142 -> 129.** Thirteen
> rows were implemented rather than whitelisted; the per-class sections below
> carry a **[FIXED]** line each. What changed, in the order Oleg asked for it:
>
> | # | class | rows | outcome |
> |---:|---|---:|---|
> | 1 | Q sparse `spdot_general` | 2 | **NOT ours** — diagnosed to MLX's compiled-kernel FUSION (`METALJAX_MLX_COMPILE_MODE=no_fuse` makes both pass, the eager walk is right). Still failing; MLX bug #8, evidence below |
> | 2 | C async collectives | 5 | `async_start`/`async_done` implemented (2 pass; 3 now fail only on an *unoptimized-HLO* expectation, with class E) |
> | 3 | I complex scatter-multiply, complex 0-spatial conv | 2 | both implemented — and the conv one uncovered that **shipped Stage 1 computes it wrong** |
> | 4 | F memory spaces | 4 | a second `PjRtMemorySpace` (`pinned_host`) |
> | 5 | L double donation | 1 | XLA's own clash check |
> | 6 | E PJRT surface | 8 | aot topology 2 (message), `dce_sink` 1 (compiles again), `OptimizedProgram` implemented — `XlaTransform` 4 and the inline row NOT done |
> | 7 | B export allowlist / A2 f64 pass-through | 49 + 9 | both investigated, neither implemented — findings below |
> | 8 | M token representation | 1 | a token result is `xla::TOKEN` now |
> | 9 | J skew/harness | 6 | both mechanisms nailed down; jax-side, stay whitelisted |
>
> Affected files, re-run natively before and after: `async_collectives_test`
> 5->3, `aot_test` 2->0 (both become SKIPs), `api_test` 7->5, `lax_test` 2->0,
> `lax_numpy_indexing_test` 1->0, `memories_test` 2->0, `export_test` 7->5,
> `sparse_bcoo_bcsr_test` 2->2. Whole pinned suite: **28,068 passed / 129
> failed = 99.54 %**, past Stage 1's 99.51 % on this tree and the 0.11.0
> release artifact's 99.53 %. Batteries: `execute_test` 493 -> **502 checks**,
> `texmo_gate` **106 ok / 0 decline / 0 FAIL**, smoke, wheel PoC from a fresh
> 3.13 venv, `bazel test //...`.

Manual review of **all 142 failures** of the pinned `jax-v0.11.0` suite through
the **native** PJRT plugin (HEAD `6dddbff`, 99.50 %: 28,057 passed / 142 failed
/ 6,158 skipped / 35 errors), one entry per test id, for sign-off on every
whitelisting.

Definitive list: `notes/data/p12-14-native-failures.txt` (142 lines, count
cross-checked). Set arithmetic against Stage 1 on the same tree:
`p12-14-native-only.txt` (12), `p12-14-stage1-only.txt` (7).

## How every row here was measured (not copied from a table)

1. **Re-ran all 142 ids** against the native dylib, grouped one pytest process
   per file, `--tb=short`:

   ```
   METALJAX_PLUGIN_PATH=plugin-native/bazel-bin/metal/libmetal_pjrt_native.dylib \
   JAX_PLATFORMS=metal,cpu JAX_ENABLE_X64=0 \
     .venv/bin/python -m pytest <ids of one file> -q --tb=short -rf
   ```

   139 reproduced id-for-id. **Three did not reproduce in a selective run** and
   were re-measured whole-file, where they do: `compilation_cache_test::test_jit`,
   `export_back_compat_test::test_custom_call_coverage`, and the two
   `sparse_bcoo_bcsr` rows (the known position-dependence).
2. **Re-ran the 130 shared ids on the Stage 1 plugin** (same tree, no
   `METALJAX_PLUGIN_PATH`) so that "shared" is a measurement, not an
   inheritance: every one of them fails there too, with the same three
   order-dependent exceptions. Their Stage 1 messages are on file, and are
   quoted below wherever they differ from the native one (`lax_test::dce_sink`,
   `layout_test`, `shard_alike_test`, `api_test`, `nn_test` — the rest match in
   substance).
3. **Re-ran the 12 native-only ids on the Stage 1 plugin**: 12 passed, 0 failed
   — the native-only split is confirmed first-hand.
4. **CPU control.** Cross-checked all 142 against `notes/data/cpu_parity.json`:
   127 are recorded `cpu_pass`, 5 `cpu_fail`, 10 unlisted (the native-only rows
   plus `export_back_compat`, which did not exist as a failure when that file
   was built). The `cpu_fail` claims for `logging_test` and
   `profiler_session_test` were re-verified by running those files on
   `JAX_PLATFORMS=cpu` today.
5. Every test's *purpose* below comes from reading its source in
   `jax-v0.11.0/tests/`, not from its name; parameterised rows had their actual
   parameter dicts introspected where the index mattered
   (`testStaticIndexing5`, `test_dtype_warning{0,1,3}`,
   `testFloatArrayCreation3`).

Raw logs (session scratch, not committed):
`…/scratchpad/reruns/*.log` (native), `…/scratchpad/reruns-stage1/*.log`.

## Executive summary

| # | class | count | now | verdict |
|---:|---|---:|---:|---|
| A | `[f64-policy]` | 48 | 48 | intentional platform constraint — **but 9 of them are pass-through, not compute** ([VERIFY]; design written, not implemented) |
| B | `[export-platform-allowlist]` | 49 | 49 | test/jax-side hard-coded `(cpu, gpu, tpu)`; **sorted: 0 reachable from a plugin, 49 upstream** |
| C | `[async-collectives-unimplemented]` | 5 | **3** | `async_start`/`async_done` implemented; the 3 left want optimized HLO (class E) |
| D | `[jax-side-allowlist]` (callbacks, `check`) | 4 | 4 | jax hard-codes the platform list; not reachable from a plugin |
| E | `[PJRT-surface]` | 8 | **5** | topology message + `dce_sink` + `OptimizedProgram` landed; `XlaTransform` 4 + the inline row remain |
| F | `[memory-spaces]` | 4 | **0** | a second `PjRtMemorySpace` (`pinned_host`) over the one unified pool |
| G | `[layout-column-major]` | 2 | 2 | MLX storage is row-major; declined by name |
| H | `[better-than-reference]` | 5 | 5 | we fail by succeeding where the harness demands a refusal |
| I | `[intentional-native]` (stricter than Stage 1) | 2 | **0** | both gates lifted: sequential complex scatter-multiply, complex 0-spatial conv |
| J | `[skew/harness]` | 6 | 6 | fails identically on CPU / is an absl-under-pytest artifact — both mechanisms nailed down |
| K | `[denormals]` | 1 | 1 | GPU flushes subnormals to zero |
| L | `[donation-contract]` | 1 | **0** | XLA's own `TestBufferDonationClashes`, its messages word for word |
| M | `[token-representation]` | 1 | **0** | a token result is `xla::TOKEN`; the device array stays the empty bool one |
| N | `[numerics/FD-reference]` | 1 | 1 | the test's f32 reference is meaningless at this tolerance |
| O | `[sdy-op-unimplemented]` | 1 | 1 | `sdy.sharding_group` declined ([VERIFY]) |
| P | `[jax-generic-LU × shard_map]` | 1 | 1 | jax's own generic LU is incompatible with `shard_map` vma typing |
| Q | `[tracked-open]` — **not benign** | 2 | 2 | sparse `spdot_general` wrong values — **diagnosed to MLX's kernel fusion**, not this tape |
| R | `[registration-artifact]` | 1 | 1 | our own `metaljax_*` custom-call targets are "declared stable, untested" |
| | **total** | **142** | **129** | |

**Classes from earlier reviews that no longer have any rows.** *Complex
inf/NaN pole semantics* (~24 at 0.4.2, reclassified INTENTIONAL at the 0.11.0
review) and *ordered-effect tokens* (~50 at 0.4.2) do **not** appear anywhere in
these 142: the pole rows were fixed or absorbed during the 0.11.0 campaign, and
P12's token work took `jaxpr_effects_test` 15 → 1 and
`debugging_primitives_test` 32 → 0. What survives of the token family is one
negative test (class M) and one internal-callback allowlist row (class D). The
*windowed scatter-apply*, *i4 bitcast* and *dilation-corner* buckets of the
0.4.2 README list are likewise all green now.

**Provenance of the whitelist.** The 130 shared rows are *not* byte-identical
to the 130 Oleg approved at 0.11.0. Two rows entered and two left:

* entered: `sparse_bcoo_bcsr_test::test_bcoo_spdot_general{0,6}` — the
  TRACKED-OPEN pair Oleg ruled on separately on 2026-08-05 (class Q);
* left: `core_test::test_reference_cycles{,_jit}` — **fixed** by the native
  stack (no Python engine to hold references), together with the four
  `debug_print`-in-a-while-cond rows and `lax_test::testScatter1`
  (`p12-14-stage1-only.txt`, 7 rows).

So: **128 rows are literally on Oleg's approved 0.11.0 list**, 2 are the
tracked-open sparse pair, and 12 are native-only.

## Rows flagged **[VERIFY]** — 22 rows, 8 entries (7 doubts + 1 finding)

| rows | what to scrutinise |
|---:|---|
| 9 | **native f64 is stricter than the documented policy.** CLAUDE.md's f64 policy says pass-through is OK (stored f32, bit-identical) and only *compute* fails. The native lowering has no f64 tape dtype at all, so an f64 **constant or parameter that is never computed on** declines. All 9 native-only f64 rows are pure creation (`jnp.array(np.float64(0))`). Is refusing pass-through the intended Stage-2 policy, or an unintended tightening? |
| 5 | **`async_collectives_test`** — `stablehlo.async_start` is unimplemented on both stacks, passes on jax-CPU with one device, and has **no disposition recorded in any note or in the README**. It was swept in with "single-device collectives". |
| 2 | **`aot_test`** — the two rows fail only because our decline message does not contain the substring the test's own skip-detector looks for. Rewording the message to include "topology not implemented" turns both into SKIPs, for free. |
| 2 | **`memories_test::test_compute_on2_out_mem_space*`** — an `out_memory_spaces=Host` annotation is **silently ignored** (values right, `memory_kind` stays `device`). The two `export_test` memory-space rows fail loudly; these two do not. |
| 1 | **`lax_test::test_dce_sink_prevents_xla_dce`** — verdict unchanged, but the native mechanism is *different* from the historical one, and it hides a Stage 1 → native functional regression: `lax.dce_sink(..., prevent_mlir_dce=True)` compiled on Stage 1 and now declines. |
| 1 | **`lax_numpy_indexing::testStaticIndexing5`** — the native refusal is conservative: this indexer's indices *are* unique and Stage 1's answer was checked against numpy. `.at[...].multiply()` on a complex array is now refused wholesale. |
| 1 | **`shard_alike_test`** — `sdy.sharding_group` is a metadata-only op with zero results that we could treat as an identity; no disposition recorded. |
| (1) | **`lax_test::testConvGeneralDilated0D2`** is flagged for the opposite reason: the native decline is right, and measuring it uncovered that **shipped Stage 1 computes complex 0-spatial convolutions wrong** (numbers below). |

---

# A. `[f64-policy]` — 48 rows

**Mechanism (all 48).** Metal has no f64 ALUs. Native declines at
`Lowering::CheckValue` with `UNIMPLEMENTED: metaljax-native: element type f64`
whenever any value in the program carries `f64` (`metal_lowering.cc:1797`).
Stage 1 declines with `program computes in float64/complex128, unsupported on
Metal` from `interpreter.py::_check_no_f64_compute`, which whitelists
data-movement ops (`_F64_DATA_MOVEMENT`: constant, convert, reshape, gather,
while, …) so a value that merely passes through is allowed and stored as f32.

**That difference is the entire native-only split**: 39 rows do real f64
arithmetic and fail on both stacks; 9 rows only *create* an f64 array and fail
only natively. Approved category: README "Intentional (platform constraints):
No float64", `notes/jax-test-suite-2026-07.md` release review (f64/x64 =
intentional platform constraint; lobpcg 27, x64_context 7 named explicitly).

### A1. f64 *compute* — shared with Stage 1 (39)

`lobpcg_test.py::F64LobpcgTest` (27) — the whole `@jtu.with_config(jax_enable_x64=True)`
class: LOBPCG eigensolver runs in f64. `testLobpcgConsistency*` checks residual
norms and relative error against analytic eigenvalues; `testLobpcgMonotonicity*`
checks the last 20 % of iterations beat the first 20 %; `testCallableMatrices*`
runs the same solver with a matrix given as a callable / BCOO operator. All 27
fail at compile with `element type f64`, before any numerics.

| # | test id (all: `UNIMPLEMENTED: metaljax-native: element type f64`) |
|---:|---|
| 1 | `testCallableMatricesF64geom_cond_100k` |
| 2 | `testCallableMatricesF64geom_cond_1k` |
| 3 | `testCallableMatricesF64id` |
| 4 | `testCallableMatricesF64linear_cond_100k` |
| 5 | `testCallableMatricesF64linear_cond_1k` |
| 6 | `testCallableMatricesF64randn` |
| 7 | `testCallableMatricesF64ring_laplacian` |
| 8 | `testCallableMatricesF64sparse_10_` |
| 9 | `testCallableMatricesF64sparse_1_` |
| 10 | `testLobpcgConsistencyF64cluster_k_1__n100` |
| 11 | `testLobpcgConsistencyF64cluster_k_2__n100` |
| 12 | `testLobpcgConsistencyF64cluster_k__n100` |
| 13 | `testLobpcgConsistencyF64geom_cond_100k_n100` |
| 14 | `testLobpcgConsistencyF64geom_cond_1k_n100` |
| 15 | `testLobpcgConsistencyF64id_n100` |
| 16 | `testLobpcgConsistencyF64linear_cond_100k_n100` |
| 17 | `testLobpcgConsistencyF64linear_cond_1k_n100` |
| 18 | `testLobpcgConsistencyF64ring_laplacian_n100` |
| 19 | `testLobpcgMonotonicityF64cluster_k_1__n100` |
| 20 | `testLobpcgMonotonicityF64cluster_k_2__n100` |
| 21 | `testLobpcgMonotonicityF64cluster_k__n100` |
| 22 | `testLobpcgMonotonicityF64geom_cond_100k_n100` |
| 23 | `testLobpcgMonotonicityF64geom_cond_1k_n100` |
| 24 | `testLobpcgMonotonicityF64id_n100` |
| 25 | `testLobpcgMonotonicityF64linear_cond_100k_n100` |
| 26 | `testLobpcgMonotonicityF64linear_cond_1k_n100` |
| 27 | `testLobpcgMonotonicityF64ring_laplacian_n100` |

`x64_context_test.py::X64ContextTests` (7) — that `jax.enable_x64()` as a
context manager gives f64 *results* for real work:

* `test_custom_jvp` — a `custom_jvp` function traced under `enable_x64`; asserts
  primal/tangent dtypes across three grad levels. f64 `x ** 2`, `sin`.
* `test_custom_vjp` — same for `custom_vjp`.
* `test_jit_cache` — `random.uniform(key, (1,), 'float64', -1, 1)` cached across
  an x64 flip.
* `test_mul` — `lax.mul` on two f64 scalars; asserts result dtype f64.
* `test_sin` — `lax.sin` on an f64 scalar.
* `test_while_loop0`, `test_while_loop1` — `lax.while_loop` counting to 10 in
  f64 (both `JIT_IMPLEMENTATION` variants).

`api_test.py::APITest` (3) — `test_dtype_warning0`, `test_dtype_warning1`,
`test_dtype_warning3`: that requesting an explicit 64-bit dtype
warns/raises/allows per `explicit_x64_dtypes` mode.
Introspected parameters: `test_dtype_warning0` = (WARN, x64=**True**),
`test_dtype_warning1` = (ERROR, x64=**True**),
`test_dtype_warning3` = (ALLOW, x64=**True**) — exactly the three x64-enabled variants, which are
the only ones that actually *execute* an f64 constructor
(`jnp.array([1,2,3], dtype="float64")`, `jnp.eye`, `linspace`, …). Variants 2, 4, 5
(x64 off) pass. Stage 1 declines on `stablehlo.divide : tensor<49xf64>`; native
declines earlier on the constant.

`api_test.py::AutodidaxTest::test_autodidax_smoketest` (1) — executes
`docs/autodidax.py` end to end as a smoke test. That tutorial runs under x64
(its printed jaxprs are `float64[]`) and calls
`backend.compile_and_load` directly; declines with `element type f64`.

`nn_test.py::NNFunctionsTest::testDotProductAttentionFloat64MaskDebugInfs` (1) —
regression test for jax #37422: under `jax.enable_x64()` and
`jax.debug_infs(True)`, `nn.dot_product_attention` on f64 zeros with a boolean
mask must not warn. f64 attention math; declines at compile.

### A2. f64 *pass-through only* — native-only (9) **[VERIFY]**

These programs contain an f64 constant/parameter and a convert, and no f64
arithmetic. Stage 1 runs them (storing f32) and all nine **pass** there —
verified today by re-running the 12 native-only ids on the Stage 1 plugin.

* `dtypes_test.py::TestPromotionTables::testFloatArrayCreation3` — asserts
  `jnp.array([1.0, 2.0], dtype=d).dtype == d` for every float dtype under
  `explicit_x64_dtypes('allow')`; index 3 of `float_dtypes` is `float64`
  (`[bfloat16, float16, float32, float64, …]`).
* `lax_numpy_test.py::LaxBackedNumpyTests::testArrayExplicitDtypes` — asserts
  `jnp.array(1, dtype=jnp.int64)`, `jnp.array(1.0, dtype=jnp.float64)` and
  `jnp.array(1j, dtype=jnp.complex128)` keep their dtype under
  `explicit_x64_dtypes('allow')`.
* `pickle_test.py::PickleTest::testPickleX64` — pickles an f64 array made under
  `enable_x64(True)` and unpickles it under `enable_x64(False)`, asserting the
  round trip downcasts to f32.
* `x64_context_test.py::X64ContextTests::test_make_array0`, `test_make_array1` —
  `jit(lambda: jnp.array(np.float64(0)))` under both `JIT_IMPLEMENTATION`
  variants; asserts the dtype tracks the ambient x64 flag.
* `x64_context_test.py::X64ContextTests::test_correctly_capture_default0`,
  `…::test_correctly_capture_default1`, `…::test_correctly_capture_default2`,
  `…::test_correctly_capture_default3` — the same one-line jitted constructor,
  defined inside an `enable_x64(True/False)` block, then called outside it
  (4 = 2 jit impls × 2 flag values); each asserts the dtype follows the ambient
  flag at call time, not at definition time.

All nine fail with `UNIMPLEMENTED: metaljax-native: element type f64`.

**Why this is [VERIFY]**: CLAUDE.md item 4 states the approved policy as
"f64 pass-through OK (stored f32, bit-identical), f64 *compute* fails at
compile naming the op". The native plugin does not implement that split — it
has no f64 dtype code, so `CheckValue` declines a program that only carries the
value. P12-14 classified all nine as "intentional (the f64 policy)", which is
true of the *spirit* (no f64 on Metal) but stricter than the letter. It is a
one-line-per-site question for Oleg: keep the tighter native rule (and update
the policy text), or teach the native lowering the same pass-through
narrowing Stage 1 does.

**Oleg's verdict: match Stage 1 exactly. NOT implemented in this batch** — the
9 rows still fail, and the design is written down here so it is an afternoon
rather than a rediscovery. Half the machinery is already in place and proven:
the BUFFER side does f64 pass-through today (`metal_dtypes.cc`
`case xla::F64: WireType{mx::float32, 8, /*widen=*/true, 1}`, and
`execute_test`'s `f64 passes through, f64 math not` row asserts both halves), and
`PrimitiveTypeOf` already answers `xla::F64`. What is missing is the LOWERING:

1. a pre-pass over the module, before the walk, that is Stage 1's
   `interpreter._check_no_f64_compute` verbatim — every op outside
   `_F64_DATA_MOVEMENT` (constant, convert, reshape, broadcast, transpose,
   slice, ds/dus, concatenate, reverse, gather, scatter, select, iota, pad,
   while/if/case, composite, custom_call, the two sdy identities, the calls and
   returns) that touches an f64 or complex<f64> value declines, keeping the
   current message so nothing that greps `element type f64` changes;
2. with that gate passed, `TapeDtypeCode` may map f64 → the f32 code and
   complex<f64> → complex<f32> (this is exactly Stage 1's arrangement: the
   value is STORED as f32 and the policy check is what keeps it honest);
3. `LowerConstant` needs an f64 arm — its raw-data fast path assumes the IR
   bytes ARE the device bytes, which is false at 8 bytes per element, so f64
   constants have to go through `APFloat` like the emulated grids do (splat and
   rank-0 included);
4. `C128` needs a `WireType` beside `F64`'s for
   `lax_numpy_test::testArrayExplicitDtypes`, the one row of the nine that
   carries a complex128 value.

The risk is contained by the pre-pass: with it in place no arithmetic can reach
an f64 value, so the widening cannot silently compute in f32 — which is the
failure mode CLAUDE.md item 20 records for c128 on Stage 1 ("the f64 checker
didn't match c128 — compute silently ran in c64").

---

# B. `[export-platform-allowlist]` — 49 rows

**Mechanism (all 49).** `jax.export` records the platforms an artifact was
lowered for, and `_export.py::_call_exported_lowering` (line 1727) raises when
the artifact is *called* under a lowering platform that is not in that list:

```
ValueError: Function 'dyn_fun' was exported for platforms '('cpu', 'cuda', 'tpu')'
            but it is used on '('metal',)'.
```

The platform tuples are hard-coded **in the tests**, not derived from the
available devices. Nothing a plugin can do reaches this: the artifact would
have to be exported for `metal` in the first place.

**Sorted, per Oleg's ask (2026-08-11): (a) fixable from a registration hook —
NONE; (b) upstream — all 49.** The check has exactly two escapes and a plugin
owns neither. `platform in exported.platforms` compares against a tuple the
CALLER passed to `jax.export`, and the override,
`DisabledSafetyCheck.platform() in exported.disabled_safety_checks`, is also a
per-export argument — there is no config flag, no environment variable and no
registry behind it (`_export.py` reads the field and nothing else). The one
remaining idea, registering `metal` as an ALIAS of `gpu` in
`xla_bridge._alias_to_platforms`, would make `exported.platforms` match by
making jax believe this backend IS a GPU everywhere else too
(`jtu.test_device_matches(["gpu"])`, every `skip_on_devices`, every
GPU-specific expectation) — a lie with hundreds of consequences, for 49 rows.
The upstream-patch pile therefore gets: `export_harnesses_multi_platform_test`
should export for the platforms actually present rather than the literal
`("cpu","gpu","tpu")` (44 rows), `export_test`'s four hard-coded tuples (4
rows), and `test_multi_platform_mlir_lower_fun_with_platform_specific_primitives`,
whose expectation table is `dict(cpu=2, gpu=3, tpu=4)[jtu.device_under_test()]`
and raises `KeyError: 'metal'` (1 row). Verified `cpu_pass` for all
49 in `cpu_parity.json` — they pass on CPU because `cpu` is inside the
hard-coded tuple. Approved category: `notes/jax-test-suite-2026-07.md`
("export_harnesses_multi_platform (44) — jax-side platform allowlist, cpu_pass
verified"; release review: "Export-harness/callback allowlists: verified
cpu_pass … Functionality covered for metal by
`scripts/run_export_harnesses.py`: 5,460/5,587").

### B1. `export_harnesses_multi_platform_test.py::PrimitiveTest` (44)

Each row is `test_prim(harness)`: export the harness for `("cpu","gpu","tpu")`,
run it natively on each available device, and compare with `exp.call(...)`.

**Why only these 44 of the file's 3,206 executed rows** (3,162 passed / 44
failed / 2,392 skipped): I read the harness definitions in
`jax/_src/internal_test_util/test_harnesses.py`, and every failing one has
**no dynamic arguments** — `StaticArg` only (`_make_iota_harness`,
`_make_iota_2x32_shape_harness`, `random_split`, `random_uniform`,
`random_randint`, and `sign_special_0`, whose operand is
`StaticArg(np.zeros((2,2)))`). With no array arguments, `exp.call()` has no
device to infer a lowering platform from, so it lowers on the **process default
backend = metal** and trips the check; harnesses with real operands lower on the
CPU device their arguments were `device_put` to, and pass. The 44/3,162 split is
exactly along that line.

All 44 fail with the identical `ValueError` quoted above; the third column says
what the harness computes.

| # | test id | harness |
|---:|---|---|
| 1 | `test_prim_iota_2x32_shape_shape_100_100_` | `prng.iota_2x32_shape_p.bind(shape)` |
| 2 | `test_prim_iota_2x32_shape_shape_3_` | same, shape (3,) |
| 3 | `test_prim_iota_2x32_shape_shape_5_7_4_` | same, shape (5,7,4) |
| 4 | `test_prim_iota_broadcasting_shape_float32_4_8_1_1_dimension_1` | `lax.iota` along a non-major dim |
| 5 | `test_prim_iota_broadcasting_shape_float32_4_8_1_1_dimension_2` | `lax.iota` along a size-1 dim |
| 6 | `test_prim_iota_dtypes_shape_bfloat16_2_3_dimension_0` | `lax.iota` per dtype |
| 7 | `test_prim_iota_dtypes_shape_complex64_2_3_dimension_0` | " |
| 8 | `test_prim_iota_dtypes_shape_float16_2_3_dimension_0` | " |
| 9 | `test_prim_iota_dtypes_shape_float32_2_3_dimension_0` | " |
| 10 | `test_prim_iota_dtypes_shape_int16_2_3_dimension_0` | " |
| 11 | `test_prim_iota_dtypes_shape_int32_2_3_dimension_0` | " |
| 12 | `test_prim_iota_dtypes_shape_int8_2_3_dimension_0` | " |
| 13 | `test_prim_iota_dtypes_shape_uint16_2_3_dimension_0` | " |
| 14 | `test_prim_iota_dtypes_shape_uint32_2_3_dimension_0` | " |
| 15 | `test_prim_iota_dtypes_shape_uint8_2_3_dimension_0` | " |
| 16 | `test_prim_random_randint_shape_int16_` | `random.randint(key(42), shape, -5, max, dtype)` |
| 17 | `test_prim_random_randint_shape_int16_32_` | " |
| 18 | `test_prim_random_randint_shape_int16_5_4_` | " |
| 19 | `test_prim_random_randint_shape_int32_` | " |
| 20 | `test_prim_random_randint_shape_int32_32_` | " |
| 21 | `test_prim_random_randint_shape_int32_5_4_` | " |
| 22 | `test_prim_random_randint_shape_int8_` | " |
| 23 | `test_prim_random_randint_shape_int8_32_` | " |
| 24 | `test_prim_random_randint_shape_int8_5_4_` | " |
| 25 | `test_prim_random_split_` | `random.split(random.key(42), 2)` |
| 26 | `test_prim_random_uniform_shape_bfloat16_` | `random.uniform(key(42), shape, dtype)` |
| 27 | `test_prim_random_uniform_shape_bfloat16_32_` | " |
| 28 | `test_prim_random_uniform_shape_bfloat16_5_4_` | " |
| 29 | `test_prim_random_uniform_shape_float16_` | " |
| 30 | `test_prim_random_uniform_shape_float16_32_` | " |
| 31 | `test_prim_random_uniform_shape_float16_5_4_` | " |
| 32 | `test_prim_random_uniform_shape_float32_` | " |
| 33 | `test_prim_random_uniform_shape_float32_32_` | " |
| 34 | `test_prim_random_uniform_shape_float32_5_4_` | " |
| 35 | `test_prim_sign_special_0_dtype_bfloat16` | `lax.sign_p.bind(StaticArg(np.zeros((2,2))))` |
| 36 | `test_prim_sign_special_0_dtype_complex64` | " |
| 37 | `test_prim_sign_special_0_dtype_float16` | " |
| 38 | `test_prim_sign_special_0_dtype_float32` | " |
| 39 | `test_prim_sign_special_0_dtype_int16` | " |
| 40 | `test_prim_sign_special_0_dtype_int32` | " |
| 41 | `test_prim_sign_special_0_dtype_int8` | " |
| 42 | `test_prim_sign_special_0_dtype_uint16` | " |
| 43 | `test_prim_sign_special_0_dtype_uint32` | " |
| 44 | `test_prim_sign_special_0_dtype_uint8` | " |

### B2. `export_test.py::JaxExportTest` (5)

* `test_multi_platform_and_poly` — exports a symbolic-shape function for
  `("cpu","tpu")`, calls it, then re-exports the call. Fails with
  `ValueError: Function '<lambda>' was exported for platforms '('cpu', 'tpu')'
  but it is used on '('metal',)'`.
* `test_multi_platform_nested_inside_single_platform_export` — exports
  `_testing_multi_platform_func` for `("cpu","tpu","cuda","rocm")` and then
  serialises a call to it for the current platform. Same `ValueError`, naming
  the four-platform tuple.
* `test_multi_platform_mlir_lower_fun_with_platform_specific_primitives` —
  primitives with per-platform lowering rules routed through `mlir.lower_fun`
  (the Pallas pattern). Fails on the test's **own** expectation table:
  `expected = x * np.float32(dict(cpu=2, gpu=3, tpu=4)[jtu.device_under_test()])`
  → `KeyError: 'metal'`.
* `test_ordered_effects_multi_platform_and_poly_v_9`,
  `test_ordered_effects_multi_platform_and_poly_v_10` — ordered effects +
  `platform_index` + symbolic shapes, exported for
  `("cpu","tpu")` at calling-convention versions 9 and 10. Both:
  `ValueError: Function 'f_jax' was exported for platforms '('cpu', 'tpu')'
  but it is used on '('metal',)'`. (The token plumbing itself works — P12's
  ordered-effect support cleared `jaxpr_effects_test` 15 → 1.)

---

# C. `[async-collectives-unimplemented]` — 5 rows **[VERIFY]**

**Mechanism.** `jax.experimental.parallel`'s `psum_start(...).done()` family
lowers to `stablehlo.async_start` / `async_done` wrapping a collective. Both
stacks decline the wrapper:

* native: `UNIMPLEMENTED: metaljax-native: op stablehlo.async_start (it carries a region)`
* Stage 1: `UnsupportedOpError: op 'stablehlo.async_start' not implemented by metaljax`

The file is **0 passed / 5 failed / 7 skipped**: the other 7 (the `test_lower_*`
ones, which only inspect the emitted StableHLO text) are all
`@jtu.with_explicit_mesh((2,), ('i',))` and skip for want of a second device.
The five that run size their mesh by `jax.device_count()`, so they execute on
one device — and every one of them dies on the same op.

* `async_collectives_test.py::AsyncCollectivesTest::test_async_all_gather` —
  `all_gather_start(...).done()` under `shard_map` matches the sync `all_gather`.
* `…::test_async_all_to_all` — same for `all_to_all_start`.
* `…::test_async_ppermute` — same for the permute start/done pair.
* `…::test_async_psum` — `psum_start(x,'i').done()` equals `lax.psum`, with a
  matmul in the program to give the async collective something to overlap.
* `…::test_async_psum_scatter` — same for `psum_scatter_start`.

**Whitelist rationale on file: none.** These are `cpu_pass` (CPU runs them with
one device), they are not mentioned in `notes/jax-test-suite-2026-07.md`, not in
`cpp-p12-14-parity.md`, and not in the README's itemised list. They were
absorbed into the "single-device collectives" family without being named. On one
device the async pair is trivially the synchronous collective plus an identity —
i.e. this looks implementable in the same way P12's collectives were, and the
5 rows are a real feature gap rather than a platform constraint. **Needs Oleg's
explicit ruling.**

**[FIXED] 5 -> 3.** Oleg's verdict was to emulate async with sync, and that is
what `Lowering::LowerAsyncStart` / `LowerAsyncDone` do: the `async_start`
region is INLINED where the start sits (its collective is one of
`LowerCollective`'s single-device arms, so it is an alias or a constant), the
`!stablehlo.future` never gets a slot — the lowering remembers which slots it
stands for, in `futures_` — and `async_done` aliases its results back. The
order is the block's order, which is what XLA's own erasure of a single-device
async pair would leave. A start whose region holds something else declines from
inside, naming that op. `execute_test` grew an `async collectives (start/done)`
module (all-reduce and all-gather pairs around a dot, against jax-CPU).

`test_async_ppermute` and `test_async_all_to_all` now pass. The other three
compute correctly and then fail on a different assertion: they read
`compiled().as_text()` for both the sync and the async form and require that if
the SYNC text names `all-gather(` / `all-reduce(` / `reduce-scatter(`, the ASYNC
one does too. Ours does not, because `GetHloModules` hands back the program as
GIVEN (class E): the sync form keeps its plain collective and the async form its
`all-gather-start(` / `-done(` pair, where an optimizing backend would have
rewritten or erased both. They belong with the inline row: what they need is an
HLO optimization stage, not an async collective.

---

# D. `[jax-side-allowlist]` (callbacks and `check`) — 4 rows

**Mechanism.** jax itself hard-codes which platforms may use these paths
(`callback.py`, `buffer_callback.py`: cpu/cuda/rocm/tpu). A plugin cannot join
the list. Approved category: README "*Ordered-effect residue* (~3):
`buffer_callback` and `emit_python_callback` are rejected by jax-side platform
allowlists … not reachable from a plugin; verified passing on CPU because cpu is
inside those hard-coded lists. Ordered `debug.print`/`io_callback` work."

* `buffer_callback_test.py::BufferCallbackTest::test_side_effect` — a
  `buffer_callback` with `has_side_effect=True` must run its Python callback.
  `ValueError: `buffer_callback` not supported on metal backend.`
* `jaxpr_effects_test.py::EffectOrderingTest::test_can_execute_python_callback` —
  binds the internal `callback_p` twice and asserts the log is `[2., 3.]` after
  `effects_barrier`. `ValueError: `EmitPythonCallback` not supported on metal
  backend.` (This is the *internal* emit path; the public
  `debug.print`/`pure_callback`/`io_callback` all work — P13 wired them through
  `metaljax_callback`, which is why the rest of the file passes 62/1.)
* `checkify_test.py::AssertPrimitiveTests::test_assert_primitive_lowering` —
  asserts that lowering a `checkify.check(False, "hi")` inside `jit` raises
  `ValueError: Cannot abstractly evaluate…`. We raise a *different* error first:
  `NotImplementedError: MLIR translation rule for primitive 'check' not found
  for platform metal`. Same on Stage 1.
* `checkify_test.py::AssertPrimitiveTests::test_debug_check_noop` — same
  primitive, asserting `debug_check` is a no-op; same `NotImplementedError`.

The `checkify` pair is named in `cpp-p12-14-parity.md` as part of the 24
whitelist rows the family slice keeps. The error is a shape mismatch on a
*negative* test — nothing computes wrongly.

---

# E. `[PJRT-surface]` — 8 rows

Optional PJRT extensions and debugging surfaces the plugin does not implement.
Approved category: README "*`test_dce_sink_prevents_xla_dce`*: needs
optimized-HLO text retrieval (`PJRT_Executable_OptimizedProgram`), a debugging
surface we have not implemented"; release review "OptimizedProgram: wontfix (no
optimized HLO exists in our architecture; jax hands plugins unoptimized
StableHLO)"; and the "PJRT surface ~40" bucket of the README's under-review list.

* `aot_test.py::JaxAotTest::test_get_topology_from_devices` **[VERIFY]** —
  builds a PJRT topology for the current platform and asserts
  `topo.platform_version` matches. The test *wants* to skip when topology is
  unsupported: it catches `(ValueError, NotImplementedError)` and asserts the
  message contains `'topology_name is not specified'` **or**
  `'topology not implemented'`. Ours says
  `UNIMPLEMENTED: metaljax: ahead-of-time topology compilation is not supported.`
  → `AssertionError: assert ('topology_name is not specified' in '…' or
  'topology not implemented' in '…')`. The decline is right; only the wording
  keeps it from being a SKIP.
* `aot_test.py::JaxAotTest::test_topology_jit_serialize` **[VERIFY]** — same
  skip-detector, then AOT-compiles against a topology. Identical assertion.
* `api_test.py::APITest::test_inline_optimized_hlo` — checks `jax.Inline` modes
  by reading `.lower(...).compile().as_text()`. We return no optimized HLO text,
  so `get_hlo(...)` is `None`:
  `TypeError: argument of type 'NoneType' is not a container or iterable`
  (identical on Stage 1).
* `lax_test.py::LaxTest::test_dce_sink_prevents_xla_dce` **[VERIFY]** — checks
  that `lax.dce_sink(y, prevent_mlir_dce=True)` survives into the compiled HLO
  text (and that the default form is DCE'd). **Native fails earlier and
  differently from the historical disposition:**
  `UNIMPLEMENTED: metaljax-native: custom call target 'dce_sink'` — the program
  no longer compiles at all. Stage 1 compiles it and fails on the missing text
  (`TypeError: … 'NoneType' …`), which is the failure the README describes. The
  verdict is unchanged (no optimized-HLO text ⇒ unpassable either way), but the
  native decline is a small **functional regression vs Stage 1**: a user program
  containing `lax.dce_sink` compiles on the Python engine and does not on the
  native one. Worth a decision (accept, or lower `dce_sink` to a no-op).
* `xla_transform_test.py::XlaTransformTest::test_sin_to_cos0` (PRE_SCHEDULER),
  `…::test_sin_to_cos1` (POST_SCHEDULER) — register a user HLO pass that
  rewrites `sin`→`cos` and check the numerics change.
  `RuntimeError: Cannot register XLA transform 'sin_to_cos_*_test': PJRT plugin
  does not support the XlaTransform extension.`
* `xla_transform_test.py::XlaTransformTest::test_clear_transform` — registers,
  verifies, clears, re-verifies. Same `RuntimeError`.
* `xla_transform_test.py::XlaTransformE2ETest::test_simple_transform_preserves_donation_and_aliasing` —
  the same extension, checking donation/aliasing survive a transform. Same
  `RuntimeError` at registration.

The `XlaTransform` extension is an XLA-compiler pass-pipeline hook: there is no
HLO pipeline in this backend to insert a pass into (the release review already
noted that adopting XLA's pass pipeline as a library is a Stage-2 idea; that is
the only route that would make these four implementable).

**[FIXED] 8 -> 5, in three pieces.**

* **aot topology (2) -> SKIP.** The decline now reads `metaljax: topology not
  implemented (ahead-of-time compilation against a described machine; metaljax
  compiles for the attached GPU)`, which is the substring both tests' own
  skip-detector looks for. `aot_test` is 21 passed / 7 skipped, 0 failed.
* **`dce_sink` (1) -> PASS, and the Stage 1 regression is gone.**
  `lax.dce_sink(...)` lowers to nothing again: the marker keeps its operand
  alive through a compiler's DCE, this tape has no DCE (it lowers every op the
  block holds), so the marker's job is already done. The test's other half —
  `assertNotIn("add", hlo)` for the default form — passes because XLA's own
  parse, which runs before `CompileAndLoad`, has already dropped the dead add.
* **`OptimizedProgram` (1) -> implemented, the row still fails.**
  `MetalExecutable::GetHloModules` now answers: the compile keeps the StableHLO
  it was handed as MLIR bytecode (written once, cheap) and converts it to HLO
  on demand, cached, degrading to `UNIMPLEMENTED` — the code jax's `as_text`
  turns into `None` — if the module has no HLO form. That is what unblocked two
  `async_collectives_test` rows and it is the surface the planned XLA
  optimization layer will publish through. What comes back is the program this
  executable RUNS, **unoptimized**: nothing here optimizes at the HLO level (the
  recognizers and the compile decisions work on the tape, below this
  representation). So `test_inline_optimized_hlo` still fails — it asserts XLA
  inlined an `inlineable = "xla_early"` call and preserved an `xla_late` one,
  which is a property of XLA's pass pipeline, and running `CallInliner` here
  purely to satisfy it would claim an optimization this backend does not
  perform.
* **`XlaTransform` (4) — NOT attempted in this batch.** The route is now clear
  and cheap to describe: `pjrt::CreateXlaTransformExtension()` in
  `xla/pjrt/c/pjrt_c_api_xla_transform_internal.h` is a complete implementation
  of the extension (registry included), so the plugin only has to add it to its
  extension chain and, in `CompileAndLoad`, call
  `xla::ApplyXlaTransformsToModule` on the converted HLO **when a transform is
  registered** (`GetHloXlaTransforms(...)` is empty otherwise, so a normal
  compile pays nothing) and lower the result back through
  `ConvertHloToStablehlo`. That would take `test_sin_to_cos{0,1}` and
  `test_clear_transform`; `XlaTransformE2ETest` also needs
  `GetCompiledMemoryStats`, which this executable does not answer.

---

# F. `[memory-spaces]` — 4 rows

**Mechanism.** `MetalDevice` exposes exactly one memory space, `device`. jax's
host/pinned memory kinds have nothing to map to.

* `export_test.py::JaxExportTest::test_memory_space_from_arg` — exports a
  function whose *input* is on `pinned_host` and checks the memory space
  survives into `in_avals`/`in_shardings_jax`.
  `ValueError: Could not find memory addressable by device Apple GPU. Device
  Apple GPU can address the following memory kinds: device. Got memory kind:
  pinned_host` — raised while building the sharding, before export.
* `export_test.py::JaxExportTest::test_memory_space_from_out_shardings` — same,
  with the host memory space coming from `out_shardings`. Identical `ValueError`.
* `memories_test.py::StreamAnnotationTest::test_compute_on2_out_mem_space`
  **[VERIFY]** — `@compute_on2(compute_type='device_host',
  out_memory_spaces=jax.memory.Space.Host)` over `x * 2.0`; asserts values and
  `out.sharding.memory_kind == 'pinned_host'`. Values are right; the assertion
  fails `- device / + pinned_host`.
* `memories_test.py::StreamAnnotationTest::test_compute_on2_out_mem_space_tuple`
  **[VERIFY]** — the tuple form `(Host, Device)`; the first output's
  `memory_kind` is again `device`.

The two `memories_test` rows are flagged because there the annotation is
**silently ignored** rather than refused: the program runs and returns correct
numbers with a memory placement the caller did not ask for. Everywhere else this
backend's policy is "decline loudly by name". Same on Stage 1 (identical
`- device / + pinned_host`), so this is a long-standing behaviour, not a P12-14
change — but it has no written disposition either.

**[FIXED] 4 -> 0.** Of the three shapes this could take — a metadata field per
buffer, a second memory SPACE, or keeping the silence where it is provably
inert — the middle one is Oleg's and is what landed, because it adds **no
per-tensor state at all**: the client exposes a second `MetalMemorySpace` of
kind `pinned_host` (`kind_id` 1) beside `device`, and a buffer points at
whichever space it was asked for through the `memory_space_` pointer it already
carries. Apple silicon's memory is unified, so both spaces name one physical
pool and the placement costs nothing; what changes is that the plugin now SAYS
which one a buffer is on. `mhlo.memory_kind` on main's arguments and results is
read at lowering into `ValueSpec::host_memory` (compile-time, one bit per
boundary value, not per buffer), `GetParameterMemoryKinds` /
`GetOutputMemoryKinds` answer from it, and `RunOnce` builds the result buffer on
the space the module asked for. Any OTHER kind — XLA also spells
`unpinned_host` — declines by name rather than being ignored. Silence was not
an option for any of the four: two of them assert `sharding.memory_kind`
directly and two need `memory_space_by_kind('pinned_host')` to exist before
export can even build the sharding.

---

# G. `[layout-column-major]` — 2 rows

**Mechanism.** `jax.Layout(major_to_minor=(1,0))` at rank 2 is *column*-major
(`mhlo.layout_mode = "{0,1}"` in minor-to-major spelling). MLX storage is dense
row-major, so P12-14 added a guard that reads `mhlo.layout_mode` on main's
parameters and results and declines anything else **by name**. The note records
this as a correctness guard, not only a diagnostic: without it, jax would not
compare, and the engine would compute on row-major bytes the caller believes are
transposed.

* `layout_test.py::LayoutTest::test_device_put_user_concrete_layout` (shared) —
  `jax.device_put(np_inp, Format(Layout(major_to_minor=(1,0)), sharding))` and
  asserts the resulting array reports that layout.
  Native: `UNIMPLEMENTED: metaljax-native: a result layout of {0,1} (metaljax is
  row-major: {1,0})`. **Stage 1 fails differently and worse**:
  `AssertionError: Tuples differ: (0, 1) != (1, 0)` — it accepts the request
  silently and reports the wrong layout back.
* `layout_test.py::LayoutTest::test_in_layouts_jit_jnp_input` (**native-only**) —
  `jax.jit(f, in_shardings=Format(Layout(major_to_minor=(1,0)), …))` called with
  jnp and numpy inputs.
  `UNIMPLEMENTED: metaljax-native: a parameter layout of {0,1} (metaljax is
  row-major: {1,0})`. It became reachable only because P12-14 implemented
  `GetDefaultLayout` (which unblocked 10 tests in three other files, and let
  this one get far enough to ask for a layout we do not have). Net on the file:
  1 failure before, 2 after, against 10 tests gained elsewhere — recorded in
  `cpp-p12-14-parity.md`.

Not flagged: the trade is documented, quantified, and the new row fails loudly.

---

# H. `[better-than-reference]` — 5 rows ("we fail by succeeding")

Approved category: README "*Better-than-reference cases* (4): shape-polymorphic
`jnp.insert` / `jnp.nonzero` — the harness asserts `NotImplementedError` because
jax's CPU path cannot lower them; ours can… We fail these tests by succeeding",
plus the bf16/f16 linalg entry ("**including bfloat16/float16 inputs, which
jax's CPU backend itself rejects**").

* `linalg_test.py::NumpyLinalgTest::testQrInvalidDtypeCPU` — regression test for
  jax #10530: `jnp.linalg.qr` on **float16** must raise (`NotImplementedError,
  "Unsupported dtype float16"` on CPU; `Exception, "Unsupported dtype"`
  elsewhere). Our host LAPACK handlers (the `geqrf`/`orgqr` FFI targets served
  in-process) take f16 by upcasting to f32, so nothing raises:
  `AssertionError: Exception not raised`. Identical on Stage 1. Checked that the
  answer is real and not garbage — f16 5×6 input, factors returned in f32:
  `max|QR − A| = 1.3e-3`, `max|QᵀQ − I| = 1.3e-3`, i.e. f16 input precision.
* `shape_poly_test.py::ShapePolyHarnessesTest::test_harness_jnp_insert_insert_constant`
* `…::test_harness_jnp_insert_insert_poly`
* `…::test_harness_jnp_nonzero_size_constant`
* `…::test_harness_jnp_nonzero_size_poly`

  All four: the harness declares
  `expect_error = (NotImplementedError, "associative scan over axis of
  non-constant size")`, and the test body **skips** the case on `cpu`/`gpu`
  ("native serialization with shape polymorphism not implemented for
  window_reductions on CPU and GPU"). `metal` matches neither skip, so the case
  runs, our lowering computes it, and the harness reports
  `AssertionError: NotImplementedError not raised`. P14 re-measured the file at
  2,342 passed / 4 failed — the same four Stage 1 fails, "the family closed
  without a line of code".

---

# I. `[intentional-native]` — 2 rows, stricter than Stage 1 **[VERIFY]**

* `lax_numpy_indexing_test.py::IndexedUpdateTest::testStaticIndexing5` — a
  `sample_product` case; introspected parameters:
  `name='TupleOfIntAndSliceAndIntArray', shape=(3,2,3),
  indexer=(array(2), slice(None), array([0,1,2])), update_shape=[2],
  op=UpdateOps.POW, dtype=complex64, update_dtype=int32, mode='drop'`.
  It checks `x.at[idx].power(y)` against numpy and then under jit.
  Native: `UNIMPLEMENTED: metaljax-native: complex scatter multiply without
  unique indices`.
  Rationale (P10 §2): MLX has no complex scatter, so a complex scatter is done
  by parts; set/add/subtract are componentwise and exact, but **multiply is
  not** — it is rewritten as gather-combine-set, which equals the combiner only
  if no two updates hit one slot. Stage 1 assumes that silently; the native
  lowering *checks* it against the op's `unique_indices` flag and declines
  otherwise. P10 records it as "intentional (stricter than Stage 1)".
  **[VERIFY]**: the guard keys off the flag, not the index structure. Here the
  index array is `[0,1,2]` — unique in fact — and Stage 1's answer is
  verified against numpy by this very test. So the cost is real: any
  `.at[...].multiply()` / `.power()` on a **complex** array without a uniqueness
  promise now fails to compile where it used to give the right answer. Accept, or
  refine the guard (e.g. accept static/`iota`-derived indices)?
* `lax_test.py::LaxTest::testConvGeneralDilated0D2` — `lax.conv_general_dilated`
  with **zero spatial dimensions** (`("NC","OI","NC")`, empty strides/padding),
  over `lax_test_util.all_dtypes`; this index is the complex one.
  Native: `UNIMPLEMENTED: metaljax-native: conv: complex with no spatial
  dimensions`.
  Rationale (P7): the 0-spatial arm is a matmul, and the Python handler runs its
  operands through `astype(mx.float32)`, which **drops the imaginary part**; a
  faithful port would be silently wrong, so phase 2 refuses the combination.
  **[VERIFY] — and this one is a finding, not a doubt.** The test passes on
  Stage 1 only because `_CompileAndCheck` compares metal against metal. Measured
  directly today (complex64, `(2,3)·(2,3)`):

  | | result |
  |---|---|
  | Stage 1 metal | `[ 5.+0.j, 14.+0.j, 14.+0.j, 50.+0.j]` |
  | jax CPU | `[15.-5.j, 42.-14.j, 42.-14.j, 150.-50.j]` |

  i.e. released Stage 1 (v0.11.0, on TestPyPI) computes complex 0-spatial
  convolutions **wrong, silently** — real·real, imaginary discarded. The native
  refusal is strictly better and this test failing is the correct outcome.
  Flagged so Oleg sees the Stage-1 bug, which is *not* covered by any existing
  note beyond P7's one-line remark.

**[FIXED] 2 -> 0, both gates lifted per Oleg.**

* **complex scatter multiply.** Asking `unique_indices` was the wrong question
  twice over: jax sets it FALSE for every plain `.at[i].multiply(u)`, literal
  indices included, and what actually matters is whether two updates land on
  one slot — which the lowering cannot read off a computed index array. So a
  missing promise no longer refuses and no longer guesses: it takes the
  SEQUENTIAL apply arm over the op's own multiply body (`method_code = 8`, the
  arm `.at[].apply()` already uses), one update at a time in XLA's order, which
  is exact whether or not the indices repeat. It keeps that arm's cap — a
  constant number of MLX ops per update — and above it declines by name
  (`complex scatter multiply with duplicates: N updates`). With the promise the
  fast gather-multiply-set rewrite is unchanged. Four new `execute_test` rows,
  including the duplicate-index one the rewrite would get wrong.
* **complex 0-spatial convolution.** Implemented as four real matmuls, exactly
  as the spatial arm is four real convolutions; the matmul arm now carries the
  same `mode` field the spatial one does. Measured against jax-CPU on the
  report's own case: both now `[15.-5.j, 42.-14.j, 42.-14.j, 150.-50.j]`.
  **Stage 1 is still wrong here and is frozen** — `plugin/` keeps the f32 cast
  that drops the imaginary part, and `lax_test::testConvGeneralDilated0D2`
  cannot see it because `_CompileAndCheck` compares metal against metal. The
  two new `execute_test` rows compare against CPU, which is the whole
  difference.

---

# J. `[skew/harness]` — 6 rows

Backend-independent: these fail for environment reasons, re-verified on CPU
today.

* `compilation_cache_test.py::CompilationCacheDisabledTest::test_jit` — asserts
  that with the compilation cache disabled, `jit(lambda x: x*x)(1)` writes no
  cache items. Fails only in a whole-file run:
  `AssertionError: Test changed the compilation cache object: before test it was
  <InMemoryCache …>, now it is None`.
  Chain: 33 tests in the file skip on metal with jax's own message
  **"serialize executable only works on tpu,gpu,cpu"** (another hard-coded
  allowlist, this time on executable serialisation); each skip leaves jax's
  `assert_global_configs_unchanged` teardown guard tripping (those are the 33
  "errors" the suite reports for this file), and the cascade leaves the cache
  object `None` for this test's guard. Whole file on `JAX_PLATFORMS=cpu`:
  **35 passed, 1 skipped**, no errors.

  **Diagnosed (2026-08-11), jax-side, stays whitelisted.** The allowlist is in
  the TEST — `compilation_cache_test.py:145`, `supported_platforms = ["tpu",
  "gpu", "cpu"]`, then `raise SkipTest(...)` — and it is raised from `setUp`,
  *after* `CompilationCacheTestCase.setUp` has already run
  `cc.reset_cache(); cc._cache = InMemoryCache()`. unittest does not call
  `tearDown` when `setUp` raises, so the matching `cc.reset_cache()` never runs
  and every one of the 33 skips leaves the module-level cache object where it
  was; jtu's config guard reports that as an error, and by the time
  `CompilationCacheDisabledTest::test_jit` compares the object before and after
  itself, it is `None`. Nothing here is about compilation caching on this
  backend, and nothing a plugin can register joins that list
  (`jtu.test_device_matches` compares `device_under_test()`, which is `metal`).
* `logging_test.py::LoggingTest::test_subprocess_stderr_info_logging` — runs a
  subprocess with `JAX_LOGGING_LEVEL=INFO` and asserts stderr contains `"INFO"`
  and not `"DEBUG"`. Fails: `AssertionError: 'INFO' not found in 'I0811 …
  pjrt_api.cc:119] GetPjrtApi was found for metal …'` — in this environment no
  record carries the literal string `INFO`: jax's python logs come out as
  `DEBUG:2026-…:jax._src.…` and the C++ ones as `I0811 …`. **Reproduced on
  `JAX_PLATFORMS=cpu` today, identically.**
* `logging_test.py::LoggingTest::test_subprocess_stderr_debug_logging` — same
  with `JAX_LOGGING_LEVEL=DEBUG`, asserting both `INFO` and `DEBUG` appear.
  Same missing-`INFO` assertion; **also reproduced on CPU**.

  **Is the absl log format ours to set up? No** (re-measured 2026-08-11). The
  subprocess the test runs emits FOUR stderr lines under
  `JAX_LOGGING_LEVEL=INFO`, and on `JAX_PLATFORMS=cpu` they are:
  `I0811 … pjrt_api.cc:119] GetPjrtApi was found …`,
  `I0811 … profiler_utils.cc:40] PJRT_Profiler_Extension is not registered …`,
  and two `pjrt_client.cc` device lines — all four from XLA's C++ absl logger,
  whose severity glyph is `I`, never the word `INFO`. jax's own Python loggers
  emit nothing at INFO level in this environment at all, which is what the
  assertion is really about; the format they would use (`INFO:2026-…:jax._src…`)
  is set by `jax._src.logging_config`, not by a backend. A plugin has no hook
  into either, and both tests fail identically with no metal plugin in the
  picture.
* `profiler_session_test.py::ProfilerSessionTest::test_programmatic_profiling_without_session_id`
* `…::test_programmatic_profiling_with_empty_session_id`
* `…::test_programmatic_profiling_with_custom_session_id`

  All three: `jax.profiler.trace(tmpdir, …)` around a `pmap` and then a check for
  one `*.xplane.pb` under `plugins/profile/<session>`. They die on the first line,
  `self.create_tempdir()`:
  `absl.flags._exceptions.UnparsedFlagAccessError: Trying to access flag
  --test_tmpdir before flags were parsed.` — an absltest-under-pytest artifact.
  **Verified: the file fails identically on `JAX_PLATFORMS=cpu` when run alone**
  (3 failed), and passes when some other module in the same process has called
  `jax.config.parse_flags_with_absl()`. The suite runs one file per process, so
  it always fails. Nothing profiler-specific is being measured here.

---

# K. `[denormals]` — 1 row

* `lax_numpy_operators_test.py::JaxNumpyOperatorTests::testSpacingSubnormals0` —
  builds the 11 float32 values nearest zero (5 negative denormals, 0, 5 positive)
  with `np.nextafter` and compares `jnp.spacing` against `np.spacing` at `tol=0`.
  `Mismatched elements: 11 / 11`; ours are `∓0.0` where numpy gives
  `∓9.1835e-41`. The GPU flushes subnormals to zero, so the difference of
  adjacent denormals is zero.
  Approved category: README "**Denormals flush to zero** on the GPU (hardware
  behavior); tests asserting subnormal outputs (e.g. `jnp.spacing`) differ from
  CPU." Identical on Stage 1.

---

# L. `[donation-contract]` — 1 row

* `api_test.py::JitTest::test_double_donation` — passes the *same* array in a
  donated and a non-donated position (`jit(add, donate_argnums=(0,))(x, x)`) and
  asserts a `RuntimeError`. We raise nothing: `AssertionError: RuntimeError not
  raised`.
  Rationale (`cpp-p12-14-parity.md`, reviewer-scrutiny item 5): "Donation deletes
  the buffer even when the caller passed it twice (one position donated, one
  not). XLA raises on that; here the donated position wins and the other
  reference is invalidated. `test_double_donation` is the test, it is on the
  shared whitelist (Stage 1 does not raise either), and the caller has broken
  the contract in any case." Donation itself works end-to-end (P13; 21 jax tests
  unskipped at 0.11.0, `tests/test_donation` 3 → 0 on the native plugin).

**[FIXED] 1 → 0.** `RunOnce` calls XLA's own `TestBufferDonationClashes`
(`xla/pjrt/utils.h`) while it walks the arguments, so the three messages are
word for word the ones a jax user sees on cpu/cuda/tpu — `f(donate(a), a)`,
`f(a, donate(a))`, `f(donate(a), donate(a))` each have their own — and
`non_donatable_input_indices` takes the promise back for one call exactly as it
does for the deletion. The refused call consumes nothing: the check runs while
the arguments are being read, before the tape does anything. `execute_test`
grew a `double donation raises` contract that asserts both halves (it raises,
and the buffer survives).

---

# M. `[token-representation]` — 1 row

* `api_test.py::JitTest::test_print_token_buffer_error` — asserts that reading
  `jax.lax.create_token()._buf._value` raises
  `RuntimeError: Cannot convert a token-shape buffer to a numpy array.`
  We raise nothing (`AssertionError: RuntimeError not raised`) because a token
  **is** an ordinary empty bool array here: `dtypes.token_value` on Stage 1 and
  `PRED[0]` in the native lowering (`CheckValue` accepts `!stablehlo.token`,
  `Dims` answers `[0]`). That representation is not a convenience — jax's own
  `dispatch.RuntimeTokenSet` hands the runtime `np.zeros(0, np.bool_)` for a
  token argument and expects the same back — and it is what makes ordered
  effects work at all (P12 §2; `jaxpr_effects_test` 15 → 1,
  `debugging_primitives_test` 32 → 0). Converting the buffer is therefore
  legitimate on this backend; the test asserts a property of XLA's token
  representation.

**Why jax "asserts tokens can't be arrays", precisely** — it is not jax, it is
jaxlib, and it keys off the buffer's DTYPE. `PyHostValue::AsNumPyArray`
(`jax/jaxlib/py_array.cc:1832`) refuses when `ifrt_array->dtype().kind() ==
ifrt::DType::kToken`, with the comment "The only `jax.Array` with token-shape
buffer is the one wrapped by `jax.core.Token`. Since it is an internal
implementation detail, we don't support converting it to a numpy array." So the
test asserts that the BACKEND hands jax a buffer whose element type is XLA's
`TOKEN` — the empty bool array could never satisfy it, whatever the engine did,
because the refusal never looks at the shape.

**[FIXED] 1 → 0, and the empty-bool representation survives it.** Only main's
boundary SPEC changed: a token parameter or result is now
`ValueSpec{xla::TOKEN, …}` and `ValueSpec::shape()` answers
`ShapeUtil::MakeTokenShape()` for it (a token shape is not an array shape —
`MakeShape(TOKEN, {0})` is invalid and was the first thing that broke). The
device side is untouched: the array is still `mx::bool_` of extent 0, which is
what `RunOnce` checks against and what jax's `RuntimeTokenSet` seeds a token
with (`np.zeros(0, np.bool_)`), so a token argument still arrives as a plain
bool buffer and is still accepted. Ordered effects were re-measured on the
change: `jaxpr_effects_test` + `debugging_primitives_test` + `api_test::JitTest`
= 237 passed, 1 failed (the class D `EmitPythonCallback` allowlist row, which is
not this).

---

# N. `[numerics/FD-reference]` — 1 row

* `lax_control_flow_test.py::LaxControlFlowTest::testScanGrad_jit_scan=False_jit_f=False_impl=unroll0` —
  compares `jax.grad` of a 5-step `scan` (body: `c = sin(c * (Σsin a + Σsin c +
  Σsin d))`) against an unrolled reference, then `jtu.check_grads` down to a
  finite-difference check. The failing assertion is the *second-order* one:

  ```
  Not equal to tolerance rtol=0.5, atol=0.001   VJP of VJP cotangent projection
   ACTUAL: array(0.001592)   DESIRED: array(-0.000398, dtype=float32)
  ```

  Approved category: README "*`testSincInfinities`, FD-reference gradient
  corners*: fail on the CPU backend too, or the test's finite-difference
  reference is numerically meaningless in f32 (documented with numbers)";
  release review: "CPU-also-fails: allowed by default. FD-reference scanGrad
  singleton accepted (number table above)." The quantities compared are ~1e-3
  and ~4e-4 with a *0.5 relative* tolerance — a sign flip in a second-order
  finite difference of a chaotic-ish composition of sines in f32. Identical
  numbers on Stage 1 (same ACTUAL/DESIRED), so this is arithmetic association
  order, not a control-flow bug.

---

# O. `[sdy-op-unimplemented]` — 1 row **[VERIFY]**

* `shard_alike_test.py::ShardAlikeTest::test_sharding_preserverd_single_device` —
  `shard_alike(x, jnp.arange(8))` with a 1-device mesh; asserts the second output
  inherits the first's sharding.
  Native: `UNIMPLEMENTED: metaljax-native: op sdy.sharding_group with 0 results`.
  Stage 1: `UnsupportedOpError: op 'sdy.sharding_group' not implemented by
  metaljax` on `sdy.sharding_group %arg0 group_id=0 : tensor<8xi32>`.
  The whole file is 0 passed / 1 failed / 13 skipped (the rest need >1 device).

  We already treat `sdy.sharding_constraint` and `sdy.reshard` as identities;
  `sharding_group` is the one member of that family we do not, and it produces
  **no results at all** — on a single device it is pure metadata. No disposition
  is recorded for it anywhere (not in the release review, not in the README).
  Either it is a 3-line identity arm, or it should be written down as declined
  on purpose.

---

# P. `[jax-generic-LU × shard_map]` — 1 row

* `shard_map_test.py::ShardMapTest::test_custom_linear_solve_rep_rules` —
  regression test for jax #20162: `jnp.linalg.solve(a, b)` inside a `shard_map`
  over a 1-device mesh should "not crash".
  Fails at trace/lower time inside jax:
  `TypeError: scan body function carry input and carry output must have equal
  types … loop_carry[1][0] has type int32[1] but the corresponding output carry
  component has type int32[1]{V:i}, so the manual axis types do not match`.

  Cause, confirmed by reading our registration
  (`src/jax_plugins/metal/__init__.py:304-322`): on `metal`, static-shape LU is
  lowered with **jax's own generic python blocked factorisation**
  (`ll._lu_python` through `mlir.lower_fun`), because the alternative — routing
  everything to our host `getrf` — rounds differently, "and at bfloat16 that
  difference is visible, so the host path must not take work the device path can
  do". That generic path traces a `scan`, and jax's `shard_map` vma checker
  rejects the resulting carry types. On CPU the same `lu_p` is a LAPACK custom
  call, so no scan exists and the test passes — this is a jax-side
  incompatibility that only non-cpu/gpu/tpu platforms reach. Listed as a shared
  whitelist row in `cpp-p12-14-parity.md`. Not our arithmetic, and the
  alternative costs bf16 accuracy.

---

# Q. `[tracked-open]` — 2 rows — **NOT benign**

`test_bcoo_spdot_general` checks `sparse.bcoo_dot_general` against dense
`lax.dot_general` (values, batching, and forward-mode grads). Both failing rows
are the same shapes and dtype — introspected params: `lhs_shape=(5,7)`,
`rhs_shape=(7,)`, `dimension_numbers=(([1],[0]),([],[]))`, `n_batch=0`,
`dtype=float32`; row 0 is `swap=False`, row 6 is `swap=True` (operands and
dimension numbers reversed). In both, the failing assertion is inside
`_CheckGradsSparse` — the **forward-mode gradient** of the sparse product has
nonzero entries where the dense reference has exact zeros.

* `sparse_bcoo_bcsr_test.py::BCOOTest::test_bcoo_spdot_general0` —
  `Not equal to tolerance rtol=1e-06, atol=1e-06; Mismatched elements: 8 / 90
  (8.89%)`; first mismatches `[0,11]: 1.9702634811401367 (ACTUAL) vs 0.0
  (DESIRED)`, `[1,3]: 1.5024161338806152 vs 0.0`, `[1,5]`, `[1,8]`, `[2,10]`;
  `Max absolute difference among violations: 1.9702635`.
* `sparse_bcoo_bcsr_test.py::BCOOTest::test_bcoo_spdot_general6` —
  `Mismatched elements: 5 / 20 (25%)`; the whole last column is wrong:
  `[0,3]: 0.7336747646331787 vs 0.0`, `[1,3]: 1.5729273557662964 vs 0.0`,
  `[2,3]: 2.1404266357421875 vs 0.0`, `[3,3]: -0.4724152684211731 vs 0.0`,
  `[4,3]: -4.0442891120910645 vs 0.0`.

  **Position-dependent**: both pass when run alone (native *and* Stage 1) and
  fail in a whole-file run (2 failed / 418 passed / 25 skipped, re-measured
  today at 165 s).

  Disposition (verbatim from `notes/data/pinned-0.11.3-failures.txt`, Oleg
  2026-08-05): "**NOT whitelisted-benign** — serious class, wrong values — but
  non-blocking for 0.11.3; scheduled for fix soon after the C++ migration,
  together with the blocked model rows 8/10/12/15. Pre-existing in released
  0.11.0 (approval tree fails the same 283-test-prefix repro identically);
  position-dependent; lottery-classification experiment is the first step. See
  TASKS.md 'position-dependent silent wrongness in sparse'." Not flagged
  [VERIFY] because the ruling is explicit and recent — but it is the one class
  in this report that is a known wrong answer rather than an accepted
  limitation, and the C++ migration it was deferred behind is now landing.

## Diagnosed (2026-08-11): MLX's compiled-kernel FUSION, not this tape

**It is not ours, and it is not the test.** Still 2 failing rows, but they are
no longer unexplained, and the whole chain is reproducible in seconds instead of
in a 283-test prefix.

*The repro is two tests.* Bisecting the file's 445 ids by prefix (30 ok, 31
FAIL) named a single predecessor: `test_bcoo_concatenate5` followed by
`test_bcoo_spdot_general{0,6}` fails in 2.5 s, and either row alone passes. The
data is not what changes — `jtu.JaxTestCase._rng` is seeded from the test's own
NAME, so both rows draw the same matrices whatever ran before them. **The whole
file passes on `JAX_PLATFORMS=cpu`** (420 passed, re-measured), so it is
metal-side.

*Which program.* `METALJAX_VERIFY_COMPILE=1` (new, off by default,
`metal_executable.cc`) runs every executable a SECOND time op by op and compares
the two answers. Exactly one of the ~390 executables in the repro diverges:
`jit_scatter-add`, a one-entry tape,
`(f32[22,5], i32[22,5,2], f32[22,5]) -> f32[22,5]`, an XLA scatter with
`inserted_window_dims = [0,1]`, `index_vector_dim = 2`, no window dims. Its
indices are the full `(i, j)` grid, so nothing is out of bounds and nothing
repeats. 18 of its 110 output elements differ; the compiled answer keeps the
updates at flat positions 0 and 14 and ZEROES every one from 16 on — the shape
of an out-of-bounds mask that is true for everything past element 15, i.e. a
fused elementwise kernel that stopped writing.

*Where it lives.* Four knobs, each a fresh process on the same two tests:

| setting | result |
|---|---|
| default | 2 failed |
| `METALJAX_COMPILE=0` (no tape compiles) | **passes** |
| `MLX_DISABLE_COMPILE=1` | **passes** |
| `METALJAX_MLX_COMPILE_MODE=no_fuse` | **passes** |
| `METALJAX_MLX_COMPILE_MODE=no_simplify` | **passes** |
| `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` at 1 or at 10⁸ | 2 failed |

so it is neither the command-buffer split (the 0.32 corruption this plugin
already pins budgets against) nor the tape: it is MLX's `compile`, and
specifically its kernel FUSION. `METALJAX_MLX_COMPILE_MODE` is new and exists
for exactly this attribution — MLX's `set_compile_mode` has no environment
variable of its own, so the earlier `MLX_COMPILE_MODE=...` probes in this
report's own methodology were silently no-ops.

*What it is NOT.* Two hypotheses were tested to destruction and both are
innocent: the strided VIEW that `index_component` slices out of the coordinate
vector (materializing it changes nothing), and the process's buffer pool (200
allocate-compute-free rounds before the rows do not reproduce it).

*A real hazard found on the way, and fixed.* At the divergence the scatter's
OPERAND argument was `size=110, data_size=1, strides=[0,0]` — a broadcast VIEW
that a previous executable had handed jax as a PJRT buffer: 4 bytes presenting
themselves as 440. Every consumer that reads a buffer as dense memory is
exposed to that, `unsafe_buffer_pointer` included. `RunOnce` now materializes
any output whose `data_size != size` or which is not row-contiguous, after the
eval that makes those flags meaningful (`mx::contiguous`, not `fresh_copy` —
the select `fresh_copy` builds keeps the broadcast's strides, measured). It did
**not** fix these two rows: they fail with fully dense arguments. It is kept on
its own merits, with an `outputs own their bytes` contract in `execute_test`.

*Open, for Oleg.* A minimal MLX-level reproducer (a C++ one — MLX's Python API
exposes no general scatter) is the next step, and then an upstream report; this
is **MLX bug #8** on CLAUDE.md item 20's tally. Until then the only sound
workarounds are global (`no_fuse`, or refusing to compile any tape holding a
scatter), both of which cost far more than these two rows are worth, so the two
stay failing and named.

---

# R. `[registration-artifact]` — 1 row

* `export_back_compat_test.py::CompatTest::test_custom_call_coverage` — walks
  jax's registry of custom-call targets "declared stable" and asserts every one
  has a back-compat test in that file. Only fails in a whole-file run:

  ```
  AssertionError: {'metaljax_eigh', 'metaljax_hessenberg', 'metaljax_callback',
  'metaljax_tridiagonal_solve', 'metaljax_lu', 'metaljax_tridiagonal',
  'metaljax_svd', 'metaljax_triangular_solve', 'metaljax_eig', 'metaljax_schur'}
  has length of 10. : The following custom call targets are declared stable but
  are not covered by any tests
  ```

  These are *our* ten targets, registered by
  `src/jax_plugins/metal/__init__.py` so that linalg and callbacks lower on
  `metal` at all (P9, P13). The test is jax's internal hygiene check over its own
  corpus; a third-party plugin registering targets necessarily trips it, and the
  only "fix" would be to stop registering platform lowerings. Rest of the file:
  3 passed / 99 skipped. Not in the 0.11.0-era annotations because it is a
  consequence of the lowerings added during the parity campaign; mechanism
  confirmed here first-hand.

---

## Appendix: the 12 native-only rows, and the 7 the native stack fixes

Native-only (all pass on Stage 1, verified today — 12 passed in one run):

| row | class |
|---|---|
| `dtypes_test::TestPromotionTables::testFloatArrayCreation3` | A2 [VERIFY] |
| `lax_numpy_test::LaxBackedNumpyTests::testArrayExplicitDtypes` | A2 [VERIFY] |
| `pickle_test::PickleTest::testPickleX64` | A2 [VERIFY] |
| `x64_context_test::X64ContextTests::test_make_array0`, `…1` | A2 [VERIFY] |
| `x64_context_test::X64ContextTests::test_correctly_capture_default0…3` | A2 [VERIFY] |
| `lax_numpy_indexing_test::IndexedUpdateTest::testStaticIndexing5` | I [VERIFY] |
| `lax_test::LaxTest::testConvGeneralDilated0D2` | I [VERIFY] (native right, Stage 1 silently wrong) |
| `layout_test::LayoutTest::test_in_layouts_jit_jnp_input` | G |

Fixed by the native stack (fail on Stage 1, pass natively):
`core_test::test_reference_cycles`, `…_jit`;
`debugging_primitives_test::test_can_print_inside_while_loop_cond{0,1}`,
`…test_can_print_in_batched_while_cond{0,1}`; `lax_test::testScatter1`.
