# Stage 2 P11: the emulated grids, reduce_precision, and the scatter tail

Follows [`cpp-p10-scatter-sort.md`](cpp-p10-scatter-sort.md).  Three families
P8's census put next on the ladder — **126** extended/emulated-dtype tests,
**66** `reduce_precision`, **30** in the scatter tail P10 left behind — plus
the `tests/`-on-native file that has been red since the fork
(`test_subbyte_float.py`).  Everything lands in the forked runtime
(`plugin-native/runtime/`) and the plugin's lowering; `native/` and `src/` are
untouched.

| file | change | what |
|---|---:|---|
| `plugin-native/runtime/program.h` | +46/-8 | `kReducePrecision`, `kSelectAndScatter`, `Entry::regrid`, the attribute layouts |
| `plugin-native/runtime/dtypes.cc` | +171/-15 | the 13 emulated element types and `quantize_emulated` |
| `plugin-native/runtime/program.cc` | +16/-3 | the ONE site that spends a regrid |
| `plugin-native/runtime/ops_elementwise.cc` | +73/-7 | `reduce_precision`'s four arms; convert onto a grid |
| `plugin-native/runtime/ops_reduce.cc` | +128 | `select_and_scatter` |
| `plugin-native/runtime/ops_index.cc` | +81 | the scatter apply body, both arms |
| `plugin-native/runtime/ops_shape.cc` | +59/-12 | `bitcast_convert`'s four 4-bit arms |
| `plugin-native/runtime/config.cc` | +14/-4 | two op names |
| `plugin-native/metal/metal_dtypes.{h,cc}` | +179/-6 | the wire table: MLIR names, PJRT enums, the encode/decode pair |
| `plugin-native/metal/metal_client.cc` | +39 | ingest decodes |
| `plugin-native/metal/metal_buffer.cc` | +49/-9 | egress encodes |
| `plugin-native/metal/metal_lowering.cc` | +459/-32 | the regrid rule, `reduce_precision`, `select_and_scatter`, the 4-bit bitcast, the scatter apply and the rank-0 scatter |
| `plugin-native/execute_test.py` | +301/-16 | 98 new checks, three declines retired, three added |
| `plugin-native/{smoke,wheel_poc}_test.py` | +34/-24 | the decline sentinel re-aimed |

## 1. The emulated grids

`src/metaljax/dtypes.py` is the spec, and it says something the phase-2 dtype
table was built to forbid: an `i4` lives in a whole `int8`, an `f8` in a
`float16` and `f8E8M0FNU`'s exponent range in a `float32`, holding the VALUE
rather than the encoding.  M5c declined the whole family for exactly that
reason ("a faithful port would have to re-grid after every arithmetic op — a
per-site flag on every elementwise entry, where one missed site is silent
wrongness rather than a decline").

**The port removes the per-site flag.**  `Entry::regrid` is one field, and
`Program::step` spends it in one place — after the family handler has written
its results and before the drop list runs:

```cpp
if (e.regrid >= 0)
  for (int s : e.outs)
    if (env[s]) env[s] = quantize_emulated(*env[s], e.regrid);
```

So the question "does this handler have to round?" is never asked of a
handler.  It is asked once, of the op's NAME and its RESULT type, by
`RegridOf` in the lowering, which is a transliteration of the three sites the
Python engine spends its rule at (`_regrid`, `_maybe_wrap4`, `_convert`):

* a **convert** onto any emulated type quantizes — that is what puts a value
  on the grid in the first place;
* an **arithmetic** result re-grids only on the OCP FP4/FP6 formats (whose 16
  or 64 points diverge from XLA after one operation: on `f4E2M1FN`, 4 + 4 is
  6, not 8) and, for `i4`/`ui4`, only on the three ops the 4-bit wrap is
  visible through;
* a **bitcast onto i4/ui4** rides the same field, because `ops/shape.py`'s
  `_from_nibbles` IS `quantize_emulated`;
* the float8 family deliberately stays unrounded between ops, exactly as it
  has since the emulation landed in Stage 1.

`dtype_of` answers with the STORAGE, so every handler that builds a result of
the declared type is right without knowing a grid exists — including the
small-int reduce guard P8.5 added, which now covers `i4` for free.

**One bug the first draft had, and it is the one to look for in a review.**
`kConvert` cast to the storage dtype and *then* re-gridded.  That is two
roundings (`f32 -> f16 -> f8E4M3FN` is not `f32 -> f8E4M3FN`) and, for the
integer grids, a saturating float→int cast in front of a wrap.  The handler
now passes an emulated convert's operand through untouched — `quantize_emulated`
reads the operand's own value and ends in the storage dtype itself, which is
what `_convert` does.  Measured before the fix: `float32(1e5).astype(int4)`
came back `-1` where the Python engine says `0`.

### the host transfer

XLA's default layout gives a sub-byte type a whole byte (`primitive_util::
ByteWidth` rounds up and nothing sets `element_size_in_bits` here), so the
wire is one byte per element holding the type's own encoding, and the transfer
is a per-element CONVERSION rather than a copy — the third thing in this
plugin, after `bf16` and `f64`, whose wire is not its storage.

`llvm::APFloat` does the encoding: it carries all eleven float semantics
(`Float8E4M3FN`, `Float6E2M3FN`, `Float4E2M1FN`, ...) with the right
non-finite behaviour per format, which is the same specification `ml_dtypes`
implements and therefore what the CPU backend's answers come from.  Two
formats needed more than a call:

* **`f8E8M0FNU`** — exponent only, unsigned, with no zero.  Converting THROUGH
  APFloat returns an infinity for the NaN code, and `ml_dtypes` maps a zero, a
  negative, an infinity and an out-of-range exponent all to the single NaN, so
  the format is spelled out (two lines, measured against numpy).
* **the OCP FP4/FP6 pair** — no NaN to convert to, and `ml_dtypes` maps one to
  a ZERO whose sign is the OPPOSITE of the NaN's (measured: `+NaN` becomes
  `-0`, `-NaN` becomes `+0`).  Reproduced rather than reasoned about; APFloat,
  which has no rule to follow, returns `+0` for both.

The gate is exhaustive: for each of the thirteen formats, EVERY canonical bit
pattern is device_put and read back, and separately read as f32.  That is what
says a subnormal or NaN encoding nobody thought about survives.

### where this engine and XLA:CPU still disagree

All three are Stage 1's answers, unchanged, and all three are visible in the
`execute_test` rows' input arrays (the specials are trimmed per family, with
the reason written beside them):

| | CPU | here (and Stage 1) |
|---|---|---|
| FP6 overflow | a ZERO | saturates to `±max` |
| FP6 `convert` at all | **crashes** its own fusion compiler (`llvm_module != nullptr`) | computes |
| `f32 -> i4/ui4` out of range | saturates (`1e5 -> 7`) | wraps (`((v+8) mod 16) - 8`, so `0`) |
| a NaN's payload through an f8 round trip | preserved | lost in the f16 storage (2 of 256 codes on `f8E5M2`, 6 on `f8E4M3`, 14 on `f8E3M4`) |

The fp6 rows are CLAUDE.md item 20's finding ("XLA:CPU's fp6 is itself broken
— ml_dtypes is the reference"), so ours is the right answer there.  The NaN
payloads are strictly BETTER than Stage 1's, which canonicalizes 5 / 13 / 29
of the same codes.

## 2. `reduce_precision`

A new opcode, and the whole of `ops/elementwise.py::_reduce_precision`.  Which
of the four arms runs — identity, the bf16 grid, the f16 grid, or the general
any-e/m rounding — is a question about two attributes and the operand's
storage dtype, all static, so the lowering answers it and the handler carries
only the arithmetic: RNE mantissa rounding in f32 bit space, then a clamp to
the `e`-bit exponent range (overflow to an infinity, underflow to a zero;
XLA's `reduce_precision` has no subnormals, so there is nothing between).  An
`e`/`m` outside the ranges the Python handler implements declines with the
same message, so the two engines refuse the same programs.

## 3. The scatter tail

**The apply body** (`jax.lax.scatter_apply`, every `.at[i].apply(f)`).
`ops/gather.py` has TWO executions and the mission's brief was to port the
gate honestly, so both are here:

* under the op's `unique_indices` (method **7**), gather the current values,
  run the body on `(old, update)` and SET the result — one shot, the same
  shape as P10's complex-multiply rewrite;
* without it (method **8**), one update at a time in row-major update order,
  because a computed body need be neither associative nor idempotent and a
  duplicate index really does mean `f(f(x))`.

**The gate is not a decline.**  jax emits `unique_indices = false` for every
`.at[].apply()`, so declining on the missing promise would have closed zero
tests; what `ops/gather.py` actually gates on is the update COUNT (its
sequential arm costs a constant number of MLX ops per update), and that cap —
1024 — is what declines here, with the Python handler's own message.  The
sequential arm takes no drop strategy: a dropped update leaves the slot alone,
which is a per-update `where` and not something a mask over the whole update
array can say once the body has run.

**`select_and_scatter`** is `ops/reduction.py`'s handler line for line: pad
with the select's identity, one `as_strided` view of every window, `argmax`/
`argmin` for the winner (first hit, which is XLA's in-order GE/LE select), the
winner's absolute flat position by digit arithmetic, and one scatter-add (or
max/min, for the `or`/`and` bodies jax emits on PRED pooling gradients).  Both
regions are read STRUCTURALLY at lowering — a compare, then an add/or/and — so
the entry carries no sub-Program, and a body outside those shapes declines
naming what it found.  M5c declined this family on the grounds that its
scatter-add over overlapping windows is order-nondeterministic on the GPU and
"no byte differential could hold it"; the phase-2 disposition is exactly that,
with a tolerance instead — so the `execute_test` rows carry one.

**A rank-0 operand.**  `stablehlo.scatter` on a 0-d array arrives with an
EMPTY coordinate vector (`tensor<0xi32>`) and one update, and there is no axis
to index: the whole op degenerates to its combiner.  A `set` aliases the
update, an arithmetic combiner becomes that binary op, and an apply body is
spliced into the enclosing frame the way a `func.call` is (`InlineBlock`,
factored out of `Inline`).  jax reaches this through
`jax.experimental.sparse` on a 0-d array, where the reduction over the updates
has already happened before the scatter.

## 4. The decline sentinel moved to `stablehlo.rng`

Three checkpoints — `smoke_test`, `wheel_poc_test` and one `execute_test`
decline (a while body holding an unlowered op) — watch that an op outside the
set declines by NAME.  P9 pointed them at `convolution`, P7 at the LAPACK
targets, P10 at `reduce_precision`; this milestone gave the last of those an
executor, so they now watch **`stablehlo.rng`**, which `jax.lax.rng_uniform`
emits.  It is a better sentinel than its predecessors: XLA's
non-deterministic RNG is implemented by NEITHER engine and is on no phase's
ladder, so it is not one milestone away from being retired again.

## Validation

| | result |
|---|---|
| `plugin-native/execute_test.py` | **482 checks**, 0 failures (384 at P10) |
| `plugin-native/texmo_gate.py` | **106/106** ok (21 via sensitivity scaling), 0 decline, 0 FAIL |
| `plugin-native/smoke_test.py` | 4/4 checkpoints |
| `plugin-native/decline_census.py` | 34 of 35 lower (unchanged; the one left is `debug_print`, a JAX-side registration gap — P13) |
| `plugin-native/wheel_poc_test.py` | 4/4 from a fresh 3.13 venv with the native wheel |
| `bazel test //...` | PASSED (`//metal:runtime_gil_free_test`) |
| eager (`METALJAX_COMPILE=0`) vs compiled | all cases agree |
| dylib | 165,943,720 -> **165,997,624 B** (+53,904, **+0.032 %**) |

### the census slice

The 14 files P8's reasons file names for the three families, before and after
on this tree, one process per file, sequentially.

| file | before (pass/fail) | after | delta |
|---|---:|---:|---:|
| `dtypes_test` | 749/68 | 816/**1** | −67 |
| `lax_test` | 2468/53 | 2519/**2** | −51 |
| `api_test` | 659/60 | 694/**25** | −35 |
| `lax_vmap_test` | 287/30 | 317/**0** | −30 |
| `shape_poly_test` | 2330/16 | 2342/**4** | −12 |
| `lax_numpy_indexing_test` | 362/10 | 371/**1** | −9 |
| `lax_numpy_test` | 3191/19 | 3199/**11** | −8 |
| `export_test` | 102/20 | 108/**14** | −6 |
| `jax_jit_test` | 10/6 | 16/**0** | −6 |
| `sparse_bcoo_bcsr_test` | 413/7 | 418/**2** | −5 |
| `lax_autodiff_test` | 678/4 | 681/**1** | −3 |
| `mutable_array_test` | 130/2 | 132/**0** | −2 |
| `debugging_primitives_test` | 22/31 | 24/**29** | −2 |
| `batching_test` | 230/2 | 232/**0** | −2 |
| **TOTAL** | **11,631/328** | **11,869/90** | **−238** |

**Zero regressions**: the set difference `after − before` over the failing
test ids is empty.

**No numeric mismatch on a lowered path.**  All 90 were re-run with
`--tb=line` (87 reproduced; 3 are the position-dependent
`test_bcoo_spdot_general` class and passed):

| # | reason | phase |
|---:|---|---|
| 29 | `debug_print` has no rule for platform metal | P13 |
| 12 | donation: "Some donated buffers were not usable" | P13 |
| 9 | `a value that is not a ranked tensor` (effect tokens) | P12 |
| 9 | `unsafe_buffer_pointer` identity (`testArrayCopy*`) | P13 |
| 9 | assertions that a platform SHOULD have failed | shared whitelist |
| 6 | `element type f64` | intentional |
| 5 | export for other platforms / `KeyError: 'metal'` | version skew |
| 2 | cross-memory-space copies | P13 |
| 2 | memory-space addressing | P13 |
| 2 | cost analysis / optimized HLO (PJRT surface) | P13 |
| 1 | `custom call target 'dce_sink'` | P13 |
| 1 | `conv: complex with no spatial dimensions` | intentional (P7) |
| 1 | `complex scatter multiply without unique indices` | intentional (P10) |

### `tests/` through the native plugin

The Stage 1 suite with `METALJAX_PLUGIN_PATH` at the native dylib, the second
standing leg: **90 -> 84 failures**, and `test_subbyte_float` is the only file
that moved (**6 -> 0**, which is this milestone's acceptance row).  What
remains, by file: `test_moe` 28, `test_qmm` 26, `test_qmm_mxfp4` 16,
`test_pjrt_surface` 10, `test_donation` 3, `test_engine_gc` 1 — the
recognizer-emit families (whose Python-side pack building the plugin has no
path to) and P13's PJRT surface.  `test_pjrt_surface` is FLAKY on both dylibs
in the same band (5/9/5 on this one, 7/5/8 on HEAD over three runs), which is
P10's finding about `test_buffer_pointer_of_broadcast`, not a movement.

## Reviewer-scrutiny list

1. **`Entry::regrid` is the whole safety argument.**  A handler that produces
   a value on a grid cannot forget to round it, because no handler rounds; but
   `RegridOf` can be wrong about WHICH ops round, and being wrong there is a
   value that drifts off the grid rather than a decline.  The two lists
   (`IsUnaryRegridOp`, `IsBinaryRegridOp`) are `ops/elementwise.py`'s `_UNARY`
   and `_BINARY` keys, and the ops NOT in them (select, clamp, compare,
   reduce, gather, iota, constant) do not round in the Python engine either.
2. **The convert must not cast first** (section 1).  A future reader who
   "simplifies" `kConvert` back to a single `astype` reintroduces a double
   rounding that no round-trip test can see — the grid values all survive
   `f32 -> f16` — and only an off-grid or out-of-range input reveals it.
3. **The FP4/FP6 NaN sign is a reproduced quirk**, not a derivation.  If
   ml_dtypes ever stops inverting it, the `execute_test` encode rows are what
   will say so.
4. **`f8E8M0FNU` rests on denormal flush.**  `quantize_emulated` maps a zero
   (and every negative) to `2^-127`, which is SUBNORMAL in f32 and therefore
   flushed to zero on this GPU, which is what makes the host encode produce
   the NaN code XLA produces.  Stage 1 has the same dependency and the same
   answer; on hardware without FTZ both engines would say `2^-127`.
5. **The sequential apply is O(updates) MLX ops.**  The 1024 cap is
   `ops/gather.py`'s and it is enforced at lowering, so a big
   duplicate-index `scatter_apply` DECLINES rather than emitting thousands of
   entries' worth of work — there is an `execute_test` row for the refusal.
6. **`select_and_scatter` is order-nondeterministic by construction** and its
   rows carry a tolerance.  A reviewer who tightens them to 0 will get a flake,
   not a bug.
7. **An emulated buffer's `unsafe_buffer_pointer` hands out the STORAGE** (an
   f16 array under a one-byte f8 shape), the same pre-existing hole `f64` has.
   `CopyRawToHost` declines for both; `AcquireExternalReference` does not,
   because jax reads it for object identity rather than for bytes.
8. **The opcode enum shifted twice** (`kReducePrecision` after `kConvert`,
   `kSelectAndScatter` after `kScatter`) and the dtype codes gained thirteen
   entries at the END.  Nothing serializes either — the plugin and the runtime
   are one build and both registries are keyed by name — but a reviewer should
   satisfy themselves of that, as P10's list asked.
