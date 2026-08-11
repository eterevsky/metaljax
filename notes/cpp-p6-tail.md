# Stage 2 P6: the decline tail whose executors already exist

Follows [`cpp-p5-compile.md`](cpp-p5-compile.md). P5 ended with the plugin at
25 of 35 census probes and a list: *sort/top_k, then reduce_window,
convolution, fft, then the host-op custom calls*. Four of those five families
are executed by the runtime today and were missing only a lowering; this
milestone ports them. The fifth, **convolution, is not** — see the section of
its own below, which is the one finding a reader should not skim.

```
$ .venv/bin/python plugin-native/decline_census.py    # P5 -> P6
25 of 35 programs lower.                31 of 35 programs lower.
  2  op stablehlo.sort                    1  op stablehlo.convolution
  2  op stablehlo.reduce_window           1  op stablehlo.cholesky
  1  op stablehlo.rng_bit_generator       1  op stablehlo.custom_call (qr)
  1  op stablehlo.fft                     1  debug_print (a JAX-side gap)
  1  op stablehlo.convolution
  ...
```

Everything left is P7's (LAPACK on Accelerate), JAX's own (no `debug_print`
lowering rule is registered for platform `metal` through this plugin — the
program never reaches us), or convolution.

## What was built

| file | lines | what |
|---|---:|---|
| `metal/metal_lowering.cc` | +835/-11 | the five lowerings, the window plan, the sort comparator recognizer |
| `execute_test.py` | +436/-9 | 72 new cases, 4 new modules, 4 new declines, module-text declines |
| `smoke_test.py` | +15/-6 | the decline checkpoint moved from sort to convolution |
| `wheel_poc_test.py` | +18/-6 | same, plus a sort that now COMPUTES from a wheel |

Dylib: 165,735,912 -> **165,790,280 B** (+54,368, **+0.033 %**). Edit loop
unchanged (~5 s to recompile `metal_lowering.cc`, ~2 s to link).

## The four ports, and where each one comes from

Same doctrine as P2-P5: `src/metaljax/tape.py` is the specification, the C++
handler's `Cursor` reads are the ground truth, and a tape dump diff is the
proof.

| written | Stage 1 anchor |
|---|---|
| `LowerRng` | `tape.py::_lower_rng` -> `native/ops_rng.cc` |
| `BuildWindowPlan` | `tape.py::_window_plan` -> `read_window_plan` |
| `LowerReduceWindow` | `tape.py::_lower_reduce_window` (cum peephole / monoid / select_and_gather_add / generic) |
| `LowerGenericBody` | `tape.py::_generic_reduce_attrs` |
| `LowerReduce`'s third arm | `tape.py::_lower_reduce`'s `_generic_reduce` fall-through |
| `LowerFft` | `tape.py::_lower_fft` (both MLX workarounds) |
| `LowerSort` / `LowerTopK` | `tape.py::_lower_sort` / `_lower_top_k` **plus** `ops/sort.py::_sort`'s key-chain arm (below) |
| `TaintFromAll` | tape.py's `if regions or name in _TAINTING_OPS` rule |

Four transliteration details are wrong NUMBERS rather than build errors, and
are the ones to read the diff against:

1. **`FloorDiv`.** `_window_plan` computes `(shape - span) // stride + 1` and
   clamps at 0. A window that does not fit its padded axis makes that
   numerator NEGATIVE, and C++'s truncation rounds it towards zero where
   Python's floors: `(-1) // 2` is -1 in Python and 0 in C++, which after the
   `+ 1` is **one window versus none**. Written out as a helper, with the
   reason on it.
2. **The threefry split index.** With no even dim the split goes to the
   largest, and Python's `max(range(n), key=...)` returns the FIRST maximum.
   A strict `>` in the scan is what reproduces that; `>=` would pick the last
   and shift every word of the output.
3. **`_TAINTING_OPS` is one op and it is rng.** With an empty output XLA
   consumes no blocks and the handler returns the STATE OPERAND'S OWN ARRAY,
   so without the taint that state could be handed out as an output aliasing
   an argument. `execute_test`'s "rng empty output returns the state" is that
   case.
4. **reduce_window's generic arm reduces the WINDOW rank, not the operand's.**
   What that reduce sees is the extracted window view: rank
   `len(out_sizes) + 1`, reduced dim `len(out_sizes)`. Passing the operand's
   rank would compute a fold over the wrong axis, silently.

Two Python-side arms are deliberately absent, because this plugin cannot reach
them: `stablehlo.triangular_solve` / `cholesky` (host ops, P7) and the msl
plans (`METALJAX_MSL=0` is the setting the cross-check runs Stage 1 under).

## sort: the recognizer, ported rather than declined

This is the one family where the mission's premise ("only the native lowering
is missing") was half true, and it is worth being precise about why.

`tape.py::_lower_sort` lowers exactly one comparator shape: the one that IS a
bare compare on the `(lhs, rhs)` block-argument pair. That is an integer sort,
and the `sort(values, iota)` a `chlo.top_k` decomposes to. **jax's float sort
is not that shape.** It emits a comparator that computes a KEY first:

```mlir
^bb0(%a: tensor<f32>, %b: tensor<f32>):
  %1 = stablehlo.compare EQ, %a, %zero, FLOAT      // -0 -> +0
  %2 = stablehlo.select %1, %zero, %a
  %3 = stablehlo.compare NE, %a, %a, FLOAT         // NaN -> canonical qNaN
  %4 = stablehlo.select %3, %qnan, %2
  ... the same four ops for %b ...
  %9 = stablehlo.compare LT, %4, %8, TOTALORDER
```

Stage 1 handles that in `ops/sort.py`, by *running the comparator's block on
whole arrays* — the chain is elementwise, so scalar block code computes the
key of the entire operand — and `tape.py` declines it, because the Python
ENGINE is there to catch what the tape drops. Phase 2 has no Python engine
behind it, so the recognizer is ported. What moves is WHEN it runs: what the
Python does with an interpreter at execute time, this does with tape entries
at compile time.

* `ArgDeps` / `SerializeKey` are `ops/sort.py`'s `_arg_deps` / `_serialize`:
  each side must depend on exactly one block argument, they must be the
  `(2k, 2k+1)` pair, and the two def-DAGs must be EQUAL with each side's own
  argument renamed. The symmetry check is not decoration — an asymmetric
  comparator has no key array that orders the operand the way it does, and
  sorting by the left side's chain anyway is the silent-wrongness failure
  this family offers. `execute_test` has a hand-written module for it.
* The cone of ops feeding the key is then lowered **into the enclosing
  frame** through the ordinary `LowerOp`, with the comparator's arguments
  aliased to the operand slots. The sort entry gains the chain's output as a
  trailing input and keys on it (`at[2] = n`); the handler's own
  `total_order_key` then applies, exactly as `_gather_sorted` applies it to
  the key the Python evaluated.
* Two guards make "scalar block code, run on arrays" a fact rather than an
  assumption: an allowlist of elementwise op names, and a check that every
  operand and result of every cone op is rank-0. A `broadcast_in_dim` or a
  `reduce` inside the chain would be lowered against its rank-0 IR type and
  then handed a whole array — a wrong answer, so it declines instead
  (`sort: comparator op stablehlo.broadcast_in_dim`).
* The entries' `bytes` are corrected to the OPERAND's size after lowering.
  The IR says four bytes; what a chain entry materializes is a whole array,
  and the eager flush cadence meters device bytes.

`jnp.sort` on f32 becomes six entries plus the sort:

```
[tape] stablehlo.constant  -> 1 [] const        <- XLA hoisted these out
[tape] stablehlo.constant  -> 2 [] const           of the comparator region
[tape] stablehlo.compare 0,2 -> 3 [0,0]
[tape] stablehlo.select 3,2,0 -> 4 []
[tape] stablehlo.compare 0,0 -> 5 [1,0]
[tape] stablehlo.select 5,1,4 -> 6 []
[tape] stablehlo.sort 0,6 -> 7 [1,0,1,1]        <- key = slot 6, kind 1
```

and the same thing happens inside a `lax.scan` body, which still compiles
(`max_repeat 740`).

**Still declining, named**: the two lexicographic select trees. Both are a
DIFFERENT execution shape rather than a different key — `_lex_sorted` threads
a permutation through successive stable argsorts, and the complex arm packs
canonicalized `(re, im)` order keys into one u64 — and the sort entry computes
one argsort and gathers with it. There is no opcode that gathers by a
permutation the tape computed, so expressing either would need runtime work
(a `take_along_axis` entry, or a key-packing one), not lowering work. That
costs `jnp.lexsort`, `jnp.unique` over rows, and complex sorts; the messages
are `sort: comparator ends in stablehlo.select, not compare` and
`sort: complex lexicographic comparator`.

## convolution: NOT a lowering gap

Every other family in this milestone had a handler waiting in `native/`.
Convolution does not: there is **no `kConv` opcode in `native/program.h`**, no
case in any `step_*`, and no name in `config.cc`'s registry.
`src/metaljax/ops/conv.py` (283 lines: `mx::conv_general` for every layout,
batch groups, `im2col`+int64 for exact integer convs, four real convs for
complex, 0-spatial as a matmul, the grouped fallbacks) has never been
transliterated — CLAUDE.md's own migration ledger says so twice
("`convolution` (never in the op set)" in the M5c census, and
"convolution (2): PORT" in the phase-2 decline dispositions).

Porting it is therefore a change to the SHARED executor: an enum value, a
registry name, a new `ops_conv.cc`, and the full P1 battery behind it
(differential pytest on both engines, the texmo gate, the command-buffer
canaries) because Stage 1's engine would run the new code too. That is outside
this milestone's file boundary and it is not a small job. It stays declined,
named, and it is now the LAST op between this plugin and jax's dense-model
surface.

The one thing worth recording for whoever picks it up: texmo's `conv.4` layer
does **not** produce `stablehlo.convolution` (the gate is 106/106 with zero
declines, `mid11` included), so this is not on the training path — it is on
the vision/audio path and the JAX test suite's.

`select_and_scatter` also stays declined, unchanged from Stage 1 and for
Stage 1's reason: its scatter-add over overlapping windows is
order-nondeterministic on the GPU, so no byte differential can hold it.
`select_and_gather_add` DID land (it is a reduce_window arm, and the jvp of a
max pool exercises it).

## Validation

| | result |
|---|---|
| `plugin-native/execute_test.py` | **228 checks**, 0 failures (156 in P5) |
| `plugin-native/texmo_gate.py` x2 | 106 ok, 0 decline, 0 FAIL, 0 error |
| `plugin-native/decline_census.py` | **31 of 35** (25 in P5) |
| `plugin-native/smoke_test.py` | 4/4 checkpoints |
| `plugin-native/wheel_poc_test.py` | 4/4, from a native wheel in a fresh 3.13 venv |
| `bazel test //metal:runtime_gil_free_test` | PASSED |
| `pytest tests/ -q` | 1258 passed (nothing under `native/` or `src/` was touched) |
| tape cross-check | **146 lines over 19 probes, byte-identical** |
| eager vs compiled | 190 of 191 bit-identical, the odd one P5's fusion finding |

### The new execute_test rows, by family

* **sort** — f32/f16/bf16/i32/u8/bool; ties, signed zeros, NaNs and
  infinities in one row (total order puts NaN last, and -0 must tie with +0);
  axis 0 and a rank-3 middle axis, which arrive as strided views (the shape
  MLX 0.32's argsort reads wrong elements from); argsort stability, once with
  every key equal and once with repeats — an unstable sort is free to return
  anything there, which is exactly what that catches; an ARGSORT over signed
  zeros and NaNs, which is the only row that can see the canonicalization run
  at all (a values-only sort cannot: -0 and +0 are numerically equal); `sort_key_val`, median,
  percentile, partition; top_k with ties, on integers, and over the leading
  axis (the non-last-axis case that was a silent-wrongness bug in 0.4.x); a
  sort inside a scan body.
* **rng** — Philox and ThreeFry, u32/u8/u16 outputs, odd counts, a shape with
  no even dim, rank 3, a scalar output, the empty output that returns the
  state, two draws chained through the state, and `jax.random` on an `rbg`
  key (uniform and normal). Every bits row is compared **EXACT**, not within a
  tolerance: `_canonical` widens unsigned words to int64, so those rows are a
  word-for-word compare against XLA's CPU backend. The 64-bit arm is a
  hand-written module (jax refuses a u64 output without x64), with the u64
  state BUILT inside it from a u32 argument because a u64 host buffer cannot
  cross `device_put` either.
* **reduce_window** — the cum peephole (sum/prod/max/min, last axis, reverse);
  max/sum/min pooling with VALID, SAME and explicit padding; window dilation,
  base dilation, and both together; a window wider than its axis (the
  zero-size guard); bool and integer monoids; the jvp of a max pool
  (select_and_gather_add); one inside a scan body. Generic bodies — a monoid
  neither table knows — are two hand-written modules, one `reduce_window` and
  one `reduce`, since jax only ever emits the recognized forms.
* **fft** — fft/ifft/rfft/irfft, odd lengths, `fft2`/`rfft2`, a complex input,
  a transform inside an elementwise chain, and all three unit-last-length
  rewrites (rfft, rfft2, irfft).
* **declines** — lexsort, complex sort, an asymmetric comparator, a
  non-scalar key chain, convolution, negative reduce_window padding, and a
  while body holding an unlowered op. The decline harness now accepts a module
  TEXT as well as a jitted function, because two of those encodings are ones
  jax's own lowerings cannot produce.

### Tape cross-check, family by family

Method is P5's: `METALJAX_DUMP_MODULE=1` prints the module XLA's parse handed
the plugin, `METALJAX_DUMP_TAPE=1` prints the finished tape, and a scratch
driver lowers THAT module through `tape.py` with a recorder standing in for
`engine.NATIVE.Program`. Stage 1 runs at `METALJAX_MSL=0`, compile ON.

| probe | lines | verdict |
|---|---:|---|
| sort i32 (bare compare) | 3 | identical |
| top_k | 6 | identical |
| rng philox (7 words) | 7 | identical |
| rng threefry (3, 5) | 7 | identical |
| rng inside a fori_loop | 20 | identical |
| rng philox 64-bit (module) | 7 | identical |
| cumsum | 5 | identical |
| cumprod / cummax / cummin | 10 | identical |
| max pooling | 5 | identical |
| base + window dilation | 5 | identical |
| zero-size window | 5 | identical |
| bool reduce_window | 6 | identical |
| select_and_gather_add (jvp) | 7 | identical |
| reduce_window in a scan | 23 | identical |
| reduce_window generic body (module) | 8 | identical |
| reduce generic body (module) | 8 | identical |
| fft | 4 | identical |
| rfft unit length | 7 | identical |
| rfft2 | 3 | identical |

**146 lines over 19 probes, byte for byte** — including every window-plan
field, both rng schedules, and the compile decisions on the loops.

Four more probes (`jnp.sort` f32, `jnp.argsort` f32, `jnp.median`, a sort in a
scan) have **no Stage 1 tape to diff**: `tape.py` declines them, which is the
whole point of the recognizer port. Those are held by the CPU differential
instead, which is the stronger bar anyway.

## Gotchas

1. **Python's floor division is not C++'s**, and `_window_plan` depends on it.
   See transliteration note 1 above. Any port of a Python shape calculation
   with a possibly-negative numerator wants the same treatment.
2. **XLA's parse hoists a comparator's constants OUT of its region** — into
   the enclosing block, not all the way to main. So the key chain's constants
   are usually already slots when the sort is lowered, and the in-region
   constant path is reachable only through hand-written IR. Both work; only
   one is exercised by jax.
3. **`engine.NATIVE.opcodes()` returns a dict**, name -> code. The
   cross-check driver's first version enumerated it as a list and produced
   thirteen convincing, entirely wrong tapes (`stablehlo.select` where
   `stablehlo.convert` belonged). If a cross-check reports every probe as
   differing, suspect the renderer before the lowering.
4. **XLA:CPU cannot compile a philox with a u32 state and a u64 output**
   ("Binary op shift-right-logical with different element types: u32[] and
   u64[]", out of its own rng expander). The lowering covers that
   combination; nothing in a CPU-differential harness can prove it does, so
   there is no case for it and this note is the record.
5. **Two harness files asserted that `jnp.sort` declines.** `smoke_test.py`
   and `wheel_poc_test.py` both watched that decline as their "an op outside
   the set is refused, naming itself" checkpoint. Both now watch convolution.
   A milestone that CLOSES a gap has to go and find the tests that were
   watching it.
6. **A multi-result op has to be dispatched before the arity check.** In
   `LowerOp` the `getNumResults() != 1` decline sits above the attribute
   chain, so sort / reduce_window / top_k / rng bind and emit themselves the
   way scatter does — and `stablehlo.sort` and `stablehlo.reduce_window` had
   to join the region-carrying allowlist, or they would have declined as
   "it carries a region" before any of this ran.

## What P7 should pick up

* **`convolution`**, which now means a new opcode and an `ops_conv.cc` in
  `native/` — the first P-milestone whose work is executor work rather than
  lowering work, with the full P1 battery behind it.
* **LAPACK on Accelerate** (`cholesky`, `qr`, `eigh`, `svd`, ...): the
  remaining census declines, and the reason the plugin has no host-op path at
  all. Stage 1 runs these through Python/numpy; phase 2's plan is direct C.
* The two lexicographic sorts, if a `take_along_axis` entry lands — one
  opcode would close `jnp.lexsort`, `jnp.unique` over rows, and complex sorts
  together.
* `select_and_scatter`, which needs goldens with a tolerance rather than a
  byte differential (its GPU scatter-add is order-nondeterministic).
* Still open from P5: asynchronous execute + donation, and MLX's global
  command-encoder map.
