# Stage 2 P7: convolution, executor and lowering

Follows [`cpp-p6-tail.md`](cpp-p6-tail.md), whose last section says the thing
this milestone had to act on: **convolution was not a lowering gap.** Every
other family P6 closed had a handler waiting in `native/`; this one had no
`kConv` opcode at all, no case in any `step_*`, no name in the registry.
`src/metaljax/ops/conv.py` had never been transliterated — CLAUDE.md's ledger
says so twice ("`convolution` (never in the op set)", "convolution (2): PORT").

So P7 is the first P-milestone whose work is **executor** work: a new op
family in the shared runtime, with the full P1 battery behind it, plus the
lowering that feeds it.

```
$ .venv/bin/python plugin-native/decline_census.py     # P6 -> P7
31 of 35 programs lower.               32 of 35 programs lower.
  1  op stablehlo.convolution            1  op stablehlo.cholesky
  1  op stablehlo.cholesky               1  op stablehlo.custom_call (qr)
  1  op stablehlo.custom_call (qr)       1  debug_print (a JAX-side gap)
  1  debug_print (a JAX-side gap)
```

Everything left is LAPACK-on-Accelerate or JAX's own.

## What was built

| file | lines | what |
|---|---:|---|
| `native/ops_conv.cc` | **+329** (new) | the family: four arms of `ops/conv.py`, the im2col integer path, the two shape guards |
| `native/program.h` | +9/-2 | `kConv`, `step_conv`, the layout pointer, the file-layout table |
| `native/config.cc` | +9 | the registry name — under a pseudo-name, and that is the finding below |
| `native/program.cc` | +1/-1 | `step_conv` in the dispatch chain |
| `native/build.sh`, `BUILD.runtime` | +2/-1 | the two build lists (clang and bazel) that share these sources |
| `metal/metal_lowering.cc` | +252/-2 | `LowerConv`: the layouts, the window plan, the negative-pad rewrite, the arm |
| `execute_test.py` | +302/-11 | 46 new checks, 2 new modules, 2 new declines, 3 repointed |
| `smoke_test.py`, `wheel_poc_test.py` | +50/-25 | their decline checkpoint moved from convolution to cholesky |

Dylib: 165,790,280 -> **165,827,320 B** (+37,040, **+0.022 %**). The nanobind
extension, which links the same runtime: 680,232 -> **700,328** (+20,096,
+2.95 % — the runtime is small, so one family is visible in it).

## The attribute layout

There was no `tape.py` conv lowering to copy — Stage 1 declines the op and
runs it on the Python ENGINE — so the layout is this milestone's own. It is
documented at the handler (`native/ops_conv.cc`) and reads with a `Cursor`,
flat, like its neighbours:

```
kConv:
  [empty?]
    1 -> [out dtype, [out shape]]          the result is zeros; attrs stop
    0 -> [out dtype, rank,
          [lhs perm], [rhs perm], [out perm], fgc, bgc,
          rank == 0 ? (nothing more)
                    : [strides], [lhs dilation], [rhs dilation],
                      [pad lo], [pad hi],
                      crop?, [crop start], [crop stop],
                      flip?, mode, native groups?, [want]]
```

Four things it says, and why each is an attribute rather than a runtime
question:

* **The three permutations.** Every layout XLA can spell — `NCHW`/`OIHW`,
  `NHWC`/`HWIO`, anything else — reaches the executor as the transpose into
  MLX's layouts (input `(N, *spatial, C_in)`, weight `(C_out, *spatial,
  C_in)`) and the transpose back out. The handler never sees a dimension
  number. Each is checked to BE a permutation at lowering, where it costs
  nothing: a repeated dim would otherwise be a silently wrong layout.
* **`rank`** is the number of SPATIAL dims, and `0` selects the matmul arm —
  a convolution with no window to slide is a contraction over features, and
  the attrs stop before the window fields because there are none.
* **`mode`** (0 float / 1 exact integer / 2 complex) and **`native groups?`**
  are the arm and the group strategy, both static properties of the result
  dtype and the rank that `ops/conv.py` recomputes per call.
* **`want`** is the result shape in MLX's layout: the guard, not a hint (see
  below).

`crop` carries the negative-padding rewrite, and its two vectors are written
whether or not the flag is set — the same cursor-alignment discipline
`LowerPad` uses.

## The arms, ported and declined

| arm of `ops/conv.py` | here |
|---|---|
| `mx.conv_general` for float layouts | `mx::conv_general`, all layouts, strides, both dilations, `flip` |
| feature groups | MLX's own `groups` for 1-D/2-D floats; one convolution per group otherwise (`native groups?`) |
| batch groups | split batch and kernel output features, concatenate along features |
| exact integer conv | `int_conv`: dilate with zero holes, pad, ONE `as_strided` im2col view, multiply and sum in int64 (bool folds with `any`) |
| complex64 | four real convolutions, recombined by `make_complex` |
| 0 spatial dims | a (grouped) matmul over features |
| negative padding | the operand-slice rewrite, resolved at lowering |
| zero-size operands / results | zeros of the declared shape, folded into the `empty?` arm |
| `window_reversal`, all axes | `flip` on the float path, an index reversal of the weights on the integer one |

**Declined, named.** `window_reversal` on SOME axes and not others
(`conv: mixed window_reversal`): MLX's flip is all-or-nothing and so is the
Python handler, which raises. And a COMPLEX convolution with no spatial dims
(`conv: complex with no spatial dimensions`) — this one is a deliberate
divergence rather than a transliteration. The Python matmul arm runs its
operands through `astype(mx.float32)`, which drops the imaginary part; a
faithful port would be silently wrong against jax-CPU, so phase 2 refuses the
combination instead. Both are execute_test decline cases.

## The two shape guards, which are the whole point of the family

`ops/conv.py` carries two checks that look like belt-and-braces and are not.
At a zero-size spatial extent XLA computes the dilated extent as 0 where MLX
computes `(0-1)*d+1`, so `mx.conv_general` returns a NARROWER array than the
result type declares — and whoever reads that result reads past the end of
the buffer, out of uninitialized device memory. That is CLAUDE.md item 20's
"conv short-buffer overread (uninit memory, flaky)": a wrong answer that
comes and goes, not a crash.

Both came across:

1. An empty operand or an empty result never reaches MLX — the lowering emits
   the zeros entry instead. The crop case folds into it too: when the
   negative-pad rewrite empties the operand, the lowering returns the same
   zeros attrs the top guard does, which is what the Python handler's
   post-crop `if x.size == 0` returns.
2. Everything MLX does produce is measured against `want` before it is handed
   on. A disagreement throws (an `InternalError` from Execute), exactly where
   the Python handler raises `UnsupportedOpError`.

## Transliteration details worth reading the diff against

1. **The pseudo-name, and it is the finding of this milestone.** The obvious
   registry entry — `{"stablehlo.convolution", kConv}` — is WRONG, and the
   suite says so within a second. `tape.py` lowers an op it finds in the
   opcode table with whatever its `_HANDLERS` entry returns, and
   `handler(...) if handler else ([], None)` means an op with no entry gets an
   **empty attribute vector**. Stage 1 has no conv lowering, so registering
   the StableHLO name made Stage 1 emit a `kConv` with no attrs; the
   `Cursor`'s underrun throw caught it at RUN time and the engine fell back to
   Python, so the answers stayed right — but the program had already been
   accepted, and two tests in `tests/test_native_tape.py` that use convolution
   as their "an op with no opcode" stand-in failed outright.
   The opcode is therefore registered as **`metaljax.conv`**, the convention
   the M4 emits and `metaljax.msl_scan` already use, and `LowerOp` asks for it
   by that name. `stablehlo.convolution` stays absent from the table, so
   Stage 1 declines the op at COMPILE time with `op stablehlo.convolution` and
   runs it on the Python engine — which is exactly the arrangement the
   milestone wanted, now enforced by the registry rather than by a run-time
   throw.
2. **Python's floor division, again.** `_int_conv` sizes its windows with
   `max(0, (n - span) // stride + 1)`, and a kernel wider than its padded axis
   makes that numerator negative — `(-1) // 2` is -1 in Python and 0 in C++,
   which after the `+ 1` is none versus one. `floor_div` with the reason on
   it, on the RUNTIME side this time (the Python computes these from the
   array, not from the IR, so they could not move into the lowering).
3. **The im2col reduction axis.** `p2[..., None, :] * w2` broadcasts to
   `[N, *out, C_out, K]` and the sum is over `K` — index `p2.ndim()` after the
   inserted axis, not `p2.ndim() - 1`. Every integer row in P6-era style would
   have been 1-D and blind to it; the suite has 2-D and 2-D-strided-dilated
   integer rows for exactly this.
4. **The pad value is typed.** `mx::pad`'s default is `array(0)`, which is
   int32 and could promote an int8 operand; the Python binding treats its
   `0` as weak. `mx::array(0, x.dtype())`, like `extract_windows` does.
5. **The negative-pad rewrite is arithmetic, and it is where silent wrongness
   would live.** XLA pads AFTER lhs dilation, so a negative pad crops the
   DILATED array; MLX would crop the operand. Dropping
   `q = ceil(k / dilation)` operand elements removes `q*dilation` dilated
   entries, and the excess `q*dilation - k` is that many interior holes —
   zeros — so it is added back as a non-negative pad. Two CPU-differential
   rows hold it (with and without dilation), plus an integer one, because the
   integer arm reaches `mx::pad`, which has no negative widths at all.

## Validation

| | result |
|---|---|
| `bash native/build.sh` + `pytest tests/ -q` | **1258 passed** (unchanged: the runtime grew an opcode nothing Python-side emits) |
| `scripts/texmo_check.py` (`METALJAX_ENGINE=native`) | **106 ok, 0 FAIL** |
| `plugin-native/texmo_gate.py` | **106 ok, 0 decline, 0 FAIL, 0 error** |
| `plugin-native/execute_test.py` | **274 checks**, 0 failures (228 in P6) |
| `plugin-native/decline_census.py` | **32 of 35** (31 in P6) |
| `plugin-native/smoke_test.py` | 4/4 checkpoints |
| `plugin-native/wheel_poc_test.py` | 4/4, from a native wheel in a fresh 3.13 venv |
| `bazel test //metal:runtime_gil_free_test` | PASSED |
| eager vs compiled | 234 of 235 bit-identical (the odd one is P5's fusion finding, not a conv row) |

### The conv rows, by what they cover

* **Layouts** — `NCH`/`OIH`, `NCHW`/`OIHW`, `NHWC`/`HWIO`, `NCDHW`/`OIDHW`;
  1-D, 2-D and 3-D.
* **Windows** — VALID, SAME and explicit padding; stride 2; rhs dilation
  (atrous); lhs dilation (the transposed convolution); both dilations at once.
* **Groups** — feature groups, depthwise (`groups == channels`), a 3-D grouped
  convolution (which MLX will not take, so it takes the expanded path), batch
  groups in 1-D and 2-D.
* **Padding corners** — negative padding, negative padding WITH lhs dilation,
  and a negative pad that empties the operand; a kernel wider than its axis
  (zero-size output); a zero-size batch; zero-size channels.
* **Dtypes** — i32/i8/u8 EXACT, integer with both dilations, integer with
  feature groups, integer 2-D and 2-D strided+dilated, integer with negative
  padding; complex64; f16; bf16.
* **0 spatial dims** — plain, feature-grouped, batch-grouped.
* **jax's own wrappers** — `jnp.convolve`, `jnp.correlate`, `lax.conv`,
  `lax.conv_with_general_padding`, `lax.conv_transpose`.
* **Gradients** — `jax.grad` of a strided 1-D convolution and of a 2-D one.
  This is the real test of the lhs-dilation arm: the gradient wrt the input is
  a transposed convolution and the gradient wrt the weights is a
  BATCH-GROUPED one, so one grad exercises the two arms jax's forward
  spelling barely reaches.
* **In regions** — a convolution inside a `scan` body and inside a
  `fori_loop` body.
* **Hand-written modules** — `window_reversal` on the float path and on the
  integer one. jax's `conv_general_dilated` has no parameter for reversal (it
  flips the kernel itself when it wants one), so the op has to be written out.

## Cross-check: there is none to run, and that is expected

Every P-milestone since P2 has proved its lowering by dumping the tape and
diffing it against the one `src/metaljax/tape.py` builds from the same module.
**Convolution has no Stage 1 tape to diff.** `tape.py` declines it — that is
the whole premise of this milestone — so the only reference is the CPU
differential, which is the stronger bar anyway, plus the eager-vs-compiled
arm, which says the 46 conv rows compute the same bits with `mx::compile` on
and off.

Stage 1 itself is unchanged and stays that way: with `METALJAX_ENGINE=native`
a convolution prints

```
[metaljax] native tape declined: op stablehlo.convolution
```

and runs on the Python engine, matching jax-CPU to 9.5e-07. Nobody should
wonder why the two engines disagree about which opcode exists — the runtime
offers `kConv` under a name Stage 1 never asks for.

## Gotchas

1. **A registry entry is a contract with EVERY tape builder**, not just the
   one you are writing. See transliteration note 1: adding an op under its
   StableHLO name silently enrolls Stage 1's lowering, which knows nothing
   about the attribute vector. A new opcode whose handler reads attrs and
   whose Python-side lowering does not exist wants a pseudo-name.
2. **A milestone that closes a gap has to find the tests watching it** —
   P6's own lesson, and it bit twice as hard here. `smoke_test.py` and
   `wheel_poc_test.py` both used convolution as their "an op outside the set
   is refused, naming itself" checkpoint (they had moved there from sort in
   P6); execute_test's "while loop over an unlowered op" decline used it too.
   All three now watch `stablehlo.cholesky`, which is the next milestone's —
   so P8 will have to move them again.
3. **`jax.lax.conv_general_dilated` refuses string padding with lhs
   dilation** ("String padding is not implemented for transposed
   convolution"), on CPU and metal alike. A transposed-convolution case has
   to spell its padding out, or go through `lax.conv_transpose`.
4. **MLX's `groups` is 1-D and 2-D only.** A grouped 3-D convolution is not a
   decline, it is the expanded path — one ungrouped convolution per group,
   concatenated — and it is only exercised if a case actually goes to rank 3.
5. The `precision_config` attribute is ignored, as Stage 1 ignores it: matmul
   precision is pinned process-wide (`MLX_METAL_GPU_ARCH`), so there is
   nothing per-op to honour.

## What P8 should pick up

* **LAPACK on Accelerate** (`cholesky`, `qr`, `eigh`, `svd`, ...): the
  remaining census declines, and the reason the plugin has no host-op path at
  all. It is now the ONLY family between this plugin and the census.
* The two lexicographic sorts, if a `take_along_axis` entry lands.
* `select_and_scatter`, which needs goldens with a tolerance rather than a
  byte differential.
* Still open from P5: asynchronous execute + donation, and MLX's global
  command-encoder map.

## Review observation (main agent): threaded execute under cross-process load

During the review battery, `execute_test.py`'s threaded-execute row FAILED
once while BOTH texmo gates (Stage 1 and plugin, 106 configs each) ran
concurrently in other processes -- and passed 5/5 standalone immediately
after, plus the agent's own runs. Not reproduced, detail line not captured.
Plausible cause: Metal's ~499k live-buffer limit is DEVICE-wide, and three
heavy processes can jointly exceed it where each alone cannot; the in-process
recovery ladder cannot see other processes' buffers. Disposition: observation
recorded; single-process operation (the supported mode) is clean. If it
recurs under normal load, capture the detail line first -- the test prints
the exception text.
