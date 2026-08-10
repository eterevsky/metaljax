# Stage 2 P2: the first end-to-end native execute

Follows [`cpp-p1-runtime.md`](cpp-p1-runtime.md), which put the executor
runtime behind `native/program.h` into the plugin's dylib without anything
calling it. P2 connects the two halves: a C++ tape builder over the parsed
StableHLO module, a `PjRtLoadedExecutable` that replays the tape, and the
buffer path widened from f32 to the whole M1 dtype table.

**Milestone zero, on the new stack:**

```
$ JAX_PLATFORMS=metal METALJAX_PLUGIN_PATH=.../libmetal_pjrt_native.dylib \
    .venv/bin/python -c "import jax.numpy as jnp; print(2 * jnp.array([1,2,3]))"
[2 4 6]
```

No Python below `jax` — `metaljax.engine` is never imported (smoke_test.py
checkpoint 2 asserts it), and the dylib contains no CPython.

## What was built

| file | lines | what |
|---|---:|---|
| `metal/metal_lowering.{h,cc}` | 1,194 | StableHLO -> `metaljax::Program`; the whole decline discipline |
| `metal/metal_executable.{h,cc}` | 421 | `MetalLoadedExecutable` + the `MetalExecutable` that answers the metadata |
| `metal/metal_dtypes.{h,cc}` | 174 | one element-type table, read by the transfer path, the lowering and the result metadata |
| `metal/metal_stream.{h,cc}` | 59 | `BindThread`: engine.py's per-thread MLX stream, in C++ |
| `metal/metal_names.h` | 30 | the identity strings, extracted so client and executable can share them |
| `metal/metal_buffer.{h,cc}` | +134/-30 | all dtypes, widened f64/c128 egress, contiguity discipline, external references |
| `metal/metal_client.{h,cc}` | +109/-51 | strided/typed ingest; `CompileAndLoad` builds an executable |
| `execute_test.py` | 534 | the differential suite (new, permanent) |

Dylib: 165,481,672 -> **165,596,936 B** (+115,264, +0.07 %). Build times
unchanged: a touched `.cc` recompiles in ~5 s, the link is ~2 s.

## What lowers

Op for op, and each one is a transliteration of the `_lower_*` in
`src/metaljax/tape.py` that writes the same attribute vector:

* **elementwise, no attributes** (43 names): `abs cbrt ceil cosine erf
  erf_inv exponential exponential_minus_one floor is_finite log log_plus_one
  logistic negate not round_nearest_afz round_nearest_even rsqrt sign sine
  sqrt tan tanh`, `chlo.{erf,erf_inv,square}`, `add multiply subtract maximum
  minimum and or xor divide remainder power atan2`, `select clamp`, and the
  complex trio `real imag complex`;
* **with attributes**: `compare` (direction + the TOTALORDER key arm),
  `convert` (dtype + the complex->real arm), `reshape`, `transpose`,
  `broadcast_in_dim` (including the unsorted-dims transpose arm), `slice`,
  `concatenate`, `iota`, `pad` (all three stages: interior dilation, edge
  pads, negative-pad crop), `dot_general` (all four arms: float matmul,
  exact-f32 K-chunks, int64 outer product, bool), `constant`;
* **`reduce`**, in the two forms `ops/reduction.py` recognizes structurally:
  the single-operand monoid (`sum prod max min any all`, with the bool table
  chosen by the operand's element type) and the `(values, indices)` pair jax
  lowers argmax/argmin to;
* **aliases**, lowered by binding a slot and emitting nothing:
  `optimization_barrier`, `sdy.sharding_constraint`, `sdy.reshard`;
* **calls**, spliced in rather than lowered: `func.call` and
  `stablehlo.composite`, exactly as tape.py's `_inline` does. Not optional in
  practice — jax lowers `jnp.pad`, `jnp.clip`, `argmax` and half of
  `jax.nn` through `func.call`, so without inlining those all declined on the
  call rather than on anything about their arithmetic.

Dtypes: `bool s8 s16 s32 s64 u8 u16 u32 u64 f16 f32 bf16 c64`, plus f64/c128
as **pass-through** (stored f32/c64, widened back on egress). An f64
*computation* declines naming the element type, which is the Stage 1 STRICT
policy unchanged. complex64 is in the table because its storage IS its bits;
the C99 arms in `native/ops_elementwise.cc` do the rest, and `jnp.abs` on a
complex64 array matches jax-CPU to 0 ULP in the suite.

## What declines

Everything else, naming the op, whole-program: control flow (`while`, `if`,
`case`), `gather`, `scatter`, `sort`, `top_k`, `rng_bit_generator`, `fft`,
`reduce_window`, `dynamic_slice`/`dynamic_update_slice`, `bitcast_convert`,
`reverse`, `shift_*`, `popcnt`/`count_leading_zeros`, `custom_call` of every
kind (so no host ops, no LAPACK, no callbacks), general reduce bodies, and
every recognizer emit (qmm/sdpa/moe). Also declined: a module with no `@main`,
a multi-block function, a recursive call, a dynamic dimension, and any element
type outside the table.

Deliberately NOT built in P2, and each is a phase of its own: `mx::compile`
of a lowered program (`set_compile` is left off — the cost and byte
estimators that decide it are Stage 1's analysis, and they move with the
recognizers), the counted-loop and msl_scan machinery, donation, and the
asynchronous execute.

## The lowering, and where it is stricter than tape.py

Structure is tape.py's: slots by a monotone counter, one `Pending` entry per
op, drop lists derived at the end by "highest index that reads it", and the
same two aliasing taints (`arg_alias`, `const_view`) driving the output-copy
rule. Two deliberate differences:

1. **The taints are booleans here, not sets.** tape.py tracks WHICH argument
   a slot may be because a region maps its taints back through a call's
   operands; P2 lowers no regions, and the only question an output position
   asks is whether the slot may be an argument's array at all.
2. **A direct argument return is copied too.** tape.py exempts an output whose
   terminator syntactically names a block argument, because `engine.execute`
   copies those on the way out whatever engine ran. There is no
   `engine.execute` here, so the plugin copies them itself. Strictly more
   copies than Stage 1, never fewer, and `execute_test.py`'s
   `unsafe_buffer_pointer` check on a jitted identity is what holds it.

`Program::run` handles the other half of XLA's no-alias contract on its own
(two outputs reading one slot are one array; it copies the duplicate).

## Attribute encodings a reviewer should scrutinize

The tape's attribute vectors are the contract between this builder and
`native/ops_*.cc`, which also decodes tape.py's. A disagreement is a wrong
number, not a build error. The four with real content:

* **`kBroadcastInDim`** `[transpose?, in_rank, perm..., out_rank, interim...,
  out_shape...]`. The perm is `sorted(range(n), key=dims.__getitem__)` and
  must be built with a STABLE sort to match Python's `sorted` (StableHLO's
  verifier makes broadcast_dimensions unique, so the tie case cannot arise —
  but relying on a verifier's promise for a bit-level agreement is exactly the
  kind of thing that later stops being true).
* **`kDotGeneral`** `[lrank, lperm..., rrank, rperm..., B, M, K, N, out_dtype,
  out_rank, out_shape..., kind, chunk]`. `kind` and `chunk` are resolved from
  the operand dtypes here, once; `ExactF32Chunk` is `_exact_f32_chunk`
  including its `bit_length() - 1` rounding (i8xi8 -> 1024, i8xu8 -> 512,
  u8xu8 -> 256).
* **`kPad`** three `(flag, vector, vector)` groups, read by the handler
  whether or not their flag is set, so the cursor stays aligned. The shapes
  each stage produces are computed in sequence — the crop's `end` is measured
  against the shape AFTER dilation and edge padding, not against the operand.
* **`kReduce` / `kArgReduce`**: which one an op becomes is decided by reading
  the body block structurally, in tape.py's order (monoid first, then the
  compare-carrying pair). A body neither recognizes declines rather than being
  guessed at — the generic pairwise arm needs a sub-Program, which is a later
  phase.

`stablehlo.constant` is the one place P2 does something tape.py cannot:
`DenseElementsAttr::getRawData()` hands us the elements as bytes, so bf16 and
f16 cross with no decoding at all (the Python bindings cannot cast a bf16
dense attr to numpy, which is why `_ir.dense_to_np` has a text/hex path).
Three rules are kept from `ops/elementwise.py::_constant`: a splat of more
than one element is a broadcast from a ONE-element buffer and never
materialized; `i1` is read through the typed iterator because dense i1 data is
bit-packed; everything else is a raw copy whose length is checked against
`numel * itemsize` and declines if it disagrees.

## The executable: two classes, and why

`PjRtLoadedExecutable` is **not** a `PjRtExecutable`. It owns one (a
`PjRtExecutableForwarder` by default) and the C-API wrapper asks THAT object
for the output element types and dimensions it hands jax
(`PJRT_Executable_OutputElementTypes` -> `executable->get()->...`). XLA's
default answer derives them from `GetHloModules`, which a plugin holding a
tape cannot give.

Overriding `GetOutputElementTypes` on the *loaded* executable does not help —
that is not the object being asked, and the forwarder would bounce the
question back into an infinite recursion. The fix is `MetalExecutable`, a real
`PjRtExecutable` answering from the MLIR function's own signature (recorded at
compile), returned by `GetExecutable()`. Symptom before that landed, and it
is a confusing one because it surfaces at EXECUTE time on an unrelated
primitive:

```
jax.errors.JaxRuntimeError: UNIMPLEMENTED: metaljax-native: the executable
  is a tape, not an HLO module.     # ...raised from convert_element_type
```

Also implemented because jax asks for them: `GetParameterLayouts` /
`GetOutputLayouts` (dense major-to-minor, the only layout this backend has),
`GetParameterMemoryKinds` / `GetOutputMemoryKinds`, and
`AcquireExternalReference` on the buffer, which is what
`unsafe_buffer_pointer` goes through.

**Execute is synchronous in P2**: `Program::run` then `mx::eval` on the
outputs, so every buffer handed back is honestly ready and `GetReadyFuture` is
not a lie. The Stage 1 engine instead leaves outputs lazy and submits with
`async_eval` — that, plus `set_compile`, is the first thing to revisit when
this plugin is measured rather than tested.

**Thread binding**: `metal_stream.cc`'s `BindThread` is engine.py's
`bind_thread`, called at every PJRT entry that touches MLX (execute, ingest,
egress). A function-local `thread_local` initializer is the "once per thread"
the Python side gets from `threading.local`. `execute_test.py` runs 32
executes of one executable across 8 threads; without the binding this is the
shape that aborts the process rather than failing a test.

## Validation

**`plugin-native/execute_test.py`** — 77 checks, all green, `exit 0`. Every
expression is compared against jax on the CPU backend, computed in a
subprocess (a process with `JAX_PLATFORMS=metal` can see no other platform, so
an in-process reference would compare metal with itself). Integers and bools
compare EXACTLY; floats widen to f64 first, and NaN/inf placement is checked
before the finite parts.

| case | max error | | case | max error |
|---|---:|---|---|---:|
| 2*x int32 (milestone zero) | 0 | | pad | 0 |
| elementwise chain f32 | 1.6e-07 | | pad with interior dilation | 0 |
| unary mix f32 | 1.9e-06 | | pad with negative edges (a crop) | 0 |
| logistic/erf/sqrt f32 | 2.4e-07 | | rank-0 scalar | 0 |
| elementwise chain f16 | 0 | | empty array | 0 |
| elementwise chain bf16 | 0 | | matmul 2D | 0 |
| integer arithmetic | 0 | | batched matmul | 0 |
| unsigned arithmetic | 0 | | dot with batch and free dims | 0 |
| bool logic | 0 | | vector dot | 2.4e-07 |
| sum over one axis | 0 | | matmul 128x128 | 1.9e-05 |
| sum over two axes | 0 | | int32 matmul | 0 |
| max/min/prod | 0 | | int8 matmul | 0 |
| any/all | 0 | | select / compare / clamp | 0 |
| sum of everything | 6.0e-08 | | comparison ladder | 0 |
| argmax / argmin | 0 | | maximum / minimum | 0 |
| argmax with a NaN | 0 | | f32 / bf16 / int / bool constants | 0 |
| integer sum | 0 | | splat + bf16 splat constants | 0 |
| transpose + reshape | 0 | | convert chain, bool convert | 0 |
| transpose rank 3 | 0 | | complex arithmetic | 0 |
| broadcast | 0 | | complex from parts | 0 |
| broadcast unsorted dims | 0 | | several outputs / identity / dup | 0 |
| strided slice | 0 | | constant output / no argument | 0 |
| concatenate (axis 0 and 1) | 0 | | dense + norm + gelu | 3.6e-07 |
| iota | 0 | | softmax | 3.0e-08 |

Plus: three declines that must name their op (`stablehlo.sort`,
`stablehlo.gather`, `stablehlo.while`); the no-alias contract on a jitted
identity; a host round-trip for all 13 dtypes and for a negative-stride numpy
view; 32 executes on 8 threads; and the f64 policy (buffer passes through,
`a * 2` declines with "element type f64") in a subprocess, since
`jax_enable_x64` is a global switch.

**Tolerances.** f32 elementwise 1e-6; halves 5e-3; contractions 1e-5. That
band is narrow on purpose: the M5's low-precision f32 matmul (CLAUDE.md's
~4e-3 neural-accelerator path) is **not** in play, because
`src/jax_plugins/metal/__init__.py` pins `MLX_METAL_GPU_ARCH=applegpu_g16g`
before dlopening the plugin — on the native branch too, and it says so. I
checked rather than assumed: 512x512 f32 matmul is 7.6e-7 relative against an
f64 reference here, where jax-CPU itself is 1.3e-6, and setting the variable
by hand changes nothing.

**The other suites.**

| | result |
|---|---|
| `plugin-native/smoke_test.py` | 3/3 checkpoints (checkpoint 4 rewritten: it now computes) |
| `plugin-native/wheel_poc_test.py` | 4/4, from a native wheel in a fresh Python 3.13.5 venv |
| `bazel test //metal:runtime_gil_free_test` | PASSED |
| `pytest tests/ -q` | 1258 passed, unchanged (nothing under `native/` or `src/` was touched) |

Checkpoint 4 of the two smoke tests used to assert that compilation FAILED.
Both now assert that `jit(a * 2)` computes, that `2 * jnp.array([1,2,3])` is
`[2 4 6]`, and that `jnp.sort` still declines naming `stablehlo.sort` — the
decline half kept deliberately, because "it compiles now" is only half the
contract.

## Tape cross-check against Stage 1

`METALJAX_DUMP_TAPE=1` prints the finished tape one entry per line
(`opcode ins -> outs [attrs]`). The Stage 1 lowering was dumped in the same
format by wrapping `Program.add`/`set_outputs` (scratch script, not checked
in), and the two were diffed over six programs chosen to cover the encodings
with content: `tanh(a@b+a).sum(0)`, a broadcast/slice/concatenate mix, a
pad+iota+compare+convert chain into bf16, an argmax pair with a transpose, a
clip/convert/prod, and a batched einsum.

**51 tape lines across 6 programs, byte-identical** — same opcodes, same slot
numbering, same attribute vectors, same output and copy sets. The only
difference in the raw dumps is that the Stage 1 recorder sees `set_outputs`
twice (tape.py's `_build` calls it once bare, `run` again with the copies).

```
### tanh(a@b+a).sum(0)
[tape] stablehlo.constant  -> 2 [] const
[tape] stablehlo.dot_general 0,1 -> 3 [2,0,1,2,0,1,1,4,4,4,10,2,4,4,0,0]
[tape] stablehlo.add 3,0 -> 4 []
[tape] stablehlo.tanh 4 -> 5 []
[tape] stablehlo.reduce 5,2 -> 6 [0,1,0]
[tape] outputs 6 copies  slots 7
```

Entry-for-entry agreement was not the bar (Stage 1 may fold differently); it
is what happened, and it is the strongest evidence available that the two
builders write the same attribute layouts.

## Gotchas

1. **The metadata recursion** (above). `PjRtLoadedExecutable`'s forwarding
   methods and `PjRtExecutableForwarder`'s forward to each other; every one
   of them must be answered on one side or the other. Overriding on the
   loaded executable alone is silently useless for the output-shape family
   and an infinite loop for the rest.
2. **`func.call` is everywhere.** jax's lowering of `jnp.pad`, `jnp.clip`,
   `argmax`, `jnp.sort` and much of `jax.nn` goes through it. Before inlining
   landed, six of the probe's twenty cases declined on `func.call` and not one
   of them on anything to do with its own arithmetic — a decline list read
   before that point would have been badly misleading.
3. **`MLX_METAL_GPU_ARCH` is already handled**, by the loader, on the native
   branch (`src/jax_plugins/metal/__init__.py` sets it before `dlopen`). It
   could not be fixed from inside the dylib anyway: libmlx is a dependent
   library, so its initializers run before ours.
4. **Do not filter duplicate lines when diffing the two tape dumps.** The
   Stage 1 recorder prints `set_outputs` twice; `uniq` is the right tool,
   dropping the line is not (it hides a real disagreement in the copy set).
5. **jax canonicalizes 64-bit integers on ingest** with x64 off, so a
   round-trip test that asserts `back.dtype == src.dtype` fails on int64/uint64
   for reasons that have nothing to do with the plugin. Ask
   `jax.dtypes.canonicalize_dtype` what to expect.
6. **A cancelling f32 sum is not a threading test.** The first version of the
   8-thread check compared `tanh(x*2).sum()` against numpy and failed at
   1.0e-5 relative — the sum is near zero, so relative error says nothing.
   An all-positive reduction measures what was meant.
7. **`GetOnDeviceSizeInBytes` reports the WIRE size**, matching the Stage 1
   plugin: for an f64 buffer the device really holds half that, but a caller
   sizing a host transfer from the smaller number comes up short.

## What P3 should pick up

* `set_compile` and the analysis behind it (cost, bytes, purity), then the
  counted-loop recognizer — the tape supports all of it already, and none of
  the decisions may be re-derived in C++ independently of Stage 1's while both
  engines exist.
* Control flow (`while`/`if`/`case`) as sub-Programs, which is the single
  biggest decline class and the one every training step needs.
* `gather`/`scatter`, whose index plans are the longest attribute encodings in
  `program.h` and where the differential suite will earn its keep.
* Asynchronous execute (`async_eval` + a real `GetReadyFuture`), and donation.
* The runtime cadences: `metaljax::configure` is never called, so the eager
  flush/clear budgets sit at their compiled-in defaults and
  `METALJAX_EAGER_FLUSH_MB` and friends are ignored by this plugin. Stage 1
  reads them in Python and copies them in; a second reader in C++ would be a
  second opinion on numbers the command-buffer lottery is pinned to, so the
  right fix is a deliberate one, not an `getenv` sprinkled here.
