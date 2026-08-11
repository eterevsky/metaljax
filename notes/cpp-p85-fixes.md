# Stage 2 P8.5: the census's fix batch

Follows [`cpp-p8-jax-census.md`](cpp-p8-jax-census.md), which measured the whole
pinned suite through both stacks and classified all 1,918 native-only failures.
Four of its rows are defects rather than missing families, and this milestone is
those four: one bug in the SHARED runtime (so it is a live Stage 1 bug too) and
three in the plugin. Together they are **126 of the census's native-only
failures** plus the four Stage-1 regressions the census found.

Nothing here is a port: every fix is a defect with a one-line repro, and each
repro is written down below so the next reader can re-break it.

| # | defect | tests | where the fix went |
|---:|---|---:|---|
| a | a pipelined while runs an IMPURE cond (or body) one extra time | 4 (Stage 1) | `native/program.cc` (`reads_host`) |
| b | a small-int reduce comes back int32 | 18 | `native/ops_reduce.cc` (`reduce_apply`) |
| c | a zero-size constant declines the program | 52 | `plugin-native/metal/metal_lowering.cc` |
| d | no eager fallback when `mx::compile` refuses a trace | 56 | `native/program.cc` (`run_recovering`) |

## (a) The effect that fired twice

`METALJAX_WHILE_PIPELINE=0` cured it and so did `METALJAX_ENGINE=python`, which
is what says the bug is in the native tape's pipelined dynamic loop — the shape
M5a introduced to keep the device a token ahead of the host. That loop builds
iteration t+1's body and t+2's condition BEFORE it reads t+1's condition back,
which is free for a pure region (an MLX graph nobody evaluates is dropped) and
is a second `debug.print` for one that leaves the tape.

`run_while` already gated on exactly this: `!body->reads_host() &&
!cond->reads_host()`, with a comment saying a host read "would make *building*
it mean RUNNING it". **`reads_host()` only counted control flow.** A `kHostCall`
— which is what `debug.print`, `io_callback` and the LAPACK targets lower to —
was not in its set, so a region that calls back to the host looked pure to the
one predicate whose job is to say it is not.

```python
# JAX_PLATFORMS=metal,cpu, shipped defaults
def f(x):
    def cond(x):
        debug.print("x: {x}", x=x)   # 5 6 7 8 9 10 ... and an 11 that never ran
        return x < 10
    return lax.while_loop(cond, lambda x: x + 1, x)
f(5); jax.effects_barrier()
```

The census saw this only for prints in a COND, and that is an artifact worth
recording: `lax.while_loop(lambda x: x < 10, lambda x: x + 1, x)` is a COUNTED
loop, and the counted path never pipelines — so a print in the body of the
tests' loops is safe for a reason that has nothing to do with the body. Write
the same loop so the analyzer cannot count it (`x * 1.5` against `x < 10.0`) and
the body prints one extra line too:

```
dynamic body print, before: 2.0 3.0 4.5 6.75 10.125   (10.125 never ran)
dynamic body print, after:  2.0 3.0 4.5 6.75
```

Fix: `kHostCall` joins `kWhile`/`kIf`/`kCase` in `Program::reads_host`. One
predicate, read off the tape, so it cannot drift from what the tape holds.

Regression test: `plugin-native/metal/runtime_gil_free_test.cc` builds the loop
three ways — host call in the cond, in neither, in the body — and asserts the
handler's call count (6 / 0 / 5) AND which shape the loop took
(`serial_loops` vs `pipelined_loops`). It is written there rather than in
`tests/` because a host call cannot reach the runtime through the plugin's PJRT
surface, so this is the only process in which the rule can be stated end to
end. On the unfixed runtime it reports 7, 0, 6 and two pipelined loops.

**A pure cond must still pipeline**, and it does: the three canaries in
`tests/test_native_control.py` (`..._says_so`, `..._can_be_turned_off`,
`a_body_that_reads_the_host_is_not_pipelined`) and the `pipelined_loops` row of
the new C++ test all hold, and `tests/test_command_buffer.py`'s pipeline
detector is unchanged.

## (b) `jnp.sum(x, dtype=int8)` came back int32

```
INTERNAL: metaljax-native: jit__lambda result 0 came back as [], the module
declares s8[]
```

MLX's `sum` and `prod` ACCUMULATE WIDER, exactly as numpy does: int8 and int16
come back int32, uint8 and uint16 uint32, bool int32. `min`/`max`/`any`/`all` do
not, and no float type does (measured, all dtypes). A StableHLO reduce returns
its OPERAND's element type, so the tape was handing back the wrong type — caught
by `MetalLoadedExecutable`'s result guard, which is why it was loud.

Stage 1 never saw it because `engine.to_host` casts a result to the declared
numpy dtype on the way out. That hides the boundary case and nothing else: a
reduce whose result feeds more ops on the DEVICE was computing them in the
widened type, where XLA wraps at every step.

Fix: `reduce_apply` folds the accumulator back to the operand's dtype for
`sum`/`prod`. The value is XLA's — two's-complement arithmetic is a ring
homomorphism, so one truncation at the end is the same answer as a truncation at
every step.

The same helper serves `reduce_window`'s monoid arm, so `lax.reduce_window(...,
lax.add, ...)` over int16 was wrong in the same way and is fixed by the same
line — the census had no id for it (its 18 are `sum`/`prod`/`trace`/
`apply_over_axes`), it came out of a probe over the reduce forms.

## (c) "a constant whose raw data is the wrong size"

All 52 are ONE form. Instrumenting the decline to print the numbers and re-running
the 52 ids gives 102 hits of `raw=4 item=4 numel=0 splat=1` and 2 of `raw=2 item=2
numel=0 splat=1` — i.e. **a zero-size constant**, and nothing else.

MLIR stores a zero-size dense attribute as a SPLAT holding one raw element:
`dense<1.000000e+00> : tensor<0xf32>` has four bytes of raw data under a shape
with no elements. The lowering's `numel > 1 && isSplat()` arm did not apply
(numel is 0), so the value fell through to the branch that expects the raw data
to BE the elements, whose length check then failed.

Where they come from: chlo's decompositions, whose constants take the shape of
their operand — `jnp.sinh(x)` on an empty array is 13 ops over `tensor<0xf32>`,
two of them constants. That is why the census bucket is full of
`lax_numpy_operators_test` rows for `sinh`, `cosh`, `arcsin`, `nextafter`,
`spacing`: those tests carry a `shape=(0,)` case.

Fix: a `numel == 0` arm ahead of everything else — there is nothing to decode,
and `zeros` of a zero-size shape holds every element the constant has. It is
ahead of the BOOL arm on purpose: a zero-size bool constant would otherwise
have gone through the bit-unpacking path and handed `OwnedArray` the data
pointer of an empty vector. The remaining decline is unchanged in meaning and
now names the sizes.

Covered by three jitted `execute_test` rows (an empty int8 through `sinh`, an
empty f16 through `spacing`, an empty f32 through two chlo composites) and a
hand-written module whose constants are zero-size **bool** and **i32** — the
two the jax lowerings never hand us empty.

## (d) MLX's compiler refuses a trace, and the program died

```
INTERNAL: metaljax-native: jit_polygamma failed: [compile] Too many
inputs/outputs fused in the Metal Compiled primitive.
```

MLX's generated kernel binds one buffer per fused input, output and stride
vector, and the most argument-hungry VARIANT decides even when the call would
dispatch to a cheaper one (`notes/data/mlx-fused-args-repro`, which is the same
bug reported upstream). Stage 1 has caught this since 0.2.1: `engine.execute`'s
`except (RuntimeError, IndexError, ValueError)` arm clears `_can_compile` and
re-runs the pure program through the interpreter.

The native runtime already owns the machinery — `run_recovering` retires a
compiled path and runs eagerly — but the failure never reached it: **a compiled
call only BUILDS the graph**, MLX generates the kernels at EVAL, and the eval
was the plugin's, three lines after `run` returned.

Fix: `run_recovering` settles the outputs itself on the FIRST compiled call of a
program, inside the ladder. Once per program, exactly like `BodyRunner`'s probe
of a freshly bound body: an executable's shapes are fixed for its life, so one
buildable call proves every later one and the steady state keeps handing back
lazy arrays. Not while an msl plan is unproven — `settle_msl` owns that eval,
and a generated kernel that fails to build must be retired rather than blamed on
the graph that traced it.

Repro and `execute_test` row: `jax.scipy.special.polygamma(2, x)` (334 entries
of elementwise chain) is refused today, falls back, and matches CPU to 1.3e-05.
`METALJAX_DEBUG=1` prints `compiled native tape failed; running eagerly`, once.

## (e) `lax_test.py::LaxTest::testScatter1` — classified

**A race the test is not entitled to win, not an order dependence in the
plugin.** Ruled out first, cheaply:

* the inputs are the same wherever the test runs — `jtu` seeds each test's rng
  from `adler32(testMethodName)`, so position in the file cannot change a draw;
* the ORDER is the same too: `pytest-randomly` is not installed, and the file
  collects the same 2,573 tests every run.

So it is device state, and 14 full-file runs put the rate at ~1 in 14 (the
census's standalone 3/3 and a `-k Scatter` run of 82 tests never see it). Caught
on the 17th run with a full traceback, it is **one element of ten**:

```
actual  (jit)   [..., -1.1107875, -0.56969315, -1.7562885,  4.5564556]
desired (eager) [..., -1.1107875, -0.56969315,  2.8813667,  4.5564556]
                                                ^ index 8
```

The case is `operand (10,)`, three index rows, `update_window_dims=(1,)` — i.e.
three windows of two — and the draw (spied through `_CompileAndCheck`) is
`idxs = [9, 7, 8]`. XLA clamps a scatter start so the window fits, so those
windows are `[8,9]`, `[7,8]` and `[8,9]`: **index 8 is written by all three
updates**. Which write survives is implementation-defined in XLA and is a plain
race in a GPU kernel — the same property this repo already records for
scatter-add (CLAUDE.md item 10, "GPU scatter-add is order-nondeterministic, like
jax-CUDA"). `_CompileAndCheck` then asserts the jit'd result equals the eager
one, which is an assertion about a program that has no single answer.

Verdict: **in-suite lottery, expected to stay flaky, wontfix as a correctness
item.** 200 jit + 200 eager repeats of the same scatter in a quiet process agree
every time, which is why it takes a loaded suite to see it; making it
deterministic would mean serializing duplicate scatter writes, which is a real
cost for a program XLA does not promise anything about. It is Stage-1-visible
today and equally possible through the native plugin (which passed it in both
census runs and in P8.5's rerun).

## The delta

The census's affected files, re-run through the native plugin exactly as the
census ran them (one process per file, `--tb=no`, strictly sequential):

| file | before | after | |
|---|---:|---:|---:|
| `jet_test.py` | 1 | 0 | -1 |
| `lax_control_flow_test.py` | 8 | 6 | -2 |
| `lax_numpy_operators_test.py` | 33 | 1 | -32 |
| `lax_numpy_reducers_test.py` | 7 | 0 | -7 |
| `lax_numpy_test.py` | 57 | 48 | -9 |
| `lax_scipy_special_functions_test.py` | 15 | 0 | -15 |
| `lax_test.py` | 63 | 63 | 0 |
| `linalg_test.py` | 361 | 349 | -12 |
| `random_lax_test.py` | 36 | 32 | -4 |
| `scipy_stats_test.py` | 74 | 40 | -34 |
| `shape_poly_test.py` | 191 | 181 | -10 |
| **total** | **846** | **720** | **-126** |

Exactly the 126 predicted (18 + 52 + 56), all of them: the set difference over
the failing ids is empty in both directions — every one of the 126 census ids
passes, and **no id that passed before fails now**. Whole-suite arithmetic, if
the rest of the suite is unchanged: native-only failures 1,918 -> 1,792, pass
rate 92.70 % -> 93.15 %.

The one survivor in `lax_numpy_operators_test` is `testSpacingSubnormals0`,
which was never in the constant-decode bucket (33 - 32 = 1).

| file | what |
|---|---|
| `notes/data/p85-native-affected-summary.csv` | per-file pass/fail/skip/seconds, the 11 files, after |
| `notes/data/p85-native-affected-failures.txt` | the 720 failing ids that remain in them |

## Findings the fixes turned up

1. **The 8-thread `execute_test` contract fails ~5 % of runs with
   `There is no Stream(gpu, N) in current thread`** — and it does so on the
   unmodified tree too (2 failures in 40 rounds on both builds, at the SAME
   rounds), so it is not P8.5's. `METALJAX_COMPILE=0` makes it 0 in 40: the
   shape is a compiled graph traced on one pool thread and replayed from
   another after the tracer has exited, taking its `new_thread_unsafe_stream`
   with it. Pre-existing, reproducible, and worth a milestone of its own — it is
   the same thread-bound-stream area as P4's submission lock.
2. MLX's integer reduce promotion is a whole table, not one case (b above), and
   the only reason it had never bitten Stage 1 is a cast at the host boundary.
3. The census's "prints in a cond" framing is a counted-loop artifact (a above):
   the pipelined loop was wrong about bodies too.
4. A manufactured fused-arg refusal is harder to write than it looks. 40
   elementwise inputs, 24 outputs, 40 array constants: MLX fuses all of them
   without complaint, because the fusion pass caps `max_compile_arrays` at 24 on
   its first traversal. The refusal needs the graph shape that defeats the
   RE-collection pass (`notes/data/mlx-fused-args-repro`), so the
   `execute_test` row is a real jax program that hits it —
   `jax.scipy.special.polygamma`.

## Battery

`native/build.sh` + `bazel build`; `pytest tests/ -q` **1258**; Stage 1
`scripts/texmo_check.py` **106 ok, 0 FAIL**; plugin `texmo_gate.py` **106 ok, 0
decline, 0 FAIL**; `bazel test //...` (the GIL-free runtime test, now with the
loop-shape section); `execute_test.py` 274 -> **284 checks**, all matching CPU;
`smoke_test.py`; `wheel_poc_test.py` in a fresh 3.13 venv off a freshly built
native wheel; `debugging_primitives_test.py` **56 passed** on Stage 1 (was 52
passed / 4 failed); `tests/test_native_control.py` + `tests/test_command_buffer.py`
**83 passed**, which is what says a PURE cond still pipelines.
