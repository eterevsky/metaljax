# Stage 2 P5: the compile decisions, ported

Follows [`cpp-p4-gather-scatter.md`](cpp-p4-gather-scatter.md), which ended with
one suite row still flickering: `db18-b4l1024` came back 21.3 off jax-CPU on one
run and 6.2e-3 on the next, and the Stage 1 engine reproduced it exactly under
`METALJAX_COMPILE=0`. The plugin ran everything eagerly because P3 wrote the
three compile fields of a while entry as zeros and never called `set_compile`.
P5 computes them.

```
$ .venv/bin/python plugin-native/texmo_gate.py --only db18-b4l1024      # P4
FAIL   db18-b4l1024   worst=3.11e+01     ok~  worst=1.33e-02    FAIL  worst=5.42e-03
$ .venv/bin/python plugin-native/texmo_gate.py --only db18-b4l1024      # P5
ok     db18-b4l1024   worst=3.43e-05     ok   worst=3.51e-05    ok    worst=4.11e-05
```

So this is a CORRECTNESS milestone with a performance side effect, exactly as
P4 predicted: a compiled or chunk-replayed loop puts its sync points where MLX
0.32's command-buffer split does not corrupt them, and an eager 1024-step loop
does not. The side effect is 1.7-3.5x on the same chunks.

## What was built

| file | lines | what |
|---|---:|---|
| `metal/metal_lowering.cc` | +437/-8 | the six estimators, the two decisions, the dump's compile line |
| `metal/metal_executable.cc` | +12/-2 | the compiled path in the `METALJAX_DEBUG` stats |
| `metal/metal_client.cc` | +3/-2 | `compiled=` in the lowering report |
| `metal/metal_lowering.h` | +3 | `LoweredProgram::compiled` |
| `execute_test.py` | +109 | three control-flow rows and the eager-vs-compiled arm |

Dylib: 165,732,040 -> **165,735,912 B** (+3,872, **+0.002 %**) — the decisions
are a few kilobytes of analysis in a 165 MB statically-linked LLVM. Edit loop
unchanged (~5 s to recompile `metal_lowering.cc`, ~2 s to link).

## The decisions, and where each one comes from

Nothing here is new policy. Every number is `src/metaljax/`'s, and the table is
the map a reviewer should read the diff against:

| written | Stage 1 anchor |
|---|---|
| `set_compile(true, anchors, 1)` on main | `engine.MetalExecutable.runner` -> `tape.py::run(compile_main=...)` |
| main's gate: `COMPILE_ENABLED and main_pure and cost <= _TRACE_BUDGET and _bytes_ok(blk, 1, whole=True)` | `engine.py` lines 476-495 |
| `anchors` | `ops/control._underived_outputs(block, [])` |
| `chunkable` | `tape.py::_while`: `COMPILE_ENABLED and cost <= _CHUNK_MAX_COST and pure and body not in interp._no_chunk` |
| `kmax` | `max(1, min(_TRACE_BUDGET // cost, _CHUNK_MAX, _bytes_chunks(body)))` |
| `body_compile_max` | `_body_fn`'s gates (purity, op budget, byte budget) solved for `repeat`: `max(0, min(by_cost, by_bytes))` |
| `set_compile(true, anchors, body_compile_max)` on the body | `tape.py::_while`, anchors `_underived_outputs(body_block, body_free)` |
| `BlockIsPure` | `interpreter.block_is_pure` (+ `_IMPURE_OPS`) |
| `WhileTraceable` | `ops/control._while_traceable` (the `while_traceable_hook`) |
| `OpBytes` / `BlockBytes` / `PassthroughBytes` / `ProgramBytes` | `interpreter.op_bytes`, `ops/control._block_bytes` / `_passthrough_bytes` / `program_bytes` |
| `BytesOk` / `BytesChunks` | `ops/control._bytes_ok` / `_bytes_chunks` |
| budgets from the environment | `METALJAX_{COMPILE,TRACE_BUDGET,BODY_COMPILE,CHUNK_MAX,CHUNK_MAX_COST,COMPILE_BYTES_MB}` |

Four things about the transliteration are worth a reviewer's attention, because
each is a wrong NUMBER rather than a build error:

1. **`by_bytes` must not be rounded up.** `BytesChunks` never returns less than
   1 (its callers ask "how many iterations may one trace hold"), but
   `body_compile_max` is `_body_fn`'s byte gate solved for `repeat`, and that
   gate says NO when a single iteration is over budget. Flooring is what makes
   an over-budget body run eagerly instead of compiled, and a compiled body
   holds every intermediate of its iteration instead of flushing inside it.
2. **`interp._no_chunk` / `_no_body_compile` have no term here.** They are what
   the Python ENGINE remembers about a body whose chunk or compiled call failed
   at run time; the executor keeps the same memory itself
   (`Program::set_no_chunk`, `Program::drop_compiled`). Both are empty at
   lowering on both engines, which is what a tape diff compares.
3. **The purity walk's two dead arms are written anyway.** A `custom_call` with
   a host handler and a token-carrying value both decline this plugin's
   lowering long before purity is asked, so neither can fire — they are in the
   code because a reader checks a transliteration line by line, and they cost a
   type comparison.
4. **`WhileTraceable` has no msl arm.** `_msl_plan_for`'s early yes is what the
   Python answers with kernels enabled; this plugin generates none, which is
   the neutral answer `METALJAX_MSL=0` gives — the setting the tape cross-check
   runs the Stage 1 side under.

`_splat_broadcast()` has no term either, and that one is a real (if harmless)
asymmetry: `interpreter.op_bytes` reads `METALJAX_SPLAT_CONST` so the estimate
and the runtime tell the same story, while this plugin's `LowerConstant`
broadcasts a splat unconditionally. Charging a splat one element is therefore
right here whatever that variable says — but if the plugin ever learns to
honour it, `OpBytes` has to learn with it.

## What the decisions do to a real tape

`fori_loop(0, 4, lambda i, c: c + 1.0, x)`, `METALJAX_DUMP_TAPE=1`:

```
[tape] stablehlo.while 3,0,2,1 -> 4,5 [2,1,1,1,0,0,4,7,64,1,16,2857]
[tape] region 1 {
  ...
  [tape] outputs 6,5 copies  slots 7
  [tape] compile anchors  max_repeat 2857
[tape] }
[tape] outputs 5 copies  slots 6
[tape] compile anchors  max_repeat 1
```

`chunkable=1 kmax=16 body_compile_max=2857` where P3 wrote `0,1,0`, and both
programs carry a `set_compile`. (This particular loop is small enough to be
UNROLLED into main's trace instead, which is the third thing the decisions buy
— `WhileTraceable` is what lets `BlockIsPure` see through it so main compiles
at all.)

The dump's compile line is new and deliberately its own line: a tape that
compiles nothing still diffs byte for byte against the dumps recorded in P3 and
P4.

## Flicker kill

**Standalone, 5 runs each** (`texmo_gate.py --only`, so each is the only
configuration in its process):

| run | db18-b4l1024 P4 (eager) | db18 P5 | synth-matlstm-b P4 | matlstm P5 |
|---|---|---|---|---|
| 1 | **FAIL** 3.11e+01 | ok 3.43e-05 | ok~ 2.33e-02 | ok 1.26e-04 |
| 2 | ok~ 1.33e-02 | ok 3.51e-05 | ok~ 2.02e-02 | ok 1.05e-03 |
| 3 | **FAIL** 5.42e-03 | ok 4.11e-05 | ok~ 1.24e-02 | ok 2.40e-04 |
| 4 | — | ok 1.93e-05 | — | ok 1.84e-04 |
| 5 | — | ok 4.29e-05 | — | ok 7.31e-04 |

db18 collapses from four orders of magnitude of spread to 1.9-4.3e-05 — the
band the Stage 1 engine has always been in (P4 measured 4.696e-05 there) — and
both rows stop needing the sensitivity scaling at all (`ok`, not `ok~`).
matlstm's remaining spread is the harness's, not the arithmetic's:
`texmo_check` re-samples its training data every run, so `sens` and the
tolerance move with it.

**Full suite, 3 runs** (and three more before the `execute_test` additions, same
result): 106/106 ok, 0 decline, 0 FAIL, 0 error, every time. The `ok~` count
moves 20/24/21 with the data sampling. The two rows in question, IN suite:

| run | db18-b4l1024 | synth-matlstm-b |
|---|---|---|
| 1 | ok 2.84e-05 | ok 2.46e-04 |
| 2 | ok 2.20e-05 | ok 4.28e-04 |
| 3 | ok 3.76e-05 | ok 1.03e-04 |

**And the same build with the switch off** — `METALJAX_COMPILE=0`, three runs of
`--only db18-b4l1024` — puts the flicker straight back: **FAIL** 5.36e-02,
**FAIL** 1.15e-02, **FAIL** 1.37e-02. That is the P4 plugin, reproduced by an
environment variable, which is what makes it the control for everything above.

## Tape cross-check, WITH compile on

Same method as P2-P4: `METALJAX_DUMP_MODULE=1` prints the module XLA's parse
handed the plugin (never the module jax printed), `METALJAX_DUMP_TAPE=1` prints
the finished tape, and a scratch driver lowers THAT module through `tape.py`
with a recorder standing in for `engine.NATIVE.Program`, rendering the same
format. The Stage 1 side runs at `METALJAX_MSL=0` and otherwise DEFAULT
settings — compile on, which is the point.

| probe | lines | verdict |
|---|---:|---|
| scan | 24 | identical |
| fori | 17 | identical |
| fori (4 iterations, unrolled into main) | 18 | identical |
| fori (5000 iterations) | 17 | identical |
| dynamic while | 15 | identical |
| cond | 17 | identical |
| switch | 21 | identical |
| scan over a matmul | 28 | identical |
| nested scan | 35 | identical |
| grad of an MLP | 24 | identical |
| 64-step train chunk (scan of grad + SGD) | 43 | identical |

**259 lines over 11 probes, byte for byte**, including every `chunkable`,
`kmax` and `body_compile_max`, both `set_compile` calls and their anchor lists.
The `kmax` field that differed in P3 and P4 (16 vs the dead 1) now agrees:
`tape.py` computes it unconditionally, so the two tapes are identical under
`METALJAX_COMPILE=0` as well.

## Perf, recorded not gated

One gate training chunk (8 steps), best of 3 x 20 executes, every output read
back at the end of the timed loop (`jax.block_until_ready` is a no-op on both
backends — CLAUDE.md item 9). ms per chunk:

| config | native P5 | native eager (P4) | Stage 1 msl off | Stage 1 default | P5 gain | native / s1-nomsl |
|---|---:|---:|---:|---:|---:|---:|
| db02-b4l1024 | 582.3 | 2037.5 | 566.0 | 3.2 | **3.50x** | 1.03 |
| db09-b128l128 | 106.2 | 192.8 | 107.1 | 6.0 | **1.82x** | 0.99 |
| db18-b4l1024 | 1753.9 | 3595.7 | 1797.1 | 29.8 | **2.05x** | 0.98 |
| synth-matlstm-b | 405.8 | 862.1 | 409.7 | 415.3 | **2.12x** | 0.99 |
| big02-b32l128 (`bytes\|gru.512`) | 237.9 | 403.9 | 239.0 | 135.8 | **1.70x** | 1.00 |

Two things to read out of it. The compile decisions are worth 1.7-3.5x on these
chunks, which is the "performance side effect" of a correctness milestone. And
the native plugin now sits within 1-3 % of the Stage 1 engine with kernels
disabled — parity, on the like-for-like comparison, on every row.

The gap to Stage 1's DEFAULT column is `msl_scan`, and nothing else: db02
(in-lane rnn.1), db09 (mingru), db18 (mullstm coop) and big02 (gru.512 coop)
are the configurations generated kernels were built for, and matlstm — the one
with no plan (the feature was dropped, notes/matlstm-2026-07) — is at 0.98 of
Stage 1's default. Porting msl_scan is a later phase; this table is what it
will be measured against.

## The finding: MLX does not fuse the first compiled program in a process

Landing the compiled path made something visible that nothing could see while
the plugin ran everything eagerly. Three DISTINCT executables of the same
program, in one process, native plugin:

```
0 0.9001417756080627     <- unfused
1 0.9001415371894836     <- fused
2 0.9001415371894836
```

The first compiled executable in a process runs MLX's UNFUSED graph; every
later one runs the fused kernels, and on a transcendental chain the two differ
by ~4.8e-7 (MLX bakes rank-0 constants into a fused kernel as `%.7g` literals,
CLAUDE.md item 20). It is deterministic in both directions — three runs of each
arm agree with themselves — and the eager path is stable at the unfused value.

What it is NOT: it is not the compile decision (the tape is byte-identical to
Stage 1's), not the command-buffer lottery (deterministic), and not device or
compiler warmth. Four warm-ups were tried and none of them moves it: an
`mx::eval` at client construction, an `mx::eval` on the thread's bound stream
inside `BindThread`, a compiled-and-evaluated trivial graph, and a compiled
sigmoid/erf/sqrt chain of the same shape as the program that shows it — twice
over. What DOES move it is running one real jitted program first, of any shape.
**The Stage 1 engine does not show it**: its first compiled executable already
fuses, through the same `Program::compiled` -> `mx::detail::compile` path, so
whatever the state is, importing `metaljax.engine` establishes it and this
plugin's first `Execute` does not. Left open, deliberately: an unverified
warm-up that costs a kernel launch per thread and fixes nothing is worse than a
documented measurement.

Consequences for reading results: `texmo_gate.py --only X` measures the
UNFUSED compiled path (one executable per process) while the full suite
measures the fused one for 105 of its 106 (they share the parent process). Both
were run, and db18 is inside 5e-05 either way. `execute_test.py`'s
eager-vs-compiled arm reports the same 4.8e-7 on the one case that reaches it.

## Validation

| | result |
|---|---|
| `plugin-native/texmo_gate.py` x3 (x6 in all) | 106/106 ok, 0 decline, 0 FAIL, 0 error |
| `plugin-native/execute_test.py` | **156 checks**, 0 failures (152 in P4) |
| `plugin-native/smoke_test.py` | 3/3 checkpoints |
| `plugin-native/wheel_poc_test.py` | 4/4, from a native wheel in a fresh 3.13 venv |
| `plugin-native/decline_census.py` | 25 of 35, unchanged from P4 |
| `bazel test //metal:runtime_gil_free_test` | PASSED |
| `pytest tests/ -q` | 1258 passed, unchanged (nothing under `native/` or `src/` was touched) |
| tape cross-check | 259 lines over 11 probes, byte-identical |

`execute_test.py`'s new rows:

* **chunked replay (512 x matmul body)** — a counted loop whose body is pure and
  cheap enough that `kmax` lands at the `_CHUNK_MAX` ceiling, so it really does
  run 32 compiled chunks of 16 and their remainder through `run_chunked`. That
  arm was unreachable while `chunkable` was hard-wired to 0.
* **counted loop unrolled into a compiled main** — `WhileTraceable` says yes, so
  the whole main compiles with the loop inlined into its trace and nothing
  reaches `run_while`'s eager path at all.
* **counted loop past the unroll ceiling** — the same decision one size up: 200
  iterations fit the op budget, so the lowering calls the body traceable and
  compiles main around it, and the executor then refuses to unroll more than 64
  into one trace and hands the program to the eager path
  (`Program::run_recovering`). Both engines take that route; the row checks
  that the answer survives it.
* **eager vs compiled** — a child re-runs every case through the same dylib with
  `METALJAX_COMPILE=0` and the two arms are compared. 126 of 127 are BIT-
  identical; the one that is not is `logistic/erf/sqrt f32` at 4.8e-7, which is
  the fusion difference above, and it is named in the output rather than hidden
  under a tolerance.

`METALJAX_COMPILE=0` reproduces P4 exactly, which is what makes that arm mean
something: the tape it builds is byte-identical to `tape.py`'s under the same
setting, and the perf table's `native eager` column is P4's plugin.

## Gotchas

1. **The dump comparison is only as good as its input module.** P4's gotcha 1
   still governs: XLA's parse legalizes chlo and hoists constants out of
   regions before `CompileAndLoad`, so a Stage 1 lowering fed the module jax
   printed is walking a different program. Start from
   `METALJAX_DUMP_MODULE=1`.
2. **`chlo.erf` is a 52-entry polynomial here and one `mx::erf` on Stage 1.**
   Same reason. It makes the two engines' transcendental chains differ by ULPs
   in a way that has nothing to do with compilation, and it is why
   `execute_test`'s CPU comparison — not a Stage-1 comparison — is the bar for
   arithmetic.
3. **A recorder is enough to diff a Stage 1 tape.** `tape.py` touches a Program
   through exactly `add`, `set_outputs` and `set_compile`; a Python class with
   those three methods, substituted for `engine.NATIVE.Program`, records a whole
   lowering without a device. `engine.NATIVE` must be restored afterwards.
4. **`std::min` over three budgets wants the initializer-list overload**
   (`std::min<int64_t>({a, b, c})`), and `<algorithm>` with it. The two-argument
   form silently compiles for the first two.
5. **Timing a texmo chunk needs the readback.** Both plugins return before the
   GPU is done in some paths; `np.asarray` on every output at the end of the
   timed loop is what makes the number honest, and it must be done identically
   on both arms.

## What P6 should pick up

* `sort` / `chlo.top_k`, then `reduce_window`, `convolution`, `fft` and the
  host-op custom calls — the census order P4 left.
* **`msl_scan`**, which is now the whole of the remaining gap to Stage 1 on the
  texmo suite (the perf table above is its baseline). M5b's Stage 1 port is the
  template: Python planned, C++ launched — except that here the planning has to
  be C++ too.
* Asynchronous execute (`async_eval` + a real `GetReadyFuture`) and donation,
  still untouched from P2.
* The first-executable fusion finding above, if a mechanism can be found in
  MLX's `compile_impl`.
* MLX's global command-encoder map (P4), still the one bug neither engine can
  paper over from outside.
