# Stage 2 P3: control flow through the native plugin

Follows [`cpp-p2-lowering.md`](cpp-p2-lowering.md), which lowered
straight-line StableHLO into `metaljax::Program` and replayed it. The
executor already ran control flow — `native/control.cc` has while/if/case,
the counted-loop machinery, chunked replay and the pipelined dynamic loop,
all battle-tested behind the Stage 1 engine. What was missing was the
LOWERING: `metal_lowering.cc` declined any op carrying a region except
`reduce`. P3 builds the regions.

```
$ JAX_PLATFORMS=metal METALJAX_PLUGIN_PATH=.../libmetal_pjrt_native.dylib \
    .venv/bin/python -c "
import jax, numpy as np
print(jax.jit(lambda xs: jax.lax.scan(lambda c,x:(c+x,c*2), np.float32(0), xs))(
      np.arange(6, dtype=np.float32)))"
(Array(15., dtype=float32), Array([ 0.,  0.,  2.,  6., 12., 20.], ...))
```

## What was built

| file | lines | what |
|---|---:|---|
| `metal/metal_lowering.cc` | +766/-70 | frames + regions, while/if/case, the three control-flow analyses, ds/dus, the dump as a tree |
| `metal/metal_client.cc` | +60/-5 | `ConfigureFromEnv`: the runtime cadences, parsed once at client creation |
| `metal/metal_executable.cc` | +30/-2 | `METALJAX_DEBUG` reports the executor's stats delta per execute |
| `execute_test.py` | +230 | the control-flow section, hand-written modules, the region alias contract |
| `decline_census.py` | 300 (new) | what the plugin still declines, over a representative workload |

Dylib: 165,596,936 -> **165,662,360 B** (+65,424, **+0.04 %**). A touched
`.cc` still recompiles in ~5 s, the link in ~2 s.

## What lowers now

* **`stablehlo.while`**, both arms: the counted loop jax emits for
  `lax.scan` / `lax.fori_loop` (whose condition is never executed — the trip
  count is read from the carry once) and the dynamic loop, whose condition
  goes back to the host every iteration and which the executor pipelines.
* **`stablehlo.if` / `stablehlo.case`**, branches as regions, index clamped
  into range by the executor.
* **`stablehlo.dynamic_slice` / `dynamic_update_slice`**, with XLA's index
  clamps — scan's carry stacking is these two, and they are straight-line
  ops that simply had no lowering before.
* Everything P2 lowered, now also INSIDE a region: region lowering is a
  recursive use of the same `Lowering`, so a body's op set is main's op set
  by construction, and a body holding `stablehlo.sort` declines the whole
  program naming `stablehlo.sort` rather than naming the loop.

Still declined, and the census below is the priority order: `gather`,
`scatter`, `sort`/`top_k`, `reduce_window`, `rng_bit_generator` (which
surfaces as `shift_right_logical`), `convolution`, `fft`, `bitcast_convert`,
`reverse`, the shifts, `custom_call` of every kind, and general reduce
bodies.

## The shape of the lowering after P3

P2's `Lowering` was one object for one block. It is now one object per
FRAME, which is what tape.py has always been (`_Lowering` per region), and
the pieces line up one for one:

| tape.py | metal_lowering.cc |
|---|---|
| `_Lowering.lower_block` | `Lowering::LowerBlock` |
| `_build` | `Lowering::Finish` |
| `_region` | `Lowering::LowerRegion` (constructs a CHILD `Lowering`) |
| `_control` / `_while` / `_branch` | `LowerControl` / `LowerWhile` / `LowerBranch` |
| `_tainted` / `_region_taints` | `TaintOf` / `MapTaint` |
| `interpreter.free_values` | `FreeValues` |
| `ops/control.py _analyze_counted` | `AnalyzeCounted` |
| `_block_cost` / `_flush_period` / `_static_start` / `_splat_int` / `_cond_has_effects` | the same names, in the anonymous namespace |

Three things a reviewer should look at hardest, because each is a wrong
NUMBER rather than a build error:

1. **Capture ORDER.** `FreeValues` is `interpreter.free_values`
   transliterated, including the walk order (an op's operands before its
   nested regions, results marked defined after both). The order is the
   encoding: captures become the region Program's trailing arguments in it,
   `MapTaint` indexes the parent slots by it, and a counted loop whose bound
   is captured records the bound's INDEX into the cond's free list.
2. **`arg_alias_` became a set per slot.** P2 could keep it a boolean
   because it lowered no regions; a region maps its outputs' taints back
   through the parent's operands (`MapTaint`), which needs to know WHICH
   argument, so it is now `slot -> {argument slots}` exactly as tape.py's
   is. The output-copy rule is otherwise unchanged, including P2's one
   deliberate strictness (a direct argument return is copied here, where
   tape.py leaves it to `engine.execute`).
3. **`AnalyzeCounted` is permissive in one direction only.** The counted
   path never evaluates the condition, so a loop wrongly called counted runs
   the wrong number of iterations — silently. Every test in the Python
   function is here, including `_cond_has_effects` (a cond with host-visible
   effects must take the dynamic path, because XLA runs the cond trip+1
   times) and the "body forwards the bound unchanged" test behind
   `bound_kind = 1`.

`_splat_int` deserves its own line: Python reaches the value through
`int(...)` on a decoded numpy scalar, so a rank-0 FLOAT constant truncates
to an integer there rather than declining. `SplatInt` does the same (and
returns nothing for NaN/inf, which is where the Python raises and the
`except` turns the loop dynamic). Being stricter here would have been a
silent cadence divergence between the two engines, not a safety margin.

### What is deliberately NOT computed

`chunkable`, `kmax` and `body_compile_max` — the last three fields of a
while entry — are written as `0, 1, 0`. They are the COMPILE decisions
(purity, the trace budget, the byte budget, `_bytes_chunks`), and P3's
contract is that everything runs interpreted; `set_compile` is not called on
main either. `0, 0` is what tape.py itself writes with `METALJAX_COMPILE=0`,
and `kmax` is read by `native/control.cc` only when `chunkable` is set, so
the dead `1` is a value that cannot be acted on. Computing two of its three
terms would have looked like agreement without being it.

Everything the executor needs for the EAGER path is computed for real:
`cost` (`_block_cost`, loops unrolled, callees recursed into, unknown trips
charged the pessimistic 1024) and `period` (`_flush_period`), which is what
makes the loop's flush cadence land where the Python engine's lands.

## Cadences from the environment

The plugin never called `metaljax::configure`, so every cadence sat at its
compiled-in default and `METALJAX_EAGER_FLUSH_MB` and friends were ignored
(P2 flagged it and deliberately left it). `MetalClient`'s constructor now
parses them once:

| variable | default | owner |
|---|---:|---|
| `METALJAX_EAGER_FLUSH_MB` | 1024 | `interpreter.FLUSH_MB` |
| `METALJAX_EAGER_FLUSH_SYNC` | 1 | `interpreter._FLUSH_SYNC_EVERY` |
| `METALJAX_FLUSH_CLEAR_MB` ᴾ²⁵ | 2048 | `interpreter._FLUSH_CLEAR_BYTES` |
| `METALJAX_LOOP_CLEAR_COST` | 500000 | `ops/control._LOOP_CLEAR_COST` |
| `METALJAX_WHILE_PIPELINE` | 1 | `ops/control._WHILE_PIPELINE` |
| `METALJAX_DEBUG`, `METALJAX_MEMDBG` | off | both |

ᴾ²⁵ the two stacks no longer SPEND `METALJAX_FLUSH_CLEAR_MB` the same way:
this plugin trims MLX's pool back to it at a hard flush, Stage 1 (frozen) still
dumps the whole pool. Same variable, same trigger, different reclaim —
notes/cpp-p25-cache-limit.md.

P2's reason for not doing this was that a second reader would be a second
opinion on numbers the MLX command-buffer lottery is pinned to. That reason
does not apply to this plugin: it is the ONLY engine in its process (nothing
Python-side is imported — `smoke_test.py` checkpoint 2 asserts it), so there
is no second reader to drift from, and what would drift is the defaults
themselves against the modules that document them. Garbage in a variable
keeps the default and says so under `METALJAX_DEBUG`, where Python would
raise; a plugin cannot usefully raise out of a client constructor.

This is correctness, not tuning, for exactly the workloads Stage 2 exists
for: Metal caps LIVE buffers at ~499k while MLX's cache is bounded by BYTES,
and the loop clear cadence is what keeps a multi-hour loop under it
(CLAUDE.md items 11/14).

## Validation

**`plugin-native/execute_test.py` — 102 checks, all green, `exit 0`** (77 in
P2), 3.4 s wall. New rows, all against jax-CPU:

| case | max error | | case | max error |
|---|---:|---|---|---:|
| scan (cumulative) | 0 | | dynamic_slice | 0 |
| scan (carry only) | 2.4e-07 | | dynamic_slice (index past the end) | 0 |
| fori_loop | 0 | | dynamic_slice (negative index) | 0 |
| while_loop (dynamic trip) | 0 | | dynamic_slice 2d (both clamps) | 0 |
| fori_loop (captured bound) | 0 | | dynamic_update_slice | 0 |
| cond (both branches) | 1.2e-07 | | dynamic_update_slice (past the end) | 0 |
| switch (every branch) | 0 | | dynamic_update_slice 2d (negative) | 0 |
| scan over matmul | 9.5e-07 | | scan stacking through dus | 0 |
| nested scan | 0 | | long counted loop (10k iterations) | 0 |
| scan with a stacked output | 0 | | | |

Plus a section of **hand-written StableHLO**, the same text through both
clients' `compile_and_load`, for two encodings jax's own lowerings never
produce:

* a counted while whose bound is a CAPTURE of the cond region
  (`bound_kind = 2`; jax threads every closed-over value into the carry
  instead, so `lax.fori_loop` reaches `bound_kind` 0 and 1 only) — at 6, 0
  and -3 iterations;
* `stablehlo.if`, both branches — jax lowers `lax.cond` to
  `stablehlo.case` even for a two-way branch, so the IF opcode and the
  "region 0 is the true branch" convention had no other coverage.

And two contracts: the P2 no-alias check, and a new one THROUGH a region — a
carry the body forwards untouched is still the caller's array on the way
out, and `unsafe_buffer_pointer` says the output is a fresh buffer.

The decline test changed shape: `fori_loop` no longer declines, so the row
is now a loop whose BODY holds `jnp.sort`, and the message must name
`stablehlo.sort` — the region is lowered by the same `Lowering` as main, so
its declines are main's.

| | result |
|---|---|
| `plugin-native/execute_test.py` | 102 checks, 0 failures |
| `plugin-native/smoke_test.py` | 3/3 checkpoints |
| `plugin-native/wheel_poc_test.py` | 4/4, from a native wheel in a fresh 3.13 venv |
| `bazel test //metal:runtime_gil_free_test` | PASSED |
| `pytest tests/ -q` | 1258 passed, unchanged (nothing under `native/` or `src/` was touched) |

### The long loop, and evidence the flush cadence engages

`METALJAX_DEBUG=1` now prints the executor's `g_stats` delta for each
execute, which is the only window a process with no interpreter has on them.
`fori_loop(0, N, lambda i, c: c + 1.0, x)`, body cost 4 -> period 64:

| N | wall | loop_flushes | expected (N/64) |
|---:|---:|---:|---:|
| 10,000 | 0.169 s | 156 | 156.25 |
| 50,000 | 0.261 s | 781 | 781.25 |

and the clear cadence responds to its variable — 10k iterations at the
default `METALJAX_LOOP_CLEAR_COST=500000` clear nothing (they are ~40k
op-units of work), at `10000` they clear 5 times. The dynamic loop
pipelines by default (`pipelined_loops=1 pipelined_steps=99` on a 100-step
loop) and goes serial under `METALJAX_WHILE_PIPELINE=0`
(`serial_loops=1`), which is the same knob `tests/test_command_buffer.py`
patches on the Stage 1 side.

## Tape cross-check against Stage 1

Same method as P2: `METALJAX_DUMP_TAPE=1` prints the finished tape, and a
scratch script wraps `engine.NATIVE.Program` so tape.py's own lowering
records the same lines. The dump grew a nested form for regions:

```
[tape] stablehlo.while 0,3,4,5,2,1 -> 6,7,8,9 [4,1,1,1,1,0,8,15,64,0,1,0]
[tape] region 0 {
  [tape] stablehlo.compare 1,4 -> 5 [2,0]
  [tape] outputs 5 copies  slots 6
[tape] }
[tape] region 1 {
  [tape] stablehlo.dynamic_slice 0,1 -> 5 [1,7,1]
  ...
```

Nine probes, **170 tape lines**: `scan`, `fori`, a dynamic while, `cond`,
`switch`, a bare `dus`, a scan over a matmul, a nested scan, and the
hand-written captured-bound module. With the Stage 1 side run at
`METALJAX_COMPILE=0 METALJAX_MSL=0` — the settings that match P3's contract
— **every line is identical except one field**:

| probe | verdict |
|---|---|
| cond, switch, dus | byte-identical |
| scan, fori, dynamic, matscan, capbound | identical but `kmax` (16 vs the dead 1) |
| nested | the same, on both of its while entries |

Same opcodes, same slot numbering in every frame, same capture lists and
counts, same counted encoding (`counted/k/bound_kind/bound` agree on all
five loops, including `bound_kind` 1 and 2), same `cost` and `period` — the
cost model agrees even on the nested case (46 outer, 11 inner) — and the
same output and copy sets.

Against Stage 1 at its DEFAULT settings there are two further differences,
both by design: `chunkable`/`body_compile_max` are `1`/a real budget there
and `0` here (the compile decision), and the `scan` probe lowers to
`metaljax.msl_scan` rather than `stablehlo.while` — msl_scan is a `kWhile`
in every other respect, so the attribute vector is the same one.

## The decline census (P4's priority order)

`plugin-native/decline_census.py` compiles — never executes — a probe set
shaped like the workloads the roadmap cares about, plus any StableHLO
modules named on the command line, and tabulates the distinct decline
messages. **16 of 35 probes lower.**

| n | decline |
|---:|---|
| 3 | `op stablehlo.sort` (sort, argsort, top_k) |
| 2 | `op stablehlo.gather` (embedding lookup, cross-entropy) |
| 2 | `op stablehlo.shift_right_logical` (threefry: `random.normal`, `random.split`) |
| 2 | `op stablehlo.reduce_window` (cumsum, max pooling) |
| 2 | `op stablehlo.scatter` (segment_sum, `.at[].set`) |
| 1 each | `convolution`, `fft`, `cholesky`, `custom_call` (qr), `bitcast_convert`, `reverse`, `shift_left` |
| 1 | `jax.debug.print`: no MLIR translation rule for platform metal — a JAX-side registration Stage 1 does in `src/metaljax/ops/callbacks.py`, not a plugin decline |

What already lowers is worth as much: the whole texmo-shaped middle (a
grad-of-MLP, an SGD and an Adam update, a scanned train chunk, layernorm,
softmax+argmax, a GRU-shaped scan) and the whole LLM decode shape
(attention, a KV-cache `dus`, a decode `while` over that cache, rope,
dequant matmul).

**On real texmo.** `scripts/texmo_check.py` drives `metaljax.engine`
directly (`compile_program` / `execute`), never PJRT, so it cannot exercise
this plugin at all — the mission's `--limit 1` invocation does not exist and
would not have measured the native path. The equivalent was done instead: a
scratch script dumps texmo training-chunk modules (the same ones
`texmo_check` builds) and the census feeds them to
`client.compile_and_load`. All five configs tried — `bytes.emb.512|mgru.512`,
`bytes.emb.256|gru.256-gru.256-gru.256`, `bytes|gru.512`, at two batch/length
shapes — decline on **`stablehlo.gather`**, matching what M5b found for the
Stage 1 tape.

A decline names the FIRST op that stopped the program, so `--ops` also
histograms what those modules CONTAIN. The whole vocabulary of a texmo train
chunk is 33 ops, and exactly two of them are outside this plugin's set:

```
   30  stablehlo.gather        28  stablehlo.scatter
```

Everything else — `while` (23), `dynamic_slice`/`dynamic_update_slice` (18
each), `reduce` (103), `dot_general` (95), `pad` (32), `select`, `slice`,
`transpose`, `chlo.square` — lowers today. So **gather and scatter are what
stand between this plugin and a texmo training step**, and P4 should take
them first; `sort`/`top_k` and the shifts (which is really
`rng_bit_generator`) are next by breadth.

## Gotchas

1. **jax never emits `stablehlo.if`.** `lax.cond` lowers to
   `stablehlo.case` with two branches, even for a boolean predicate. The IF
   arm of the executor is reachable only from hand-written IR, which is why
   `execute_test.py` grew a module section rather than trusting the `cond`
   rows to cover it.
2. **jax never emits `bound_kind = 2` either.** `while_loop`'s cond consts
   become CARRIES of the `stablehlo.while`, so a dynamic bound arrives as
   `bound_kind = 1` (`fori_loop(0, n, ...)` measured). The capture form is
   legal StableHLO — regions are not isolated-from-above — and the executor
   reads it, so it is tested by module text.
3. **`fori_loop(0, 0, ...)` is folded away by jax**, before any plugin sees
   it: there is no `while` in the module at all. The zero-trip path
   (`static_trip == 0`, where a result's taints come from its INIT rather
   than from the body) is reached through the captured-bound module with
   `n = 0` instead.
4. **A rank-0 `dynamic_slice` has no index operands**, and
   `ops/shape.py` hands the operand array straight back. tape.py aliases the
   slot (`_rank0_passthrough`) and so does this; emitting a `kDynamicSlice`
   with rank 0 instead would have been a handler call with empty axis
   vectors. It sits BEFORE the dtype checks, exactly where tape.py's does.
5. **`Taint` the struct vs `Taint` the method.** P2 had a member function
   named `Taint`; introducing a struct of that name makes every `Taint t;`
   inside the class resolve to the function. It is a compile error rather
   than a silent one, but the method is now `TaintResults` and the name is
   worth leaving alone.
6. **`timeout(1)` is not on this machine** (macOS): wrapping a long bazel
   or test invocation in it fails with "command not found" and looks like a
   build failure.
7. **The decline message for an op that carries a region names the region,
   not the op set**: `op stablehlo.sort (it carries a region)`. It names the
   op, which is the contract the tests check, but a reader may take it for a
   region-support gap when sort simply is not in the op set. Left as P2 had
   it; worth rewording when the region-carrying ops (sort, scatter,
   reduce_window) land.

## What P4 should pick up

* **`gather` and `scatter`** — the only two ops between this plugin and a
  texmo training step, and the longest attribute encodings in `program.h`.
  The index-plan quads and the OOB-drop strategies are where a differential
  suite earns its keep.
* `sort`/`top_k`, then `rng_bit_generator` (the shifts in the census are
  threefry), then `reduce_window`, then the singletons.
* The **compile decisions** — `set_compile` on main, and the three while
  fields P3 writes as zeros. That is a port of `ops/control.py`'s cost /
  byte / purity estimators and `_underived_outputs`; until it lands, this
  plugin runs every program interpreted, which is a performance question and
  never a correctness one.
* Asynchronous execute (`async_eval` + a real `GetReadyFuture`) and
  donation, still untouched from P2.
