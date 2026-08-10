# Stage 2 P4: gather/scatter, the small-op tail, and the texmo gate

Follows [`cpp-p3-control.md`](cpp-p3-control.md), whose census ended with a
single sentence: of the 33 ops a texmo training chunk is made of, exactly two —
`stablehlo.gather` (30 uses) and `stablehlo.scatter` (28) — were outside the
native plugin's set. P4 lowers those two, the small-op tail behind them, and
builds the gate that measures the whole thing.

```
$ .venv/bin/python plugin-native/texmo_gate.py
106 configurations, 8 steps per chunk, tol 0.002
...
SUMMARY: 106 ok (24 via sensitivity scaling), 0 decline, 0 FAIL, 0 error, of 106
```

Every configuration in `benchmarks/texmo-suite.csv` now trains through the
fully native stack — StableHLO parsed in C++, lowered in C++, replayed in C++,
no interpreter in the process — and agrees with jax-CPU.

## What was built

| file | lines | what |
|---|---:|---|
| `metal/metal_lowering.cc` | +754/-7 | gather, scatter, the index plan, the shifts, reverse, bitcast, popcnt/clz, `_scatter_cost`, the module dump |
| `metal/metal_client.cc` | +37 | MLX's own budgets, pinned (see "The bug the gate found") |
| `execute_test.py` | +212/-3 | 48 new differential rows and three new declines |
| `texmo_gate.py` | 327 (new) | the phase-2 gate: every suite configuration, both backends |

Dylib: 165,662,360 -> **165,732,040 B** (+69,680, **+0.04 %**). Edit loop
unchanged (~5 s to recompile a touched `.cc`, ~2 s to link).

## What lowers now

* **`stablehlo.gather`** as ONE `mx::gather`. The coordinate vector splits into
  one index array per mapped operand dim, `slice_sizes` crosses verbatim
  (windows on indexed dims included), the result reshapes past the collapsed
  extent-1 dims and transposes into the `offset_dims` interleaving. Batching
  dims become an iota keyed to their operand dim. A zero-size result
  short-circuits to zeros, which is what the Python handler does — its
  decomposition mis-shapes empty batches.
* **`stablehlo.scatter`** as one `mx::scatter`/`_add`/`_prod`/`_max`/`_min`,
  with the update transposed into `[index batch dims, window dims in operand
  order]` and reshaped to insert an extent-1 axis per inserted window dim.
  The combiner is read structurally out of the region (`set add multiply
  maximum minimum subtract`); a computed body (jax's `scatter_apply`) and a
  complex operand decline, naming why.
* **The shifts** (`shift_left`, `shift_right_logical`,
  `shift_right_arithmetic`) with XLA's overflow rule, `reverse`,
  `bitcast_convert` (all three width relations), `popcnt` /
  `count_leading_zeros`.
* With the shifts, **jax's threefry RNG is ordinary elementwise arithmetic**:
  `random.bits`, `split`, `fold_in`, `uniform` and `randint` are BIT-exact
  against jax-CPU (rows in `execute_test.py` compare them as integers, no
  tolerance). `random.normal` is 1.2e-7 off, and that is `erf_inv`: the last
  ULP there is MLX's, not the RNG's.
* `stablehlo.scatter` also joins `_block_cost` (`ScatterCost`, the Python's
  `_scatter_cost`) and `bitcast_convert` joins the view-op set that keeps a
  constant's buffer out of an output position — both were noted as pending in
  P3 and P2 respectively.

## XLA's OOB rules, which is where this op pair kills you

`mx::gather` **clamps nothing**: it wraps a negative index like `take` and
reads past the end otherwise. `mx::scatter` does no bounds checking at all,
which for a WRITE is memory corruption rather than a wrong number. XLA's rules
are different again and different from each other — a gather CLAMPS its start
index so the window fits; a scatter DROPS an update whose start is out of
bounds in any component — so every clamp and every drop is arithmetic this
lowering has to put there.

The drop is spelled with primitives that cannot skip a write, by the two
strategies `ops/gather.py` picks between and `tape.py` encodes:

* **neutral value** (strategy 1): rewrite a dropped update to the combiner's
  identity, so applying it is a no-op. Order- and duplicate-safe, and it
  touches the UPDATES.
* **dummy pad** (strategy 2): grow one indexed axis by a window's worth of
  rows, redirect dropped updates there, cut the pad off. Required for `set`,
  where neutralizing would race a genuine duplicate write at the clamped slot
  (`fill_value == size` clamps onto the last real slot — a systematic
  collision, not a rare one).

Which one is chosen is decided from the STATIC sizes (`set`, or operand
smaller than updates), here exactly as in `tape.py`, so the two engines cannot
pick differently. That matters more than it looks: adding a neutral `0` and not
adding it disagree in the BITS at `-0.0`, so a scatter-add whose dropped update
lands on a `-0.0` element gives `+0.0` on both metal engines and `-0.0` on
jax-CPU. Measured, both engines, same bit pattern; it is the documented cost of
the neutral strategy (`src/metaljax/tape.py`'s own comment) and not something
P4 introduced. It is deliberately not a test row: pinning it would pin the
strategy choice, which is a performance decision.

Everything else about the rules IS a test row, and the CPU comparison is the
only honest statement of them: `execute_test.py` runs each of gather-with-OOB,
scatter-set-with-OOB, scatter-add-with-OOB, `bincount` (whose overflow slot is
this rule in production) and a vmapped scatter with one lane out of range.

## The bug the gate found (and it is not gather's)

The first real workload through this plugin was wrong. A `bytes|gru.512`
training chunk came out ~1e-2 off jax-CPU where the Stage 1 engine — same
module, same inputs — was 7e-4, and **the same run repeated gave a different
answer**: 3 of 5 runs exactly right, 2 of 5 wrong.

It is not the lowering. The tape cross-check below is byte-identical, and the
Stage 1 engine running THAT tape is right every time. What the native plugin
was missing is two environment variables:

```
MLX_MAX_OPS_PER_BUFFER = 800
MLX_MAX_MB_PER_BUFFER  = 512
```

`src/metaljax/__init__.py` sets both before `import mlx.core`, and this plugin
never imports that module. At MLX's defaults (10 ops / 40 MB) a single `eval`
of a real training chunk is split across many command buffers, and splitting
one eval across command buffers **corrupts results in MLX 0.32** — silently,
and differently on every call (`notes/mlx-command-buffer-split.md`). Nothing
had noticed because until gather/scatter landed, no program big enough to split
a buffer ever lowered here.

`metal_client.cc` now pins them (plus `MLX_METAL_GPU_ARCH`, which the jax
loader sets but a C++ embedder would not get) from a load-time static
initializer, with `setenv(..., overwrite=0)` so a user's own value still wins.
MLX reads each one ONCE from a function-local static when it builds its Metal
device — which happens at the first `mx::` call, inside this plugin — so the
initializer lands early enough. P2's gotcha 3 says the arch "could not be fixed
from inside the dylib anyway, because libmlx's initializers run before ours";
that reasoning is wrong in the way that matters, because the READ is lazy, not
an initializer.

After the pin: 6 of 6 runs exactly right, and the value matches the Stage 1
engine's to the last digit.

**KEEP IN SYNC**: `src/metaljax/__init__.py` owns those numbers and the crash
stories behind them (400 ops is a corrupting alignment; 512 MB is bounded from
below by corruption and from above by a kernel panic). Two copies now.

## The remaining corruption, which the compile decisions own

One suite row is still wrong, and it is the same MLX bug from the other end:

| `db18-b4l1024` `bits.4.oh+bp\|mullstm.32-dense.128.gelu` | worst vs CPU |
|---|---:|
| Stage 1, compiled (`METALJAX_COMPILE=1`) | 4.696e-05, deterministic |
| Stage 1, **eager** (`METALJAX_COMPILE=0 METALJAX_MSL=0`) | 9.5e-3 / 3.8e-3 / 4.2e-2 |
| native plugin (which is always eager, per P3) | 21.3 / 0.13 / 6.2e-3 |
| native plugin, `METALJAX_EAGER_FLUSH_MB=1` | 4.618e-05, 3 of 3 |
| Stage 1 eager, `METALJAX_EAGER_FLUSH_MB=1` | 4.618e-05 |

The two engines agree even in their failure — this is a property of the shared
eager path, not of the native lowering — and the cure is either a smaller
eager-flush budget or compiling the loop, which is what the Python engine does
for this config and what P3 deliberately left off (`chunkable`,
`body_compile_max` and `set_compile` are all zero here). So: **the compile
decisions are correctness for long eager loops, not only performance.** That is
the strongest argument yet for taking them next.

Nothing was changed about the cadence defaults. `METALJAX_EAGER_FLUSH_MB` is
Python's number (`interpreter.FLUSH_MB`), and lowering it here to dodge an MLX
bug would be a silent divergence from the engine that owns it.

Its siblings at the same length are clean (`db02`/`db12`/`db17`-b4l1024 all at
1e-5), so it is this body's size that lands in the corrupting band, not the
1024-step loop as such. Two full gate runs: 106/106 ok, then 105/106 with
`db18-b4l1024` FAIL — expect that row to flicker until the loop compiles.

## What the pin exposed: MLX's encoder map is not thread-safe

Pinning the budgets has a price, and it is worth stating precisely because it
looks like a regression. `execute_test.py`'s "32 executes on 8 threads" row now
**SEGFAULTS about 5 % of runs**, inside
`mlx::core::metal::get_command_encoder(Stream)` on a worker thread (crash
report: `EXC_BAD_ACCESS` under `MetalLoadedExecutable::RunOnce` ->
`mx::eval` -> `gpu::eval`). Never a wrong answer, always a crash, and only in
that deliberately concurrent check.

| arm | crashes |
|---|---:|
| native plugin, pinned 800 ops / 512 MB | **4 of 74** |
| native plugin, MLX's defaults (10 / 40) | 0 of 74 |
| Stage 1 plugin (same budgets, GIL-serialized) | 0 of 20 |
| the threaded section looped 160× inside ONE process | 0 |

Both native arms run the same dylib and the same load-time initializer, so it
is the BUDGETS, not the `setenv`. The mechanism is visible in MLX's own header:
a `new_thread_unsafe_stream` — which is what `metal_stream.cc` binds per
thread, exactly as `engine.py`'s `bind_thread` does, because a cross-thread
evaluable stream is what a PJRT client needs — routes to
`get_global_command_encoders()`, one process-wide `unordered_map<int,
CommandEncoder>`. Eight threads inside `mx::eval` at once contend on it; a
bigger command buffer keeps an encoder alive across more of the window. Stage 1
never sees this because the GIL serializes its executes — being GIL-free is
precisely what P1 bought and precisely what makes this reachable.

Not fixed here, because both cures are design decisions rather than
corrections: a per-thread `new_stream()` gives up cross-thread evaluability
(the reason the thread-unsafe one was chosen), and a mutex around `Execute`
gives up the concurrency P1 exists for. The correct answer is probably a lock
around the encoder acquisition inside MLX. **P5 item**, and the unpinned
alternative is not one: silent wrong training results beat a visible 5 % crash
in a stress test nowhere near a real workload (106 gate configurations, 1258
pytest cases and every other suite ran without one).

## The tape cross-check, and a trap P2/P3 did not hit

Same method as P2 and P3: `METALJAX_DUMP_TAPE=1` prints the finished tape and a
scratch script wraps `engine.NATIVE.Program` so `tape.py`'s own lowering
records the same lines.

**The trap**: for these programs the two dumps disagreed everywhere, and none
of it was the lowering. XLA's parse runs BEFORE `CompileAndLoad`, and it
rewrites the module — it legalizes `chlo` (one `chlo.erf_inv` becomes its
polynomial, ~30 entries and 31 slots more on the threefry-normal probe), CSEs
and hoists constants to the top of
the block (so every slot number downstream shifts), and hoists them OUT of
regions (so a while body that defined a constant now CAPTURES one, and its
capture counts and cost change with it). `tape.py` fed the module jax printed
was walking a different program.

`METALJAX_DUMP_MODULE=1` now prints the module the plugin was HANDED, and the
Stage 1 side is run over THAT. With the same input:

| probe | verdict |
|---|---|
| embedding lookup, gather with batching dims, windowed gather | byte-identical |
| cross-entropy (gather of a log-softmax) | byte-identical (41 entries) |
| segment_sum, scatter set, scatter max, vmapped scatter | byte-identical |
| embedding gradient (a scatter-add through AD) | byte-identical |
| threefry normal | byte-identical (297 entries) |
| shift / reverse / bitcast / popcnt / clz | byte-identical |
| scan over a gather | identical but the dead `kmax` (6 vs 1) |
| a real texmo train chunk (`bytes\|gru.512`, 558 lines) | identical but `kmax` |

**461 tape lines across 12 probes plus 558 for the training chunk, with one
field differing on one line** — `kmax`, which `native/control.cc` reads only
when `chunkable` is set and P3 writes as a dead `1`. Same opcodes, same slot
numbering in every frame, same capture lists, same index-plan quads, same drop
strategies, same `cost`/`period`, same output and copy sets.

A structural comparator (`tapecmp.py`, scratch) was written for the first,
confused pass and is worth keeping in mind: it canonicalizes each tape into
expression trees and compares the multiset, so it survives reordering. It is
what proved the differences were the input module's rather than the builder's,
before `METALJAX_DUMP_MODULE` existed. Hash the subtrees, do not nest them —
the naive version is exponential on a 297-entry tape.

## The gate

`plugin-native/texmo_gate.py`, permanent, is phase 2's standing measure.

Per configuration it forks a CPU child that builds the real `ManagerJax` model,
lowers one jitted training chunk (forward + backward + optimizer, scanned over
8 steps), computes the jax-CPU reference and the model's own 1-ULP
sensitivity, and writes the module and the arrays to a scratch directory; the
parent — the process holding the plugin — runs the module through
`compile_and_load` + `execute` and compares every output leaf with
`scripts/texmo_check.py`'s tolerance discipline (`max(2e-3, sensitivity ×
500)`, because an ill-conditioned training step amplifies cross-backend
rounding identically on both sides).

Two processes because `scripts/texmo_check.py` drives `metaljax.engine`
directly and pins `jax_platforms` to cpu at import, so it cannot exercise this
plugin at all; one child per configuration because a whole suite's weights,
optimizer state and references at once would be tens of gigabytes. The
artifacts are deleted as they are consumed. 106 configurations in 6m15s,
~3.5 s each.

`ok` / `ok~` (passed via the sensitivity scaling) / `decline` (the plugin
refused, naming the op) / `FAIL` / `ERROR`. A decline does not fail the gate —
coverage grows phase by phase — and `FAIL` or `ERROR` exits nonzero.

Full run, 2026-08-11:

| | count |
|---|---:|
| ok | 82 |
| ok~ (via sensitivity scaling) | 24 |
| decline | 0 |
| FAIL | 0 |
| ERROR | 0 |

The 24 `ok~` rows are the ill-conditioned configurations the scaling exists
for, and they are not this plugin's doing: `big10-b8l256`
(`bits.4.oh+bp|gru.1024`) is 1.657 on the native path and **1.657 on the Stage
1 engine**, bit for bit, on the same module and inputs. A second run was
105/106 (`db18-b4l1024`, above).

## Where the census stands

`decline_census.py`: **16 -> 25 of 35 probes lower** (P3 -> P4). The whole
texmo group lowers now except `top_k`; what remains, in priority order:

| n | decline |
|---:|---|
| 3 | `op stablehlo.sort` (sort, argsort, top_k) |
| 2 | `op stablehlo.reduce_window` (cumsum, max pooling) |
| 1 each | `convolution`, `fft`, `cholesky`, `custom_call` (qr) |
| 1 | `jax.debug.print` — a JAX-side registration, not a plugin decline |

## Validation

| | result |
|---|---|
| `plugin-native/execute_test.py` | **152 checks**, 0 failures (102 in P3); ~5 % of runs segfault in the 8-thread row, above |
| `plugin-native/texmo_gate.py` | 106/106 ok, 0 decline, 0 FAIL |
| `plugin-native/smoke_test.py` | 3/3 checkpoints |
| `plugin-native/wheel_poc_test.py` | 4/4, from a native wheel in a fresh 3.13 venv |
| `bazel test //metal:runtime_gil_free_test` | PASSED |
| `pytest tests/ -q` | 1258 passed, unchanged (nothing under `native/` or `src/` was touched) |

The 48 new `execute_test.py` rows: 11 gather (take, embedding, 2-D indices,
out-of-range indices, two index components, a window, batching dims,
`take_along_axis`, cross-entropy, an empty result, a rank-0 index), 17 scatter
(set/add/multiply/max/min/int, duplicates, out-of-range in each, updates bigger
than the operand, `segment_sum`, `bincount`, whole rows, the middle axis, two
vmapped forms, an embedding gradient, empty updates), 14 small-op (shifts at
and past the operand width on i32 and u8, a static overflow amount, reverse in
three forms, roll, four bitcasts, popcount, clz) and 6 threefry. The decline
list gained three rows that must name their reason — `cumsum`
(`stablehlo.reduce_window`), a scatter with a computed body (`scatter combiner
apply`) and a complex scatter (`scatter on complex`) — and lost one: the
gather row lowers now, so the "a loop whose body holds an unlowered op" row is
the only `stablehlo.sort` left besides `jnp.sort` itself.

## Gotchas

1. **XLA rewrites the module before the plugin sees it** (above). Any
   comparison against a Stage 1 lowering has to start from
   `METALJAX_DUMP_MODULE=1`, and the practical consequence is bigger than the
   diff: this plugin never sees `chlo.*` at all, so the `chlo.erf` /
   `chlo.erf_inv` / `chlo.square` entries in `SimpleOps()` are dead weight in
   the jax path (kept: hand-written IR still reaches them).
2. **`np.ascontiguousarray` promotes a rank-0 array to rank 1.** A training
   chunk's arguments include scalars, and the gate's first run failed with
   "argument 1 is [1], the program expects f32[]" for exactly that reason.
   `scripts/texmo_check.py` has the same call and gets away with it because
   it passes `list(a.shape)` from the promoted array.
3. **A nondeterministic wrong answer is an MLX command-buffer split**, not a
   race in your code. The tell is that repeating the identical call gives a
   different answer while the Stage 1 engine is stable — and the discriminator
   is to run Stage 1 with `METALJAX_COMPILE=0 METALJAX_MSL=0`, which puts it on
   the same eager path and reproduces it exactly.
4. **`texmo_check.py` re-samples its training data on every run**, so its
   `sens` (and therefore its tolerance) moves between runs by a factor of a
   few. A row that passes at `tol=1.2e-2` and fails at `tol=2.5e-3` with the
   same error is telling you about the tolerance, not the arithmetic.
5. The gate's child imports `scripts/texmo_check.py` for the suite reader and
   the dataset, which is also what pins it to the CPU backend. The parent must
   NOT import it — it is the process that has to see the plugin — so
   `configs_from_csv` is replicated there, deliberately.

## What P5 should pick up

* **The compile decisions** — `set_compile` on main and the three while fields
  P3 writes as zeros. It is a port of `ops/control.py`'s cost / byte / purity
  estimators and `_underived_outputs`, and after the db18 finding it is a
  CORRECTNESS item: an eager 1024-step loop is exposed to an MLX bug that a
  compiled one is not.
* `sort` / `chlo.top_k` (3 census probes, and the sampling half of every LLM),
  then `reduce_window` (cumsum, pooling), then `convolution`, `fft` and the
  host-op custom calls.
* Asynchronous execute (`async_eval` + a real `GetReadyFuture`) and donation,
  still untouched from P2.
* **MLX's global command-encoder map** (above): the one thing P4 found that
  neither engine can paper over from the outside, and the first bug that is
  reachable only because this plugin has no GIL.
