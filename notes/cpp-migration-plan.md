# Stage 2: C++ migration plan (started 2026-08-05, post-0.11.3)

Per Oleg: start with the code that may be called at RUNTIME; the final
goal is everything in C++. This document is the working plan — edit as
milestones land.

## Why (measured, from the 0.11.3 anchors)

- LLM decode is Python-dispatch-bound: ~120 ms/tok of overhead at
  gemma-31B scale (dtype-independent; f32 vs bf16 only 1.34× apart).
- mlx-lm gap band on dense decode: 1.7–2.6× (same Metal library
  underneath — the gap IS our dispatch).
- texmo sub-crossover dispatch floor: ~0.7–1.0 ms/step flat across
  sizes; metal wins only 53/163 configs vs CPU because of it.
- Tracked metrics: benchmarks/texmo.md (topconfs geomean vs the
  2026-08-05 anchor) and benchmarks/models.md (mlx-lm band).

## Shape of the migration

Phase 1 — native replay engine (runtime path), Python keeps compiling:

    Python (per executable, once):                 C++ (per call):
    parse StableHLO → analyze → recognizers  →     prepared-program
    → prologue packs → PREPARED PROGRAM      →     interpreter: eager
      (flat op tape, resolved indices,             handlers, mx::compile'd
      attrs decoded, plan annotations)             regions, native while,
                                                   native emits, flush
                                                   discipline — no GIL on
                                                   the hot path

Phase 2 — compile path native too: StableHLO/MLIR parsed in C++
(llvm-project clone is staged for this), recognizers/analysis ported,
PJRT plugin calls the C++ engine directly, Python trampoline retired.

## Build facts (verified 2026-08-05)

- The installed MLX wheel ships `include/` (mlx + metal-cpp),
  `lib/libmlx.dylib`, and `share/cmake/MLX` — linking extensions
  against the wheel is MLX's supported extension path. Pin: the
  extension is rebuilt when the mlx wheel version changes (assert the
  version string at import; ABI skew must fail loudly at load, not
  corrupt at runtime).
- nanobind: not yet installed; add as a build-time dependency only.
- Build style: extend the plugin/build.sh pattern (clang, no CMake)
  for phase 1 unless MLX's cmake config proves necessary; revisit at
  phase 2.

## Milestones (each gated: 687-test suite + texmo 104 gate; perf
milestones also run the topconfs sweep vs the 2026-08-05 anchor)

- **M0 — scaffolding**: `native/` extension (nanobind) linked against
  the wheel's libmlx; version handshake; METALJAX_ENGINE=py|native
  flag defaulting py; CI target `pytest tests/ --engine=native` (env).
  Deliverable: a C++ mx::array round-trips through the extension.
- **M1 — buffer path**: MetalBuffer from_host/to_host/donation in
  C++ (bitcast bf16 rules, negative-stride base offsets, buffer
  protocol contract from CLAUDE.md item 20). Differential tests vs
  the Python path, bit-exact.
- **M2 — prepared-program serializer + eager core**: Python-side
  lowering of an analyzed executable into a flat tape (op codes,
  operand indices, decoded attrs, constants); C++ eager interpreter
  for the core op set (elementwise, dot/matmul, shape ops,
  reductions, gather/scatter basics). ANY program containing an
  unsupported op falls back to the Python engine wholesale — the
  native set grows monotonically with zero correctness risk.
  Differential harness: every tests/ case runs BOTH engines when
  native supports the program; outputs must match bit-exact (float
  tolerance only where the Python engine itself is
  nondeterministic).
- **M3 — control flow + compile** (landed 2026-08-07): while/if/case,
  counted-loop detection reuse (Python analysis annotates the tape;
  C++ executes), mx::compile of mains and bodies from C++ closures
  (`mx::detail::compile` with ids this engine owns and erases),
  chunked replay, the body-probe contract (24e93f0), body-failure
  fallback, loop flush cadence + clear/retry discipline (0.3.2/0.4.x
  semantics), eager flush cache-clear (METALJAX_FLUSH_CLEAR_MB).
  Every cadence and budget is READ FROM PYTHON (`tape.configure` and
  the lowering's annotations), never re-parsed in C++, so the two
  engines cannot drift on a number the command-buffer lottery is
  pinned to. The eager-only production gate is gone: a program whose
  tape lowers takes the native path whether or not it compiles.
  Canaries re-pointed — tests/test_command_buffer.py grew a native
  detector and two sweeps, and the native engine's corrupting bands
  are NOT the Python engine's (2026-08-07 addendum in
  notes/mlx-command-buffer-split.md). The aliasing rule changed with
  it: an output that reads a constant the Program holds, or an
  argument's array through no-ops, is COPIED in C++ instead of
  declining the whole program.
- **M4 — recognizer emits native** (landed 2026-08-10): qmm/sdpa/moe
  emits become native calls (mx::quantized_matmul,
  mx::gather_qmm/gather_mm, mx::fast::scaled_dot_product_attention).
  Pack BUILDING stays in Python (compile-time/prologue, once per
  process — the build cache makes it cheap); the tape references pack
  slots by index. What landed with it:
  - The lowering follows `interpreter._rewrite_plan`, in `lower_block`
    AND in `_inline` (a root or an absorbed op is as likely to sit
    inside a callee): absorbed ops get no entry and no slot, so no
    static last use can land on one; roots lower to the new opcodes.
    moe's plan of pair-space nodes becomes a RUN of entries in the
    plan's own order, with the root's bytes charged only to the tail,
    so the eager flush cadence lands where the Python engine's lands.
  - Packed arrays are trailing INPUTS (never captures — mx.compile
    bakes those), of main and of any region that holds a pack-reading
    root. A repack that changes mode/group/bits/perm ARITY re-lowers:
    `invalidate_traces` drops the Program on qmm's `changed`, and
    `engine._native_ready` re-checks the arity every execute anyway.
  - `stablehlo.sort` (comparator == one compare, which is what every
    top_k lowers to) and `chlo.top_k` had to come with it: without
    them every MoE program declines on its router. jnp.sort's float
    comparator computes a key first and still declines.
  - The region-capture rule gained ops/control._captures' stand-in: an
    op absorbed in the ENCLOSING block still shows up in a region's
    syntactic `free_values`, has no slot, and is never read.
  - Emit-time diagnostics (moe's gather kind, sdpa's `fused`) are
    counted at lowering, once, rather than per execute.
  Recognizer-family census (test_qmm/test_qmm_mxfp4/test_moe/
  test_sdpa): 11 -> 87 of 102 executables lowered; whole suite 454 ->
  556 of ~670. The 15 that still decline are all `stablehlo.gather`
  (an M2-era op-set gap, not an emit gap).
- **M5 — msl_scan + host ops** (M5a landed 2026-08-10, M5b below):
  generated kernels via the C++ metal_kernel API; host LAPACK/callbacks
  stay Python (impure ops already leave traces — the tape marks host-op
  sites and the C++ engine calls back only there).
- **M6 — flip the default**: METALJAX_ENGINE=native default with
  per-program fallback; full release-style gates; perf acceptance:
  texmo sub-crossover floor ≤0.3 ms/step (from 0.7–1.0), dense-decode
  mlx-lm gap ≤1.3× on the 8B rows (from ~1.9×), 31B ≤1.5× (from
  1.7×). Numbers to be revisited against M2/M3 profiling.

## Correctness doctrine (unchanged from Stage 1)

- The Python engine is the reference until phase 2 ends: differential
  testing engine-vs-engine on the whole suite, plus the existing
  CPU-reference gates.
- Every silent-wrongness lesson carries over explicitly: command-
  buffer canaries re-pointed (M3), token-agreement policy for
  quantized rows, the sensitivity-scaled texmo gate.
- Post-migration fix list rides with M6 (per Oleg, 2026-08-05):
  sparse spdot_general pair (tracked-open), model rows 8/10/12/15.

## Open decisions

1. (resolved) Link against the wheel's libmlx — supported extension
   path; loud version handshake.
2. Tape format: nanobind-built C++ objects constructed from Python
   (no serialization format to invent) vs a flat binary buffer.
   Start with nanobind objects; flatten only if construction time
   shows up in profiles.
3. Threading/GIL: phase 1 releases the GIL for the duration of a
   native execute (host callbacks re-acquire). Decide during M3
   whether while-loop host syncs need finer-grained handling.

## M4 real-model verdict (2026-08-10)

Correctness: row 7 native 23.3 ms/tok (anchor 22.2), row 5 native
59.5 (anchor 57.8) — ok, memory identical, tokens in the ladder
class. Perf: PARITY, not a win. The decode hot loop was already a
compiled replay under the Python engine; stripping the remaining
Python dispatch bought ~nothing on these rows, so the 1.9-2.5x
mlx-lm gap does NOT live in Python dispatch. Where it lives (M3's
decode-shaped benchmark said the same: 420 -> 392 us/step, ~7%):
the PER-TOKEN pipeline stall — our lax.while decode evaluates the
cond on the host every iteration (a full submit-wait per token),
where mlx-lm's Python decode loop pipelines tokens with async_eval
and never blocks on a device condition; plus KV-cache
dynamic_update_slice copies vs mlx-lm's in-place cache.

CONSEQUENCE for M5/M6: reorder priorities. M5a = decode-loop
pipelining (async cond: dispatch iteration N+1 speculatively while
N's cond settles — the cond is `i < N`, its value is knowable
host-side for counted segments; for dynamic conds, double-buffer
the carry) + KV donation/in-place update on the native tape. M5b =
msl_scan port (the texmo floor). M6 targets unchanged but their
path runs through M5a, not dispatch removal.

## M5a landed (2026-08-10): pipelined dynamic while + carry donation

Both halves of M4's verdict, on the native tape. Nothing in the
Python engine's execution changed — it stays the differential
reference; `ops/control._WHILE_PIPELINE` is the shared knob
(METALJAX_WHILE_PIPELINE=0 restores the old shape), read by
`tape.configure` like every other loop cadence.

**Donation (the bigger win).** The dynamic loop built its condition's
argument vector at the top of the iteration and let it live to the
BOTTOM — a second handle on every carry, alive while the next
iteration's `dynamic_update_slice` evaluated. `mx::array::is_donatable`
is a use_count test, so MLX copied the whole KV cache per token.
Scoping that vector to the condition (`cond_of`) is the entire fix.
Measured on a decode-shaped loop whose carry is one cache: cost per
token was 4.6 us per megabyte of cache (a full copy at ~435 GB/s);
it is now FLAT — 16 MB 152 us/step, 512 MB 165.

**Pipelining.** The loop now builds iteration t's body and t+1's
condition BEFORE reading t's condition back, blocks on that host read
(the single sync point), then releases the carry, submits it with
`async_eval`, and goes round. Two host round trips per iteration
become one, and the graph building for the next iteration overlaps the
device work already submitted. Deeper speculation — SUBMITTING the
body before its condition resolves — is what would close the last
gap, and it is mutually exclusive with donation: the pipeline would
have to hold the carry the loop may return, which is exactly the
handle that stops MLX writing in place. Donation is worth far more
(a cache copy per token vs one round trip), so the pipeline
speculates on BUILDING only.

Gated on `!body->reads_host() && !cond->reads_host()` (no nested
while/if/case): with a host read inside, "building" the body means
running it, and a nested dynamic while need not even terminate at a
carry the outer loop is about to abandon.

Numbers (us/step, slope over two trip counts so per-execute costs
cancel; native engine, standalone, M5 Max):

| benchmark                                | main  | M5a serial | M5a  |
|------------------------------------------|-------|-----------|------|
| M3 decode shape (4 matmuls, no cache)    | 331.5 | 328.7     | 176.4|
| realistic decode (qkv+attn over KV+MLP)  | 236.4 | 197.3     | 160.5|
| cheap body, 16 MB cache                  | 339.9 | —         | 151.7|
| cheap body, 512 MB cache                 | 2376  | 328.4     | 165.0|

Projected on the model rows: rows 5/7 are dense decode whose bodies
are one compiled replay, so the win is the per-token round trip
(~150-170 us of the 23.3 ms/tok row 7, ~0.7%) PLUS whatever the KV
copy was costing — the latter is the one to measure, since it scales
with context length and the benchmark above says it was the whole
size-dependent term. Model rows are the main agent's to run.

**Mechanism worth keeping** (it explains a second finding):
`mx::async_eval` holds handles on its root arrays until the task
completes, so a graph built on those roots and submitted before
anything WAITS cannot donate them. That is why the pipelined loop
donates (its host read of the condition is the wait) and why the
COUNTED loop's chunked replay does not: `_run_chunked` async-evals
every chunk and blocks only every `sync_every`, so the first
`dynamic_update_slice` of each chunk copies. Measured on the 512 MB
cache with a counted loop: K=16 157 us/step, K=4 735, K=1 (no
chunking) 23 — one full cache copy per chunk, and uncompiled chunks
behave identically (165), so it is the chunk loop's shape, not
`mx::compile`. Not fixed here: `chunkable` requires body cost <=
`_CHUNK_MAX_COST` (1500), so a real decode body never chunks — the
shape that loses is a CHEAP body with a huge in-place carry.

Canaries: tests/test_command_buffer.py grew a fourth detector
(`pipeline`) — the same 28-layer init scan with its condition
rewritten into the non-counted form, pipelined vs serial. Clean at
the shipped budgets 3/3; corrupts at 128 MB (5 of 66 outputs) and 64
MB per buffer, clean at 100 and 50 kernels — so only the BYTE axis
bites this layout, where the serial native detector's band is the
other way round. The three existing detectors' bands did not move.

## M5b landed (2026-08-10): msl_scan kernels + host ops on the tape

Phase-1 doctrine held exactly: **Python compiles, C++ runs**. msl_scan's
planning, pattern matching and MSL source generation did not move at all;
what moved is the LAUNCH, and the one thing added to `msl_scan.py` is
`Plan.kernel_name` (the native side rebuilds the same kernel through
`mx::fast::metal_kernel` under the same name, so both engines share MLX's
compiled-library cache).

**kMslScan.** `tape.py::_while` now asks `ops.control._msl_plan_for` — the
same function, the same cache, the same eligibility questions the Python
engine asks — and lowers a `metaljax.msl_scan` entry when it answers. A
plan the tape cannot express declines the whole PROGRAM rather than running
the loop op by op: a loop that took a kernel on one engine and an
interpreted loop on the other would compute its carries by different
arithmetic, and no byte-level differential could hold. The entry is a
`kWhile` in every other respect — same attrs, same cond/body sub-programs —
so a kernel Metal rejects falls back to the interpreted loop *in the same
call* (the carries are untouched; `msl_scan.try_run` does exactly this),
and the plan stays dead for the process. Where the body is outside the
native op set the loop lowers with no fallback regions at all, and a
failure there hands the program back to the Python engine instead.

`Plan.run` is transliterated, resolved statically: its `bufs` loop becomes
slots plus a per-source weight-normalization recipe (as_strided ->
transpose -> contiguous, the layout canonicalization the backward pass
needs), its `feed` becomes the unpacked list plus the per-dtype pools of
0.4.3's input packing, and its `vals` assembly becomes four static lists
(pass-through carries, affine counters, stacked outputs, final states) plus
the accumulator recipes of loop fission, which are encoded as little trees
(hidden stack / buffer slice / post-kernel sum / post-kernel batched
matmul). The lowering checks that every carry position is produced by
exactly one rule.

Two disciplines carried over verbatim: the first launch of an unproven
plan evaluates SYNCHRONOUSLY (a Metal build error on an async worker aborts
the process), and when the kernel was traced into an `mx::compile` graph —
where nothing can be evaluated at the point it is built — the plan goes on
a pending list and `Program::run` settles the call synchronously, kills the
plans, drops the compiled graphs that embedded them and reruns. That is
`engine.execute`'s `_msl_pending` + `disable_msl` on the native side; it is
skipped (and the failure rethrown) for programs holding a host call, since
a rerun repeats what the first attempt already did to the world.

**kHostCall.** A `stablehlo.custom_call` whose target has a host handler
(`ops.lapack.TARGETS`, which includes `metaljax_callback`), plus
`stablehlo.cholesky` and `stablehlo.triangular_solve`, lowers to an entry
holding a Python callable bound at lowering time; C++ reacquires the GIL
there and nowhere else, so the arithmetic around a host op stays native.
`Sharding`/`annotate_device_placement` lower as aliases and
`shape_assertion` as nothing, exactly as the Python handler treats them.
Ordering is the tape's order, which is the block's order; ordered-effect
tokens are now ordinary (empty bool) values with `create_token`/`after_all`
as entries, so a program with ordered effects lowers instead of declining
on a non-tensor type.

**Census** (whole pytest suite, `METALJAX_ENGINE=native`): 605 -> 622
executables lowered, 116 -> 99 declined. The decline classes that remain
are all M2-era op-set gaps, itemized: complex 26, rng_bit_generator 23,
`stablehlo.gather` 19, fft 9, popcnt 2, sub-byte float types 5, general
reduce bodies 4, and singletons (scatter, pad, real, sort comparator,
TOTALORDER compare, recursive call, f64). Not one msl or host decline is
left.

**Floor probe** (3 db-class sub-ms configs, `scripts/texmo_topconfs.py
--bench-only`, ms/step):

| config | native (M5b) | python engine | jax-CPU |
|---|---:|---:|---:|
| db02-b4l1024 `bits.1+bp\|split.cat(rnn.1.tanh, pass)-norm-dense.1.tanh-suffix.2` | 0.764 | 0.761 | 0.115 |
| db09-b128l128 `tokens.32.fold.emb.8\|mingru.4-latent.8.2` | 1.407 | 1.405 | 2.149 |
| db11-b64l256 `bits.4.oh+bp\|rnn.16.gelu-dense.8.gelu-rmsnorm` | 0.645 | 0.648 | 2.951 |

Unmoved, and the reason is not msl: **every texmo train chunk declines on
`stablehlo.gather`** (the cross-entropy target gather; the tokenized inputs
add more), so the native tape never runs one. The floor cannot move until
gather (and scatter, for the optimizer's index updates) join the op set —
that, not kernel invocation, is what M6 needs next. On a gather-free
texmo-shaped chunk (256 training steps, each a scan + its AD transpose +
an SGD update, lowering with 2 msl entries inside a 256-step while) the two
engines are at parity: 0.1962 vs 0.1959 ms/step at L128/H32/B32, 0.0499 vs
0.0497 at L32/H16/B16 — consistent with M4's verdict that a body which is
already one compiled replay has no Python dispatch left to remove.

**Canaries** (tests/test_command_buffer.py): all 10 pass, and the init-scan
detectors are untouched by this milestone — that scan's body contains RNG
shifts msl_scan rejects (`not eligible (op stablehlo.shift_right_logical)`),
so its sync-point layout is the same one M3/M5a measured. No band moved.

Not covered by a test: a plan source that msl_scan HOISTED out of the body
(an invariant op inside the loop). The tape lowers its defining ops into
the enclosing frame, which is what `Plan.run`'s `hoisted` does by
re-evaluating them, but no program in reach produces one — jax's `lax.scan`
threads every closed-over value through the carry list, so jax-lowered
plans have carry sources only, and the hand-written loop that does trigger
a hoist dies earlier inside msl_scan itself (`_source_key` keys hoisted
sources as `("free", id, value)` while `source_id` keyed them as
`("hoist", value)` — a pre-existing inconsistency, left alone here). The
capture case, one step short of it, is tested.

## M5c landed (2026-08-10): the decline tail

Everything M5b's census itemized, transliterated from `src/metaljax/ops/`
the same way: what the Python handler decides from the IR is decided at
lowering, what it computes on arrays is a C++ handler, and the differential
compares BYTES on both the eager and the mx::compile'd path.

**Census** (whole pytest suite, `METALJAX_ENGINE=native`): 83 -> 19
declined. What landed, family by family:

| family | was | now |
|---|---:|---|
| complex64 (+ `real`/`imag`/`complex`) | 29 | dtype-table entry + the C99 arms: scaled-hypot `abs`, `sign`, `exp`/`expm1`'s exact zero-sin, Kahan `csqrt`/`rsqrt`, `tan`'s pole, complex->real `convert`, complex `iota`/`dot_general` |
| `rng_bit_generator` | 23 | Philox4x32 + ThreeFry2x32, bit-exact (the whole block/half schedule is static and resolved in Python) |
| `fft` | 9 | `mx::fft::{fftn,ifftn,rfftn,irfftn}` + both MLX workarounds (empty transform, unit last length) + the eager input barrier, which reads the `in_trace` flag the interpreter already threads |
| general reduce bodies | 4 | the body becomes a sub-Program; the pairwise halving, the odd-extent init pad and the final init fold are the handler's, in its order |
| `popcnt` / `count_leading_zeros` | 2 | SWAR, u64 for 64-bit operands and u32 below |
| TOTALORDER compare | 1 | the integer keys `total_order_key` already built for sort |
| `pad` | 1 | interior dilation (slice_update into an init-valued array), edge pads, negative-pad crop |
| `reduce_window` | 1 | the cum-op peephole, the as_strided window path (base dilation, padding, the materialization MLX's strided reductions need), monoid / select_and_gather_add / generic-body folds |
| `reverse` | — | a latent op-set gap with no test behind it, found by a complex case |

**Declined, on purpose.** Sub-byte floats (7: f4/f6/f8 grids) are NOT
ported. Their values live in a WIDER storage dtype, which breaks the
invariant the dtype table exists for (storage IS the logical bits, which
is what makes `bitcast_convert` and every `mx::view` in the tape correct),
and a faithful port would have to re-grid after every arithmetic op — a
per-site flag on every elementwise entry, where one missed site is silent
wrongness rather than a decline. `dtypes.quantize_emulated` cannot be
called from the tape either: it is device computation, so binding it as a
Python callable would reacquire the GIL in the hot path and could not be
traced into `mx::compile`.

Also still declining, each for a reason the tape cannot paper over:
`select_and_scatter` (its scatter-add over overlapping windows is
order-nondeterministic on the GPU, so no byte differential could hold it
to the Python engine), scatter on complex (MLX has no complex scatter
kernels; the Python handler scatters the two parts), a sort comparator
that computes a KEY, a scatter apply-body, `convolution` (never in the op
set), f64, and the two test programs that assert a decline.

**Floor probe** (bench_spec.py, nsteps=16, ms/step; the M5b table's three
db-class configs). `stablehlo.pad` was the blocker: every texmo train
chunk declined on it, and all three now lower with no declines at all.

| config | native | python engine | jax-CPU |
|---|---:|---:|---:|
| db02-b4l1024 | 0.985 | 0.914 | 0.121 |
| db09-b128l128 | 1.598 | 1.572 | 2.238 |
| db11-b64l256 | 0.843 | 0.825 | 3.087 |

So the floor did not move — and the native path is now 2-10% SLOWER than
the Python one on these (repeated interleaved: db02 0.971/1.014 native vs
0.921/0.920 python, db11 0.834/0.818 vs 0.801/0.805 — consistent, not
noise). These chunks are one compiled replay per step on both engines, so
nothing is left for dispatch removal to win; the likely cost is the tape's
STATIC output-copy rule (a carry an optimizer leaves untouched is tainted
as an argument alias and copied, where the Python engine's `id()` check
copies nothing) plus the nanobind crossing. Worth measuring before M6
flips the default — it is the first row where native is behind.

Canaries: 10/10 on both engines, no band moved.

## The 2-10% chunk deficit, attributed (2026-08-10)

Not the tape. Measured on db02 by splitting one chunk into the parts
that can differ — `execute`'s CPU time (the graph is submitted inside it,
by `async_eval`), the wait for the first output to land, and the host
transfer of the other 22 — best of 5, ms per 16-step chunk:

| part | python | native | delta |
|---|---:|---:|---:|
| submit + GPU completion (execute CPU + wait) | 14.725 | 14.705 | -0.020 |
| host transfer of the other 22 outputs | 0.159 | 0.770 | **+0.611** |
| total | 14.884 | 15.474 | +0.590 |

The parts sum, and only one of them moves. Inside the first: native's CPU
half is 4.665 vs 5.170 ms — the tape IS cheaper to dispatch, the GPU is
simply the binding constraint, so the saving shows up as a longer wait
and nothing else. The suspected culprit is innocent twice over: these
chunks lower with **zero** static output copies (db02 23 outputs / 0,
db09 23 / 0, db11 26 / 0 — an optimizer carry is written by the update,
so no taint reaches it), and the boundary crossing is inside the part
that came out ahead. Across the tape test corpus, which is deliberately
full of aliasing edge cases, 24 of 272 programs carry any copy at all.

The cost was M1's `to_host`: it wrapped every result in a fresh
`mx::contiguous` node and evaluated THAT. The copy short-circuits to a
shared buffer, but the eval is a full round trip through the stream —
**20us per output whatever its size** (measured; `eval` on the array
itself, already available, is 0.11us). Per OUTPUT, so a program handing
back 23 small tensors paid 0.5ms. It now settles the array itself and
builds a `contiguous` only for a layout that needs gathering
(`row_contiguous` and `data_size`, both read after the eval that sets
them — the short-buffer read this guards is the conv-overread bug of
CLAUDE.md item 20, in a place with no test to catch it).

`to_host` on a settled f32: 21us -> 0.07us at 16 elements (numpy path
0.43), 24.9us -> 3.4us at 256x256 (numpy 13.7). Interleaved chunks,
ms/step, 5 rounds each:

| config | native before | native after | python |
|---|---:|---:|---:|
| db02-b4l1024 | 1.004 | 0.923 | 0.919 |
| db09-b128l128 | 1.617 | 1.565 | 1.578 |
| db11-b64l256 | 0.840 | 0.776 | 0.791 |

Suites 1255/1255 on both engines, canaries 10/10 on both, and the floor
is pinned by a test comparing the two paths' per-call cost as a ratio
(43x before the fix, 0.2x after).

Left alone, deliberately: `buffer_from_host` is 2.2us vs the numpy
path's 1.0 (one staging malloc+memcpy the numpy path gets for free from
`frombuffer`), and a non-row-contiguous output still costs its
gather-and-sync (~190us) where a host-side odometer gather like the
ingest path's would not need the GPU at all. Neither is on a measured
hot path today.

## Grand plan (Oleg, 2026-08-10)

1) Migrate ALL non-test code to C++ (phase 2 included);
2) correctness as close to perfect as possible;
3) release; 4) make every model that runs on mlx-lm/torch/llama.cpp
run on metaljax (rows 8/10/12/15 + the llama.cpp expansion);
5) tackle performance disparities one by one.
Steps 3/4 may swap depending on post-migration state.

## Dispositions (Oleg, 2026-08-10)

- f64 emulation: ASPIRATIONAL, last roadmap stage only (perf phase),
  mainly to satisfy the JAX suite's f64 entries. At double-double's
  5-15x, host-CPU execution of f64 programs (the existing host-op
  callback machinery fits) may beat on-device emulation — evaluate
  both then. Not a priority.
- Off-MLX migration: NON-GOAL. Understanding recorded above for the
  lay of the land; the only planned form is targeted hand-written
  kernels for specific ops during perf work (the msl_scan/
  metal_kernel mechanism we already use).

## Phase-2 decline dispositions (every decline must die: the Python
## fallback is deleted with the rest of the Python engine)

- sub-byte floats (7): PORT — implement quantize_emulated as a native
  mx-op sequence + a per-site regrid flag on elementwise entries; the
  silent-wrongness risk is contained by the static registry audit and
  goldens frozen while both engines coexist.
- complex scatter (3): PORT — by-parts decomposition (real/imag
  scatters), same shape as the Python handler.
- convolution (2): PORT (already queued with the census-blind-spot
  audit).
- sort-with-key-chain (2): PORT — key chain as a sub-Program feeding
  stable argsort (general-reduce-body mechanism).
- select_and_scatter (1): PORT — GPU-nondeterministic like the Python
  engine; goldens with tolerance, not byte-pins.
- scatter apply-body (1): PORT — per-update sub-Program under
  unique_indices.
- asserted declines (3: unknown custom call, recursive callee, f64
  strict): become COMPILE ERRORS — strict failure is the correct
  phase-2 semantics, matching today's f64 policy.
- Host ops: LAPACK targets move to Accelerate (direct C, no Python);
  jax user callbacks (debug.print/pure/io_callback) keep a C-level
  trampoline — that is the USER'S Python, not ours, and every PJRT
  backend needs it.
- Tests-to-engine-path migration (Interpreter-direct files) + the
  static opcode-vs-REGISTRY audit: scheduled with the M6 flip prep
  (per Oleg, 2026-08-10).

## Post-migration: XLA optimization layer (Oleg, 2026-08-10)

DECIDED: once the C++ migration is done, add a graph-level
optimization stage above the tape — the missing middle layer vs
XLA's architecture (optimized-HLO :: thunks == optimized-StableHLO ::
tape). Two candidate routes, evaluate then: (a) XLA's HLO pass
pipeline as a library pre-step (cleans unoptimized input StableHLO,
enables an honest PJRT OptimizedProgram); (b) our recognizers
graduate to StableHLO->StableHLO MLIR rewrite passes in the phase-2
compile path, producing an inspectable optimized graph. Not before
the migration completes.

## M6 flipped (2026-08-11, per Oleg — no full sweep, no release until
## migration completes). Default engine = native, graceful py fallback
## when unbuilt. Phase 2 begins: StableHLO/MLIR C++ compile path,
## recognizers as passes, exotic-decline ports (conv first), fallback
## retirement, goldens freeze, then the XLA optimization layer.

## Phase 2 architecture — REVISED (Oleg, 2026-08-11)

TWO COMPLETE PLUGINS side by side, not migration-under-the-trampoline:
- plugin/ (current, trampoline) FREEZES as the reference; selected any
  time via METALJAX_PLUGIN_PATH for feature-by-feature comparison.
- plugin-native/ built TOP-DOWN from scratch: PJRT C API -> native
  StableHLO/MLIR parse -> analyses/recognizers (as passes) -> tape
  build -> the SHARED executor (native/ tape modules; program.h is the
  interface line). Free to structure for C++, unconstrained by the
  Python architecture. Forking the executor allowed if ever needed.
- No releases until migration completes -> half-working plugin-native
  is acceptable; validation = suite-vs-suite via the jaxlib PJRT
  client (same StableHLO through both plugins, bytes compared), not
  per-program fallback.
- Sequencing: P0 StableHLO/MLIR C++ build proof from the llvm-project
  clone (deserialize + round-trip a portable artifact — this phase's
  M0 handshake) + plugin-native skeleton; then parse/analyses (bit-
  identical lottery-pinned numbers, canaries), recognizers-as-passes,
  pack building (Accelerate for LAPACK), exotic ports + goldens
  freeze + trampoline retirement, XLA optimization layer last.
- tape.cc split into modules (in flight) — its report is the
  executor's structural doc for the plugin-native author.
- **P0 landed** (notes/pjrt-native-p0.md): the xla::PjRtClient route,
  plus the wheel PoC. **P1 landed** (notes/cpp-p1-runtime.md): the
  executor runtime is Python-free and links into the plugin.
  **P2 landed** (notes/cpp-p2-lowering.md, 2026-08-11): a C++ tape
  builder + MetalLoadedExecutable — `2 * jnp.array([1,2,3])` computes on
  the GPU through the native plugin with no interpreter under jax.
  Elementwise / shape / reduce / dot_general / constant lower (calls
  inlined; bf16 constants read straight out of DenseElementsAttr, which
  the Python bindings cannot do); everything else declines naming its
  op. 77-case differential suite vs jax-CPU
  (plugin-native/execute_test.py), and a tape dump byte-identical to
  tape.py's over six programs. Next: the compile decisions, control
  flow, gather/scatter, async execute.
## The native engine's file layout (2026-08-11)

`native/tape.cc` had grown to 4,128 lines holding everything from the
opcode enum to the msl launch recipe, with `Program::step` a single
1,500-line switch. It is now one translation unit per concern behind
one header. Movement only: the code is the same lines in the same
order, and the differential suites are the proof (1258/1258 on both
engines before and after, canaries 11/11, per-entry dispatch
unchanged at ~105 ns on a 2,000-entry elementwise tape).

**`program.h` is the interface line.** A phase-2 plugin that wants to
build tapes and run them includes this and nothing else: the opcode
enum, the attribute layouts, `Entry`, `Cursor`, `Program`, the dtype
predicates and the runtime disciplines. `msl.h` is separate on
purpose — a tape carries a generated kernel only as an opaque handle
(`Entry::msl`), so nothing but msl.cc and the nanobind registration
needs to know what a launch recipe is made of.

| file | lines | what lives there |
|---|---:|---|
| `program.h` | 585 | the interface: opcodes, attrs layouts, Entry, Cursor, Program, dtype + runtime declarations |
| `msl.h` | 114 | `MslPlan` + `AccNode`, for the two files that touch a kernel |
| `program.cc` | 306 | Program: slots, the environment, the walk, the eager flush, the recovery ladder |
| `config.cc` | 231 | the Python-facing surface: op-name registry, `configure`, `stats`, `register_tape` |
| `dtypes.cc` | 211 | dtypes.py: the element-type table, predicates, weak literals, complex64 |
| `runtime.cc` | 134 | interpreter.py + ops/control.py cadences: flushes, loop accounting, host reads |
| `compile.cc` | 111 | mx::compile integration: cache ids, output anchoring, one-way retirement |
| `ops_elementwise.cc` | 492 | ops/elementwise.py (+ compare/select/convert, fft, popcnt/clz) |
| `ops_shape.cc` | 199 | ops/shape.py (+ `stablehlo.constant`) |
| `ops_reduce.cc` | 373 | ops/reduction.py: monoid, arg pair, general bodies, reduce_window |
| `ops_index.cc` | 264 | ops/gather.py + ops/sort.py: index plans, OOB-drop strategies, sort/top_k |
| `ops_linalg.cc` | 118 | ops/linalg.py: dot_general's three arms |
| `ops_rng.cc` | 205 | ops/rng.py: Philox + ThreeFry, bit for bit |
| `emits.cc` | 485 | the M4 recognizer emits: qmm, sdpa, moe |
| `control.cc` | 449 | ops/control.py: while/if/case, chunked replay, pipelining, `BodyRunner` |
| `msl.cc` | 495 | M5b: MslPlan's launch + the entry that dispatches one + settle/retire |
| `host.cc` | 93 | ops/callbacks.py: the two entries that take the GIL, and `gc_collect` |

The op files mirror `src/metaljax/ops/` one for one, which is the
point: every handler is a transliteration of the Python one, and the
differential test compares output BYTES, so a reader (or a reviewer of
a phase-2 port) can put the two engines' files side by side.

**How dispatch works now.** `Program::step` holds no handler. It offers
the entry to one `step_*` per family in turn; each switches on the
opcodes it owns and returns false for the rest, and the `throw` at the
end of the chain is what asserts the partition. Adding an op is
therefore: an enum value in `program.h`, a name in `config.cc`'s
registry, a case in the family file. Nothing else, and no table that
can disagree with a switch. Measured cost of the chain: nil (an
elementwise entry hits the first probe; the deepest, a host call, is
eight predictable branches on a path that is about to acquire the GIL).

**What a phase-2 author should know about the coupling.**

* `Program` is the only class with private state, and the op families
  are its private member functions — not free functions — because the
  handlers reach four things: `shape()`/`axes()` (attribute readers),
  `generic_reduce` (a sub-Program run pairwise), `run_while`/`run_msl`
  (control flow) and `write_results`. A family in a new file is a new
  method declared in `program.h`; that is the whole ceremony.
* Region programs are reached through `Entry::regions` and called with
  the public API (`call`, `interpret`, `may_compile`, `compiled`,
  `drop_compiled`), so a loop body is compiled, replayed and recovered
  exactly like a top-level program. `control.cc`'s `BodyRunner`,
  `run_body` and `run_chunked` are file-local for that reason: they
  need nothing private.
* The runtime globals (`g_cfg`, `g_stats`, `t_msl_pending`) are
  declared in `program.h` and defined once — config.cc for the first
  two, msl.cc for the pending list. They were file-statics inside one
  anonymous namespace before; everything is now in `namespace
  metaljax`, and helpers used by exactly one file stayed file-local in
  an anonymous namespace of their own.
* `register_tape` remains the single symbol between the extension's
  two halves (metaljax_native.cc declares it; config.cc defines it at
  global scope with a `using namespace metaljax` inside).
* The nanobind stl casters are included by `program.h` deliberately:
  they must be visible in every TU that crosses the boundary, and a
  missing one is silent (nanobind falls back to an opaque type).
## The native engine's file layout (2026-08-11)

`native/tape.cc` had grown to 4,128 lines holding everything from the
opcode enum to the msl launch recipe, with `Program::step` a single
1,500-line switch. It is now one translation unit per concern behind
one header. Movement only: the code is the same lines in the same
order, and the differential suites are the proof (1258/1258 on both
engines before and after, canaries 11/11, per-entry dispatch
unchanged at ~105 ns on a 2,000-entry elementwise tape).

**`program.h` is the interface line.** A phase-2 plugin that wants to
build tapes and run them includes this and nothing else: the opcode
enum, the attribute layouts, `Entry`, `Cursor`, `Program`, the dtype
predicates and the runtime disciplines. `msl.h` is separate on
purpose — a tape carries a generated kernel only as an opaque handle
(`Entry::msl`), so nothing but msl.cc and the nanobind registration
needs to know what a launch recipe is made of.

| file | lines | what lives there |
|---|---:|---|
| `program.h` | 585 | the interface: opcodes, attrs layouts, Entry, Cursor, Program, dtype + runtime declarations |
| `msl.h` | 114 | `MslPlan` + `AccNode`, for the two files that touch a kernel |
| `program.cc` | 306 | Program: slots, the environment, the walk, the eager flush, the recovery ladder |
| `config.cc` | 231 | the Python-facing surface: op-name registry, `configure`, `stats`, `register_tape` |
| `dtypes.cc` | 211 | dtypes.py: the element-type table, predicates, weak literals, complex64 |
| `runtime.cc` | 134 | interpreter.py + ops/control.py cadences: flushes, loop accounting, host reads |
| `compile.cc` | 111 | mx::compile integration: cache ids, output anchoring, one-way retirement |
| `ops_elementwise.cc` | 492 | ops/elementwise.py (+ compare/select/convert, fft, popcnt/clz) |
| `ops_shape.cc` | 199 | ops/shape.py (+ `stablehlo.constant`) |
| `ops_reduce.cc` | 373 | ops/reduction.py: monoid, arg pair, general bodies, reduce_window |
| `ops_index.cc` | 264 | ops/gather.py + ops/sort.py: index plans, OOB-drop strategies, sort/top_k |
| `ops_linalg.cc` | 118 | ops/linalg.py: dot_general's three arms |
| `ops_rng.cc` | 205 | ops/rng.py: Philox + ThreeFry, bit for bit |
| `emits.cc` | 485 | the M4 recognizer emits: qmm, sdpa, moe |
| `control.cc` | 449 | ops/control.py: while/if/case, chunked replay, pipelining, `BodyRunner` |
| `msl.cc` | 495 | M5b: MslPlan's launch + the entry that dispatches one + settle/retire |
| `host.cc` | 93 | ops/callbacks.py: the two entries that take the GIL, and `gc_collect` |

The op files mirror `src/metaljax/ops/` one for one, which is the
point: every handler is a transliteration of the Python one, and the
differential test compares output BYTES, so a reader (or a reviewer of
a phase-2 port) can put the two engines' files side by side.

**How dispatch works now.** `Program::step` holds no handler. It offers
the entry to one `step_*` per family in turn; each switches on the
opcodes it owns and returns false for the rest, and the `throw` at the
end of the chain is what asserts the partition. Adding an op is
therefore: an enum value in `program.h`, a name in `config.cc`'s
registry, a case in the family file. Nothing else, and no table that
can disagree with a switch. Measured cost of the chain: nil (an
elementwise entry hits the first probe; the deepest, a host call, is
eight predictable branches on a path that is about to acquire the GIL).

**What a phase-2 author should know about the coupling.**

* `Program` is the only class with private state, and the op families
  are its private member functions — not free functions — because the
  handlers reach four things: `shape()`/`axes()` (attribute readers),
  `generic_reduce` (a sub-Program run pairwise), `run_while`/`run_msl`
  (control flow) and `write_results`. A family in a new file is a new
  method declared in `program.h`; that is the whole ceremony.
* Region programs are reached through `Entry::regions` and called with
  the public API (`call`, `interpret`, `may_compile`, `compiled`,
  `drop_compiled`), so a loop body is compiled, replayed and recovered
  exactly like a top-level program. `control.cc`'s `BodyRunner`,
  `run_body` and `run_chunked` are file-local for that reason: they
  need nothing private.
* The runtime globals (`g_cfg`, `g_stats`, `t_msl_pending`) are
  declared in `program.h` and defined once — config.cc for the first
  two, msl.cc for the pending list. They were file-statics inside one
  anonymous namespace before; everything is now in `namespace
  metaljax`, and helpers used by exactly one file stayed file-local in
  an anonymous namespace of their own.
* `register_tape` remains the single symbol between the extension's
  two halves (metaljax_native.cc declares it; config.cc defines it at
  global scope with a `using namespace metaljax` inside).
* The nanobind stl casters are included by `program.h` deliberately:
  they must be visible in every TU that crosses the boundary, and a
  missing one is silent (nanobind falls back to an opaque type).

## Phase 2 ledger (2026-08-10..11)

Route decision (Oleg): the **C++ PJRT API** — subclass xla::PjRtClient,
let pjrt_c_api_wrapper_impl manufacture the C surface. Chosen over the
hand-rolled C API after the P0 measurement (notes/pjrt-native-p0.md);
the deciding synergies: one bazel workspace supplies the wrapper,
StableHLO/MLIR at jax's exact pin, real protos, and the later XLA
optimization layer as a deps entry. Runtime is bazel-compiled too (Oleg);
fork authorized but not needed.

* **P0 landed** (603c092): plugin-native/ bazel workspace, xla via
  local_repository at 131bf41a (= jax-v0.11.0's pin), MetalClient stub
  through the wrapper; jax.devices() serves from it; f32 round trip;
  CompileAndLoad receives a PARSED stablehlo module (the VHLO dance is
  free). Cold build 399 s, warm 3 s, edit loop 4-6 s, dylib 157 MB.
* **Wheel PoC landed** (8f2511b): METALJAX_WHEEL_PLUGIN=native bundles
  the native plugin; fresh-venv full stack on py3.13 passes; rpath
  order is load-bearing (consumer's libmlx must win over the build
  venv's); --incompatible_strict_action_env vs uv's per-invocation
  PATH; identical-filename + unbounded-mlx-pin traps recorded.
* **P1 landed** (c8736db): program.h is Python-free (HostFn, g_gc_hook,
  GIL contract on the caller); bindings.cc is the adapter;
  @metaljax_runtime = new_local_repository over native/ (SHARED sources,
  15 TUs); runtime linked into the plugin dylib; GIL-free cc_test proves
  interpreted + mx::compile execution with no interpreter in the
  process. pytest 1258/1258, gate 106/106.
  Supersedes two lines of the file-layout section above: register_tape
  now lives in bindings.cc (not config.cc), and the nanobind stl casters
  ride in bindings.cc (program.h includes no nanobind at all).
* **P2 landed** (4efd2ee, notes/cpp-p2-lowering.md): native lowering
  (tape.py's attr encodings are the spec), MetalLoadedExecutable::Execute,
  M1 dtype table on the buffers, a 77-case differential execute_test vs CPU.
* **P3 landed** (notes/cpp-p3-control.md): control flow. Regions are a
  recursive use of the same `Lowering`, so while/if/case, ds/dus (with
  XLA's index clamps) and the counted-loop encoding reach the executor —
  `lax.scan`/`fori_loop`/`while_loop`/`cond`/`switch` all run. The runtime
  cadences are read from the environment at client creation (this plugin
  is the only engine in its process, so the "two readers would drift"
  argument that held in P2 does not apply, and the loop clear cadence is
  correctness for long loops). Compile decisions stay OFF: the three
  compile fields of a while entry are written as zeros, which is what
  tape.py writes under METALJAX_COMPILE=0. execute_test 102 checks;
  cross-check vs the Stage 1 tape over 9 probes / 170 lines is identical
  but the dead `kmax` field. Census: **gather and scatter are the only two
  ops between this plugin and a texmo training step**.
* **P4 landed** (notes/cpp-p4-gather-scatter.md): gather, scatter (both OOB
  drop strategies, batching dims, windowed updates) and the small-op tail —
  the shifts, reverse, bitcast_convert, popcnt/clz — so jax's threefry RNG
  lowers as plain elementwise and is BIT-exact vs CPU. **All 106 texmo suite
  configurations now train through the native stack and match jax-CPU**
  (`plugin-native/texmo_gate.py`, phase 2's standing gate: 106 ok, 0 decline,
  0 FAIL, 6m15s). execute_test 152 checks; the tape cross-check is
  byte-identical over 12 probes plus a real training chunk, but the dead
  `kmax`. Two findings worth the ledger: (a) the plugin was missing MLX's own
  command-buffer budgets (`MLX_MAX_{OPS,MB}_PER_BUFFER`, which
  src/metaljax/__init__.py pins for Stage 1) and was therefore hitting MLX
  0.32's split corruption — nondeterministically wrong training steps, 2 runs
  in 5; now pinned in metal_client.cc — at the price of exposing MLX's
  process-wide command-encoder map, which segfaults ~5% of the 8-thread
  execute_test row (0 of 74 at MLX's smaller defaults, 0 of 20 through the
  GIL-serialized Stage 1 plugin: being GIL-free is what makes it reachable).
  (b) The same corruption still reaches
  one suite row through the EAGER path, and the Stage 1 engine reproduces it
  exactly under METALJAX_COMPILE=0 — so the compile decisions are correctness
  for long loops, not only performance. Also: XLA's parse rewrites the module
  before CompileAndLoad (chlo legalized, constants CSE'd and hoisted out of
  regions), so tape diffs now start from METALJAX_DUMP_MODULE=1.
* **P5 landed** (notes/cpp-p5-compile.md): the compile decisions. `set_compile`
  on main and on while bodies, and the three fields P3 wrote as zeros
  (`chunkable`/`kmax`/`body_compile_max`), computed by transliterations of
  `interpreter.block_is_pure` / `op_bytes` and `ops/control`'s
  `_block_bytes` / `_passthrough_bytes` / `_bytes_ok` / `_bytes_chunks` /
  `_while_traceable` / `_underived_outputs`, under the same six environment
  budgets. This was CORRECTNESS, as P4 predicted: `db18-b4l1024` went from
  FAIL 3.1e+01 / ok~ 1.3e-02 / FAIL 5.4e-03 to `ok` at 1.9-4.3e-05 on five
  standalone runs, `synth-matlstm-b` likewise, and six full gate runs are
  106/106 with zero FAIL. `METALJAX_COMPILE=0` puts the flicker back, which is
  the control. The tape cross-check is byte-identical over 11 probes / 259
  lines with compile ON — including the `kmax` field that differed in P3 and
  P4. Perf, recorded not gated (one 8-step gate chunk, ms): the decisions are
  worth 1.7-3.5x over the eager plugin, and the native path now sits within
  1-3 % of the Stage 1 engine with `METALJAX_MSL=0` on every row measured —
  the whole remaining gap to Stage 1's default is msl_scan. One finding left
  open: MLX does not fuse the FIRST compiled executable in a process (a few
  ULP on transcendental chains; four warm-ups tried, none moves it; Stage 1
  does not show it).
* **P6 landed** (notes/cpp-p6-tail.md): the decline tail whose executors already
  existed — `sort`/`chlo.top_k`, `rng_bit_generator` (Philox + ThreeFry, bit-exact
  vs CPU on every output width), `reduce_window` (cum peephole, monoid,
  select_and_gather_add, generic bodies, base/window dilation, zero-size guards)
  and `fft` (all four kinds plus both MLX rewrites), with `stablehlo.reduce`'s
  general-body arm riding along on the same sub-Program mechanism. `sort` needed
  more than a transliteration: `tape.py` lowers only the bare-compare comparator,
  and jax's FLOAT sort computes a key (-0 -> +0, NaN -> qNaN, TOTALORDER), so
  `ops/sort.py`'s recognizer was ported — dep analysis, the structural symmetry
  check, and the key chain lowered into the ENCLOSING frame as ordinary entries
  (scalar elementwise code computing the key of the whole operand), with the sort
  entry keyed on its output. Guards: an allowlist plus an all-rank-0 check, so a
  non-elementwise chain declines instead of being run against its rank-0 IR type.
  Census 25 -> **31 of 35**; execute_test 156 -> **228 checks**; cross-check
  **146 lines over 19 probes byte-identical** (the four float-sort probes have no
  Stage 1 tape — `tape.py` declines them, which is the point); gate 106/106 x2;
  pytest 1258; dylib +0.033 %.
  **The mission's premise failed on one family and it is worth repeating here:
  `convolution` is NOT a lowering gap.** There is no `kConv` opcode in
  `native/program.h` at all — `src/metaljax/ops/conv.py` has never been
  transliterated (M5c's census says "never in the op set"), so porting it is
  executor work with the full P1 battery behind it, not lowering work. It is now
  the last op between this plugin and jax's dense-model surface, and it is off
  texmo's path (the gate is 106/106 with `mid11`'s `conv.4` included).
  Still declining and named: the two lexicographic sort comparators (a different
  execution shape — a permutation threaded through successive stable argsorts —
  which would need a `take_along_axis` entry), `select_and_scatter`, negative
  reduce_window padding (Stage 1 raises too), LAPACK, and `debug_print` (a
  JAX-side registration gap: no lowering rule for platform `metal`).
* **P7 landed** (notes/cpp-p7-conv.md): **convolution, both halves** — the first
  P-milestone whose work is EXECUTOR work. `native/ops_conv.cc` (329 lines) is
  `src/metaljax/ops/conv.py` handler for handler: `mx::conv_general` for every
  float layout, XLA's feature and batch groups (MLX's own `groups` where it
  serves, one convolution per group where it does not), the exact integer path
  (zero-hole dilation + pad + ONE `as_strided` im2col view, summed in int64),
  complex as four real convolutions, the 0-spatial matmul, the negative-pad
  rewrite, and both zero-size guards — the second of which is the conv
  short-buffer overread of CLAUDE.md item 20, kept as a `want`-shape check on
  whatever MLX produced. The plugin's `LowerConv` resolves the three layout
  permutations, the window attributes and the arm; `precision_config` is
  ignored, as Stage 1 ignores it. Declined and named: a MIXED
  `window_reversal` (MLX's flip is all-or-nothing, and the Python handler
  raises) and a COMPLEX 0-spatial convolution (the Python matmul arm runs its
  operands through f32 and would drop the imaginary part — a deliberate
  divergence rather than a transliteration).
  **The finding: the opcode is registered as `metaljax.conv`, a pseudo-name.**
  `tape.py` lowers any op it finds in the registry, with an EMPTY attribute
  vector when it has no `_HANDLERS` entry — so the StableHLO name would have
  enrolled Stage 1's lowering, which knows nothing about this layout (caught
  by the two `tests/test_native_tape.py` cases that use convolution as their
  "an op with no opcode" stand-in). With the pseudo-name Stage 1 declines the
  op at COMPILE time and keeps running it on the Python engine, which is what
  the milestone wanted and is now enforced by the registry.
  Census 31 -> **32 of 35**; execute_test 228 -> **274 checks** (46 conv rows:
  1/2/3-D, four layouts, all paddings and dilations, feature/batch/depthwise
  groups, exact integer incl. 2-D strided, complex, 0-spatial, zero-size,
  jnp.convolve/correlate, the jax wrappers, two grads — which is where the
  transposed and batch-grouped arms really get tested — and conv inside scan
  and fori bodies, plus two hand-written `window_reversal` modules); both gates
  106/106; pytest 1258 unchanged; eager vs compiled 234/235 bit-identical;
  dylib +0.022 %. **No tape cross-check exists for this family** — `tape.py`
  declines convolution, so the CPU differential is the only reference, and that
  is the point rather than a gap. `smoke_test`/`wheel_poc_test`/one
  execute_test decline all moved their checkpoint from convolution to
  `stablehlo.cholesky`.
* **P8 landed** (8d99421, notes/cpp-p8-jax-census.md): the whole pinned suite
  through both stacks, sequential, same tree — native 26,133/2,059 (92.70 %)
  against Stage 1's 28,062/137 (99.51 %), with all 1,918 native-only failures
  classified and every one of them LOUD. The phase ordering below is that
  census's, by measured test count. Second finding: a live Stage 1 bug (M5a's
  pipelining runs an impure while cond once extra), fixed in P8.5.
* **P8.5 landed** (notes/cpp-p85-fixes.md): the census's fix batch — one bug in
  the SHARED runtime and three in the plugin, **126 native-only failures** plus
  the four Stage-1 regressions.
  (a) `Program::reads_host` counted control flow but not `kHostCall`, so the
  pipelined dynamic while built a region that LEAVES the tape one iteration
  ahead of the condition that decides whether it runs — a second `debug.print`
  per loop. The bug was wider than the census could see: prints in a BODY are
  wrong too, and the tests miss it only because their loops are counted.
  (b) MLX's `sum`/`prod` accumulate wider for small integers (int8 -> int32),
  so a `stablehlo.reduce` handed back the wrong element type; Stage 1 hid it
  with a cast in `to_host`, the plugin's result guard caught it. Same line fixes
  `reduce_window`'s monoid arm. (c) A zero-size constant is stored by MLIR as a
  splat with ONE raw element, which the lowering's length check rejected — all
  52 census rows are that one form, and they come from chlo decompositions over
  an empty operand. (d) `mx::compile` refuses some traces at EVAL, and the eval
  was the caller's, outside `run_recovering`'s ladder; the ladder now settles
  the first compiled call of each program itself, exactly as `BodyRunner`
  probes a freshly bound body. Also classified, not fixed:
  `lax_test::testScatter1` is a duplicate-write scatter race (index 8 written by
  three updates; XLA leaves the winner implementation-defined), an in-suite
  lottery at ~1 run in 14, and a wontfix.
  Affected files re-run natively: 846 -> **720 failures, -126 exactly, zero
  regressions**. execute_test 274 -> **284**; the GIL-free runtime test grew a
  fourth section (loop shape + host-call counts, the one place a host call can
  be stated end to end); both gates 106/106; pytest 1258.
  Open, pre-existing, found while measuring: the 8-thread `execute_test`
  contract fails ~5 % of runs with `There is no Stream(gpu, N) in current
  thread` — a compiled graph traced on a pool thread and replayed after that
  thread exited with its `new_thread_unsafe_stream`. Same rate on the
  unmodified tree, 0/40 under `METALJAX_COMPILE=0`.
* **P9 landed** (notes/cpp-p9-linalg.md): the linalg family, **both halves**.
  (a) The JAX-side registrations: `_initialize_native` now calls the same
  `_register_linalg_lowerings` the trampoline calls, minus callbacks and
  donation (P13's) — without them eigh/svd/eig/schur/hessenberg/tridiagonal
  die at TRACE time on this platform and never reach the plugin at all, which
  is 319 of the census's 823. (b) The execution path:
  `plugin-native/runtime/host_lapack.cc` (1,158 lines, the fork's first new
  file) is `src/metaljax/ops/lapack.py` on **Accelerate's LAPACK**, LP64 —
  twelve factorizations, batched, real and complex, halves computing in f32
  and cast back (the exceeds-CPU property survives: bf16 `eigh` runs here and
  raises on jax-CPU). The plugin's `LowerCustomCall`/`LowerHostLinalg` bind a
  `HostFn` into a `kHostCall` entry, `stablehlo.cholesky` and
  `stablehlo.triangular_solve` among them, and `BlockIsPure` grew the
  custom-call arm P5 recorded as "nothing to call" — a block holding a LAPACK
  target is impure, so no trace can contain one. ApproxTopK rode along as a
  DEVICE op (`kApproxTopK`, exact top-k satisfies any recall target).
  Census slice (20 files, sequential, before/after on this tree):
  **1,007 -> 152 failures, -855, zero regressions**, and every one of the 152
  is a loud decline (92 complex scatter, 28 f64, 10 lexicographic sort, ...)
  or a shared-whitelist assertion — no numeric mismatch anywhere.
  `linalg_test` 349 -> 54, `eigh_test`/`svd_test`/`ann_test`/`random_lax_test`
  to zero. execute_test 284 -> **357 checks** (70 linalg rows, 62 of them
  BIT-identical to jax-CPU: both stacks call LAPACK). Gate 106/106, wheel
  test green from a fresh venv (which is what proves `-framework Accelerate`
  is really in the shipped dylib), dylib +0.070 %.
* **P10 landed** (notes/cpp-p10-scatter-sort.md): the scatter/sort tail, plus a
  live correctness defect the plugin had re-exposed. (a) `mx::compile` bakes a
  RANK-0 constant into generated Metal source as a `%.7g` literal, which costs
  an f32 its last ULP (CLAUDE.md item 20); Stage 1's rule — round-trip test at
  decode, one-element buffer for what fails it — is ported, with the reshape
  moved into the ENTRY, because `eval` detaches a reshape into a leaf and a
  rank-0 leaf is bakeable again. It cleared both `tests/test_elementwise.py`
  regression tests and P8's last numeric census row (`testCauchyIsf1`). P5's
  "first compiled executable is unfused" finding is NOT this and did not move
  (two probes fail to reproduce it on this tree; the `execute_test`
  eager-vs-compiled 4.8e-07 is unchanged by the fix, and that program's two
  paths are bit-identical standalone). **New finding, not acted on**: MLX's
  `is_scalar` is a SIZE-1 test, so every SPLAT constant is baked the same way,
  on both engines — the fix is the same shape but costs a kernel argument per
  lossy constant, which is Metal's 31-buffer limit, so it waits for Oleg.
  (b) Complex scatter by parts (set/add/subtract componentwise; multiply as
  gather-multiply-set, gated on the op's own `unique_indices` where the Python
  handler assumes it; max/min and a broken promise decline). (c) The two
  lexicographic comparators: `kLexSort` (successive stable argsorts through a
  permutation) and the complex key packed as (re, im) totalOrder halves, both
  recognized structurally with a vocabulary guard the Python does not have —
  a tree holding a GT declines rather than sorting the wrong way silently.
  Census slice (18 files): **608 -> 109 failures, −499, zero regressions**,
  every remainder a loud decline or a known non-P10 row. execute_test 357 ->
  **384 checks**; gate 106/106; decline_census 32 -> **34 of 35**; dylib
  +0.011 %.
* **P11 landed** (notes/cpp-p11-dtypes.md): the emulated grids,
  `reduce_precision` and the scatter tail. (a) `src/metaljax/dtypes.py`'s
  thirteen EMULATED element types (i4/ui4 and the f8/f6/f4 grids), whose values
  live in a WIDER storage dtype — the invariant M5c declined the family over.
  The per-site regrid it feared is one field, `Entry::regrid`, spent by
  `Program::step` in ONE place after the family handler has written its
  results, so no handler rounds and none can forget to; the lowering's
  `RegridOf` is the Python engine's three sites (`_regrid`, `_maybe_wrap4`,
  `_convert`) asked once, of the op's name and its result type. The host
  transfer is a per-element CONVERSION (one wire byte per element, the type's
  own encoding) through `llvm::APFloat`, with `f8E8M0FNU` and the OCP FP4/FP6
  NaN spelled out where APFloat and ml_dtypes disagree; the gate is every
  canonical bit pattern of every format, device_put and read back. **The bug
  worth remembering: a convert must NOT cast to the storage dtype before
  re-gridding** — `f32 -> f16 -> f8E4M3FN` is not `f32 -> f8E4M3FN`, and for
  the integer grids it puts a saturating float->int cast in front of a 4-bit
  wrap (`float32(1e5).astype(int4)` came back -1 where the Python engine says
  0). (b) `reduce_precision`'s four arms, resolved at lowering. (c) The
  scatter tail: the apply body in BOTH of `ops/gather.py`'s executions — the
  one-shot arm under `unique_indices` and the sequential arm without it, which
  is the one that runs, since jax emits `unique_indices = false` for every
  `.at[].apply()` (so the honest gate is the Python handler's 1024-update cap,
  not a decline on the missing promise); `select_and_scatter`, with a
  tolerance rather than a byte pin, since its scatter-add over overlapping
  windows is order-nondeterministic; and the rank-0 scatter, which degenerates
  to its combiner. Census slice (14 files): **328 -> 90 failures, −238, zero
  regressions**, every remainder a loud decline, a P12/P13 family or a
  whitelist assertion — no numeric mismatch anywhere. `tests/`-on-native
  90 -> **84** (`test_subbyte_float` 6 -> 0, the only file that moved).
  execute_test 384 -> **482 checks**; gate 106/106; decline_census 34 of 35;
  dylib +0.032 %. The decline sentinel moved off `reduce_precision` to
  **`stablehlo.rng`** (`jax.lax.rng_uniform`), which neither engine implements
  and no phase is scheduled to.
* **P12-P14 landed** (notes/cpp-p12-14-parity.md): the parity tail, all six
  families of the census's remainder, and THE measurement.
  **Collectives** are `ops/collectives.py` with two guards the Python handler
  does not have (replica-group size, and a result shape that is not the
  operand's — aliasing is how the identity arms lower, so a shape change must
  decline rather than hand back the wrong array). **Tokens** are the empty bool
  array in three places (`CheckValue`, `Dims`, and main's boundary specs — the
  buffer jax's `RuntimeTokenSet` actually passes), plus `create_token`/
  `after_all` on the runtime's existing `kToken`. **Callbacks** are a new
  `runtime/host_callback.{h,cc}` and the dylib's SECOND exported symbol: the
  registry of Python callables moved out of `metaljax.ops.callbacks` into
  `src/jax_plugins/metal/__init__.py` (the native branch), which installs a
  ctypes callback the tape calls through a C ABI — so the GIL enters this
  plugin inside a user callback and nowhere else, and the runtime still names
  no Python symbol. **Donation** is jax's `_platforms_with_donation` plus the
  plugin collecting the promise (`tf.aliasing_output` / `jax.buffer_donor` read
  at lowering; the buffers deleted after a successful run, minus
  `non_donatable_input_indices`). **The PJRT surface**: `unsafe_buffer_pointer`
  became STABLE (a settled view is now kept — a broadcast gathered afresh per
  call handed out a new address, which jax asserts on), `GetDefaultLayout`,
  `CopyToMemorySpace` as a real copy, compile-option validation, and a cost
  analysis of the tape's own facts. **Shape-poly** needed nothing: its four
  remaining rows are Stage 1's four.
  Two live defects found on the way: the submission lock had to become
  RECURSIVE (a callback that touches a metal array re-enters the plugin on the
  same thread — measured: the pre-fix dylib hangs), and a callback operand must
  be settled and checked for `data_size` before its pointer crosses (the
  short-buffer overread of CLAUDE.md item 20, in a new place).
  Family slice (17 files): **200 -> 26 failures**, of which 24 are the shared
  whitelist. execute_test 482 -> **493 checks**; decline_census 34 -> **35 of
  35**; gate 106/106; `tests/`-on-native 84 -> **71** (`test_pjrt_surface`
  10 -> 0, `test_donation` 3 -> 0; the 70 that stay are the recognizer-emit
  families); dylib +0.027 %.
* **THE measurement** (notes/data/p12-14-*): the whole pinned suite through the
  native plugin, sequential, exactly as P8 ran it — **28,057 passed / 142
  failed = 99.50 %**, against Stage 1's 28,062 / 137 = 99.51 % on this tree and
  the 0.11.3 release artifact's 99.53 %. Shared with Stage 1: 130 (the
  whitelist). **NATIVE-ONLY: 12** (from P8's 1,918), and every one is an
  intentional decline by name — 9 `element type f64` (the f64 policy;
  `x64_context` 6, `pickle` 1, `dtypes` 1, `lax_numpy` 1), 1 complex scatter
  multiply without unique indices (P10), 1 complex 0-spatial convolution (P7),
  1 non-default parameter layout. No numeric row, no unclassified row.
  STAGE1-ONLY: 7 — the reference-cycle pair, `testScatter1`, and the four
  `debug_print`-in-a-while-cond rows that were P8's live Stage 1 bug.
  Errors (35) are identical to Stage 1's, file for file.

## The runtime fork (2026-08-11, per Oleg)

Stage 1 is FROZEN ("essentially no longer modified; don't test it too
much"). The shared-runtime arrangement (P1's new_local_repository over
native/) therefore forked at 6c2bb5e: plugin-native/runtime/ holds the
16 core TUs + program.h/msl.h as //runtime, and every pure-native
change from P9 on lands there ONLY. native/ stays exactly as P8.5
left it -- still building the Stage 1 nanobind extension via
build.sh, sealed by its final battery (pytest 1258, texmo_check 106,
the pipelining bug fixed on both sides of the fork). Backporting to
native/ is an explicit decision that re-opens the Stage 1 battery,
not a reflex. Standing battery from P9: execute_test + plugin
texmo_gate + affected jax-suite files (+ tests/-on-native leg once
added); Stage-1 legs retired.

Primary objective (Oleg): jax-suite parity with Stage 1's 99.53%.
Census ladder: P9 Accelerate+linalg registrations (867), P10 complex
scatter + lexicographic sort (542), P11 dtypes + reduce_precision
(192), P12 collectives+tokens (109), P13 callbacks/PJRT/donation
(62), P14 shape-poly (17). Native at 93.15% after P8.5;
**P9 removed 855 of the 867** (its slice: 1,007 -> 152), **P10 499 of
the 542** (608 -> 109), **P11 238** (its 14-file slice: 328 -> 90) --
more than the 192 the ladder budgeted for it, because the scatter tail
and the `<unknown>`-element-type wall shared files with rows the
per-family counts did not separate.

**MET, and the ladder is finished** (P12-P14, 2026-08-11): the whole
pinned suite natively is **99.50%** (28,057/142) against Stage 1's
99.51% on this tree and 99.53% at release, with **12** native-only
failures left of P8's 1,918 -- nine of them the f64 policy and the
other three declines this migration chose. Nothing on the census is
unported; what is left is the f64 decision (Oleg: aspirational, last
stage) and the recognizer-emit families in `tests/`, which are the
performance phases'.

## P15: Oleg's review verdicts on the 142 (2026-08-11)

`notes/parity-whitelist-report.md` reviewed all 142 pinned-suite failures and
flagged 22 for scrutiny; this milestone is Oleg's rulings implemented. **142 ->
129 (99.54 %)**, past Stage 1's 99.51 % on this tree and the 0.11.0 release
artifact's 99.53 %. Thirteen rows flipped, and the two that did not are the ones
worth reading about.

**Class Q (sparse `spdot_general`, the only known wrong answers): NOT this
tape.** Bisecting the file's 445 ids gave a two-test repro
(`test_bcoo_concatenate5` then `test_bcoo_spdot_general{0,6}`, 2.5 s, and either
alone passes; the data is identical either way -- jtu seeds its rng from the
test NAME). A new `METALJAX_VERIFY_COMPILE=1` runs every executable a second
time op by op and compares: exactly ONE of ~390 diverges, a one-entry
`jit_scatter-add` tape whose compiled answer drops 18 of its 20 updates. The
attribution knob is new too (`METALJAX_MLX_COMPILE_MODE=no_fuse|no_simplify|
disabled`, since MLX's `set_compile_mode` has no environment variable): the rows
pass under `no_fuse`, under `MLX_DISABLE_COMPILE=1` and under
`METALJAX_COMPILE=0`, and fail at every command-buffer budget -- so it is MLX's
kernel FUSION, **MLX bug #8**. Two hypotheses were tested and cleared (the
strided index view; the buffer pool). Left failing on purpose: the only sound
workarounds are global.

*Found on the way and fixed*: the scatter's operand argument was a broadcast
VIEW (`size=110, data_size=1, strides=[0,0]`) that an earlier executable had
handed jax as a PJRT buffer -- 4 bytes presenting themselves as 440. `RunOnce`
now materializes any output whose `data_size != size` or which is not
row-contiguous, after the eval that makes those flags readable. It must be
`mx::contiguous`: the select `fresh_copy` builds keeps the broadcast's strides
(measured, `data_size` stays 1), which is fine for its de-aliasing job in
`Program::run` and useless here.

**What landed, by class.**

* **C, async collectives (5 -> 3).** `async_start` inlines its region where the
  start sits and the `!stablehlo.future` never gets a slot -- `futures_` records
  which slots it stands for and `async_done` aliases them back. On one device
  the collective inside is already an alias or a constant, so the pair is the
  synchronous form plus two aliases, in the block's order.
* **I, both gates lifted (2 -> 0).** A complex scatter MULTIPLY without
  `unique_indices` (which is every plain `.at[i].multiply(u)`, literal indices
  included) takes the SEQUENTIAL apply arm over the op's own body instead of
  refusing -- exact whether or not indices repeat, capped like every other use
  of that arm. And the complex 0-spatial convolution is four real matmuls; the
  matmul arm now carries the same `mode` the spatial one does. **Frozen Stage 1
  computes that convolution wrong and silently** (its f32 cast drops the
  imaginary part; the jax test never saw it because `_CompileAndCheck` compares
  metal against metal).
* **F, memory spaces (4 -> 0).** A second `MetalMemorySpace` of kind
  `pinned_host` beside `device`, and a buffer points at whichever was asked for
  through the pointer it already carries -- no per-tensor state. `mhlo.memory_kind`
  on main's arguments and results is read at lowering into
  `ValueSpec::host_memory`; any other kind declines by name. Unified memory
  makes the physics free, so this is honest metadata rather than a placement.
* **L, double donation (1 -> 0).** XLA's own `TestBufferDonationClashes` in
  `RunOnce`, so the three messages match cpu/cuda/tpu exactly.
* **M, token representation (1 -> 0).** A token boundary value is
  `xla::TOKEN` with `ShapeUtil::MakeTokenShape()`; the device array stays the
  empty bool one jax's `RuntimeTokenSet` passes. jaxlib's refusal
  (`py_array.cc:1832`) keys off the IFRT dtype, so nothing else could have
  satisfied it.
* **E, PJRT surface (8 -> 5).** The topology decline says "topology not
  implemented", which both `aot_test` rows' own skip-detector reads (2 -> SKIP);
  `lax.dce_sink` lowers to nothing again, undoing a Stage 1 -> native
  regression; and `GetHloModules` answers -- the compile keeps its StableHLO as
  bytecode and converts it to HLO on demand, cached, degrading to UNIMPLEMENTED
  (the code jax's `as_text` turns into `None`). That is the surface the planned
  XLA optimization layer will publish through, and it is what unblocked two
  async-collective rows. It reports the program **as given**: no HLO-level
  optimization happens here, which is why `test_inline_optimized_hlo` and the
  three remaining async rows still fail -- all four assert that XLA's pipeline
  ran (an early-inlined call, a rewritten collective).
* **B and A2, investigated not implemented.** Every one of the 49 export rows is
  upstream: `exported.platforms` and `disabled_safety_checks` are per-export
  arguments with no registry behind them, and aliasing `metal` to `gpu` would
  lie to every other test in the suite. The 9 f64 pass-through rows have a
  written design in the report (policy pre-pass -> f64 storage widening ->
  constant `APFloat` arm -> a C128 wire type); the buffer half already works.
* **J, both mechanisms nailed down.** `compilation_cache_test` skips in `setUp`
  after the base class has already replaced the cache object, and unittest skips
  `tearDown` on a `setUp` raise -- so 33 skips leave it changed and a later
  test's guard reports it. `logging_test` asserts on the word "INFO", which
  never appears: XLA's absl logger writes `I0811 …` and jax's Python loggers
  emit nothing at INFO level here, on CPU too.

**Battery** (the standing one, all green): `execute_test` 493 -> **502 checks**
(complex 0-spatial conv incl. groups, four complex scatter-multiply shapes, the
async start/done module, and four new contracts -- double donation, outputs own
their bytes, the host memory space, the optimized program); `texmo_gate`
**106 ok / 0 decline / 0 FAIL**; `smoke_test`; `decline_census` 35 of 35;
`bazel test //...`; the native wheel built and run from a fresh 3.13 venv
(`wheel_poc_test`). Affected jax files re-run natively: `async_collectives_test`
5->3, `aot_test` 2->0, `api_test` 7->5, `lax_test` 2->0,
`lax_numpy_indexing_test` 1->0, `memories_test` 2->0, `export_test` 7->5,
`sparse_bcoo_bcsr_test` 2->2, with `lax_numpy_test` / `nn_test` / `random_test`
unchanged as controls.

Open, for scrutiny: `api_test::test_concurrent_device_get_and_put` failed once
in two whole-file runs and passes standalone 3/3 -- the same intermittent
multi-thread row P8.5 left open (`There is no Stream(gpu, N) in current
thread`), not a new one, but it now has a second sighting.

## P16: the performance baseline, native vs Stage 1 (2026-08-12)

The first end-to-end perf measurement of the phase-2 plugin, both texmo suites
and every non-embargoed model row, on tree 845ab89:
**[benchmarks/perf-2026-08-native-baseline.md](../benchmarks/perf-2026-08-native-baseline.md)**
(raw data under `~/.cache/metaljax-bench/logs/native-baseline/`; the PJRT-route
runner is `scripts/bench_texmo_pjrt.py`, since `texmo_topconfs.py` drives
`metaljax.engine` and can only ever measure Stage 1).

Headline: **the parity work is done and the emits are the whole remaining
gap.** Where no recognizer emit and no msl plan fires, the two stacks are
indistinguishable (top_confs at parity 0.98-1.00x, gemma4-E2B decode exact,
maxtext decode/train within 2%) and on large matmul-bound texmo configs the
native path is *ahead* (`big09-b8l256` 0.68x, transformer d512 0.82x) - P5's
`METALJAX_MSL=0` finding, confirmed across 269 configurations. Everything else
is a missing emit, ordered by measured gap:

1. **msl_scan** - top_confs geomean **36.5x** slower (median 46x, worst 175x),
   texmo suite `db` class 14.6x, whole suite 4.24x. It costs every one of the
   54 top_confs where metaljax beats jax-CPU (native: 0 of 163).
2. **qmm** - gpt-oss-20b **434x** (9.5 s/token vs 21.9 ms; memory is fine at
   21 GB, so it is pure compute), E2B-int4 5.6x.
3. **MoE expert gather** - gemma4-26B-A4B 6.9x (300.5 vs 43.7 ms/tok), memory
   identical; reproduces the pre-gather Stage 1 number.
4. **sdpa** - SD 3.5 @1024 3.1x, SigLIP b32 1.8x, 31B decode 1.24x, LoRA 1.63x.

Two blockers found on the way, neither of them the tape:

* **Static-protobuf/LLVM symbol collision.** The dylib cannot be dlopened into
  a process holding TensorFlow or array_record - weak-def coalescing binds the
  plugin's `AddDescriptors` to the other image's protobuf (SIGSEGV), or, with
  the plugin first, TF's LLVM aborts on duplicate CommandLine options. It
  blocked *every* model row (keras directly, gemma-lib through `kauldron`,
  maxtext through `array_record_module.so`). Fixed by linking with an
  exported-symbols list of exactly `_GetPjrtApi` and
  `_metaljax_native_set_callback_trampoline` (166 -> 46 MB, coexists with TF,
  perf-neutral on five suite configs, execute_test clean bar the known
  thread-stream flake). The change is not in the tree - it belongs in
  `plugin-native/metal/BUILD` as a deliberate commit.
* **KERNEL PANIC #8** on the row-9 native attempt (65 GB streaming load,
  watchdog wedge with every memory metric healthy - panic #7's signature).
  The native plugin has no equivalent of Stage 1's load-phase clear cadence.
  Row 9 native is embargoed until it does; the campaign was halted there.

## Performance era goals (Oleg, 2026-08-12)

1. Completion parity: everything that completed on 0.11.3 completes
   natively (row 9 via cadence fix + ladder + supervised retry;
   exported-symbols relink lands properly with its validation re-run).
2. The big gaps, measured order: qmm 434x/5.6x -> msl_scan 36.5x
   geomean -> MoE 6.9x -> sdpa 3.1x. Runtime already executes the
   fused opcodes; only recognizers/lowerings are missing.
Then: NEW RELEASE at ~parity with the old implementation; then the
remaining STATUS.md rows (all 20 must eventually complete); then
perf parity with other frameworks. Also open: Stage-1-vs-anchor
regressions (qwix-int8 1.85x, maxtext train 2.17x) to investigate
despite the freeze; row 11 token divergence to classify.

## The ingest cadence (2026-08-12, P16's row-9 blocker)

`notes/ingest-cadence-2026-08-12.md`. The plugin now has the reclamation
point a model load needs — `runtime.cc::ingest_account`, charged by every
transfer (`BufferFromHostBuffer`'s `wrap`, `CopyToMemorySpace`), clearing
every `METALJAX_INGEST_CLEAR_MB` (default 8192 = the bench harness's
`BENCH_STREAM_CLEAR_GB=8`, which is where Stage 1's `"clears": 16` came
from; the Stage 1 *engine* has no transfer-denominated cadence at all).
Counters ride in `g_stats` (`ingest_bytes` / `ingest_clears`, in the
per-execute `METALJAX_DEBUG` line) and each clear narrates its cache before
and after, so a flight log can prove it engaged.

Two findings the ladder produced, both of which cut against the premise:

* **The transfer path never touches MLX's buffer cache.** The staging block
  becomes the array's storage through the alien-buffer path, so it is freed
  to the C allocator; `get_cache_memory()` stays at 0 through 4 KB, 256 KB
  and 256 MB transfers, 9.8 GB of churn, and a real 15.26 GB checkpoint.
  The cadence reclaims what the work AROUND the transfers leaves (2.58 GB
  on a cast-per-tensor load; nothing on a plain one) and costs nothing.
* **Panic #8 is not the missing cadence.** At the moment it wedged, the
  native run's memory was no worse than the Stage 1 run that had passed the
  same point twice 8 minutes earlier (54.0/54.6/64.5 GB vs 51.5/62.7 at
  t≈150 s), only ~20 % faster to fill. It belongs with #4/#7's wedge class,
  and the 65 GB retry should vary the RATE, not the cache.

Ladder: `plugin-native/ingest_test.py` (synthetic 8/8, plus `--checkpoint`
for a real one) and, through the temporarily-relinked dylib, the panic's own
shape at 10 GB — `gemma4-e2b-bf16` keras streaming, harness clears off,
3 plugin clears, peak footprint 12.00 GB against a 10.2 GB model, decode
26.8 ms/tok. P16's exported-symbols relink is STILL not in the tree, and it
is what every keras/gemma-lib row is blocked on.

## P17: the recognizer emits, natively (2026-08-12,
## notes/cpp-p17-emits.md)

All three of Stage 1's rewrites are now recognized and emitted by the phase-2
lowering -- `metaljax.qmm` (affine int4/int8, per-channel, MXFP4, batched, with
the regrouping pack), `metaljax.moe.*` (the expert gather, `gather_mm` and
`gather_qmm`) and `metaljax.sdpa` -- so items 2, 3 and 4 of the P16 frontier
are closed and `msl_scan` (item 1) is the whole of what is left there.

The architecture is a TWO-PHASE compile, forced by the fact that a quantized
weight is an ARGUMENT and its pack must be a tape INPUT (`mx::compile` bakes a
captured constant by value): `CompileAndLoad` lowers the plain tape as before,
and the FIRST `Execute` re-parses the kept StableHLO, runs the analyses, builds
the packs and the router checks on the real buffers and lowers a second tape
with the emits.  Any decline at any point leaves the plain tape in place, so
the fallback is structural rather than a policy.  `Lowering::LowerCone` -- the
cone of a set of values over @main's arguments, as a Program of its own -- is
what evaluates an operand subtree on concrete buffers, and its `bound` argument
pins values in the middle of a graph, which is how the MoE router check runs on
synthetic logits (without that, a dispatch inside a decode loop can never be
verified: its logits are a loop carry).

Deliberately not ported: qmm's row-blocked `_Source` evaluation and its
cross-executable build cache (memory/latency optimizations over an evaluation
the tape already stages with last-use pruning), and an sdpa or moe root that
lives wholly inside a callee.

Battery: `execute_test` 502 -> **520 checks** (18 new differential rows -- the
FUSED answer against the literal chain jax-CPU runs); `texmo_gate` 106/106;
`smoke_test`; `decline_census` 35 of 35; `bazel test //...`; `tests/` on native
1187/71, the same 71 rows as before.  Those 71 do not move and cannot: 70 of
them assert Stage 1 PYTHON counters (`qmm.stats()`, `moe.stats()`) that a
plugin with no interpreter can never tick, and `src/`/`tests/` are frozen --
with the counters neutralized by an instrument the same files go 70 -> 8, and
each of the 8 asserts one of the internals above.

Measured (`notes/data/p17-emits-*`, machine lock held; the same native plugin
under `METALJAX_RECOGNIZE=0` is the control): qmm mxfp4 decode at gpt-oss's
gate_up shape **7.63x**, qmm int4 decode **6.62x**, moe decode **4.43x**, sdpa
1024x1024 **2.61x**, and against Stage 1 the native path is at or ahead on four
of seven rows.  The MODEL rows were not run: they all import TensorFlow, and
the dylib cannot be dlopened into such a process without the exported-symbols
relink P16 found and left out of the tree (a separate deliberate commit).  One
row is worth carrying forward: at `Tq=1` the FUSED attention is slower than the
literal chain (0.85x), on both stacks.

## P18: the relink lands, and the emits meet the model rows (2026-08-12)

Two things, in that order: P16's exported-symbols relink became a real,
default part of the build with its validation re-run from scratch, and the
model rows were re-measured on the P17 emits through the relinked plugin.
Evidence: `notes/data/p18-relink-battery-2026-08-12.txt` (relink) and
`notes/data/p18-relink-models-2026-08-12.jsonl` (rows); Table 3 of
`benchmarks/perf-2026-08-native-baseline.md` and STATUS.md footnote 27 carry
the numbers.

**The relink.** `plugin-native/metal/exported_symbols.exp` plus a
`linkopts`/`additional_linker_inputs` pair on
`//metal:libmetal_pjrt_native.dylib`; nothing else was needed (ld64 strips what
the two exports cannot reach on its own). Dylib 166 -> 46 MB, wheel 42.2 ->
11.8 MB, `nm -gU` reports exactly `_GetPjrtApi` and
`_metaljax_native_set_callback_trampoline`. `plugin-native/coexist_test.py` is
the standing contract: TensorFlow and array_record, each in both load orders,
each in a fresh subprocess because the failure is a SIGSEGV. On the kept
pristine dylib all four cases are `rc=-11`; on the relinked one all four pass
and run a jit on the GPU. Perf neutrality was re-measured A/B/A on P16's five
suite configs (relinked/pristine 0.976-1.001, inside the relinked passes' own
2.4 % spread). The battery came out *better* than P16's transcript claimed:
execute_test **520 of 520** (the intermittent 8-thread stream row passed),
texmo_gate 106/106, smoke, decline_census 35/35, ingest_test 8/8,
`bazel test //...`, wheel from a fresh 3.13 venv.

**The rows.** Items 2, 3 and 4 of the P16 frontier are closed wherever a row
could be re-run: MoE 6.88x -> 0.99x, sdpa 3.14x -> 1.13x (SD 3.5 @1024) and
1.78x -> 0.96x (SigLIP b32), with SD 3.5 @512 at **0.81x** and the dense/int8
rows at 0.95-0.99x -- native is now at or past Stage 1 on six of the eight rows
measured. Row 3's greedy tokens are 64/64 identical to Stage 1 (it is the P16
*dense* run that diverges, at token 52 -- footnote 16's ladder class), and so
are row 13's.

Two things did NOT close, and both are one mechanism each:

* **Row 7 (gpt-oss) is now a memory block.** The emits fire (94 qmm recognized,
  47 gathered expert dispatches, 188 packs) but the two optimizations P17
  deliberately skipped -- qmm's row-blocked `_Source` evaluation and its
  cross-executable pack cache -- put a full pack set per compiled shape in
  memory: guard kill at 46 GB under the row's 45 GB budget, 62 GB under 60,
  against Stage 1's 25 GB. Not escalated further; 62 GB is panic #7/#8
  territory.
* **The fused lowering's compile decisions read the UNFUSED IR.** Row 13 fuses
  every one of its 777 quantized dots and its PREFILL is already ahead of
  Stage 1 (218.3 vs 241.0 ms), yet the fused program reports `compiles=0
  compiled_calls=0 serial_loops=1`: `BlockCost` walks the StableHLO block and
  charges the dequant chain the emit absorbs, so `body_compile_max` solves to
  zero and the decode loop runs op by op. The byte budget is not it
  (`METALJAX_COMPILE_BYTES_MB=1e8` alone: 274.6 ms/tok) and the packs are not it
  (`METALJAX_RECOGNIZE=0` also reports `compiles=0`); `METALJAX_TRACE_BUDGET=1e7`
  gives **85.5 ms/tok = 1.06x of Stage 1**. Cost and byte accounting must follow
  the rewrite plan, as Stage 1's does. Worth 2.9x on that row and a candidate
  for any emit row short of parity.

Scrutiny carried forward: row 5's greedy tokens now diverge from Stage 1 at
token 61 of 64 where they agreed before the sdpa emit (tie-flip class, but new);
the two SD 3.5 stacks produce different images at the same prompt (pixel_std
61.1 vs 77.5 at 512); Stage 1's own SD 3.5 @512 has drifted 1389 -> 1520.7
against its 0.11.3 anchor, a third Stage-1-vs-anchor regression beside rows 14
and 19.

## P19: the two pack optimizations P17 deferred (2026-08-13,
## notes/cpp-p19-packing.md)

Item 7 of the P18 frontier -- "pack building has no memory discipline" -- is
the block on row 7 and the reason row 13 peaks at 48 GB for a 3 GB model.
Both halves are now ported from `src/metaljax/qmm.py`:

* **Row-blocked evaluation** (`_Source`). The operand subtrees are evaluated
  one slice of the weight's leading axis at a time and packed as they go, so
  the reconstruction -- several times the size of the weight -- never exists
  at once. The shape of the port is the thing worth knowing: there is no
  second evaluator. `RowSource` does only the ANALYSIS (qmm.py's `_op` rules,
  producing a SET of values that carry the row axis) and
  `Lowering::LowerConeBlocked` builds the ordinary cone with those values
  declared `c` rows tall -- one override inside `Dims()`, the single place a
  declared shape enters a lowering, plus `stablehlo.slice`, whose extent is an
  attribute. A block is therefore computed by exactly the code a whole
  evaluation would have used.
* **The cross-executable build cache**. A canonical serialization of the
  reconstruction (qmm.py's `_Fingerprint`, transliterated) plus the identity of
  the buffers it reads, so keras-hub's per-sequence-length generate programs
  build each weight once between them. Bounded by
  `METALJAX_QMM_BUILD_CACHE` (512, LRU, dead-leaf sweep); `0` restores P17.
  The one difference from the Python: an `mx::array` cannot be rebuilt from a
  weak handle, so the PACK is held strongly and the LEAVES weakly (through
  `data_shared_ptr`), and leaf identity is re-proven three ways on every hit
  because `mx::array::id()` is a recyclable address.

**A wrongly narrowed value is silent wrongness** -- the exactness checks pass
happily on the wrong rows -- so there are two locks in two files: `RowSource`
decides what may be narrowed, and `Lowering::LowerOp` asserts the consequence
at emission (an op that reads a block must hand back a block; only
`RowLocalOps()` may be narrowed). The second lock earned itself immediately:
`func.call` was missing from the set, so every MXFP4 weight (jax lowers
`jnp.take` as a call) fell back to a whole evaluation, and the guard named it
instead of failing silently.

Deliberately narrower than Stage 1, each a fall-back to the P17 behaviour and
each reported under `METALJAX_DEBUG=1`: a value read BOTH whole and blocked
declines (a tape has one slot per value where Stage 1 keys on (value, demand)),
as does a callee invoked twice in one subtree; and a callee's body is swept
after the walk, because the lowering splices a callee whole, dead ops included.

**Row 7 is unblocked** (`notes/data/p19-packing-models-2026-08-13.jsonl`).
gpt-oss-20b completes at its historical 45 GB budget -- 35 GB peak, 128 tokens,
**25.3 ms/tok = 1.16x** of Stage 1's 21.9 (re-measured same-day, reproducing its
anchor exactly), four samples inside 25.3-25.5. P18 was guard-killed at 46 GB
under 45 and 62 GB under 60.

**The ablation contradicts the P18 diagnosis in a useful way.** One knob at a
time, same budget: cache off / blocking on is killed at 46 GB (P18's number to
the gigabyte); blocking off / cache on completes at 36 GB; both on, 35 GB. So
the **build cache is the load-bearing half** -- three executables were each
building their own ~10 GB pack set -- and row-blocking is worth a further
gigabyte. P17's argument that "the tape already stages op by op with last-use
pruning" was substantially right about the per-weight TRANSIENT; what it did not
cover was the same pack set built three times. Only the pair clears the line.
Mechanism from the run's log: 94 built / **188 reused**, all 94 blocked (47 in
16 row blocks, 47 in 32), pack-wave peak 33.9 GB then **0.000 GB** twice.

Row 7 does not share row 13's compile bug: `METALJAX_TRACE_BUDGET=1e7` returns
the same 25.3 ms/tok with bit-identical compile decisions (16 compiles / 354
compiled calls either way), so item 6 of the P18 frontier does not touch it.

**Row 13 is timing-neutral under P19** (275.6 vs a P19-off control of 271.7 on
the same binary; P18's own byte-cap control read 274.6, so its 249.0 headline
was the low end of the row's spread). What P19 changes there is the steady state
**4.2 -> 3.2 GB** and 518 of the 777 pack builds -- 259 built, 518 reused across
three executables, 0 fingerprint declines, all 259 packing whole because a keras
`[K, N]` weight fails `_blocking`'s no-transpose precondition on both stacks.
Its 46 GB peak is now attributed rather than open: it is the keras streaming
LOAD transient (Stage 1's own is 44 GB), not the packs, whose wave peaks at
6.6 GB.

Scrutiny carried forward: row 7's greedy tokens diverge from Stage 1 at index 52
of 64. P18 never completed this row, so there is no prior native record and this
is a first observation rather than a change; same late-divergence ladder class
as rows 3, 5 and 11.

Battery: `notes/data/p19-packing-battery-2026-08-13.txt`. execute_test 520 ->
**524** checks (blocked-vs-whole bit equality over five quantized graphs incl.
one inside a decode loop; cross-executable reuse; the cache's off switch; and
the pack-wave peak, 0.76 -> 0.13 GB on one 8192x4096 MXFP4 weight), `texmo_gate`
**106 ok / 0 decline / 0 FAIL**, smoke, decline_census 35/35, ingest_test 8/8,
coexist_test, `bazel test //...`, native wheel from a fresh 3.13 venv.

## P20: the four named regressions (2026-08-13)

Oleg's list, in order, with the raw runs under
`~/.cache/metaljax-bench/logs/p20-regressions/` and the tables in
`benchmarks/perf-2026-08-native-baseline.md`. Two are plugin bugs and are fixed
here; one is a shared-runtime mechanism and is reported, not fixed; one was
never a regression.

**1. The compile gate read the unfused IR (P18 frontier item 6) — FIXED, and it
was worth more than the diagnosis said.** `BlockCost`/`BlockBytes` in
`metal_lowering.cc` now consult `ctx.plan`, which was already in scope and set
before `Run`: an absorbed op is charged nothing and not recursed into, a qmm or
moe root costs 2 units and an sdpa root 3 (the Python merges qmm and moe into
one State and charges the pair 2), and a root's bytes are its own result plus
what the emission really builds — `qmm.emit_bytes`'s one activation copy and
`moe.emit_bytes`'s pair-space plan, both transliterated (`MoeEmitBytes`,
`MoeNodeItemsize`, `MoeTrailingElems`). sdpa declares no `emit_bytes` in the
Python either, and that is right: the [B,H,T,T] scores it absorbs are never
written.

Row 13 **275.6 -> 79.7 ms/tok** with **no env override** — past P18's
`METALJAX_TRACE_BUDGET=1e7` proxy of 85.5, and past Stage 1 (0.99x of 80.6,
0.98x of the anchor). The decode body compiles: `compiles=1 compiled_calls=127`
where P18/P19 reported `0/0`. Greedy tokens 64/64 identical to the P19 run.

Row 7 **25.3 -> 22.2** (1.01x of Stage 1's same-day 21.9, 1.00x of anchor),
which P19 had explicitly cleared: its probe lifted the OP-COUNT budget only, and
what moved here is the BYTE term (`by_bytes`, `kmax`, `BytesOk`). Its greedy
tokens now diverge from the P19 native run at index 51 where P19 diverged from
Stage 1 at 52 — carried to scrutiny: changing a compile decision changes fusion
boundaries, so a late tie-flip is expected, but it is a change and belongs on
the logit-delta ladder rather than in a footnote.

**2. Row 18 (LoRA) was the output-copy rule ignoring donation — FIXED.** The
lowering copied every output that may alias an argument. A keras LoRA training
step **donates 2,255 of its 2,262 arguments** and threads the frozen parameters
straight through, so it copied **1,952 outputs, ~10 GB per step**.
`engine.py::_dealias` has always exempted donation ("aliasing is exactly what
donation licenses"), and the plugin now does: an output whose every aliased
argument is donated is exempt, and because donation is retractable per CALL the
exempted outputs travel with the arguments they alias
(`LoweredProgram::donated_output_aliases`) so `RunOnce` copies the ones a call
takes back through `non_donatable_input_indices`. `Program::run`'s duplicate-
output pass already handles two outputs that land on one array, dynamically.

**656.3 -> 397.5 / 396.2 ms/step** (1.00x of Stage 1's 398.9 measured the same
day, 0.98x of the 407 anchor), `0 output copies`, and the row's peak drops
**55 -> 37 GB** — below Stage 1's own 56. What it was NOT, each measured: not
the compile decisions (both stacks refuse this main with *identical* numbers,
`cost=27308 bytes=40779.0MB` — an incidental cross-check of the item-1 fix), not
sdpa (neither stack fuses attention here, which those equal costs prove), and
not the flush clear (raising `METALJAX_FLUSH_CLEAR_MB` to 8 GB moves 597.1 ->
593.1). The flush CADENCE is a separate ~1.13x on both stacks (native 597 ->
501, Stage 1 399 -> 351 at `METALJAX_EAGER_FLUSH_MB=8192`) and is left alone.

**3. Row 19 (maxtext train), the 2.17x shared drift — ROOT-CAUSED, reported.**
The bisect is in the record: the same harness measured 440.0 on 2026-08-03 and
964.2 on 2026-08-05 with losses identical to the last digit. On today's machine
and today's (unchanged since 0.11.3) Stage 1 dylib, 0.11.2's `src/metaljax` on
`PYTHONPATH` gives **448.2** against the current tree's **969.1** — so it is
metaljax's own code; the harness, the maxtext venv and its checkout are all
untouched since before the anchor.

The cause is `4d34bff`'s cache clear on the eager flush
(`METALJAX_FLUSH_CLEAR_MB`, default 2048). This program's `@main` is over the
trace budget (`cost=24870 > 20000`, unchanged since 0.11.2), so it runs op by
op with ~105 GB of traffic per step against a live set of a few hundred MB: **82
flushes and 7 clears per step**, each clear returning MLX's whole pool to the OS
so the next ~2 GB of allocations are cold. Stage 1 **478.8** and native
**468.0** with the clear off; 446.2 with the flush off entirely. It is shared
because `runtime/program.cc::eager_flush` is the transliteration of
`interpreter._eager_flush` and both read the same budgets.

NOT fixed, deliberately: the clear is a live memory bound (with it off, row 18
blew an 81 GB peak through a 70 GB guard, and Stage 1 with the flush off was
killed on trajectory at a projected 95 GB). The fix that keeps the bound without
the cliff is `mx::set_cache_limit(flush_clear_bytes)` — MLX reclaims only the
excess, on the next allocation, so the pool is bounded at every instant (tighter
than clearing at flush points) and reuse below the limit survives. That is a
change to the shared memory discipline and Stage 1's copy is frozen: Oleg's
call, with a memory ladder behind it.

**4. Row 14 (qwix-int8) was never regressed.** Standalone under the lock:
**32.9** and **32.7** ms/tok against a 32.5 anchor. P16's 60.1 was measured 12
minutes into a sequential campaign — the suite-context trap of CLAUDE.md item
12, reaching the model harness. Native today is 35.0 (1.06x), and 32.1 with the
flush clear lifted, so its residual is item 3's mechanism too.

**Report hygiene.** Table 3 of the baseline now carries `S1/anchor` and
`native/anchor` beside `native/S1`, and STATUS.md's native cells carry both
ratios. Row 19 read "1.01x of Stage 1" through P16 and P18 while both stacks sat
2.2x off the anchor; a same-day ratio cannot see a shared drift, and that is the
column that was missing.

**Battery** (final binary, both fixes): `execute_test` **524/524**,
`texmo_gate` **106 ok / 0 decline / 0 FAIL**, `smoke_test`, and the rows touched
re-measured on it (13: 79.7 twice; 7: 22.2; 18: 397.5/396.2). Not re-run and
worth watching, since `cost` also sizes the loop flush period: rows 3, 5, 6, 16,
17 were at parity before this change.

## P21: msl_scan, natively (2026-08-14, notes/cpp-p21-msl.md)

Item 1 of the P16 frontier, the last and largest: the generated persistent
kernels. All three modes at once -- `scalar` (affine), `vector` (in-lane
matvecs + loop fission) and `coop` (threadgroup per batch element) -- because
they are one recognizer with three emitters. `metal_msl.{h,cc}` +
`metal_msl_emit.cc` are `src/metaljax/msl_scan.py` transliterated; the LAUNCH
half is M5b's `runtime/msl.cc`, which has executed these plans since May.
`LowerMslPlan` is `tape.py::_lower_msl`, and `MslPlanFor` is the one cache the
cost walk (8 units for a planned loop), `WhileTraceable` (planned = traceable)
and `LowerWhile` all ask, exactly as `ops/control._msl_plan_for` is in Stage 1.

**The census is identical to Stage 1's, plan for plan and decline for decline**
over the whole 106-configuration suite: 146 coop / 52 vector / 12 scalar, and
136 declines whose reasons match one for one (104 `stablehlo.gather`, 18 the
coop work cap, ...). Two recognizers agreeing about every loop in 106 real
training chunks is the evidence the port is faithful.

Measured standalone, same binary, `METALJAX_MSL` flipped (ms/step of one 8-step
chunk): `db02-b4l1024` 588 -> **8.0** (73x), `db11-b64l256` 231 -> **7.0**
(33x), and **nothing** on the rows that take no kernel (`db00` 1.73 -> 1.76,
`big14` 143.2 -> 143.5, `big16` 480.8 -> 483.5, `db04` 4.03 -> 3.97) -- compile
time included. One row LOSES: `big09-b8l256` (`rnn.1024`) 202 -> 308, because
its single 1024x1024 dot is 1.05M elements, under `METALJAX_MSL_COOP_CAP`'s
2.2M, and per-threadgroup weight re-streaming loses to the compiled matmul at
that width. `METALJAX_MSL_COOP_CAP=1000000` returns it to 201.6; not applied,
because it would put the native census out of step with Stage 1's -- a policy
question rather than a port decision (CLAUDE.md item 12e already says F=1024
loses).

One runtime fix rode with it, and it is phase-2-specific: `settle_msl`'s
recovery assumed a second kernel failure could hand the program to the Python
engine. It cannot here, and the second failure is reachable (a retired kernel
runs its body, whose own loop then hands back a second unproven plan). The
ladder is bounded now -- second failure retires every plan in the program
(`disable_msl_deep`) and reruns once. Found by the new
`METALJAX_MSL_FORCE_BUILD_FAIL` arm of `execute_test`.

**RESOLVED in P22**: not by the work cap. See the P22 entry below --
`METALJAX_MSL_COOP_MAX_F`, a width cap, changes exactly these two rows.

Battery: `execute_test` 524 -> **534** checks (8 msl cases, 2 msl contracts,
and two whole-suite arms: `METALJAX_MSL=0` -- 466 of 469 bit-identical, the
three that differ being exactly the fissioned weight-gradient rows -- and
`METALJAX_MSL_FORCE_BUILD_FAIL=1`, the recovery ladder end to end);
`texmo_gate` **106 ok / 0 decline / 0 FAIL** twice on the final binary;
`smoke_test`; `decline_census` 35 of 35; `bazel test //...`. One flake seen on
an earlier binary (`big10-b8l256`, `inf` in-suite, passes standalone 3/3,
builds no msl plan at all) is P4's recorded lottery for that row's class.

## P22: the coop width cap + THE RELEASE MEASUREMENT (2026-08-15,
## notes/cpp-p22-release.md)

**Part 1 -- the phase-2 lowering's first deliberate divergence from
`msl_scan.py`.** P21 left `big09`'s `rnn.1024` open: its single 1024x1024 dot
is 1.05M elems/step, under `METALJAX_MSL_COOP_CAP`'s 2.2M, so coop takes it and
loses 1.5x to the compiled matmul; `COOP_CAP=1e6` fixed the row and the
question was what else it costs. **It costs a great deal.** The census (every
coop candidate narrates its work before the cap decision) says a 1e6 cap takes
coop away from **22 of 106** configurations to fix 2, and the stopwatch says
what that is worth: `mgru.512` **1.73x**, `gru.512` 1.08-1.39x, `lstm.512`
1.15x, `gru.512-gru.512` 1.16x. CLAUDE.md item 12e's "lstm.512 ties" was a
vector-vs-coop comparison, not coop-vs-no-kernel.

The cap that costs nothing is on the WIDTH, because the loss mechanism is
per-lane weight re-streaming and that scales with F, not with total work: at
F=1024 a square cell loses at 1.05M elements, at F=512 the same 1.05M wins.
`METALJAX_MSL_COOP_MAX_F` (default **1024**, `0` restores Stage 1's policy)
declines coop at F >= 1024 when the plan has any dot. **Collateral proven, not
argued**: re-running the census on the new binary and diffing plan for plan,
**2 of 106 configurations change (4 plans of 210)** and every other decline
reason is identical -- by construction, since every other F>=1024 cell in
reach (`gru.1024` 3.1M, `lstm.1024` 4.2M/11.5M) is already over the work cap.
Worth **1.52x** on `big09-b8l256` (38.40 -> 25.32 ms/step) and 1.03x on
`big09-b32l128`; every other row in the nine-configuration collateral set is
within noise. `top_confs` cannot be reached at all: coop work is bounded by
the cell's weight count and the largest of the 163 models has 1,888 weights.

**Part 2 -- the release measurement.** One lock hold 22:41:34-23:33:37,
strictly sequential, nothing else on the machine, native arms on a FROZEN copy
of the dylib (`frozen-release-208ca0d1`, sha256 `208ca0d1...558d61`) so P21's
halt cannot repeat. Five runs, all complete: suite-106 both stacks (488/489 s),
top_confs Stage 1 on the anchor's engine route (1104 s, **163/163 ok, 0 FAIL**),
top_confs native on the PJRT route (507 s) -- **the pairing that had never
completed** -- and a top_confs Stage 1 PJRT run (503 s) so the campaign
measures its own route factor. The analysis reproduces every published P16
aggregate from P16's artifacts before being applied to the new ones.

| | n | P22 | P16 |
|---|---:|---:|---:|
| top_confs native / Stage 1 (engine route) | 163 | **0.998** | 36.46x |
| top_confs native / Stage 1 (same PJRT route) | 163 | **1.001** | — |
| route factor today (engine/PJRT) | 163 | **1.002** | 1.009 |
| top_confs Stage 1 / anchor, native / anchor | 163 | **1.071 / 1.073 faster** | 1.046 / — |
| top_confs beating jax-CPU | 163 | **native 59**, S1 55 | native 0, S1 54 |
| suite-106 native / Stage 1 (geomean / median) | 106 | **1.011 / 1.000** | 4.24 / 3.37 |
| suite-106 rows within 1.2x / at or above 10x | 106 | **103 / 0** | 33 / 32 |

**Five of nine in-suite anomalies were the suite itself.** Every row outside
+/-10 % was re-measured standalone before being reported, and
`big14-b32l128` (1.198 -> 1.000), `big12-b8l256`, `big07-b8l256`,
`big00-b32l128` (0.807 -> 0.994) and `mid11-b64l128` are Stage-1-side
in-suite variance -- the same rows whose Stage 1 column moved +-15-20 %
against P16 while their standalone numbers sit on their P16 values. They cut
both ways; two flattered native. CLAUDE.md item 12's trap is the majority of
the outliers, not a footnote.

**The one qualifier on the parity claim**: three `db*-b256l512` rows are
genuinely slower natively (`db16` **1.77x**, `db17` 1.60x, `db11` 1.31x),
reproducible standalone, and P21's preliminary column saw the same two. Ruled
out, each measured: not the surrounding graph (`METALJAX_MSL=0` levels the
stacks -- 83.99 vs 83.82 on `db16` -- so the whole gap is on the msl path),
not the plans (identical narration: mode, lanes, trip, stacked, and the same
kernel name hence the same MLX library), not the flush cadence, not the
compile budget. **Identical kernels, dispatched differently** -- the per-call
launch work in `runtime/msl.cc` (weight normalization, input pooling) is where
to look. Narrow pocket: the largest `db` shapes only; `db11-b64l256` is at
exact parity.

Battery: `execute_test` 534 -> **535**, the delta verified as exactly one
check row against the pre-change file on the same binary (a new contract pins
the width cap: an
F=1024 cell must build no coop plan and narrate the width decline by default,
must build one under `COOP_MAX_F=0`, and the two answers must agree),
`texmo_gate` **106 ok / 0 decline / 0 FAIL / 0 error**, census diff 2
configurations. Model rows not re-run: nothing in P22 touches their paths (no
model row builds an msl plan).

## P23: the qualifier, closed — a planned loop the byte gate could not see
## (2026-08-16, notes/cpp-p23-dispatch.md)

P22's parity claim carried one qualifier: three `db*-b256l512` rows at
**1.77 / 1.60 / 1.31x**, with identical plans, identical kernels and
`METALJAX_MSL=0` levelling the stacks — so "the launch, not the plan", and
`runtime/msl.cc`'s per-call work was the named suspect. **It was neither.**
Nothing in `runtime/msl.cc` changed.

`ops/control._block_bytes` charges a `stablehlo.while` that became one
generated msl kernel **its outputs only** ("its per-timestep state lives in
registers, not in buffers"); `BlockCost` has the same case (`cost += 8`) and
P21 ported that one, while `BlockBytes` was ported without it and charged a
planned loop `trip x body` — the traffic of a loop that does not run. On
`db16-b256l512` that made the per-step estimate **163 GB instead of 2.05 GB**,
over `METALJAX_COMPILE_BYTES_MB` (64 GB), which is read by three decisions at
once: the loop body's compile (`by_bytes` -> 0), the chunked replay
(`BytesChunks` -> K=1) and the whole-main compile. So every training step ran
op by op, with the byte-denominated eager flush firing on the same inflated
numbers — 128 blocking `mx::eval` per chunk. The fix is one line in
`BlockBytes`: `if (MslPlanFor(ctx, &o) != nullptr) continue;`, after which the
native estimate is Stage 1's number **exactly** (134,234.4 MB, digit for
digit), flushes are 0 and the chunk replays 4 compiled calls of K=16.

**Found by diffing narration, not by profiling.** P22's own probe logs had it:
`cost` agreed to the unit (39113 — the walk WITH the msl case) while `bytes`
was 80x apart (the walk without), and the native line said in the same breath
`compiles=0 compiled_calls=0 flushes=128`. Lesson for a transliteration
project: diff what the two engines SAY about a program before profiling what
they do.

**The cost, split** (`db16-b256l512`, ms/step, all on P22's *released* binary
bar the last two): shipped **7.923** -> `EAGER_FLUSH_MB=0` **7.185** (the
flushes, 0.74 ms) -> `COMPILE_BYTES_MB=1048576` **4.469** (the gate raised past
the inflated estimate: the fix reproduced by knob, on unmodified code) -> P23
binary **4.464**, Stage 1 **4.471**. The remaining 2.71 ms was op-by-op
dispatch of ~611 ops per step, ~4.4 us each.

| | P22 | **P23 (RC)** |
|---|---:|---:|
| `db16` / `db17` / `db11-b256l512`, standalone | 1.774 / 1.594 / 1.307 | **0.998 / 0.999 / 0.985** |
| suite-106 geomean (median) | 1.011 (1.000) | **1.0050 (1.0012)** |
| suite-106 `db` class | 1.030 | **1.0013** |
| suite-106 rows within 1.2x | 103 | **106 of 106** |
| top_confs, same PJRT route | 1.001 | **1.0016** (native arm vs P22's: 0.9999) |
| `execute_test` | 535 | **536** |

The census is **EMPTY**: 568 narration lines identical in content and order,
142 coop / 52 vector / 12 scalar and every decline reason — as it must be,
since the change is to a compile decision and nothing else. Gate 106/106 twice
on the fixed code (one flake attributed: `mid03-b16l256` is a sensitivity
lottery row, 3/3 standalone on BOTH binaries). New contract `msl loop charged
as one kernel` pins the mechanism no correctness test can see — the same cell
planned vs `METALJAX_MSL=0`, 3.1 MB against 290.1 MB.

**Measurement protocol, changed.** The suite pair was measured twice: the first
ran straight after a 263 s `texmo_gate` in the same hold and came back at
geomean 1.047 with 21 rows outside +-10 %, all large-batch `l128` rows, **7 of
the 11 worst building no msl plan at all** (byte-identical tapes). Re-run in a
hold of its own, every one returns to its P22 value. From here: the suite
first, the gate afterwards, and every outlier standalone before it is named —
six of this campaign's seven were the suite itself.

Frozen RC binary: `~/.cache/metaljax-bench/frozen-rc-ed355691.dylib`, sha256
`ed355691…94a16` (tree d70499b + this fix).

## P25: the eager flush trims instead of dumping (2026-08-16,
## notes/cpp-p25-cache-limit.md)

P20's last open item, approved by Oleg. A hard eager flush that found MLX's
buffer cache over `METALJAX_FLUSH_CLEAR_MB` (2048) called `mx::clear_cache()`
-- the whole pool to the OS, 7 times a step on maxtext's training row at ~70 ms
each. It TRIMS to the watermark now (`runtime.cc::trim_cache`: set the cache
limit, poke the allocator with one byte so MLX reclaims the excess "on the next
allocation", restore the limit). Same variable, same cadence, same programs;
`flushes=N(+clear M)` in the stats line is now `(+trim M)`, and
`METALJAX_MEMDBG` prints the pool at every hard flush -- the meter the ladder
is argued on, since a flush point is reachable from nowhere outside the dylib.

**The approved shape was the GLOBAL one -- `mx::set_cache_limit` at plugin
init -- and it was measured and rejected.** A global bound also bounds the
paths that never reach a flush and were never bounded: a compiled decode step
whose transients exceed it re-allocates from the OS every step. Row 13
(E2B keras-int4) **190.0 vs 80.7 ms/tok, 2.35x**, in a clean interleaved
position; suite-106 geomean **1.0420** against P23's 1.0050 with the `mid`
class at 1.1175; and one repetition of that arm died on a **GPU address
fault**, which is the hazard an allocation-time reclaim has and a flush-point
reclaim does not. Binary kept (`frozen-p25b.dylib`) if it wants re-running.

| | shipped dump (same-day RC) | **P25 trim** |
|---|---:|---:|
| row 19, maxtext train, ms/step | 975.4 | **833.9** (1.17x; peaks 21 / 20 GB) |
| row 18, LoRA train, ms/step | 400.0 | **394.0** |
| row 13, E2B int4, ms/tok | 80.7 | **80.8 / 80.9** |
| suite-106 native/Stage 1 | 1.0050 (P23) | **0.9685**, 106/106 within 1.2x |
| suite-106 `big` class | 1.0107 (P23) | **0.9296** |
| native arm vs P23's native arm | -- | **0.9882** (Stage-1 control: 1.0254) |

**The other 1.9x of row 19 is the WATERMARK, not the dumping**, and it is a
memory trade: 512 -> 1067.0 ms, 2048 -> 833.9 (21 GB), 8192 -> 685.6 (25 GB),
32768 -> 464.1 (39 GB), unbounded -> 461.7. The 32 GB setting reaches the 440
anchor and is exactly where row 18 blows through its 70 GB guard (68 GB, killed;
P20 measured 81 GB unbounded). Left at 2048, which beats the dump everywhere at
the same peak. Raising it is one variable and Oleg's call.

Ladder (run before any timing counted): four new `execute_test` contracts --
the pool holds at 255 MB over 552 hard flushes at a 256 MB watermark, the
median flush still finds 228 MB cached (a dump leaves 0), the same program runs
to 4025 MB with the trim off, and 20k interpreted loop iterations still clear
on the op-unit COUNT cadence (104 clears, 0 recoveries); `ingest_test` 8/8;
row 18's peak 39 GB at the shipped watermark. Row 18's peak is a LOAD transient
-- 37/38/39/43/56 GB across today's runs on BOTH binaries, always at sample ~10
of ~60.

Battery: `execute_test` all cases match CPU, `ingest_test` 0 failed,
`smoke_test`, `bazel test //...`, `texmo_gate` 105 ok + 1 FAIL
(`mid03-b64l128`, P23's documented flake for that config -- 3/3 ok standalone
on this binary and 3/3 on the RC one). Shipped binary
`~/.cache/metaljax-bench/logs/p25-cache-limit/frozen-p25c.dylib`, sha256
`516e4b43…`.

**Stage 1 is untouched and still dumps** (`src/metaljax/interpreter.py:776`,
`native/program.cc:38`, both frozen): the backport is three edits and a battery
re-run, and it is Oleg's decision. Until it lands, every same-day
native/Stage-1 ratio on an eager-main row carries this difference -- which is
what the 0.86x on row 19 above now measures.

## Post-0.11.5 retirement (Oleg, 2026-08-16): confirmed FIRST post-release
milestone — full cleanup INCLUDING docs. Delete: src/metaljax engine/
interpreter/tape/msl_scan (~19.3k), plugin/ trampoline, frozen native/ +
nanobind extension, METALJAX_ENGINE machinery, engine-API tests (71
counter rows, test_native_tape, extension buffer tests), texmo_check
(texmo_gate is the gate). Slim the loader to native-only; wheel build
becomes bazel-native (METALJAX_WHEEL_PLUGIN retires; sdist story TBD).
DOCS PASS: CLAUDE.md rewritten for the native-only world (Stage-1
machinery moves to history/notes), README, RELEASING.md. Full battery
on the native-only tree is the proof. Then the framework-gap fix list
(notes/framework-gap-gemma31b.md) becomes the performance era.

## The no-panic contract (Oleg, 2026-08-17, panic #9 during the 0.11.5
gate battery -- row 9 native, throttled, LAST in a 34-row sequence,
hot page cache). REFRAME: metaljax never panics the machine; degrade
or clean-OOM instead. 0.11.5 requirements now: (1) no tested model
panics, incl. rows 9/10/12/15/20 (OOM-error acceptable); (2) prefer
degradation over OOM; (3) under that contract, the 15 pre-migration
rows work at not-worse-than-pre-migration performance. Design
direction: an in-runtime memory governor (budget from hw.memsize +
task footprint via the P27 machinery + macOS pressure signals),
degrade ladder (hard trim -> ingest throttle/stall -> eval barriers)
-> clean RESOURCE_EXHAUSTED past the hard line; page-cache discipline
(madvise consumed mmap ranges away during ingest -- the hot-cache
last-in-battery pattern is the common factor of #8/#9).

## P28: the flush watermark has to be EARNED (2026-08-17..18,
## notes/cpp-p28-benefit-gate.md)

The 0.11.5 release gate scored gate 5 as REGRESSION over one thing: P27's
watermark cost the two maxtext DECODE rows (11, 14) 17 GB and 11 GB of peak
footprint, bought them no speed, and guard-killed both at the budgets every
previous campaign used. Oleg chose option (b) -- benefit-gate it. `flush_bound`
gains a THIRD rule over P25's floor, entering as one more `min` so it can only
ever lower a bound and no program that was safe under P25/P27 becomes unsafe:

    earned = min(METALJAX_FLUSH_EARN_MULT * (live_hi - live_lo), live_hi)

over the PROGRAM's own live set (`mx::get_active_memory`, sampled at its hard
flushes, kept in `FlushState`). The discriminator came out of the two rows'
own flight logs and it is NOT the flush count -- their checkpoint load takes
134 hard flushes in a single call, so every P27 counter reads "eager main" for
it. It is the live set: the training step swings 6.8 -> 20.5 GB every flush
(pool 11.7 GB p50 over 352 flushes), while the load holds ~3 GB flat and
merely has 14 GB of freed weights land in the pool at its last flush, and the
decode step holds 1,197 MB at all 71 of its flushes. Both terms are
load-bearing: `2*swing` alone leaves the loads at 7.1 GB (row 11 killed 1 run
in 3), `peak_live` alone hands the flat decode step a pool it never reads.
`METALJAX_FLUSH_EARN_MULT` default 2 (1 breaks row 19 at 569.9 ms/step); 0
restores P27 exactly.

Result at the rows' HISTORICAL budgets: row 11 9/9 complete at 20 GB
(16.61-16.83 ms/tok), row 14 4/4 at 25 GB (31.82-32.13), row 19 holds its P27
fix (459.2/458.4/462.5 ms/step at 48 GB, loss identical to P27's to 13
digits), row 18 361.8 ms/step with its live-set spike unmoved (meter peak
57,478 MB against P27's 57,479/57,480). Suite-106 native reads 0.9963 of the
recorded rc column with 0 of 106 rows past 1.1x, and 1.0004 of P27's native
arm. Battery: `texmo_gate` 106/106, `execute_test` 553 ok, `ingest_test`,
`smoke_test`, `bazel test //...`.

Two process findings worth the ledger. (a) **A battery split across two builds
attests neither**: the contracts had been run on the pre-clamp binary and the
rows on the shipped one, and re-running them on the shipped binary failed one
of P28's own four contracts -- it still asserted the rule's FIRST DRAFT
(`earn == 2*swing`) rather than the shipped `min(2*swing, peak_live)`. The
rule was right, the contract was stale; it now checks the shipped identity and
narrates which term bound each flush. (b) The rows were re-spotted on the
COMBINED build (P28 + the vendored patched libmlx, `frozen-vendor-d651add3`):
16.60 / 31.94 ms/tok and 463.5 ms/step at the same budgets, row 19's loss
bit-identical across all eight runs of the campaign -- the rule reads a
counter the fence fix does not touch, and the measurement says so.
