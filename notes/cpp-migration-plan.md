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
* Next: LAPACK on Accelerate (the only family left in the census), then
  msl_scan (still the whole texmo gap), then async execute + donation.
  Suite-vs-suite as the standing gate. (North star per Oleg: everything
  through the new stack, correctness first, performance second.)
