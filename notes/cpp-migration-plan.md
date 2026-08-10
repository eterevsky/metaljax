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
