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
- **M5 — msl_scan + host ops**: generated kernels via the C++
  metal_kernel API; host LAPACK/callbacks stay Python (impure ops
  already leave traces — the tape marks host-op sites and the C++
  engine calls back only there).
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
