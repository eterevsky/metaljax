# metaljax — a Metal backend for JAX

## Goal
Build a JAX backend that runs on Apple-silicon GPUs via Metal, on this machine
(M5 Max, macOS 26.5, Xcode 26.6). North star: **everything the `~/texmo`
project runs on CPU must run on Metal, preferably faster** (texmo trains many
small language models: dense/GRU-style layers, norms, softmax/CE loss, Adam,
`lax.scan`, RNG). Milestone zero: `a = jnp.array([1, 2, 3]); print(2 * a)`
executes on the Metal device through plain `jax.numpy`.

## Architecture decisions (agreed with Oleg)
- **Native plugin (the only engine).** metaljax is a real PJRT plugin:
  `plugin-native/` builds `libmetal_pjrt_native.dylib`, an
  `xla::PjRtClient` subclass that XLA's `pjrt_c_api_wrapper_impl` wraps into
  the PJRT C API. StableHLO is parsed and lowered natively (C++) and
  executed against our vendored MLX; no Python is on the execute path. JAX
  sees a genuine `METAL` device; unmodified `jax.numpy` code works.
  *History:* through 0.11.5 a Stage 1 implementation also existed — a
  trampoline dylib (`plugin/`) driving a StableHLO interpreter written in
  Python (`src/metaljax/`) on `mlx.core`. Stage 2 (native) reached parity
  and shipped as the release wheel in 0.11.5; **Stage 1 was RETIRED in
  0.11.6** (~26k lines deleted). Notes and benchmark tables from that era
  keep their Stage-1 references as history.
- **Execution engine: our vendored MLX build** (linked as
  `libmlx_metaljax`, private install name). Chosen per Oleg's criteria
  (compat with arbitrary jit'd ops + vmap/etc → performance → build ease):
  MLX is Apple's maintained Metal array library, lazy by default, and
  `mx::compile` lets the engine trace lowered programs into fused, cached
  Metal graphs per executable. The op layer is kept behind an internal
  interface so hand-written Metal shaders / MPS can replace pieces later
  (msl_scan already does, for counted loops).
- **Compile path:** PJRT hands the plugin a parsed StableHLO module (XLA's
  C-API wrapper does the VHLO upgrade and deserialization); the plugin
  lowers it op by op into a tape, decides what to fuse into `mx::compile`
  graphs and what to emit as generated MSL kernels, and replays it.
- **Dtypes:** Metal/MLX have no float64. f32/f16/bf16/int/bool only;
  `jax_enable_x64` stays off. f64 in programs is an expected failure, not a
  target.

## Ground rules
- **Owned MLX build** (Oleg, 2026-08-17): metaljax vendors its own MLX,
  built from OUR FORK at the latest tagged release (currently v0.32.0 —
  the pip pin already matches). Fix branches in the fork, one per fix,
  sent upstream as patches/PRs, and incorporated into our build
  REGARDLESS of upstream acceptance. This unties: direct fixes for the
  MLX bug tally (command-buffer corruption, fusion #8, ...) instead of
  bug reports and workaround knobs. Private install name for the
  vendored dylib so pip's mlx can never collide. Pushing the fork and
  opening upstream PRs = Oleg (never push rule applies to the fork
  too); patch commits + PR-quality descriptions prepared locally.


- All changes live in `/Users/oleg/metaljax`. **Never modify `metaljax/jax/`**
  or `metaljax/llvm-project/` — read-only reference clones (gitignored).
  JAX clone HEAD ≈ 2026-07-23, matches installed jax 0.11.0.
- Git: repo remote is git@github.com:eterevsky/metaljax.git. **Commit locally;
  never push** — Oleg pushes himself. No PRs.
- Releases: build + smoke-test wheels locally, but **never upload to
  TestPyPI (or anywhere) without Oleg's explicit approval** — he decides
  version numbers based on what's being published. Plan: iterate on
  0.4.x; when the gap list is closed to his satisfaction, jump straight
  to 0.11.0 (tracking the jax pin; 0.5–0.10 intentionally skipped).
- **The no-panic contract** (Oleg, 2026-08-17, after panic #9): metaljax
  must NEVER cause a kernel panic. The OS does not keep us in check, so
  the library does: preferred = degrade performance under memory
  pressure; acceptable = a clean OOM error (RESOURCE_EXHAUSTED at the
  PJRT boundary); never a wedge. Applies to EVERY model row including
  the previously embargoed ones (9/10/12/15/20) -- they may OOM-error,
  they may not panic. A 0.11.5 release requirement alongside the test
  gates, and a standing acceptance criterion for all future big-run
  work. Amendment (same day): in extreme cases the JAX benchmark/model
  code may be MINIMALLY fixed (harness variants in scripts/model_bench,
  never ~/texmo) -- try hard on the original first; if it cannot work,
  the original may cleanly OOM or run slow (never panic) and the fixed
  variant must work as intended. Both cells reported.
- **Release rule 1 — no stale numbers** (Oleg, 2026-08-16, after the
  0.11.4 near-miss): every number in a release table must come from the
  release binary. Changes after the last benchmark run are acceptable
  only if they provably cannot move a number; otherwise re-measure the
  affected rows before release.
- **Release rule 2 — never "PASS" over a regression**: a significant
  regression on any test suite or benchmark makes the gate verdict
  REGRESSION, not PASS. Releasing over one requires (a) Oleg's explicit
  confirmation, (b) the regression stated in the gate report itself.
  Both rules go verbatim into every RC/release agent brief.
- **Wheel-only distribution** (0.11.6): the plugin needs bazel + the pinned
  XLA workspace + the vendored MLX, none of which an sdist can build at
  install time.
- **MLX A/B is build-level**: `METALJAX_MLX_DIR=<tree> bazel build
  //metal:libmetal_pjrt_native.dylib`, then pin each dylib with
  METALJAX_PLUGIN_PATH. (The venv-swap canary died with the Python engine,
  which imported the pip `mlx`; the vendoring battery's row-1 A/B is the
  worked example.)
- Python via **uv**: venv at `metaljax/.venv` (CPython 3.14.4), installed:
  `jax 0.11.0`, `jaxlib 0.11.0`, numpy, pytest (no `mlx` — the plugin ships
  its own). Run things with `.venv/bin/python`.
- Correctness bar: every implemented op/feature gets a pytest comparing Metal
  results against the CPU backend (tolerances appropriate for f32).
- **Perf-costing correctness fixes need Oleg's sign-off first.** Where
  matching XLA/C99 semantics would mean adding per-element branches to a
  hot path (complex special values at inf/NaN poles are the motivating
  case), report the finding — which ops, how many tests, what the branch
  costs — and let Oleg decide whether the compatibility is worth the
  slowdown. Don't just implement it.
- `~/texmo` is the acceptance workload (read-only; don't modify it either).
- Delegation (per Oleg): well-defined "investigate and fix" tasks go to
  Opus subagents (worktree-isolated, GPU-light validation only while
  other GPU work runs) to preserve main-context tokens. The main agent
  reviews every diff before it is committed — subagents never commit.

## Layout
- `CLAUDE.md` — this file.
- `plugin-native/` — THE ENGINE (bazel workspace): `metal/` the PJRT plugin
  (ingest, lowering, qmm/moe/sdpa recognizers, MSL codegen), `runtime/` the
  executor (op emitters, control flow, host LAPACK, memory governor),
  `third_party/mlx/` the vendored MLX wiring. Its own differential suites:
  `execute_test.py`, `ingest_test.py`, `texmo_gate.py`.
- `pyproject.toml`, `src/jax_plugins/metal/` — the loader (backend
  registration, linalg/callback lowerings, the host-callback registry the
  plugin calls back into).
- `src/metaljax/` — `__version__` and `lib/`, where the plugin dylib and the
  vendored MLX runtime land (also the bazel MLX source dir).
- `tests/` — pytest suite (Metal vs CPU, through PJRT).
- `notes/` — investigation notes worth keeping (many describe the retired
  Stage 1; they stay as history).

## Roadmap / status
1. ✅ Decisions above; env set up (uv venv, jax 0.11.0, mlx 0.32.0).
2. ✅ StableHLO→MLX interpreter (`src/metaljax/`): 122 pytest cases pass vs
   CPU — elementwise, shapes, dot_general/einsum, reductions, cumops,
   while/if/scan, threefry RNG, gelu/composites, bf16/f16.
3. ✅ PJRT plugin (`plugin/metal_pjrt.cc` → `plugin/build/libmetal_pjrt.dylib`,
   build via `plugin/build.sh`): `2 * jnp.array([1,2,3])` runs on
   MetalDevice(id=0); jit(grad) matches CPU ~1e-8; RNG bit-exact vs CPU;
   scan/matmul correct through the real backend.
4. ✅ **texmo trains on Metal.** gather/scatter (incl. windowed + batching
   dims), sdy identity ops. 131 tests. f64 policy (per Oleg): STRICT by
   default — f64 pass-through OK (stored f32, bit-identical), f64 *compute*
   fails at compile naming the op; METALJAX_F64=downcast opts into f32
   emulation (needed only under jax_enable_x64, e.g. optax AdamW's
   beta**step; NB texmo.py sets x64 FALSE — scripts/texmo_train.py must
   match it, an earlier x64=True copy skewed repro perf/dtype profiles).
   Driver: scripts/texmo_train.py (imports ManagerJax directly,
   avoids texmo.py's hardcoded platforms; torch installed just for imports).
   `bench_jax.py --platform metal` works (flag committed to texmo repo).
   Verified: `bits.1+bp|rnn.1.tanh` and `bits.1+bp|mgru.4-dense.4.gelu` train
   end-to-end with sane losses; train_and_eval incl. recurrent eval path.
5. ✅ **Performance pass 1** (counted-loop detection + mx.compile): pure
   mains and counted while bodies (scan/fori: cond `i < N`, body `i+1`)
   trace once through mx.compile, then replay fused graphs; caches keyed by
   ir.Block (pointer-stable across traversals). METALJAX_COMPILE=0 disables.
   Numbers (full train steps, f32, scripts/bench_compare.py):
   transformer d256/b32: metaljax 31.2 ms vs torch-MPS 30.1 vs jax-cpu 174;
   d512/b64: 157.6 vs torch 154.6 (within 2% — Oleg's aspirational goal met
   for transformers). GRU.256/b256: 112 vs torch fused nn.GRU 50.7 (2.2×
   gap = per-timestep python replay); texmo bench_jax b256: 147.7 vs cpu
   273.3; b64: 133.6 vs cpu 113.7. Was 502 ms/step before this pass.
   Pass 2 (loop unrolling): small statically-counted loops unroll into the
   enclosing mx.compile trace (interp._in_trace flag decides per context;
   never nest mx.compile). METALJAX_TRACE_BUDGET (20000) caps any single
   trace: MLX retains every intermediate while tracing AND each pending
   replay pins its buffers → Metal's ~500k live-buffer limit; eager counted
   loops flush (mx.eval) every ~25000/cost iterations. _block_cost must
   traverse func.call/composite callees (undercounting made the engine
   compile texmo's whole 256-step chunk → buffer exhaustion). Result: texmo
   train step = ONE graph replay (~34ms for mgru.4 — kernel-launch-bound;
   CPU wins tiny models on physics; parity at mgru.256: 5.1s vs cpu 5.3s).
   METALJAX_DEBUG=1 logs loop/compile decisions.
   Future perf ideas: prepared-closure interpreter rewrite (kills
   MLIR-wrapper overhead on eager paths + trace time); if/case branch
   compile; hoisting loop-invariant input projections (torch's fused-GRU
   trick). Remaining coverage gaps: argmax/argmin multi-result reduce
   (sampling path), sort, partial-window scatter.
6. ✅ PyPI release prep (v0.1.0): plugin ported to CPython **limited API**
   (>=3.12) → single wheel py3-none-macosx_14_0_arm64 for all Pythons;
   hatch_build.py compiles the dylib at wheel-build time. Gotcha: never use
   '#' formats in PyObject_CallMethod (ABI differs across versions — broke
   on 3.12). Wheel verified on fresh 3.12 venv; twine check passes.
   LICENSE = Apache-2.0 (flagged for Oleg's confirmation). RELEASING.md has
   the upload steps; **Oleg publishes himself** (like git pushes).
7. ✅ **msl_scan: persistent-kernel codegen** (src/metaljax/msl_scan.py) —
   counted loops with pure elementwise bodies (mingru/rglru/lrnn family,
   fwd AND AD-generated bwd) compile to ONE generated Metal kernel
   (mx.fast.metal_kernel, thread per batch×feature lane, state in
   registers). Generic IR pattern-matching, no layer-specific code (Oleg's
   requirement: nothing texmo-specific, transferable to future layers).
   Gotchas: jax hoists the loop-increment constant OUT of the body (fold
   free splat-constant captures or the counter looks non-affine); MLX
   passes 0-dim inputs by value (no subscript); first run must eval
   synchronously (Metal build errors in async workers abort the process);
   kernels ARE traceable into enclosing mx.compile graphs. METALJAX_MSL=0
   disables. Affine H256/B16/L256: fwd+grad 15.4→0.07ms (CPU 0.73).
   db09-b128l128 1.00ms — beats CPU (2.2) AND 4090 (2.3). Matvec cells
   (rnn/gru/mgru) still fall back — future: cooperative threadgroup
   variant (pass C).
8. ✅ **msl_scan vector mode**: small in-lane matvecs (rnn/gru/mgru cells,
   block-diag einsums) via register-vector lanes; AD weight-grad
   accumulations handled by loop fission (hidden stacked outputs + one
   post-kernel einsum); gate-split slice/pad; stride bookkeeping; dots
   absorb transposes (SymPerm). MSL loops count as traceable + cost ~8 so
   training-step bodies compile around them. db02-b4l1024 72.9→0.38ms
   (4090 13.8!); db09 0.63ms; db05 10.6. REMAINING: (a) texmo's lrnn
   lowers block contraction as multiply+REDUCE in-cell (db07/08/10 still
   ~600/49ms) — need SymReduce over the reg dim; (c) db05-class outer
   bodies partially eager still.
9. ✅ **msl_scan coop mode** (v0.2.0): full-width matvec cells — one
   threadgroup per batch element, feature dim = thread axis, dot data via
   threadgroup shared array + barriers. Square dots == state width F only,
   F ≤ 1024, no batching dims. Three things made it fast (see notes/):
   structural CSE (hash-consing `_canonicalizer` — AD residual outputs
   duplicate whole gate subtrees; ir.Value hashes by MLIR value, id() of
   the wrapper is NOT stable) + emitter-level dot CSE (residual dot copies
   differ by a leading unit dim); dW einsums as explicit batched matmul
   (mx.einsum was the +120ms); weight-layout canonicalization (bwd W^T
   dots read uncoalesced otherwise; materialized transposed per call).
   GRU.256/b256 full train step 51.6 ms vs torch fused nn.GRU 47.1
   (within 10%); texmo bench_jax 66.5 ms/step (beats CPU everywhere now);
   mgru.256 ~24 ms/step. MEASUREMENT TRAP: jax.block_until_ready is a
   no-op on this backend (events born ready + async_eval) — time through
   np.array() or mx.eval.
10. ✅ **openxla/xla benchmark suite + v0.2.1** (notes/xla-benchmarks-2026-07.md):
   HLO text → StableHLO via bazel-built xla-translate (--hlo-text-to-mlir-hlo,
   then jaxlib mhlo_to_stablehlo; the --hlo-to-stablehlo flag wants a proto and
   null-crashes on text) + scripts/run_stablehlo_bench.py (PJRT
   compile_and_load on cpu/metal/cuda, seeded inputs, outputs checked vs CPU).
   Metal beats M5 CPU 2-13x on all 6 runnable single-device benchmarks;
   gemma3_12b (23.5GB bf16) runs on metal, doesn't fit the 4090. New ops from
   the suite: argmax/argmin reduce (jax pattern; ties→lowest index = MLX
   first-occurrence), reduce_precision, plain stablehlo.dot. CRITICAL FIX:
   bf16 hex-splat constants (dense<0xFF80>) — bindings decode hex-as-float
   (65536, not -inf) so bf16 always takes _ir.dense_to_np's text path, and
   hex = bit pattern for all float types (ml_dtypes bf16 has np kind 'V').
   Engine falls back to eager on mx.compile IndexError (unused inputs).
   Benchmark traps: refs and runs need the SAME gen_inputs version (rng draw
   order shifts invalidate refs); GPU scatter-add is order-nondeterministic
   (like jax-CUDA); sample_loop benchmarks are vacuous under random inputs
   (decode loop runs 0 iterations). v0.2.1 on TestPyPI, verified 3.12+3.13.
11. ✅ **v0.2.2: long-run worker fixes** (from Oleg's errors.txt after
   hundreds of training cycles). (a) Metal caps LIVE buffers at ~499k
   (device_info resource_limit) while MLX's cache is bounded by BYTES —
   config-sweeping workers accumulate freed small buffers forever;
   engine now mx.clear_cache()s at every compile boundary + every 50k
   executes (METALJAX_CLEAR_PERIOD). (b) mx.compile dies
   (unordered_map::at) when two OUTPUTS bake to equal constant values
   (minimal: mx.compile(lambda x: (x+1, mx.array(.9), mx.array(.9)));
   unused inputs are NOT the problem — tolerated fine). Fix: statically
   find non-input-derived outputs (_underived_outputs) and anchor them
   with where(x==x, out, out) — bitwise exact, kills constant baking.
   This got maxtext's whole-main compiling (19.4s→10.6s/step eager→
   compiled). (c) Compiled while BODIES can still fail at call time on a
   deeper variant of the same MLX bug (equal constants colliding
   somewhere inside the tape — anchoring all outputs does NOT cure the
   lrnn.8.2 S512 chunk body); such bodies now fall back to the
   uncompiled body (blacklisted per process, iteration retried — safe:
   pure body, carries untouched). Manifests as per-op dispatch overhead
   for the affected loop level only (inner scans keep MSL/compiled
   paths); METALJAX_DEBUG=1 prints "compiled while body failed".
12. ✅ **v0.3.0: lrnn + rectangular coop dots + CRITICAL Metal compiler
   workaround.** (a) In-lane register reduces (SymRedReg), iota,
   invariant hoisting ('hoist' leaves eval IR subgraphs per call),
   nested statically-counted unroll (trip<=64): texmo's lrnn family
   compiles fwd+bwd (db08 100x, db07 40x). (b) Rectangular coop dots
   (csize/dsize multiples of F, g-loop, chunked sh writes): mullstm/
   fused-gate cells (db18 47x). (c) CRITICAL: Apple's Metal shader
   compiler miscompiles multi-iteration kernel time loops (db10/db12
   were order-1 WRONG since 0.2.0; found by whole-model gate). Only a
   fully-volatile loop counter fixes it — volatile-loads-only, opaque
   table-read counter, and single-volatile-copy ALL still miscompile
   (METALJAX_MSL_VOLATILE=t/tmap/tv/load/0 to retest on OS updates);
   costs ~1.4x on small-F kernels (db11/14/15). (d) Mode policy: vector
   needs min(csize,dsize)<=16 (widened caps stole square 32-wide dots
   from coop: db13/db16 regressed 46x/23x, caught by suite rerun).
   (e) Coop work cap 2.2M dot-elems/step (METALJAX_MSL_COOP_CAP):
   rectangular support captured gru/lstm.1024 where per-threadgroup
   weight re-streaming loses 2-2.5x to compiled matmul; measured
   crossover gru.512 wins / lstm.512 ties / F=1024 loses. Correctness
   gate scripts/texmo_check.py (whole-model vs jax-CPU, 1-ULP
   sensitivity-scaled tol): 104/104 x3 runs. Suite: 2.97x faster than
   0.2.3 total; 84/104 beat CPU (all >=10k weights; sub-ms rows are
   dispatch-floor); 41/104 beat the 4090. vs torch MPS: transformer
   d256/d512 within 1.5%, GRU.256 within 11%, bench_jax b256 59.2ms.
   Suite-context trap: sub-ms configs measure ~2x slower inside a
   104-config sweep than standalone (kernel-cache/buffer-pool growth) —
   verify regressions standalone before believing them.
13. ✅ v0.3.1: msl_scan rejects plans needing >30 buffer bindings (Metal
   caps kernels at 31; deep fused bodies like dense+rglru+slstm AD
   residuals exceeded it → "Unable to build metal library" at execute
   time, from Oleg's live sweeps). Falls back to compiled-graph path;
   spec verified vs CPU. Also 0.3.0→Beta classifier + texmo-free README.
14. ✅ v0.3.2: Metal buffer-count exhaustion INSIDE one execute (from
   Oleg's live sweeps: "[metal::malloc] Resource limit (499000)" at the
   eager while-loop flush, e.g. S32768 runs dying after ~5k steps —
   the 0.2.2 engine-level clears are per-execute, far too coarse for
   multi-hour single executes). control._loop_flush: all loop sync
   points (eager counted flush, chunked-replay sync, dynamic fallback)
   now clear MLX's cache every ~100k flushed op-units
   (METALJAX_LOOP_CLEAR_COST) + clear-and-retry once on the limit;
   same retry in the while body-failure handler and engine.execute
   (programs pure → rerun safe). Verified: both crashing specs train
   past their death points, perf unchanged (db11/mid08 match 030c).
15. ✅ **v0.4.0: close the jax/tests gap** (per Oleg's plan: document
   gaps → fix bugs by severity → add features by closeness → dtypes).
   Bugs: scatter OOB DROP semantics (jnp.nonzero/bincount/sparse
   clusters + the gather-VJP mystery — arithmetic combiners neutralize,
   set uses a dummy slot; drop strategy picked by operand-vs-updates
   size); empty reduce->init / empty gather/scatter; bitcast size-
   change/rank-0; reverse rank-0; argmax NaN-wins; TOTALORDER compare;
   negative-stride host buffers (plugin passes base offset). Features:
   sort (generic comparator recognizer: key-chain eval + structural
   symmetry -> stable argsort on total-order keys; complex lexico-
   graphic; chlo.top_k); convolution (mx.conv_general all layouts +
   batch groups; int conv exact via im2col+int64; complex via 4 real
   convs; 0-spatial as matmul); windowed scatter via index expansion;
   general reduce bodies (pairwise halving, any monoid, variadic);
   general reduce_window (as_strided windows) + select_and_scatter +
   select_and_gather_add; scatter apply-bodies (unique_indices);
   popcnt/clz (SWAR); sign(NaN); complex64 END-TO-END (dtype plumbing
   incl. PJRT C64, text-parsed constants — the bindings can't cast
   complex dense attrs, real/imag/complex/fft ops, complex scatter by
   parts, expm1=exp-1); LAPACK on host (Qr/orgqr/syevd/gesdd/geev
   FFI targets s+c variants via numpy/scipy; eigh/svd/eig lowerings
   registered for platform 'metal' in the plugin, cholesky/
   triangular_solve as impure host ops). Purity: custom_call_host_hook
   + host ops in _IMPURE_OPS keep host compute out of mx.compile.
   Known-unfixable: version skew (~330), rng_bit_generator, ApproxTopK,
   i4/f8, multi-device, denormal FTZ. Suite gains recorded in
   notes/jax-test-suite-2026-07.md.
16. ✅ **v0.4.1: crash fixes + CPU-parity batch 1** (policy: JAX-CPU
   parity is the bar — every metal-failing test rerun on CPU;
   cpu_parity.json: 1,094 targets / 87 best-effort). Crash fixes over
   released 0.4.0: windowed-scatter size-1-window transpose (vmapped
   scatters/dus — expand must include ALL mapped dims owning a uwd);
   reduce_window as_strided contiguity + zero-size guards; complex
   iota. Parity batch 1: single-device collectives (all_reduce/gather/
   to_all/permute/broadcast identities, replica_id=0 → pmap+shard_map
   work); multi-key lexicographic sort (successive stable argsorts,
   -0/NaN canonicalized keys; cleared the ENTIRE sparse family —
   BCOO lexsorts internally); ApproxTopK = exact top-k; schur/
   hessenberg/tridiagonal host targets. bf16/f16 linalg EXCEEDS CPU:
   own metal lowerings (metaljax_eigh/svd/eig custom calls) accept all
   dtypes, host handlers upcast halves to f32 (jax's CPU rules reject
   bf16 in jaxlib LAPACK tables). GOTCHA (2nd occurrence): never key
   seen-sets by id() of transient MLIR wrappers — CPython reuses freed
   addresses (truncated sort-comparator dep walks this time; kernel
   names in 0.2.0 before). Remaining parity backlog: i4/f8 emulation
   (CPU supports create/add/convert+f8 matmul), Philox
   rng_bit_generator, debug callbacks, eigh_test/lobpcg/scipy_signal
   pockets, lax_test exotics tail.
17. ✅ **v0.4.2: gap-closure campaign + final audit → 98.4%** (27,779/
   418; gate 104/104; perf at baseline). Landed: Philox + ThreeFry
   rng_bit_generator BIT-EXACT vs CPU (reverse-engineered from xla
   prng.cc: state=[key,counter]u64, 128-bit counter=(ctr+i, key+carry),
   round-robin word interleave; ThreeFry splits output shape in halves
   at first even dim); i4/u4/f8* emulation (exact values in f16/f32
   storage, grid-quantized converts w/ RNE+subnormals+FN-overflow→NaN,
   4-bit wrap in binary handlers); host callbacks (in-process registry
   + metaljax_callback custom call — debug.print/pure/io_callback);
   shape-poly export (custom calls declare result_shapes via
   eval_dynamic_shape_as_tensor); XLA shift-overflow semantics;
   QR/SVD orthonormal completion (zero-tau orgqr padding); Schur/
   Hessenberg/tridiagonal(+solve, perturb_singular); generic
   reduce_precision (any e/m); grouped 0-D convs; dilated variadic
   reduce_window; 0.3.2 LIVELOCK triple-fix: bounded resource-limit
   retries, pessimistic trip=1024 cost for unknown-trip loops,
   accdot unit-squeeze bug (67x on composite b1 specs — the old
   '10 steps/s mystery'), engine retries moved OUT of except blocks
   (tracebacks pin failed-trace arrays). Remaining 418 dispositioned
   (notes/jax-test-suite): ~33 intentional, ~150 skew/harness, ~190
   'under review' itemized in README per Oleg (real cpu-passing gaps:
   effect tokens ~50, complex pole semantics ~24, PJRT surface ~40,
   windowed scatter-apply ~13, i4 bitcast 7, dilation corners ~7,
   singletons). Versioned 0.4.2 per Oleg (no major bump until tests
   fully closed).
18. ✅ **v0.4.3: coop-over-vector mode pick + kernel input packing**
   (notes/perf-0.4.3-mode-pick-packing.md). From Oleg's search timing
   data: 0.3.2's catastrophic rows were already fixed by 0.4.x; what
   remained was (a) F<=16 square cells (rnn.16/mgru.16/mullstm.8/
   gru.16) picking msl vector mode = batch-only lanes, ONE simdgroup,
   weights re-read per timestep -> coop measured faster at EVERY batch
   (b8..b2048, no crossover; threadgroup staging is half the win). Now
   flipped when dots are square multiples of state width and F>=8
   (METALJAX_MSL_COOP_MIN_F / _COOP_PREF=0), build_plan retries as
   vector on ANY flipped-build failure. (b) >31-binding kernels
   (lmgu-class deep AD bodies): same-dtype inputs pool into per-dtype
   buffers, static offsets baked into MSL ("(pk0+123u)"), run()
   concatenates; 0-dim stay by-value. CRITICAL review catch: slot
   sizes must use the weight-norm WINDOW numel, not source numel
   (gate-slice windows differ -> silent pool shift; regression test
   fails on unfixed code). Results: db17 7-12x, db11 5-6x, db14/15
   ~4-5x, lmgu 11.2->2.1 ms/step, rnn16 7.96->0.95 (beats CPU 2.5),
   mullstm 5.89->1.09; all other suite rows 1.00x +/- noise; gate
   104/104 + 3 new-path configs vs CPU. b=1 tiny models deprioritized
   per Oleg (search optimizes for >=thousands of weights). Leftovers:
   F<=4 coop pocket unmeasured; output packing; reads&weights sid
   overlap guard (pre-existing, flagged).
19. ✅ **v0.4.4: worker buffer GC + matlstm groundwork.** (a) From a
   live worker crash (Resource limit 499000 surviving clear-and-retry):
   Python's cycle gc barely triggers under array workloads, dead
   managers in refcycles pin ~500 buffers/config (measured via new
   metaljax.diagnostics.live_buffer_floor — allocate 1-elem buffers to
   the limit and count; MLX has no count API and bytes round to zero
   at these sizes), and mx.clear_cache() can't free REFERENCED
   buffers. Fix: gc.collect() before clear_cache in every
   resource-limit recovery + at compile boundaries + the 50k-execute
   backstop; plan._last_bufs debug retention gated MJDBG_VERIFY_MSL;
   METALJAX_MEMDBG telemetry. Release chain itself verified clean
   (floor returns to baseline after config release). (b) matlstm
   diagnosis (300x vs CPU: matrix state needs 2-D register blocks —
   feature DROPPED per Oleg, off Pareto frontier; notes/matlstm-2026-07)
   yielded three generic recognizer fixes: acc-broadcast unit-dims,
   in-lane small-dot rewrite (rank>=3 gated, METALJAX_MSL_INLANE=0,
   db17 24% win kept, db02 regression found+gated), _needs_registers
   mode probe; build_plan retries with fired heuristics off. Gate
   104/104, 200 pytest, perf at baseline.
20. ✅ **jax-tests parity campaign (post-0.4.4, per Oleg: examine every
   failure, fix or establish benign; delegate investigate-and-fix to
   Opus subagents, review all diffs before commit).** Pinned-release
   suite (new jax-v0.11.0/ checkout + scripts/run_jax_tests.py --tests)
   is the honest headline; HEAD-clone suite kept as early warning.
   Landed (each gated 104/104): ordered-effect tokens (bool[0] avals;
   ordering free — in-process sync callbacks); complex make_complex
   bit-interleave + C99 special values (abs/sign/exp/expm1/tan) + Kahan
   csqrt/rsqrt (beats CPU); accurate f32 expm1 (~500 ULP -> 2.9e-7,
   MSL header helper); reduce_window/sort/conv contiguity + shape
   workarounds; donation end-to-end + XLA no-alias contract (static
   forwarded-output copies); UnsafePointer via buffer protocol +
   Py_buffer.buf (np copy=False and dlpack both return TRANSIENT
   addresses — never use); windowed scatter-apply; i4 bitcast nibble
   packing; f6/f4 OCP emulation (saturating overflow; XLA:CPU's fp6 is
   itself broken — ml_dtypes is the reference); grouped int + negative-
   pad + zero-size convs; singular solves -> inf/nan; c128 pass-through
   (f64 checker didn't match c128 — compute silently ran in c64);
   compile-options validation; shape-poly LU (symbolic MATRIX dims only
   — host getrf rounds differently, bf16-visible); fft unit-length
   rewrite + input barrier.
   SILENT-WRONGNESS bugs fixed in OUR code (7): msl lane-scalar
   broadcast (0.2.0+), msl concat pad-total width (partial unrolls),
   msl SEQUENTIAL carry assignment (rotating carries collapsed,
   [4,4,4,4] for [4,3,2,1], all 3 emitters), top_k non-last axis,
   dilated reduce_window stale memory, complex sort -0/+0 ties, conv
   short-buffer overread (uninit memory, flaky).
   MLX BUG TALLY (7): strided-view reductions, strided argsort, conv
   zero-dim short buffer, rfftn unit-last-length dropping transforms,
   FFT-vs-async_eval RACE (eager-after-jit stale reads; fixed with
   fft-scoped input barrier), %.7g rank-0 constant baking (1 ULP on
   67% of constants in EVERY fused kernel), complex sqrt cancellation.
   Harness lessons: 4-job suite runs UNDER-report failures; worktree
   agents branch from origin/main AND their diffs omit untracked files
   (check git status before harvesting); helpers.check now pins its
   reference to CPU (was self-comparing under JAX_PLATFORMS=metal,cpu).
   Wontfixes documented with numbers (FD-reference sign flip, CPU-also-
   fails, better-than-CPU shape-poly cases). Pinned suite: 213 -> 130
   failed (99.53%, 27,649 passed), zero files regressed; HEAD-clone
   suite retired per Oleg (moving target; last measured 489 -> 328
   before most of the campaign landed). Export-harness sweep
   (scripts/run_export_harnesses.py): 5,460/5,587 with every non-pass
   attributed (CPU's own f16/bf16 linalg gaps block joint artifacts).
21. ✅ **v0.11.0 released** (version jumps 0.4.x→0.11.0 per plan, tracks
   the jax pin; gap list reviewed + approved by Oleg item by item, see
   notes/jax-test-suite-2026-07.md release-review section; complex pole
   semantics reclassified intentional/MLX-level). Real-LLM validation:
   gemma-4-31B-it + 12B-it end-to-end via DeepMind's gemma library
   (HF safetensors → lib tree converter in session scratch; 31B bf16
   374 ms/tok warm, 12B bf16 189 / f32 254, CPU-f32 12B 938; README
   table). Found+fixed: bf16 host transfers staged through f32
   (2x transfer, 2x transient device mem — 31B cold run 502s→90.6s,
   THE "7-min warmup" was paging; MLX f32→bf16 astype kills NaN
   payload+sign) → bitcast both directions, tests/test_dtypes.py.
   31B f32 (123 GB) does NOT fit 128 GB: metaljax thrashes, CPU
   attempt kernel-panicked the machine (LLM decode = zero-locality
   sweep, swap can't help; add memory watchdogs to big runs).
   Measured: decode is ~120 ms/tok Python-dispatch-bound (dtype-
   independent; f32/bf16 only 1.34x apart) → Stage 2 phase 1 =
   native replay engine for the execute hot loop (C++, MLX C++ API +
   nanobind ext; keep Python compile path), phase 2 = full native
   (StableHLO/MLIR C++); also noted: running XLA's HLO pass pipeline
   as a library pre-step would clean input graphs + enable
   OptimizedProgram. Pinned suite on release tree: 27,649/130
   (99.53%), zero regressions; itemized list in
   notes/data/pinned-0.11.0-failures.txt. TestPyPI 0.11.0 uploaded by
   Oleg, install verified fresh-3.13 (+3.12 wheel pre-upload); real
   PyPI + git push = Oleg.
22. ✅ **Stage 2 COMPLETE — the fully native plugin** (`plugin-native/`,
   bazel workspace against xla at the jax pin; `xla::PjRtClient` behind
   XLA's C-API wrapper). Native StableHLO lowering, recognizers, msl_scan
   codegen, compile decisions, Accelerate LAPACK, callbacks, donation,
   emulated dtypes; the memory governor and the no-panic contract; the
   owned MLX build vendored inside the wheel. **Shipped as v0.11.5**
   (PyPI, tag v0.11.5): jax suite 99.54 % id-stable, texmo 106/106,
   19 model rows re-measured on the release binary, rows 8/10/15 with
   first-ever numbers.
23. ✅ **v0.11.6 (in progress): Stage 1 RETIRED.** The Python engine
   (`src/metaljax/*.py`, 18,828 lines), the pre-PJRT C++ engine
   (`native/`, 5,637) and the trampoline plugin (`plugin/`, 1,483) are
   DELETED — ~26k lines, commit ef5774d; the native plugin is the only
   engine. The wheel builds native-only from the REAL tree (`uv build`),
   so the 0.11.5 staged-tree workaround died; `mlx` is out of the
   dependency list; wheel-only, no sdist. tests/ re-pointed onto the
   plugin (helpers.py drives `jax.jit` on the metal device +
   `compile_and_load` for raw modules) — the old suite's "native leg" had
   been running the Python engine in-process all along, so 1053/205
   measured mostly Stage 1; retained suite 483 pass + 1 xfail, which
   RECOVERED ~134 parity tests an ad-hoc deselect had hidden. Release
   texmo gate re-pointed with re-baselined anchors (top_confs → the
   223-config 16k set, suite-106 → the 0.11.5 native arm); gatelib pins
   METALJAX_PLUGIN_PATH. Two native-vs-Stage-1 differences documented:
   f64 programs decline at compile (buffers still pass through), plain
   `stablehlo.dot` declined (jax never emits it).
   Also in 0.11.6: bf16 msl fast paths (the 25× topconfs cliff → 0.99
   geomean vs fp32; `notes/topconfs16k-sweep-2026-08-22.md`) and the
   16-bit scatter-add fix.

## Environment note
- venv is **Python 3.14.4** (texmo needs PEP-649 lazy annotations; jaxlib
  0.11.0 / mlx 0.32 ship cp314 wheels). torch (CPU wheel) installed only so
  texmo modules import; the JAX path never calls it.
- Consumers depend on metaljax via a path dependency (uv sources, editable
  or not). The wheel carries the plugin dylib + the vendored MLX runtime
  under `metaljax/lib/`; a dev checkout copies
  `plugin-native/bazel-bin/metal/libmetal_pjrt_native.dylib` there (the
  loader also falls back to bazel-bin), and `METALJAX_PLUGIN_PATH`
  overrides both — measurements pin a frozen binary that way. jax is a
  declared dependency (>=0.11,<0.12 — the PJRT header pin); **mlx is NOT**
  (0.11.6: the wheel ships its own).

## Implementation notes (hard-won)
- Select the backend with `JAX_PLATFORMS=metal` (or `metal,cpu` to keep CPU
  available for comparisons). Plugin registered at priority -1 so CPU stays
  default otherwise. Registration: `src/jax_plugins/metal/__init__.py`
  (namespace pkg + entry point); env override `METALJAX_PLUGIN_PATH`.
- PJRT programs arrive as **StableHLO portable artifacts** (VHLO bytecode);
  XLA's C-API wrapper does the upgrade + deserialization before the plugin
  sees the module. The MLIR context must know **sdy** (+ mpmd) — jax 0.11
  emits sdy attrs and `sdy.sharding_constraint` ops even single-device
  (identity for us).
- **M5 GPU (applegpu_g17s) MLX f32 matmul is low-precision** (~4e-3, neural
  accelerators). metaljax defaults to accuracy: `MLX_METAL_GPU_ARCH=`
  `applegpu_g16g` before MLX builds its device; opt out with
  METALJAX_MATMUL_PRECISION=default. Owned by
  `plugin-native/metal/metal_client.cc` (a static initializer that also
  pins MLX_MAX_OPS_PER_BUFFER=800 / MLX_MAX_MB_PER_BUFFER=512, the measured
  command-buffer bands); the loader repeats the GPU_ARCH default before
  dlopen. `src/metaljax/__init__.py` sets nothing.
- All PJRT events are born ready — `jax.block_until_ready` is a **no-op** on
  this backend, so time through `np.asarray` (or `mx::eval`), never through
  block_until_ready.
- *HISTORICAL (Stage 1, retired 0.11.6):* the trampoline shim's C-ABI
  requirements (Device_GetAttributes' non-null deleter,
  AddressableDeviceLogicalIds, the hand-encoded DeviceAssignmentProto), the
  "engine.py must import only jaxlib/mlx/numpy" rule, the bf16-constant
  decoding through the MLIR Python bindings, and the `PyObject_CallMethod`
  `'#'`-format limited-API gotcha all belonged to the deleted Python path.
  The plugin does the equivalent natively; kept here as history because the
  C-ABI facts still describe what jaxlib demands of any PJRT plugin.
