# metaljax — task tracker

*Canonical backlog. STATUS.md holds benchmark results; this file holds
what we plan to do. Roadmap history lives in CLAUDE.md.*

## Benchmarks & harness

- **llama.cpp-focused benchmark expansion** (per Oleg, 2026-08-03):
  llama.cpp is the measured kernel frontier on this hardware (~100% of
  memory bandwidth on dense decode). Take popular models that run on
  llama.cpp — including ones with no existing JAX implementation
  (Phi-family, GLM, new releases) — reimplement on JAX where needed,
  and measure head-to-head. Also extend the comparison axes where
  llama.cpp excels: prefill at long contexts, KV-cache scaling,
  speculative decoding. Goal: the C++-era metaljax is benchmarked
  against the frontier, not just against jax-CPU.
- **texmo bf16 suite**: runner already honors per-config precision;
  wire in Oleg's bf16 top-confs export when the bf16 search runs.
- **Harness nits**: gemma_lib rows report mem_gb=0 (params freed before
  sampling — move the measurement into the adapter); maxtext adapter
  records lack mem_gb entirely.

## Correctness & upstream

- **MLX command-buffer corruption — upstream report (URGENT: now
  blocks a whole model class).** All three faces (byte-budget splits
  corrupt compiled graphs; ops-boundary alignment corrupts eager
  scans; unbounded buffers wire transient intermediates until the
  machine panics) with shipping repro assets
  (tests/data/qwen3_prefill_shrunk.mlir, qwen3_init_scan.mlir,
  notes/data/qwen3_8b_prefill_36layer.mlir — the strongest one:
  real-shape 36-layer 8B prefill, corrupts in 0.3 s at the shipped
  512 MB budget, clean at 2048; notes/mlx-command-buffer-split.md).
  2026-08-03: the safe band scales with tensor size — Qwen3-8B-shaped
  maxtext workloads have NO safe budget (512 corrupts replays, 2048
  nondeterministically kernel-panics the machine at load — 4th panic;
  STATUS row 15 / footnote 8). Draft the issue for Oleg to file.
  Until fixed upstream, every finite budget is a lottery draw —
  tests/test_command_buffer.py pins the shipped values; rerun both
  tests on ANY change to the budgets, the flush cadence, or MLX.
  DO NOT run 8B-class maxtext on metal at raised budgets without a
  memory watchdog and Oleg's sign-off.
  2026-08-03 (later): PURE-MLX repro achieved via mx.export_function —
  notes/data/mlx-cbuf-repro/ (repro_a/b.py + .mlxfn, only mlx+numpy;
  corrupts at MLX's STOCK defaults). Issue draft ready for Oleg:
  notes/mlx-command-buffer-upstream-issue.md.
- ~~test_command_buffer canaries are stale~~ DONE 2026-08-04 (114b4d4):
  re-swept both axes on both assets and replaced the pinned comment
  values with SWEEP canaries — two tests that re-derive a corrupting
  budget in subprocesses (candidate list, stop at the first hit; ~3 s
  typical, ~25 s worst case) and fail loudly if none corrupts. Map:
  kernels 50/100/200/400 wrong, 450-1300 clean, 1350-10^9 wrong (the
  high tail is the byte budget cutting by itself); bytes <=400 wrong on
  the scan, <=48 wrong on the compiled decode. Shipped 800/512 clean
  24/24 reps on each asset. The bounds test now pins the shipped values
  to those measured bands (was >=64 / >=160). notes/
  mlx-command-buffer-split.md addendum 2026-08-04 has the tables.
- **Quantized-decode correctness criterion**: token-stream equality is
  not usable for quantized models (notes/int8-divergence-verdict.md);
  compare_tokens.py encodes the policy — extend the logit-ladder
  method if quantized rows multiply.
- **NEW (discovered 2026-08-05, present in RELEASED 0.11.0):
  position-dependent silent wrongness in sparse spdot_general** —
  `BCOOTest::test_bcoo_spdot_general0/6` produce WRONG VALUES (8/90
  elements, max abs diff 1.97 — not tolerance) only deep into a long
  process (a 283-test prefix reproduces; standalone passes). Proven
  NOT a 0.11.3 regression: the 0.11.0 approval tree 725bd84 fails
  identically; the approval run never saw it because the per-file
  gc-quadratic (now fixed) masked the tail of the file and the
  specific prefix. Position dependence smells like the MLX
  command-buffer lottery (process allocation state changes the draw)
  — investigate post-0.11.3; candidate datapoint for the upstream
  report. Repro command in the 2026-08-05 release-gate fixes log dir.
  DECIDED (Oleg, 2026-08-05): NOT whitelisted-benign — tracked-open,
  non-blocking for 0.11.3 (notes/data/pinned-0.11.3-failures.txt
  carries them in a separate TRACKED-OPEN section). Fix scheduled
  soon after the C++ migration, together with the blocked model rows
  (8/10/12/15). First step: the lottery-classification experiment
  (283-prefix repro under METALJAX_COMPILE=0 + shifted budgets).

## Performance — C++ era (measured targets in STATUS.md)

- **Native replay engine** (dispatch gap, 1.7–2.6× to mlx-lm): C++
  execute hot loop via MLX C++ API + nanobind; keep the Python compile
  path. The texmo topconfs geomean (anchor:
  notes/data/texmo-topconfs-final.jsonl) and the STATUS mlx-lm band
  are the tracked metrics.
- ~~Quantized storage + matmul~~ LANDED pre-migration (qmm recognizer,
  affine + MXFP4 + per-channel: rows 7/13 measured; row 20 still wants
  3-bit packed storage, deferred per Oleg).
- ~~MoE gather path~~ LANDED pre-migration (2310aa2, gather_mm/
  gather_qmm: row 7 measured 39.6→22.2; row 3 re-measure queued).
- ~~Fused attention~~ LANDED pre-migration (sdpa recognizer, e4d9f5b:
  SD3.5 512² 1389 + 1024² 5141 both measured; prefill-gap compression
  not yet re-measured on the LLM rows).
- ~~keras load path~~ LANDED pre-migration (streaming shim in
  adapter_keras_extra; rows 8/9/12 first runs queued).
- **Persistent compile cache**: cold-process warmup (31B pays ~9 s per
  process; serialized executables would amortize it). Related landed
  piece: in-process cross-executable pack-build cache (in progress —
  see "qmm per-executable pack REBUILD" below).
- **Kernel-specialization tier** (mlx→llama.cpp residual, ~1.25×):
  decode-specialized GEMV via custom Metal kernels — msl_scan
  machinery generalizes; only after the layers above land.

## PAUSED — machine-wedge class (kernel panic #7, 2026-08-04)

Per Oleg after panic #7: stop the runs implicated in reboots; for each,
build an equivalent SMALLER-model repro, verify it runs at its
predicted peak memory, and only then retry the big model.

- **Panic #7 record**: watchdogd starvation (no checkins 91 s), memory
  explicitly HEALTHY at panic (compressor 7%, swap OK) — a hard
  machine wedge, NOT a memory ramp; mem_guard is structurally blind to
  this class (row 8's flight log: footprint 53 GB, sys 58.8 GB, all
  samples "ok", log stops mid-line). Happened ~46 s into row 8's
  STREAMED LOAD (Qwen3.6-35B-A3B, arch Qwen3_5MoeCausalLM, 53 of
  71.9 GB assigned). Same class as panic #4 (load-time wedge with
  memory fine); distinct from #5/#6 (memory ramps). Distinguishing
  feature vs R1-32B — which streamed 61 GB clean TWICE the same day:
  assign rate ~1.2 GB/s (vs R1's 0.6). HYPOTHESIS REVISED by the
  small-rung ladder (2026-08-04, scripts/model_bench/wedge_repro.py):
  (a) row 8 is NOT many-small-tensors — experts arrive PRE-BATCHED,
  80 giant assigns (1.0 GB expert banks) carry 95% of the bytes;
  (b) the GPU-queue theory does not hold — mx.synchronize() after
  EVERY assign costs 22 ms total and the per-assign device work is
  ~0.0008 ms (the bytes move host-side in keras-hub's converter);
  (c) the leading indicator is FILE-BACKED PAGES: RSS runs 2.3× the
  checkpoint (mmap'd shards + anonymous weights), extrapolating to
  ~133 GB of mapped pages at row-8 scale on a 128 GB machine —
  matching the panic flight log exactly (RSS 101.9, footprint 53,
  compressor 7%). Refined hypothesis: VM reclaim/page-cache storm
  under a sustained ~1.2 GB/s mmap read+copy. mem_guard now kills on
  GUARD_RSS_GB (default 110) — the metric that WAS unhealthy.
  Residual uncertainty: two worktree agents were live at the time
  (both under lock discipline, so row 8 should have been the only
  GPU user).
- **PAUSED rows**: 8 (Qwen3.6-35B — the wedge), 12 (Mixtral 93.4 GB,
  same streamed-keras class, never attempted), 15/10 (maxtext 8B
  class, already embargoed after panics #4/#5).
- **Row 15 mitigation attempt, 2026-08-04 (Oleg-authorized, 5 guarded
  runs, zero incidents)**: METALJAX_BODY_COMPILE=0 at safe-band
  budgets. Phase 1 (Orbax restore) ballooning FIXED — the eager
  flush now returns cache above METALJAX_FLUSH_CLEAR_MB to the OS
  (the qmm _NoCache lesson engine-wide; 24→48 GB monotone became a
  6–23 GB sawtooth). Phase 2 (post-restore jit materialization)
  still demands >60 GB and climbs +4–7 GB/sample at every kill
  (45/45/60 budgets); next step would enter the 67–109 GB zone from
  the panic-#5 ledger — STOPPED per protocol. Row 15 ships blocked:
  compiled path = MLX corruption (upstream), mitigation path =
  phase-2 transient (attribute + fix post-0.11.3; the guard contains
  it reliably). Side fixes landed: step-tolerant trajectory rule,
  maxtext path wired in run_bench (was never landed from
  README_maxtext.md).
- **Ladder protocol** (per row, before any big retry) — SMALL RUNG
  GREEN 2026-08-04: scripts/model_bench/wedge_repro.py builds
  synthetic same-arch (Qwen3_5MoeCausalLM) HF-safetensors checkpoints
  at any size (shape formulas verified 1026/1026 against the real
  35B); wedge_run.sh is the lock+precheck+guard wrapper;
  BENCH_STREAM_MARK gives the load a phase fingerprint. Small
  (4.45 GB): predicted peak 6.13, measured 5.88, 7/7 guarded runs
  clean, assign rate meets row 8's. Peak model (calibrated on R1:
  predicted 65.3 vs 65.0 measured): weights + 1.2 GB + 2× largest
  tensor. MID TIER GREEN 2026-08-04: `mid` (div=2, small assigns)
  3/3 clean, footprint 19.0 vs 18.98 predicted, RSS 35.9; `wide`
  (div=1 ×9 layers, real 1.0 GB expert banks) 3/3 clean, footprint
  23.0 (point prediction 20.0, upper 28.0), RSS 40.4 — one guard
  kill in the first wide soak was the TRAJECTORY rule tripping on
  1 GB-granular assigns at a tight budget (26), not an anomaly;
  reran at the upper bound (28). A/B verdict: big assigns cost
  ~+3 GB transient and run SLOWER per byte (0.77 vs 1.24 GB/s) —
  consistent with page-cache pressure, not queue pileup. Row 8 retry
  awaits Oleg's go: budget 95, GUARD_RSS_GB≈95 (the panic profile
  crossed RSS 95 ~10 s before the wedge), supervised, single run;
  optional stronger lever first = per-shard page-cache release in
  the loader (madvise/F_NOCACHE class, not built).
- **Still allowed** (never implicated, repeatedly clean same-day):
  gpt-oss class ≤25 GB (10+ clean runs), gemma4-26b 51.6 GB (3 clean),
  SD3.5 34 GB, texmo gates, pytest, agents' synthetic validations.

## Deferred / blocked (with measured reasons)

- DeepSeek-V2-Lite metal: guard-killed at 122 GB (maxtext MoE prefill
  memory model; retry with the memory stack queued behind the row-15
  supervised run, needs Oleg's sign-off — maxtext 8B class).
- 26B-A4B CPU: guard-killed at 34 GB en route to ~150 GB.
- gpt-oss CPU: dequantized working set ~126 GB — infeasible.
- eager-path scan flush cadence: values pinned by tests; revisit only
  with the MLX fix.
- **big10-b8l256 (gru.1024) intermittent inf — lottery class, first wild
  texmo sighting** (2026-08-03): one gate run produced inf in 18/20
  outputs in suite context; standalone passes, full-gate rerun 104/104,
  and engine decisions are byte-identical with METALJAX_COMPILE_BYTES_MB
  on/off (the bytes gate never touches this config). Classified as the
  MLX command-buffer nondeterministic corruption at SHIPPED budgets —
  goes into the upstream report as the first spontaneous texmo draw.
  Release protocol: treat any single-run gate FAIL as rerun-first;
  0.11.3's step-3 gate should run 2-3x (the automation already
  re-runs cheaply) and big10 sits on the soak watchlist.
- **SD3.5 all-zero image — RESOLVED as a harness bug** (see STATUS fn 9; both engine suspects cleared by bisection): 512²/20 steps now COMPLETES
  at 18.1 GB peak (memory stack works) but pixels are all zero. Suspects:
  (a) flush-cadence shift on rewrite-carrying blocks (225408b changed
  absorbed-op byte charging → sync points moved → command-buffer lottery
  redraw on the eager sampler); (b) an sdpa value bug specific to the
  MMDiT joint-attention shape. A/B levers: METALJAX_SDPA=0,
  METALJAX_EAGER_FLUSH_MB sweep, 2-step repro at 16 GB cap.
- **bytes estimator returns 0 for gpt-oss programs** → compile gate never
  fires → trace wave guard-kills row 7 re-measure. Debug shows
  bytes=0.0MB on its mains. Estimator bug for this graph class.
- **SD3.5 1024² — DONE (2026-08-04)**: 5141 ms/step, real image, peak
  34.0 GB under the 70 GB guard (STATUS fn 9). The old 50–60 GB
  working-set estimate predated plan-aware pruning; deferred-list
  entry removed.
- **qmm pack transient — RESOLVED (e04c7fc)**: row-blocked pack
  evaluation + MLX cache off during packs; 15.9 → 1.5 GB per pack
  (notes/qmm-pack-transient-2026-08.md). Row 7 re-measured **22.2**
  ms/tok (STATUS footnote 23) — the ~24 MoE projection held.
- **qmm per-executable pack REBUILD** (in progress, agent out):
  keras-hub builds one executable per generate length; each new State's
  matches re-evaluate + re-verify every weight (~0.9 s × 94/shape,
  ~200 s build_s on row 7) even though `_share` proves the results
  content-identical. Cross-executable build cache keyed on (leaf buffer
  identity, structural subtree fingerprint); on hit skip build+verify
  (sound: the pack is a pure function of both). Also: per-pack
  gc.collect cadence (~10 s/wave) and moe._dead_sweep (8.8 s/wave).

- **Suite-only deps are undeclared** (found 2026-08-05: an Aug-1 `uv
  sync` pruned ad-hoc `flatbuffers` → 5 phantom JAX-suite failures +
  export_serialization_back_compat's 21 tests silently uncollected;
  restored + run_jax_tests.py now preflights optional deps). DECISION
  FOR OLEG: declare a test dependency group in pyproject (or a
  documented install line in RELEASING.md) so pytest/scipy/optax/
  torch/flatbuffers survive venv re-syncs.

- **Latent order-dependency in the compile cost model** (found during
  M2, pre-existing): control._block_cost discounts qmm-absorbed ops by
  reading interp._qmm, and sdpa.analyze POPULATES interp._qmm as a
  side effect (via _claimed) — so merely querying sdpa earlier flips
  the Python engine's compile decision for programs nothing rewrites.
  M2 works around it (native lowering asks sdpa last); fix properly
  by making qmm analysis explicit rather than a side effect.
- **M2 native-tape next batches**: func.call inlining (single-block
  callees — jax 0.11 wraps where/clip/round in private helpers; the
  single biggest decline family), divide/remainder/power/shifts with
  their XLA-semantics wrappers, argmax-pair reduce, gather/scatter,
  integer dot_general (exact-f32 chunk machinery), expm1 (MSL
  helper). Then M3: native mx::compile + control flow.

- **INT_MIN trunc-div latent bug** (found in M2.5, pre-existing):
  ops/elementwise._int_trunc_div abs/sign dance wraps at INT_MIN on
  MLX (int8(-128)/2 == 64, XLA says -64). Both engines pinned to it
  for now; fix is a Python-engine change + tape follows.
- M2.5 batches 1-4 DONE (call/composite inlining, wrapped elementwise
  incl. expm1, argmax-pair reduce, int dot_general). Batch 5 gather/
  scatter declined: MLX ships no C++ indexing-semantics layer; next
  frontier by census: bitcast_convert (29), fft (9), gather (6),
  dynamic_slice (kv-cache ops, cheap via mx::slice_update).

- **Pre-existing engine crash flagged in M3**: nested loop with inner
  counter captured from enclosing scope dies in _run_chunked (eval
  during function transformations; except RuntimeError misses
  ValueError) — native engine survives the same module. Fix with the
  post-C++ batch.

- PRE-M6 BLOCKERS (tail sweep, 2026-08-10): (a) ✅ FIXED — native was
  2-10% behind python on fully-lowered texmo chunks. NOT the output
  taint (those chunks lower with ZERO static copies): every
  millisecond was M1's `to_host`, which wrapped each result in a
  fresh `contiguous` node and evaluated THAT — a 20us stream round
  trip per output whatever its size, 0.5ms on a 23-output chunk. Now
  settles the array itself and copies only a non-row-contiguous
  layout; all three configs at parity or ahead. (b) census blind
  spot: Interpreter-direct tests (test_conv 13) never hit
  engine.execute — stablehlo.reverse was missing with no test
  noticing; audit op-set vs the full registry, port convolution.
  (c) ThreeFry bits stay uint32 for signed result types in
  ops/rng.py (latent; differential pins both engines to it).

- Row 15 (2026-08-17): 'known MLX-quantization bug' label WITHDRAWN (the 2026-08-03 diagnosis blamed the command-buffer split and exonerated the quantized dots). Fresh leading hypothesis H5: MLX_MAX_MB_PER_BUFFER counts ELEMENTS (~537M at '512'); row 15's untied logits_dense is 622M elements -> deterministic mid-matmul buffer split -> logit collapse (all-token-0 output). Row 14's tied embedding (155M) never crosses. Decisive 1-min probe: scripts/model_bench/row15_probe.py big (self-locking) -- run when the release-gate battery frees the lock. Full-row raised-budget arm still needs Oleg's sign-off; the standalone-replay probe does not.
