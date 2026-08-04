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
- **test_command_buffer canaries are stale** (found 2026-08-03): the
  corrupting ops alignment MOVED 400 → 200 after fdc7cde's shift
  peephole changed the threefry lowering — our own commits reshuffle
  the lottery, and the shipped ops=800 is currently pinned by nothing.
  Re-pin the canary values (and consider a sweep-style canary) before
  the 0.11.3 gate.
- **Quantized-decode correctness criterion**: token-stream equality is
  not usable for quantized models (notes/int8-divergence-verdict.md);
  compare_tokens.py encodes the policy — extend the logit-ladder
  method if quantized rows multiply.

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
  assign rate ~1.2 GB/s over ~1500+ mostly-small MoE expert tensors
  (vs R1's 0.6 GB/s over 771) → working hypothesis: GPU command-queue
  pileup during rapid small assigns starves userspace. Residual
  uncertainty: two worktree agents were live at the time (both under
  lock discipline, so row 8 should have been the only GPU user).
- **PAUSED rows**: 8 (Qwen3.6-35B — the wedge), 12 (Mixtral 93.4 GB,
  same streamed-keras class, never attempted), 15/10 (maxtext 8B
  class, already embargoed after panics #4/#5).
- **Ladder protocol** (per row, before any big retry):
  1. Same-arch repro at SMALL size — smallest same-arch HF checkpoint,
     or a synthetic one: build the backbone from a shrunken config,
     save HF-safetensors, stream-load through the identical path
     (arbitrary size dial on the exact code path).
  2. Predict peak (weights + shim overhead) BEFORE the run; guarded
     run must match the prediction; soak ×3 (the wedge class is
     plausibly nondeterministic).
  3. Step up (~4-8 GB → ~20-30 GB → full), one supervised run at full
     size only after the ladder is green.
  4. Instrument the load: per-N-tensors phase markers (a wedge must
     leave a fingerprint), and test whether tighter sync bounds the
     queue (BENCH_STREAM_CLEAR_GB smaller / explicit mx.synchronize
     cadence) at small scale first.
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
