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
- **Quantized storage + matmul** (25–33× on gpt-oss; North-star row 20):
  map dequant+matmul patterns onto mx.quantized_matmul / gather_qmm;
  packed sub-byte storage. Unblocks rows 15/20; fixes the int4
  unpack re-materialization (11.7×) and the int8 int64 cliff.
- **MoE gather path** (16.7×, row 3): lower expert dispatch onto
  mx.gather_mm instead of dense-all-experts.
- **Fused attention** (fast.scaled_dot_product_attention mapping):
  unblocks SD3.5/diffusion (row 17: ~90 GB unfused live set) and
  should compress the ~6× prefill gap.
- **keras load path**: streaming weight load (skip random-init) —
  unblocks the ≥60 GB keras metal cells (rows 8, 9, 12) and the
  122 GB LoRA load transient.
- **Persistent compile cache**: cold-process warmup (31B pays ~9 s per
  process; serialized executables would amortize it).
- **Kernel-specialization tier** (mlx→llama.cpp residual, ~1.25×):
  decode-specialized GEMV via custom Metal kernels — msl_scan
  machinery generalizes; only after the layers above land.

## Deferred / blocked (with measured reasons)

- SD3.5 at 1024² on metaljax: unfused-attention live set ~90 GB
  (STATUS footnote 9) — needs fused attention.
- DeepSeek-V2-Lite metal: guard-killed at 122 GB (maxtext MoE prefill
  memory model).
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
- **SD3.5 1024² retry**: 512² banked (1389 ms/step, fn 9); 1024² guard-
  killed clean at the 55 GB budget with the ramp still climbing —
  legit working set ~50–60 GB (4× pixels). Retry at budget ~70,
  ceiling ~85, single non-chained run; same caps (COMPILE_BYTES_MB
  16384). Next-session queue, with rows 3/9/8.
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
