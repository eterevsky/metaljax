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

- **MLX command-buffer corruption — upstream report.** All three faces
  (byte-budget splits corrupt compiled graphs; ops-boundary alignment
  corrupts eager scans; unbounded buffers wire transient intermediates
  until the machine panics) with shipping repro assets
  (tests/data/qwen3_prefill_shrunk.mlir, qwen3_init_scan.mlir,
  notes/mlx-command-buffer-split.md). Draft the issue for Oleg to
  file. Until fixed upstream, every finite budget is a lottery draw —
  tests/test_command_buffer.py pins the shipped values; rerun both
  tests on ANY change to the budgets, the flush cadence, or MLX.
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
