# Model benchmark suite — status

*(Harness and manifest: [scripts/model_bench/](scripts/model_bench/).
Release-gate records: `notes/release-gates-<version>.md`.)*

Headline metric per cell: LLM rows = warm decode ms/token; vision =
forward ms; diffusion = ms/step; training = ms/step. ✗ = established
impossible (with the measured reason). metaljax cells are the current
release's gate values, measured on that release's binary; memory in
parentheses is peak footprint where measured.

| # | benchmark | jax CPU | metaljax | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **126.1** ¹² (63 GB) | 133.1 ⁷ | 148.7 | 111.2 ¹⁰ |
| 2 | gemma4-12B bf16 | 315.2 (f32) | **57.3** ¹² (26 GB) | 58.3 ⁷ | 67.6 | 44.2 ¹⁰ |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ⁶ | **33.4** ¹² | **17.0** | — | 16.9 ¹⁰ |
| 4 | gemma4-E2B bf16 | 67.5 (bf16→f32) ⁵ | **24.0** ¹² | 10.5 ⁷ | — | — |
| 5 | Qwen3-8B bf16 | 207.0 (bf16→f32) ⁵ | **42.0** (17 GB) | 30.4 | 38.1 | 29.6 ¹⁰ |
| 6 | Llama-3.1-8B bf16 | 203.6 (bf16→f32) ⁵ | **42.2** | 29.4 | 35.5 | 29.2 ¹⁰ |
| 7 | gpt-oss-20b | ✗ ¹ | **19.8** | **8.8** (13.8 GB, native MXFP4) | — | 6.7 ¹⁶ |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | **28.5** (73 GB) | **13.7** | — | 15.3 ¹⁰ |
| 9 | R1-Distill-32B | ✗ 131 GB | **190.8** (67 GB) | 131.8 | — | 114.9 ¹⁰ |
| 10 | DeepSeek-V2-Lite (maxtext) | ✗ needs 50–105 GB ² | **25.9** ¹¹ (86 GB) | 10.5 | — | 10.7 ¹⁰ |
| 11 | Qwen3-0.6B (keras-hub decode) | 29.4 | **9.0** ¹³ | 3.2 ¹³ | — | 3.4 ¹⁰ |
| 12 | Mixtral 8×7B bf16 | ✗ | **85.6** (90 GB) | 53.5 ¹⁷ (93.4 GB) | — | — |
| 13 | gemma4-E2B keras-int4 (packed) | **67.8** ⁸ | **77.0** | 4.5 ¹⁹ | — | — |
| 14 | Qwen3-0.6B maxtext qwix-int8 | 143.4 | **29.88** | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 | **388.4** (73 GB) | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 533 | **86.68** | — | 29.8 (b32: 591) | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ⁴ | **1249.3** @512², **4961.6** @1024² | ✗ ⁹ | 553 @512², 3078 @1024² ⁹ ¹⁸ | — |
| 18 | LoRA gemma4-E2B train (ms/step) | 2048 | **362.1** | — | 135.6 ³ | — |
| 19 | Qwen3-0.6B maxtext train (ms/step) | 1402 | **444.6** | — | 818 ²⁰ | — |
| 20 | *aspirational* Qwen3-235B-A22B 3-bit (mlx quant) | ✗ | **56.2** (101 GB) | **28.0** (102.9 GB, load 12 s) | — | — |
| 21 | Qwen3.8-27B bf16 (dense hybrid) | ✗ ¹⁴ | **154.9** (56 GB) | **106.4** ¹⁵ | — | 98.2 ¹⁰ |

**mlx-lm gap band (same Metal library underneath — the optimization
target):** 31B **0.95×** (126.1 vs 133.1); 12B ~parity (57.3 vs a dated
58.3 ⁷); Qwen3-8B 1.4× (42.0 vs 30.4); Llama 1.4× (42.2 vs 29.4);
gpt-oss 2.3× (19.8 vs 8.8 — native MXFP4 both sides); MoE 2.0× (33.4
vs 17.0); 3-bit 2.0× (56.2 vs 28.0); Qwen3.8-27B 1.46× (154.9 vs
106.4). llama.cpp leads mlx-lm a further
~1.25× on bf16 — the kernel frontier. metaljax prefill trails ~5×;
load ~20–30×.

## Footnotes

1. Row 7 CPU: keras dequantizes the MXFP4-native repo to bf16 (~42 GB
   weights); the working set projects ~126 GB — established infeasible
   on a 128 GB machine.
2. Row 10 CPU: maxtext's sparse MoE path wants 50–105 GB for the
   prefill on CPU — never completes inside this machine's budget.
3. torch-MPS LoRA: MPS has no SDPA backward kernel (math fallback,
   verified by autograd node inspection). Loss series are not
   comparable across stacks (different preprocessing); step cost is
   the comparison.
4. Row 17 CPU: keras's mixed-precision layers request the F16_F16_F32
   dot algorithm, which XLA:CPU rejects (an accelerator contract).
5. CPU cells run what XLA:CPU supports: weights load bf16, matmuls
   upcast per-op (bf16→f32); the 12B row is full f32 (gemma-lib path).
6. Row 3 CPU: f32 26B is ~104 GB of weights alone, and the observed
   keras-CPU load inflation (2.9×) projects a ~150 GB peak; the guard
   killed the load once the growth trajectory made that conclusive.
7. mlx-lm caveats: released 0.31.3 cannot run gemma4_unified (12B) or
   the E-series KV-sharing layout (E2B) — those two cells are mlx-lm
   git main (2026-08-03 install), and row 2's 58.3 could not be
   re-measured since (0.31.3 refuses the cached checkpoint). Row 1's
   133.1 is a 2026-08-31 re-measure on the same manifest prompt and
   token count as the metaljax cell (the metaljax cell chat-templates it,
   ~63 tokens vs ~52 raw). The E2B cell's raw run record is lost (its
   token count is unknown); it stands as dated until re-measured.
8. Row 13: packed int4 stays packed on metaljax (2.7 vs 10.2 GB — the
   only sub-byte JAX path that keeps it), while XLA:CPU fuses the
   in-graph unpack into a small net win (67.8 vs 79.2 bf16) — which is
   why the CPU cell leads this row.
9. Row 17 comparators: torch via the ungated diffusers mirror
   (adamo1139/stable-diffusion-3.5-large-ungated @5d868ff; images
   verified at both resolutions). No ungated MLX path exists for
   SD3.5-Large (mflux is Flux-only; DiffusionKit's formats are
   gated) — that cell is closed as not-runnable.
10. llama.cpp build 221f0f63, `llama-bench -p 51 -n 128 -r 5`,
    all-Metal, reproduced within 4 % on two passes; per-provider GGUF
    pins in scripts/model_bench/README_llamacpp.md. Dense rows pin to
    439–555 GB/s effective bandwidth — the machine's kernel frontier
    (llama.cpp leads even mlx-lm ~1.25× on bf16).
    LIKE-FOR-LIKE RULE (Oleg, 2026-08-28): cross-framework cells appear
    only at the SAME precision as the metaljax cell — custom kernels
    are fair game, different quantization is not. Kept cells: bf16
    (rows 1/2/3/5/6/8/9/10/11/21, dtype-verified), native MXFP4
    experts with bf16 attention/embeddings/head (row 7: metaljax and
    mlx-lm; the ggml-org GGUF is not, fn 16), 3-bit (row 20, both
    sides). Same precision means the same weight, activation and KV
    width (a 16-bit KV cache of either format counts as the same), and
    the same computation: same checkpoint, prompt window, generated
    token count, batch, resolution and encoder set. Mixtral (row 12)
    is proven quant-only across all 9 publishing providers, so its
    llama.cpp cell is legitimately empty. Row 21 is the one bf16 row
    where llama.cpp leads mlx-lm by only 1.08× (98.2 vs 106.4).
11. Row 10 protocol: runs with `METALJAX_MEM_SYS_MB=107520` (its
    documented envelope; the shipped default sits under this row's
    restore transient). Since 2026-09-06 the cell decodes the manifest
    prompt (50 DeepSeek tokens in a 64-slot prefill) for 128 tokens,
    loop-only average of 127 steps, on the 0.11.7 release binary (25.93 /
    25.95, peak 84–86 GB) — the comparators' workload; the earlier 24.8
    used the adapter's 5-token default prompt for 8 tokens.
12. Greedy token agreement vs jax-CPU: rows 5/6 are exact 64/64, and
    row 11 is exact over 64 GENERATED tokens (checked past the
    51-token prompt — stronger than the harness's first-64-ids check);
    rows 1/2/3 each flip one 1-bf16-ULP logit tie (accepted; logit
    evidence in notes/release-gates-0.11.7.md); row 4 diverges at
    token 51 (certified-benign, MODEL_TOKEN_KNOWN). Row 21 has no CPU
    counterpart — its metal stream is recorded only (4 runs
    token-identical).
13. Row 11 harness: keras-hub `Qwen3CausalLM` on `hf://Qwen/Qwen3-0.6B`
    (bench id `qwen3-06b-keras`), replacing the maxtext decode harness
    under the best-available-implementation rule (Oleg, 2026-09-01) —
    keras-hub jits the whole sampler loop, maxtext drives a Python loop
    per token. Same model, same bf16 precision, greedy streams
    identical metal-vs-CPU. Pre-switch cells are NOT comparable (see
    models.md ᵐ); the maxtext arm stays measured beside rows 14/19
    (same-day control 12.22 ms/tok). Cell band 8.5–9.0 tracking
    machine state; 9.0 is the unguarded gate-protocol median. The mlx-lm cell is the same
    128-token window (3.2 / 3.2, 2026-09-06; the earlier 3.0 was a
    64-token generate).
14. Row 21 CPU: 55.6 GB of bf16 weights plus the checkpoint's own page
    cache reaches ~116 GB of 128; two guarded attempts (the second with
    the load throttled to 0.4 GB/s) were killed during the load at RSS
    101.4 / 100.3 GB with the free list at 0.1 / 0.8 GB.
15. Row 21 mlx-lm: mlx-lm 0.31.3 released, `mlx_lm.models.qwen3_5`
    dense path on the same bf16 checkpoint; it emits EOS at 93 of 128
    tokens, so its per-token average sits at a shallower KV depth than
    the metaljax cell's.
16. Row 7 llama.cpp: the ggml-org MXFP4 GGUF stores attention q/k/v/o,
    token embeddings and the untied head at Q8_0 (2.56 GB/token vs the
    3.70 GB/token of the bf16 attention/head metaljax and mlx-lm run):
    not like-for-like under fn 10, kept for reference only; the row's
    goal is mlx-lm.
17. Row 12 mlx-lm: on a local bf16 conversion of the mlx-community
    mirror (its tensors are float16 though its config says bf16; f16
    holds every bf16 value exactly, and all 323 converted tensors read
    BF16), so the same bf16 computation as the metaljax cell: 53.4 /
    53.6, 2026-09-10 (the f16 mirror itself read 52.8).
18. Row 17 torch: diffusers run with the T5-XXL encoder OFF
    (text_encoder_3=None, max_sequence_length 77 → the same 154-token
    MMDiT context as the keras preset's t5=None), 2026-09-06; both sides
    bf16 MMDiT with CFG 7.0 (our CLIP-L/G fp16 as the preset pins them).
    Same-session T5-on controls reproduced the previous cells (648.8 /
    3155.3 vs 654 / 2998; the 1024² session ran ~5 % slower than the
    2026-08-03 one, so the 3078 carries that offset).
19. Row 13 mlx-lm: mlx-lm's own affine 4-bit, group-64 quantization
    (`mlx_lm convert -q --q-bits 4 --q-group-size 64`, 4.5 bits/weight,
    2.6 GB; mlx-lm git main 0.32 -- 0.31.3 rejects the checkpoint) of the
    same bf16 google/gemma-4-E2B-it that keras quantizes to int4 (group
    64 on q, 128 on k/v): median of 4 cells 4.9 / 4.5 / 4.0 / 4.5 on a
    31-token generation, 2026-09-10.
20. Row 19 torch-MPS: a full-parameter AdamW step at MaxText's defaults
    (f32 master weights under bf16 autocast, lr 3e-5, betas 0.9/0.95,
    eps 1e-8, weight decay 0.1, clip 1.0, batch 1, seq 256, synthetic
    tokens, loss over the whole window; mean of 4 warm steps as the
    MaxText adapter reports it): 817.9 / 817.7, 2026-09-10.  The
    backward runs MPS's math SDPA decomposition (fn 3).
