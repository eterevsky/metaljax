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
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **123.1** ¹² (63 GB) | 133.1 ⁷ | 148.7 | 111.2 ¹⁰ |
| 2 | gemma4-12B bf16 | 314.2 (f32) | **56.2** ¹² (26 GB) | 58.3 ⁷ | 67.6 | 44.2 ¹⁰ |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ⁶ | **31.1** ¹² | **17.0** | — | 16.9 ¹⁰ |
| 4 | gemma4-E2B bf16 | 67.5 (bf16→f32) ⁵ | **16.9** ¹² | 10.5 ⁷ | — | — |
| 5 | Qwen3-8B bf16 | 212.5 (bf16→f32) ⁵ | **36.0** (17 GB) | 30.4 | 38.1 | 29.6 ¹⁰ |
| 6 | Llama-3.1-8B bf16 | 208.0 (bf16→f32) ⁵ | **38.6** ¹² | 29.4 | 35.5 | 29.2 ¹⁰ |
| 7 | gpt-oss-20b | ✗ ¹ | **13.2** ¹² (34 GB) | **8.8** (13.8 GB, native MXFP4) | — | 6.7 ¹⁶ |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | **23.4** (73 GB) | **13.7** | — | 15.3 ¹⁰ |
| 9 | R1-Distill-32B | ✗ 131 GB | **199.9** ¹² (69 GB) | 131.8 | — | 114.9 ¹⁰ |
| 10 | DeepSeek-V2-Lite (maxtext) | ✗ needs 50–105 GB ² | **19.6** ¹¹ (90 GB) | 10.5 | — | 10.7 ¹⁰ |
| 11 | Qwen3-0.6B (keras-hub decode) | 28.6 | **5.1** ¹³ | 3.2 ¹³ | — | 3.4 ¹⁰ |
| 12 | Mixtral 8×7B bf16 | ✗ | **69.0** ¹² (93 GB) | 53.5 ¹⁷ (93.4 GB) | — | — |
| 13 | gemma4-E2B keras-int4 (packed) | 67.8 ⁸ | **6.0** ⁸ (48 GB load peak) | 4.5 ¹⁹ | — | — |
| 14 | Qwen3-0.6B maxtext qwix-int8 | 143.0 | **26.32** | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 | **265.3** (52 GB) | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 361.5 | **41.77** | — | 29.8 (b32: 591) | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ⁴ | **460.6** @512², **2090.4** @1024² | ✗ ⁹ | 553 @512², 3078 @1024² ⁹ ¹⁸ | — |
| 18 | LoRA gemma4-E2B train (ms/step) | 1358 | **126.1** ¹² | — | 135.6 ³ | — |
| 19 | Qwen3-0.6B maxtext train (ms/step) | 1414 | **362.8** | — | 818 ²⁰ | — |
| 20 | *aspirational* Qwen3-235B-A22B 3-bit (mlx quant) | ✗ | **45.2** (103 GB) | **28.0** (102.9 GB, load 12 s) | — | — |
| 21 | Qwen3.8-27B bf16 (dense hybrid) | ✗ ¹⁴ | **142.3** (57 GB) | **106.4** ¹⁵ | — | 98.2 ¹⁰ |

**mlx-lm gap band (same Metal library underneath — the optimization
target):** 31B **0.92×** (123.1 vs 133.1); 12B **0.96×** (56.2 vs a
dated 58.3 ⁷); Qwen3-8B 1.18× (36.0 vs 30.4); Llama 1.31× (38.6 vs 29.4);
gpt-oss 1.5× (13.2 vs 8.8 — native MXFP4 both sides); MoE 1.8× (31.1
vs 17.0); 3-bit 1.6× (45.2 vs 28.0); Qwen3.8-27B 1.34× (142.3 vs
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
8. Row 13: every cell through 0.11.7 (metaljax 77.0, jax-CPU 67.8 and
   their history) measured a numerically broken model: keras-hub 0.30.0's
   Gemma4 decoder block computes the FFN as matmul(x, layer.kernel)
   under a "HOTFIX", and for an int4 EinsumDense `.kernel` is the raw
   unpacked codes with no scale, so the model generated one token
   repeated on every backend (upstream still carries it, 2026-09-19;
   notes/keras-hub-gemma4-int4-hotfix-issue.md).  From 2026-09-21 the
   harness routes the int4 FFN through the quantized layers and gathers
   embedding rows before unpacking (scripts/model_bench/int4_fix.py,
   METALJAX_BENCH_INT4_FIX / _INT4_EMB=0 reproduce the original): the
   CPU cell is that variant at the row protocol on jax 0.11.2 (67.8, 128
   tokens; its stream equals metaljax's 64/64); the
   0.11.8 cell (6.0, 128 tokens) is the variant on the release binary; the
   0.11.7 cell (77.0) timed the broken graph.  Packed int4
   stays packed on metaljax (2.7 vs 10.2 GB).
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
    restore transient). The cell decodes the manifest prompt (50 DeepSeek
    tokens in a 64-slot prefill) for 128 tokens, loop-only average of 127
    steps, on the 0.11.8 release binary (19.59, jax 0.11.0 maxtext venv) — the
    comparators' workload; the harness variant `MAXTEXT_RNG_PRESPLIT=1`
    (the RNG split hoisted out of the timed loop) reads 19.05 on the same
    binary. The checkpoint load peaks 99–111 GB against the 105 GB
    projected guard, so about a third of its cells die at load (never
    during decode, never a panic) and are rerun.
12. The 0.11.8 cells: gate of 2026-09-23 on the release binary, jax 0.11.2
    (the maxtext rows 10/14/15/19 on jax 0.11.0: no released flax imports on
    0.11.2); the jax-CPU cells are jax 0.11.2's (it moved LoRA 2316 -> 1358 and
    SigLIP 537 -> 362).  Greedy agreement vs jax-CPU: rows 2/5/6/11/13 exact
    64/64; row 4 parts at 51/64 (the certified-benign tie).  Against 0.11.7
    rows 4/5/6/7/20/21 are identical, row 2 flipped back to the CPU stream,
    rows 1 and 3 carry tie flips (notes/release-gates-0.11.8.md).  In-battery
    vs standalone (the suite-context class): row 3 31.1 vs 28.2 / 28.2; row 6
    38.6 vs 34.5 / 34.6; row 18 126.1 vs 113.3 / 113.9 (the in-battery cells
    are the table's, as in 0.11.7); row 12 73.3 in sequence vs 69.0
    standalone (the standalone cell is the table's); row 9 185-200 across
    draws on three binaries (a ±4 % row).
13. Row 11 harness: keras-hub `Qwen3CausalLM` on `hf://Qwen/Qwen3-0.6B`
    (bench id `qwen3-06b-keras`), replacing the maxtext decode harness
    under the best-available-implementation rule (Oleg, 2026-09-01) —
    keras-hub jits the whole sampler loop, maxtext drives a Python loop
    per token. Same model, same bf16 precision, greedy streams
    identical metal-vs-CPU. Pre-switch cells are NOT comparable (see
    models.md ᵐ); the maxtext arm stays measured beside rows 14/19
    (same-day control 12.22 ms/tok). The mlx-lm cell is the same
    128-token window (3.2 / 3.2, 2026-09-06; the earlier 3.0 was a
    64-token generate).
    0.11.8 release cell: 5.1 ms/tok on the release binary (the row's first
    release-gate cell; 128 tokens; stream identical to its HEAD record);
    jax-CPU 28.6 on jax 0.11.2.
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
