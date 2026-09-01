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
| 7 | gpt-oss-20b | ✗ ¹ | **19.8** | **8.8** (13.8 GB, native MXFP4) | — | 6.7 (native MXFP4) ¹⁰ |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | **28.5** (73 GB) | **13.7** | — | 15.3 ¹⁰ |
| 9 | R1-Distill-32B | ✗ 131 GB | **190.8** (67 GB) | 131.8 | — | 114.9 ¹⁰ |
| 10 | DeepSeek-V2-Lite (maxtext) | ✗ needs 50–105 GB ² | **24.8** ¹¹ (92 GB) | 10.5 | — | 10.7 ¹⁰ |
| 11 | Qwen3-0.6B (maxtext decode) | 89.7 | **12.33** | 3.0 | — | 3.4 ¹⁰ |
| 12 | Mixtral 8×7B bf16 | ✗ | **85.6** (90 GB) | **52.8** (93.4 GB) | — | — |
| 13 | gemma4-E2B keras-int4 (packed) | **67.8** ⁸ | **77.0** | — | — | — |
| 14 | maxtext qwix-int8 0.6B | 143.4 | **29.88** | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 | **388.4** (73 GB) | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 533 | **86.68** | — | 29.8 (b32: 591) | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ⁴ | **1249.3** @512², **4961.6** @1024² | ✗ ⁹ | 654 @512², 2998 @1024² ⁹ | — |
| 18 | LoRA E2B train (ms/step) | 2048 | **362.1** | — | 135.6 ³ | — |
| 19 | maxtext train 0.6B (ms/step) | 1402 | **444.6** | — | — | — |
| 20 | *aspirational* 235B-A22B 3-bit | ✗ | **56.2** (101 GB) | **28.0** (102.9 GB, load 12 s) | — | — |

**mlx-lm gap band (same Metal library underneath — the optimization
target):** 31B **0.95×** (126.1 vs 133.1); 12B ~parity (57.3 vs a dated
58.3 ⁷); Qwen3-8B 1.4× (42.0 vs 30.4); Llama 1.4× (42.2 vs 29.4);
gpt-oss 2.3× (19.8 vs 8.8 — native MXFP4 both sides); MoE 2.0× (33.4
vs 17.0); 3-bit 2.0× (56.2 vs 28.0). llama.cpp leads mlx-lm a further
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
   133.1 is a 2026-08-31 re-measure, same prompt and token count as
   the metaljax cell.
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
    (rows 1/2/3/5/6/8/9/10/11, dtype-verified), native MXFP4 (row 7,
    both sides), 3-bit (row 20, both sides). Mixtral (row 12) is
    proven quant-only across all 9 publishing providers, so its
    llama.cpp cell is legitimately empty.
11. Row 10 protocol: runs with `METALJAX_MEM_SYS_MB=107520` (its
    documented envelope; the shipped default sits under this row's
    restore transient).
12. Greedy token agreement vs jax-CPU: rows 5/6 are exact 64/64;
    rows 1/2/3 each flip one 1-bf16-ULP logit tie (accepted; logit
    evidence in notes/release-gates-0.11.7.md); row 4 diverges at
    token 51 (certified-benign, MODEL_TOKEN_KNOWN).
