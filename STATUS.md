# Model benchmark suite — status

*(Harness and manifest: [scripts/model_bench/](scripts/model_bench/).)*

*Last updated: 2026-08-03 (block 4: big models, MoE, packed int4). Headline metric per cell: LLM rows =
warm decode ms/token; vision = forward ms; diffusion = ms per step;
training = ms per step. ⚠ = measured but contaminated by a since-diagnosed
bug; ✗ = established impossible; TODO = not yet measured (llama.cpp
column is informational and unscheduled). All values
tentative until the final sequential re-run with finished instrumentation.*

| # | benchmark | jax CPU | metaljax | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **363** | 137 | 148.7 | TODO |
| 2 | gemma4-12B bf16 | 346 (f32) | **101** | 58.3 ¹⁵ | 67.6 | TODO |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ¹⁴ | ⚠ 473 ¹⁶ | **17.0** | — | TODO |
| 4 | gemma4-E2B bf16 | 79.2 (bf16→f32) ¹³ | **28.9** | 10.5 ¹⁵ | — | — |
| 5 | Qwen3-8B bf16 | 219 (bf16→f32) ¹³ | **60.3** | 30.4 | smoke-verified ³ | TODO |
| 6 | Llama-3.1-8B bf16 | 206 (bf16→f32) ¹³ | **58.6** | 29.4 | TODO | TODO |
| 7 | gpt-oss-20b | TODO ⁴ | **220.4** (41.8 GB, dequant bf16) | **8.8** (13.8 GB, native MXFP4) | — | TODO |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | ✗ keras load ¹⁷ | **13.7** | — | TODO |
| 9 | R1-Distill-32B | ✗ 131 GB | ✗ keras load ¹⁷ | 131.8 | — | TODO |
| 10 | DeepSeek-V2-Lite (maxtext) | ⚠ 50–105 GB ⁶ | TODO ⁶ | — | — | — |
| 11 | Qwen3-0.6B (maxtext decode) | 89.5 | **15.8** (5.7×) ⁷ | — | — | — |
| 12 | Mixtral 8×7B bf16 | ✗ | ✗ keras load ¹⁷ (93 GB — not attempted, foregone) | TODO | — | TODO |
| 13 | gemma4-E2B keras-int4 (packed) | **67.5** (beats its bf16!) | ⚠ 339.5 @ **2.7 GB** ¹⁸ | (4-bit: see row 4 stacks) | — | — |
| 14 | maxtext qwix-int8 0.6B | 146 | **48.3** ᵛ | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 (maxtext; coherent) | ✗ blocked: needs quantized-matmul path ⁸ | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 965 | **248** (3.9×; b32 4.4×) | — | TODO | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ¹² | ✗ blocked ⁹ | TODO (mflux) | TODO | — |
| 18 | LoRA E2B train (ms/step) | 2141 | **417** ᵗ (losses agree) | — | TODO ¹⁰ | — |
| 19 | maxtext train 0.6B (loss) | 247.81 | **247.78** (eager ≡ compiled) ¹¹ | — | — | — |
| 20 | *aspirational* 235B-A22B 3-bit | ✗ | ✗ needs packed-quant storage | **28.0** (102.9 GB, load 12 s) | — | — |

**Splat-fix before/after (measured today):** Qwen3-8B 268→60.3 ms/tok
(143.6→16.4 GB); Llama-8B 228→58.6 (127→16.1 GB); gpt-oss 2090→220.4
(224→41.8 GB). All three now beat jax-CPU 3.4–3.7×.

ᵈ CPU cells run what XLA:CPU supports: weights load bf16, matmuls
upcast per-op (bf16→f32); the 12B row is full f32 (gemma-lib path).
ᵉ Per Oleg: attempt behind the memory guard; observed keras-CPU RSS is
~2.9× checkpoint → ~150 GB projected, expect a guard kill but measure.

**mlx-lm gap band (same Metal library underneath — the C++-rewrite
target):** bf16 dense decode 2.0–2.6× (Qwen3-8B 60.3 vs 30.4; Llama
58.6 vs 29.4; 12B 101 vs git-main-pending; 31B 363 vs 137); gpt-oss
25× (native-MXFP4 quantized_matmul + our dispatch, the two roadmap
items compounded). metaljax prefill trails ~6×; load ~20–30×
(mlx-lm mmaps quantized/bf16 weights directly).

ᶠ Released mlx-lm 0.31.3 cannot run gemma4_unified (12B) or the
E-series KV-sharing layout (E2B) at all; these two cells measured on
mlx-lm git main (2026-08-03 install). The 12B/31B gemma-lib decode improvements vs
earlier entries (189→101, 374→363) come from the dynamic-while
body-compile fix landing in the sampler's decode loop; old CPU 938
superseded by the uniform harness (cache length now matched).

ᵍ MoE DENSE-EXPERT GAP — the largest measured: keras/XLA lowers expert
dispatch densely (streams all 51.6 GB/token) → 473 ms/tok vs mlx-lm's
gathered 17.0 (27.8×). A 4B-active MoE decodes slower than dense 31B
on our path. Top C++-era item alongside quantized matmul.
ʰ keras load path (random-init before checkpoint overwrite +
conversion transients) exceeds the machine above ~50 GB checkpoints:
R1-32B (65 GB) jetsam-killed; Qwen3.6-35B (72 GB) entered swap-death
(196 GB footprint, killed by hand); Mixtral (93 GB) not attempted —
foregone. gemma-lib's streaming loader handled 62.6 GB fine, so the
fix is a streaming load for the keras path (ledger).
ⁱ Packed int4 memory saving IS real on metaljax (2.7 vs 10.2 GB —
the only sub-byte JAX path that keeps it), but decode pays 11.7× vs
bf16 (in-graph unpack re-materializes weights per matmul). XLA:CPU
fuses the same unpack into a small net WIN (67.5 vs 79.2 bf16).
Either mx.quantized_matmul mapping or unpack-fusion closes it.

## Footnotes

1. Both backends crashed on the sentencepiece SIGABRT (`import tensorflow`
   poisons pip sentencepiece — root-caused, shim landed in run_bench.py);
   re-run queued.
2. Pre-splat-fix numbers, contaminated by 23×-splat-constant retention
   (143.6 / 127 / 224 GB "active", swap thrash). Fix applied
   (splat constants broadcast from a 1-element buffer + dynamic-while
   bodies now compile); 0.6B evidence: decode 56 → 10.8 ms/tok, memory
   13.9× → 1.0×. These three rows re-run first after the gate.
3. torch-MPS adapter validated (greedy tokens ≡ torch-CPU, 32/32);
   timings deliberately deferred.
4. CPU projected ~126 GB (panic-adjacent); keras dequantizes the
   MXFP4-native repo to bf16 (~42 GB). Re-assess after splat fix.
5. Tokenizer EOS fix verified (DeepSeek removed keras's hardcoded
   `<|endoftext|>`); first generation run pending, serialized.
6. maxtext memory model: sparse path still wants 50–83 GB for a 16B MoE
   prefill. Rescope decision pending (candidate for drop; Qwen3.6-35B
   covers the MoE class).
7. FIXED (28ad2eb): was MLX 0.32 corrupting compiled graphs split
   across Metal command buffers at the 40 MB byte default; byte cap
   raised. 3/3 runs byte-identical to CPU at 15.9 ms/tok.
8. int8 is functionally exact on both backends (tokens verified) but a
   pessimization everywhere today: jax-CPU 1.7× slower than its bf16,
   metal 3.2× at decode and ~16× at prefill (the int64 outer-product
   materialization scales with batch×seq). Nobody has a fast int path on
   this hardware — the case for mapping onto `mx.quantized_matmul`.
9. Blocked on two independent walls, both fully diagnosed. (a) MEMORY:
   we lower attention unfused (matmul-softmax-matmul), so SD3.5's
   attention logits are ~12 GB/block at 1024² — peak live set ~90 GB in
   ANY execution mode (guard-killed compiled, eager, and at 20 steps
   even 512²: intermediates pin across steps under large command-buffer
   caps). Fix: map onto mx.fast.scaled_dot_product_attention (C++-era).
   (b) CORRECTNESS: with small command buffers MLX 0.32 corrupts
   compiled graphs (the footnote-7 bug), and diffusion's GB-sized
   kernels force small-kernel-count buffers at any byte cap that is
   memory-safe — the two constraints are unsatisfiable for this graph
   shape until MLX fixes the corruption. Evidence: 512²/2-step
   diagnostic at 16 GB cap = correct full-range image; every larger
   configuration black or guard-killed.
10. torch MPS SDPA has no backward kernel (falls back to math attention) —
    any torch training comparison must disclose this.
11. FIXED (52b90a2): the eager divergence was the command-buffer bug's
    THIRD face — ops=400 landed a buffer boundary that corrupted one
    RNG key in the init scan. ops now 800 (+2–3%); eager bit-identical
    to compiled, 1.4e-4 vs CPU. All three of the week's correctness
    mysteries were one MLX 0.32 bug via different budget alignments.
ᵗ tentative (pre-splat-fix agent run): official metal re-run
    guard-killed at 122 GB during the keras load transient (the
    documented initializer waste, footnote 17 ledger) — number stands
    until the keras streaming load lands.
12. keras's mixed-precision layers request the F16_F16_F32 dot
    algorithm, which XLA:CPU rejects (plain f16 dots work; the
    algorithm spec is an accelerator contract). A strip-workaround
    would enable a CPU reference; planned.
ᵛ CERTIFIED BENIGN (notes/int8-divergence-verdict.md): the token
    divergence vs int8-CPU is an exact logit tie on metal (14.5 vs
    14.5) at a step whose CPU margin is 7 bf16 ULPs = 1.3σ of the
    quantization noise; the s8 dot+dequant is bit-identical on real
    data; even CPU-int8 vs CPU-bf16 flips the same token. Timing valid
    (3.1× vs bf16 — the int64 cliff). NB token-stream equality is not
    a usable correctness criterion for quantized decode; use the
    logit-delta ladder.
13. CPU cells run what XLA:CPU supports: weights load bf16, matmuls
    upcast per-op (bf16→f32); the 12B row is full f32 (gemma-lib path).
14. 26B-A4B CPU attempt per Oleg, behind the memory guard: killed at
    34 GB RSS during load (projected ~150 GB at the observed 2.9×
    keras-CPU ratio).
15. Released mlx-lm 0.31.3 cannot run gemma4_unified (12B) or the
    E-series KV-sharing layout (E2B) at all; these two cells measured
    on mlx-lm git main (2026-08-03 install). The 12B/31B gemma-lib
    decode improvements vs earlier entries (189→101, 374→363) come
    from the dynamic-while body-compile fix landing in the sampler's
    decode loop; old CPU 938 superseded by the uniform harness.
16. MoE DENSE-EXPERT GAP — the largest measured: keras/XLA lowers
    expert dispatch densely (streams all 51.6 GB/token) → 473 ms/tok
    vs mlx-lm's gathered 17.0 (27.8×). A 4B-active MoE decodes slower
    than dense 31B on our path. Top C++-era item alongside quantized
    matmul.
17. keras load path (random-init before checkpoint overwrite +
    conversion transients) exceeds the machine above ~50 GB
    checkpoints: R1-32B (65 GB) jetsam-killed; Qwen3.6-35B (72 GB)
    entered swap-death (196 GB footprint); Mixtral (93 GB) not
    attempted — foregone. gemma-lib's streaming loader handled
    62.6 GB fine, so the fix is a streaming load for the keras path.
18. Packed int4 memory saving IS real on metaljax (2.7 vs 10.2 GB —
    the only sub-byte JAX path that keeps it), but decode pays 11.7×
    vs bf16 (in-graph unpack re-materializes weights per matmul).
    XLA:CPU fuses the same unpack into a small net WIN (67.5 vs 79.2
    bf16). Either mx.quantized_matmul mapping or unpack-fusion
    closes it.

## Bug ledger (found by this suite)

- **Splat-constant retention** (FIXED, d9d774e, gated): whole-shape
  splat constants materialized + pinned per executable; jax
  `random.normal` carries 23 full-weight-shape splats → keras models
  retained ~9× their weights. Predicted 143.7 GB vs measured 143.6.
- **Dynamic-while bodies never compiled** (FIXED, d9d774e, gated): LLM
  decode loops interpreted op-by-op → Python-dispatch-bound (the reason
  8B decode lost to CPU pre-fix).
- **MLX command-buffer corruption, three faces** (all worked around:
  28ad2eb bytes-floor, 0da62c0 bytes-ceiling, 52b90a2 ops alignment;
  upstream report pending): footnotes 7, 9, 11. Every finite budget is
  a lottery draw until MLX fixes it; the command-buffer tests pin the
  shipped values.
- **SD3.5 black image** (RESOLVED by 28ad2eb, same MLX bug): footnote 9.
- **sentencepiece SIGABRT** (worked around): footnote 1.
- **int8 dot_general int64 cliff** (known, measured): footnote 8.
- **MoE dense-expert lowering** (open, measured 27.8×): footnote 16.
- **keras load-path memory** (open, blocks ≥60 GB keras rows): footnote 17.
- **int4 unpack re-materialization on metal** (open, 11.7×): footnote 18.
