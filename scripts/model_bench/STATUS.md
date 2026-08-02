# Model benchmark suite — status

*Last updated: 2026-08-03 (block 4: big models, MoE, packed int4). Headline metric per cell: LLM rows =
warm decode ms/token; vision = forward ms; diffusion = ms per step;
training = ms per step. ⚠ = measured but contaminated by a since-diagnosed
bug; ✗ = established impossible; TODO = not yet measured. All values
tentative until the final sequential re-run with finished instrumentation.*

| # | benchmark | jax CPU | metaljax | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **363** | 137 | TODO | TODO |
| 2 | gemma4-12B bf16 | 346 (f32) | **101** | 58.3 ᶠ | TODO | TODO |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ᵉ | ⚠ 473 ᵍ | **17.0** | — | TODO |
| 4 | gemma4-E2B bf16 | 79.2 (bf16→f32)ᵈ | **28.9** | 10.5 ᶠ | — | — |
| 5 | Qwen3-8B bf16 | 219 (bf16→f32)ᵈ | **60.3** | 30.4 | smoke-verified ³ | TODO |
| 6 | Llama-3.1-8B bf16 | 206 (bf16→f32)ᵈ | **58.6** | 29.4 | TODO | TODO |
| 7 | gpt-oss-20b | TODO ⁴ | **220.4** (41.8 GB, dequant bf16) | **8.8** (13.8 GB, native MXFP4) | — | TODO |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | ✗ keras load ʰ | **13.7** | — | TODO |
| 9 | R1-Distill-32B | ✗ 131 GB | ✗ keras load ʰ | 131.8 | — | TODO |
| 10 | DeepSeek-V2-Lite (maxtext) | ⚠ 50–105 GB ⁶ | TODO ⁶ | — | — | — |
| 11 | Qwen3-0.6B (maxtext decode) | 87 | **BROKEN ⁷** (82 w/o compile) | — | — | — |
| 12 | Mixtral 8×7B bf16 | ✗ | ✗ keras load ʰ (93 GB — not attempted, foregone) | TODO | — | TODO |
| 13 | gemma4-E2B keras-int4 (packed) | **67.5** (beats its bf16!) | ⚠ 339.5 @ **2.7 GB** ⁱ | (4-bit: see row 4 stacks) | — | — |
| 14 | maxtext qwix-int8 0.6B | 146 (vs 87 bf16) | ⚠ 308 (vs 96 bf16) ⁸ | — | — | — |
| 14b | *qwix-int8 Qwen3-8B* | 2118 (maxtext; coherent) | ✗ blocked: needs quantized-matmul path ⁸ | — | — | — |
| 15 | SigLIP 2 (fwd b1 ms) | 597 | **91** (6.6×) | — | TODO | — |
| 16 | SD 3.5 Large (ms/diff-step) | ✗ keras requests F16_F16_F32 dot algorithm, unsupported on CPU (strip-workaround planned) | ⚠ 8577, BLACK IMAGE ⁹ | TODO (mflux) | TODO | — |
| 17 | LoRA E2B train (ms/step) | 3287 | **417** (7.9×, losses agree) | — | TODO ¹⁰ | — |
| 18 | maxtext train 0.6B (loss) | 228.42 | ⚠ mismatch ¹¹ | — | — | — |
| 19 | *aspirational* 235B-A22B 3-bit | ✗ | ✗ needs packed-quant storage | TODO (103 GB, fits) | — | — |

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
7. OPEN BUG: bf16 + `mx.compile` → nondeterministic garbage tokens
   (different every run); `METALJAX_COMPILE=0` is token-identical to
   CPU; f32 unaffected; MSL not involved. Under exclusive-machine
   investigation.
8. int8 is functionally exact on both backends (tokens verified) but a
   pessimization everywhere today: jax-CPU 1.7× slower than its bf16,
   metal 3.2× at decode and ~16× at prefill (the int64 outer-product
   materialization scales with batch×seq). Nobody has a fast int path on
   this hardware — the case for mapping onto `mx.quantized_matmul`.
9. Timings real, output all-zeros at every resolution. Stage-by-stage NaN
   diagnostic ready. No CPU reference possible (XLA:CPU rejects the
   preset's f16 dots).
10. torch MPS SDPA has no backward kernel (falls back to math attention) —
    any torch training comparison must disclose this.
11. First-step loss: CPU 228.4169, metal-compiled 228.3945,
    metal-uncompiled 191.2499 — inconsistent; triage queued behind bug ⁷.

## Bug ledger (found by this suite)

- **Splat-constant retention** (FIXED, d9d774e, gated): whole-shape
  splat constants materialized + pinned per executable; jax
  `random.normal` carries 23 full-weight-shape splats → keras models
  retained ~9× their weights. Predicted 143.7 GB vs measured 143.6.
- **Dynamic-while bodies never compiled** (FIXED, d9d774e, gated): LLM
  decode loops interpreted op-by-op → Python-dispatch-bound (the reason
  8B decode lost to CPU pre-fix).
- **bf16 mx.compile garbage** (open): footnote 7.
- **SD3.5 black image** (open): footnote 9.
- **sentencepiece SIGABRT** (worked around): footnote 1.
- **int8 dot_general int64 cliff** (known, measured): footnote 8.
- **MoE dense-expert lowering** (open, measured 27.8×): footnote ᵍ.
- **keras load-path memory** (open, blocks ≥60 GB keras rows): footnote ʰ.
- **int4 unpack re-materialization on metal** (open, 11.7×): footnote ⁱ.
