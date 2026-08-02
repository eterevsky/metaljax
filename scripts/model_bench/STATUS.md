# Model benchmark suite — status

*Last updated: 2026-08-02 (night — post splat-fix re-measurements). Headline metric per cell: LLM rows =
warm decode ms/token; vision = forward ms; diffusion = ms per step;
training = ms per step. ⚠ = measured but contaminated by a since-diagnosed
bug; ✗ = established impossible; TODO = not yet measured. All values
tentative until the final sequential re-run with finished instrumentation.*

| # | benchmark | jax CPU | metaljax | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **374** | TODO | TODO | TODO |
| 2 | gemma4-12B bf16 | 938 (f32) | **189** (bf16) / 254 (f32) | TODO | TODO | TODO |
| 3 | gemma4-26B-A4B (MoE) | queued (try)ᵉ | TODO | TODO | — | TODO |
| 4 | gemma4-E2B bf16 | 79.2 (bf16→f32)ᵈ | **28.9** | TODO | — | — |
| 5 | Qwen3-8B bf16 | 219 (bf16→f32)ᵈ | **60.3** | TODO (4-bit cached) | smoke-verified ³ | TODO |
| 6 | Llama-3.1-8B bf16 | 206 (bf16→f32)ᵈ | **58.6** | TODO (4-bit cached) | TODO | TODO |
| 7 | gpt-oss-20b | TODO ⁴ | **220.4** (41.8 GB) | TODO | — | TODO |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | TODO | TODO | — | TODO |
| 9 | R1-Distill-32B | ✗ 131 GB | TODO ⁵ | TODO | — | TODO |
| 10 | DeepSeek-V2-Lite (maxtext) | ⚠ 50–105 GB ⁶ | TODO ⁶ | — | — | — |
| 11 | Qwen3-0.6B (maxtext decode) | 87 | **BROKEN ⁷** (82 w/o compile) | — | — | — |
| 12 | Mixtral 8×7B bf16 | ✗ | TODO (93 GB, hosting ok) | TODO | — | TODO |
| 13 | gemma4-E2B keras-int4 (packed) | TODO | TODO | TODO | — | TODO |
| 14 | maxtext qwix-int8 0.6B | ~ | ⚠ 308 vs 96 bf16 (16×) ⁸ | — | — | — |
| 14b | *qwix-int8 Qwen3-8B* | — | ✗ blocked: needs quantized-matmul path ⁸ | — | — | — |
| 15 | SigLIP 2 (fwd b1 ms) | 597 | **91** (6.6×) | — | TODO | — |
| 16 | SD 3.5 Large (ms/diff-step) | ✗ f16 dots | ⚠ 8577, BLACK IMAGE ⁹ | TODO (mflux) | TODO | — |
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
8. The int64-materialization cliff, measured: int8 decode functionally
   exact but 16× slower than bf16 — the case for mapping dequant+matmul
   onto `mx.quantized_matmul` in the C++-era work.
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
