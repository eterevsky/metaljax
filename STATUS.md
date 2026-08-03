# Model benchmark suite — status

*(Harness and manifest: [scripts/model_bench/](scripts/model_bench/).)*

*Last updated: 2026-08-03 — FINAL for the 0.11.2 baseline: all cells
from the sequential release-gate run at shipped defaults (ops=800,
bytes=512 MB), token agreement audited (4 bf16 rows exact; 2 certified
tie-flips within 1–2 bf16 ULPs, see footnote 21; quantized rows per the
int8 certification). Headline metric per cell: LLM rows = warm decode
ms/token; vision = forward ms; diffusion = ms/step; training = ms/step.
✗ = established impossible (with the measured reason).*

| # | benchmark | jax CPU | metaljax | mlx-lm | torch-MPS | llama.cpp |
|---|---|---|---|---|---|---|
| 1 | gemma4-31B bf16 | ✗ f32=123 GB | **350** | 137 | 148.7 | 111.2 ²⁰ |
| 2 | gemma4-12B bf16 | 315 (f32) | **97.1** | 58.3 ¹⁵ | 67.6 | 44.2 ²⁰ |
| 3 | gemma4-26B-A4B (MoE) | ✗ guard-killed @34 GB ¹⁴ | **284** ¹⁶ | **17.0** | — | 7.9 (Q4 QAT) ²⁰ |
| 4 | gemma4-E2B bf16 | 67.4 (bf16→f32) ¹³ | **29.5** ²¹ | 10.5 ¹⁵ | — | — |
| 5 | Qwen3-8B bf16 | 209 (bf16→f32) ¹³ | **60.4** | 30.4 | 38.1 | 15.7 (Q8) ²⁰ |
| 6 | Llama-3.1-8B bf16 | 200 (bf16→f32) ¹³ | **57.3** ²¹ | 29.4 | 35.5 | 15.4 (Q8) ²⁰ |
| 7 | gpt-oss-20b | ✗ ⁴ | **39.6** (23.9 GB, native MXFP4) | **8.8** (13.8 GB, native MXFP4) | — | 6.7 (native MXFP4) ²⁰ |
| 8 | Qwen3.6-35B-A3B (MoE) | ✗ 144 GB | ✗ warmup transients ¹⁷ | **13.7** | — | — |
| 9 | R1-Distill-32B | ✗ 131 GB | ✗ warmup transients ¹⁷ | 131.8 | — | — |
| 10 | DeepSeek-V2-Lite (maxtext) | ✗ needs 50–105 GB ⁶ | ✗ guard-killed @122 GB ⁶ | **10.6** | — | — |
| 11 | Qwen3-0.6B (maxtext decode) | 89.7 | **16.0** ⁷ | — | — | — |
| 12 | Mixtral 8×7B bf16 | ✗ | ✗ keras load ¹⁷ | **52.8** (93.4 GB) | — | — |
| 13 | gemma4-E2B keras-int4 (packed) | **67.8** ¹⁸ | 85.0 @ 2.7 GB ¹⁸ | — | — | — |
| 14 | maxtext qwix-int8 0.6B | 143.4 | **48.5** ²² | — | — | — |
| 15 | *qwix-int8 Qwen3-8B* | 2118 | ✗ MLX command-buffer bug ⁸ | — | — | — |
| 16 | SigLIP 2 (fwd b1 ms) | 533 | **93.4** | — | 29.8 (b32: 591) | — |
| 17 | SD 3.5 Large (ms/diff-step) | ✗ ¹² | ✗ blocked ⁹ | ✗ ¹⁹ | 654 @512², 2998 @1024² ¹⁹ | — |
| 18 | LoRA E2B train (ms/step) | 2048 | **407** | — | 135.6 ¹⁰ | — |
| 19 | maxtext train 0.6B (ms/step) | 1402 | **440** ¹¹ | — | — | — |
| 20 | *aspirational* 235B-A22B 3-bit | ✗ | ✗ needs packed-quant storage | **28.0** (102.9 GB, load 12 s) | — | — |

**Splat-fix before/after (measured today):** Qwen3-8B 268→60.3 ms/tok
(143.6→16.4 GB); Llama-8B 228→58.6 (127→16.1 GB); gpt-oss 2090→220.4
(224→41.8 GB). All three now beat jax-CPU 3.4–3.7×.

**mlx-lm gap band (same Metal library underneath — the C++-rewrite
target):** bf16 dense decode 1.7–2.6× (Qwen3-8B 60.4 vs 30.4; Llama
57.3 vs 29.4; 12B 97.1 vs 58.3; 31B 350 vs 137); gpt-oss 25×
(native-MXFP4 quantized_matmul + our dispatch compounded); MoE 16.7×
(284 vs 17.0). llama.cpp leads mlx-lm a further ~1.25× on bf16 — the
kernel frontier. metaljax prefill trails ~6×; load ~20–30×.

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
4. CPU: dequantized-bf16 working set projected ~126 GB (panic
   territory) — established infeasible; keras dequantizes the
   MXFP4-native repo to bf16 (~42 GB).
5. Tokenizer EOS fix verified (DeepSeek removed keras's hardcoded
   `<|endoftext|>`); first generation run pending, serialized.
6. maxtext memory model: sparse path still wants 50–83 GB for a 16B MoE
   prefill. Rescope decision pending (candidate for drop; Qwen3.6-35B
   covers the MoE class).
7. FIXED (28ad2eb): was MLX 0.32 corrupting compiled graphs split
   across Metal command buffers at the 40 MB byte default; byte cap
   raised. 3/3 runs byte-identical to CPU at 15.9 ms/tok.
8. int8 is functionally exact on both backends (tokens verified). The
   old int64 outer-product pessimization was fixed 2026-08-03 (chunked
   exact f32 dot, fdc7cde): 0.6B decode 48.5 → 31.8, prefill 1255 →
   37.7. The 8B row then ran for the first time (~330 ms/tok, no OOM)
   but produces garbage: the MLX 0.32 command-buffer split corruption
   at 8B scale. Single-variable proof: byte budget alone flips the
   outcome — 512 (default) corrupts decode replays (bf16 AND int8;
   first call clean, replays differ per process; correct under
   METALJAX_COMPILE=0), 2048 gives correct output with a benign KV
   curve — but 2048 has NO stability margin: an 8B load at 2048
   kernel-panicked the machine (watchdog timeout, wired-memory class)
   after an identical run had succeeded. Eager-mode mitigations at the
   default budget all ballooned (67–109 GB at load: uncompiled bodies
   pin the whole lazy load DAG) and the final attempt kernel-panicked
   the machine a second time — full attempt ledger in
   notes/mlx-command-buffer-split.md (2026-08-03 addendum). 8B-class
   maxtext is EMBARGOED on this machine; the row waits for either an
   engine-side eval-forcing mode (small-scale-validated first) or the
   MLX upstream fix. Repro:
   notes/data/qwen3_8b_prefill_36layer.mlir (0.3 s).
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
10. torch MPS SDPA has no backward kernel — substantiated by autograd
    node inspection (math fallback), disclosed in the record. Loss
    series not comparable across stacks (different preprocessing);
    step cost is the comparison.
11. Correctness FIXED (52b90a2): the eager loss divergence was the
    command-buffer bug's THIRD face — ops=400 landed a buffer boundary
    that corrupted one RNG key in the init scan; ops now 800 (+2–3%).
    Validation: eager loss bitwise ≡ compiled (247.7775), 1.4e-4 vs
    CPU (247.8117). Step TIMING not yet captured — comes with the
    final sequential run.
12. keras's mixed-precision layers request the F16_F16_F32 dot
    algorithm, which XLA:CPU rejects (plain f16 dots work; the
    algorithm spec is an accelerator contract). A strip-workaround
    would enable a CPU reference; planned.
13. CPU cells run what XLA:CPU supports: weights load bf16, matmuls
    upcast per-op (bf16→f32); the 12B row is full f32 (gemma-lib path).
14. 26B-A4B CPU: the model cannot fit — f32 26B is ~104 GB of weights
    alone, and the observed keras-CPU load inflation (2.9×) projects a
    ~150 GB peak on a 128 GB machine. The guard killed the load as soon
    as the growth trajectory made that projection conclusive (34 GB and
    climbing); the alternative was the Qwen3.6-style swap-death (196 GB
    footprint) that froze the machine.
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
17. The keras LOAD ceiling is fixed (streaming loader, 30c9717: init
    never materializes; E2B peak 25→9.5 GB; Qwen3.6-35B ports all
    1026 weights, ~70 GB resident). What remains lethal is the phase
    AFTER load on 60 GB+ models: warmup/compile transient ramps drove
    swap to 9 GB (R1, guard-killed at 95 GB budget) and a chained
    second load onto that degraded system caused kernel panic #6.
    mlx-lm holds 93.4 GB resident fine — the danger is our stack's
    allocation ramps, not static residency. Rows 8/9 (and 12) wait
    for the eager-phase eval-forcing work (TASKS), same as rows
    10/15; big-run moratorium >50 GB expected claim until it lands,
    and big runs are never chained without a system-recovered check.
18. Packed int4 memory saving IS real on metaljax (2.7 vs 10.2 GB —
    the only sub-byte JAX path that keeps it). XLA:CPU fuses the
    in-graph unpack into a small net WIN (67.5 vs 79.2 bf16). The
    mx.quantized_matmul recognizer (5fd6b2a) + interleaved-group
    K-permutation (f45bbbe) took metal 336 → 241 → 85.0 ms/tok
    (all 777 quantized dots fuse; decode body compiles). The residual
    1.25× vs CPU is the batch-1 GEMV kernel-launch floor (~2k Metal
    dispatches/token; XLA:CPU launches nothing) — C++-era.
19. torch SD3.5 via the ungated diffusers mirror
    adamo1139/stable-diffusion-3.5-large-ungated @5d868ff (official
    repo is gated); coherent images verified at both resolutions.
    MLX cell: mflux is Flux-only; DiffusionKit supports the SD3 family
    but 3.5-Large weights are gated in its formats — no ungated MLX
    path exists, cell closed as not-runnable.
20. llama.cpp build 221f0f63 (past the Gemma4 cutoff), llama-bench
    -p 51 -n 128 -r 5, all-Metal, reproduced within 4% on two passes;
    per-provider GGUF pins in README_llamacpp.md. Q8/Q4 marked where
    no bf16 GGUF exists. Dense rows all pin to 439–555 GB/s effective
    bandwidth (the bandwidth-bound signature); llama.cpp leads even
    mlx-lm ~1.25x on bf16 — the kernel frontier on this hardware.
    Deferred rows (35B/R1/Mixtral llama.cpp cells) dropped as
    redundant with the covered comparison classes.
21. Release-gate token audit: greedy streams metal-vs-CPU are exact on
    12B, Qwen3-8B, E2B-int4, and the maxtext rows; E2B-bf16 and
    Llama-8B diverge at their FIRST generated token via certified
    tie-flips — competing logits within 1–2 bf16 ULPs (Llama: exactly
    tied 11.875/11.875 on metal), the same benign class as the int8
    certification. gpt-oss/26B/31B have no CPU counterpart (recorded
    only).
22. CERTIFIED BENIGN (notes/int8-divergence-verdict.md): the token
    divergence vs int8-CPU is an exact logit tie on metal (14.5 vs
    14.5) at a step whose CPU margin is 7 bf16 ULPs = 1.3σ of the
    quantization noise; the s8 dot+dequant is bit-identical on real
    data; even CPU-int8 vs CPU-bf16 flips the same token. Timing valid
    (3.1× vs bf16 — the int64 cliff). NB token-stream equality is not
    a usable correctness criterion for quantized decode; use the
    logit-delta ladder.

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
