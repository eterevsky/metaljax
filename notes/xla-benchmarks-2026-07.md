# openxla/xla benchmark suite on metaljax (2026-07)

Suite: github.com/openxla/xla `xla/tools/benchmarks/` — HLO-text dumps +
registry + CI baselines. Single-device entries only; the llama/mixtral
1x8..64x8 shards need collectives and are out of scope. The registry's
GCS bucket (xla-benchmarking-temp) holds extra unregistered gemma3
variants (4b/12b, call + sample_loop) which we included.

## Pipeline

HLO text -> MHLO (`xla-translate --hlo-text-to-mlir-hlo
--hlo-import-all-computations --hlo-flatten-computation-args-result`,
bazel-built from the xla repo, ~12 min on the M5 Max) -> StableHLO
(jaxlib's `xc._xla.mlir.mhlo_to_stablehlo`) -> text -> compiled through
each backend's PJRT client via `client.compile_and_load` (see
scripts/run_stablehlo_bench.py). Identical seeded inputs on every
platform; wall time per execute with device sync (METALJAX_SYNC=1 on
metal — block_until_ready is a no-op there); correctness = outputs vs
the jax-CPU reference on the same inputs.

NB `xla-translate --hlo-to-stablehlo` wants a *proto*, not text (and
null-crashes on parse failure) — hence the two-step conversion.

## Results (ms per call, min over reps; M5 Max CPU / metaljax Metal / RTX 4090)

| benchmark                  | jax-CPU  | metaljax | 4090  | CI baseline |
|----------------------------|---------:|---------:|------:|-------------|
| gemma3_1b_flax_call        |     80.1 |     42.5 |  3.98 | L4 15 (device), x86-CI wall 3000 |
| gemma3_4b_flax_call        |    666.9 |     81.5 | 11.21 | — |
| gemma3_12b_flax_call       |   2187.9 |    172.7 | skip¹ | — |
| gemma2_2b_keras_jax        |    158.0 |     17.5 | 10.94 | B200 100, L4 1000, x86-CI wall 14000 |
| hlo_gemma4_2b_bf16         |    512.0 |     16.9 |  2.53 | (registry lists B200; no baseline yet) |
| nv_maxtext_1n1g train step |   101066 |   19405² | OOM³  | B200 302.2 (device) |

¹ 23.5 GB bf16 params > 24 GB VRAM (user rule: only what fits in 24G).
² metal runs this EAGERLY: mx.compile rejects the traced graph
  (unordered_map::at — unused inputs); engine falls back per the new
  broadened catch. Still 5.2x faster than CPU. Compiling around that
  is future work.
³ 18.2 GiB allocation on top of 8 GB params exceeds 24 GB VRAM.

sample_loop variants (1b/2.9ms-class on GPU vs 218-1778ms CPU) are
VACUOUS under synthetic inputs: the decode while-loop runs ~0
iterations, so they measure KV-cache passthrough (CPU pays a real
memcpy of the 2-27 GB state; metal/cuda alias buffers). All three
backends agree bit-exactly on them. Only the call variants carry
cross-platform signal here.

hlo_torax_iterhybrid_predictor_corrector_f64: MHLO->StableHLO
conversion itself fails in jaxlib (INVALID_ARGUMENT) — not runnable via
this pipeline on any backend; unregistered benchmark, left out.

## Correctness (vs jax-CPU, same seeded inputs)

- gemma2_2b, gemma4_2b: metal outputs BIT-EXACT (their checked outputs
  are small logit/score tensors). Wrong-seed probe fails loudly (err 7.0),
  so the checker is sensitive, not vacuous.
- gemma3 family: worst scale-normalized error 2.9-3.6%, concentrated in
  bf16 KV-cache outputs (2-4 bf16 ULPs after 26 layers); final logits
  <=2%. The 4090 vs CPU shows the SAME divergence class (3.5-4.2%) on
  these graphs, so this is cross-backend bf16 accumulation numerics —
  metal is inside the family spread, in fact tighter than cuda here.
- maxtext (metal, eager): single-execute outputs match CPU within 5%
  on all 39 outputs INCLUDING NaN placement (the train step genuinely
  NaNs under synthetic inputs — optimizer rsqrt of degenerate stats;
  CPU does the same). Re-executing with identical input buffers gives
  slightly different results on metal (inputs verified unmutated):
  nondeterministic accumulation order (atomic scatter-add class, as on
  jax-CUDA), amplified through the NaN-adjacent optimizer math. 4 of
  39 outputs flip their NaN pattern between runs; benchmark timing is
  unaffected.

MEASUREMENT LESSON: reference outputs and device runs must come from
the SAME input-generator version — changing gen_inputs' rng draw
pattern (f64->f32 sampling) mid-suite shifted every subsequent draw and
made maxtext look 157% wrong (a counter arg changed 7->6). Regenerate
refs after any generator change.

## Reading the baselines

Baseline config-ids encode the hardware: `_l4_` = NVIDIA L4 CI runner,
`_b200_` = B200, `_x86_` = x86 CPU runner. Values are loose CI
regression gates (30% margin), not tuned numbers — gemma2_2b "L4 1000ms"
vs our 4090's 10.9ms says gate, not measurement. Most useful anchors:
gemma3_1b L4 device time 15ms (4090: 4.0ms, metal: 42.5ms) and maxtext
B200 302ms (both consistent with hardware class).

## Engine bugs this suite flushed out (all fixed, 152 tests)

1. `dense<0xFF80> : tensor<bf16>` hex splats: the MLIR bindings return
   float(hex-integer) instead of reinterpreting bits, so -inf became
   65536.0 and every gemma3 softmax NaN'd (jax never emits the hex
   form; xla-translate does). bf16 now always takes the text decode
   path, and hex tokens decode as bit patterns for every float type
   (ml_dtypes bfloat16 has numpy kind 'V', not 'f').
2. `stablehlo.dot` (plain rank-2 form, HLO-imported only) implemented.
3. argmax/argmin multi-result reduce implemented (also unblocks texmo
   sampling); `stablehlo.reduce_precision` implemented (bf16/f16 grids
   + generic f32 mantissa rounding).
4. mx.compile IndexError (unused-input graphs) now falls back to eager
   instead of failing the executable.

## Takeaway

metaljax runs 6/6 runnable single-device suite benchmarks correctly.
Gap to 4090 on big dense forwards is ~7-10x (hardware class), CPU is
2-13x behind metal. The M5's 128 GB unified memory runs gemma3_12b
(23.5 GB weights) where the 24 GB 4090 cannot — the qualitative win of
the platform.
