# Bonsai coverage probe (2026-09-01) — off-table, decision pending

`jax-ml/bonsai` (minimal pure-JAX/Flax-NNX model implementations, direct HF
safetensors, zero Pallas) run against the 0.11.7 release binary
(`frozen-0117-combined-c0ed1a10`, pinned) as a lowering-coverage probe — an
independent JAX spelling of popular models that neither of our harnesses
(maxtext, keras-hub) exercises. Explicitly NOT table material: informal
single-shot f32 timings, Python decode loops. Bonsai pinned at `f866d0f`;
raw logs, repros and the A/B in
`~/.cache/metaljax-bench/logs/bonsai-coverage/`.

## Results — all four models exact vs jax-CPU, zero declines

| model | vs jax-CPU | metal | CPU | ratio |
|---|---|---:|---:|---:|
| qwen3-0.6B (host-upload bug fixed) | exact 64/64 tokens | 14.46 ms/tok | 199.31 | 13.8× |
| qwen3-0.6B (as shipped) | exact 64/64 | 2477.36 ms/tok | 199.31 | 0.08× |
| mamba2-130m | exact 64/64 tokens | 13.37 ms/tok | 54.79 | 4.1× |
| resnet-50 | top-1+top-5 exact (Δlogit 1.1e-5) | 9.23 ms/fwd | 66.55 | 7.2× |
| vit-base-224 | top-1+top-5 exact (Δlogit 1.1e-5) | 24.27 ms/fwd | 158.56 | 6.5× |

Decode figures per step at batch 2. gemma3 and dinov3 skipped (HF
`gated=manual`, no token on this machine — a gating issue, not size).
Zero fallbacks, zero unsupported ops, zero retries in any run.

## Findings, ranked

1. **OURS — the norm recognizer never fires on flax NNX normalization.**
   flax 0.12.8 emits RMSNorm as LayerNorm-with-a-literal-zero-mean: the
   graph contains `subtract(x, broadcast(0.0))` and the weight-apply walk
   bails. 251 norms missed in one process (226 qwen3 + 25 vit) vs 49
   matched in mamba2 (which hand-writes its norm) — same engine, same run,
   outcome decided purely by spelling. Hits `nnx.LayerNorm` too. Measured
   cost on a shape-matched norm-bound synthetic (113 norms, d=1024):
   **1.39×**. Not fixed (probe brief); minimal repro + HLO saved in the
   logs dir. → Candidate recognizer extension (same matcher the 0.11.7
   norm work rewrote; the zero-mean subtract is one more transparent hop).
2. **UPSTREAM — bonsai keeps params on the host.**
   `safe_open(framework="numpy")` + a `device_put` gated on a sharding
   existing, so with no mesh all 311 leaves stay numpy and the jitted
   `forward` re-uploads the whole model every decode step
   (`ingested=86017MB` over 32 steps). One `device_put` → 2477 → 14.46
   ms/tok. Would cripple CUDA/TPU identically. (Our memory governor's
   ingest counter is what named it.) Also: bonsai's own
   `qwen3/tests/run_model.py` is broken at HEAD (`model.init_cache` vs
   module-level `init_cache`).
3. **POSITIVE — sdpa generalizes to two unseen attention spellings**:
   bonsai's 5-D GQA einsum `BTKGH,BSKH->BTSKG` (28/28 layers) and flax's
   stock `nnx.MultiHeadAttention` (12/12).
4. **POSITIVE — exotic paths lower exactly**: mamba2's SSM mix (4-operand
   einsum, 6-D contractions, segsum, depthwise conv with cache) and the
   first convnet evidence we have — ResNet-50's 53 convs + 53 BatchNorms
   with zero narration remarks.

## Verdict + decision menu (Oleg)

Agent's verdict: **yes as a recurring coverage probe, no as a benchmark
row** — cheapest third-party JAX surface available (one venv at
`venvs/bonsai`, no conversion), produced findings our own harnesses
structurally cannot (they are the spellings we already tuned against);
but unmaintained, broken entry point, non-comparable timings.

Options on the table:
- (a) Extend the norm recognizer to the flax-NNX zero-mean spelling
  (repro ready; queue behind the row-11 attention item).
- (b) Adopt bonsai@f866d0f as a recurring coverage probe (e.g. re-run
  against the narration whenever recognizers change).
- (c) File the two upstream bonsai issues (host-params device_put, broken
  entry point) — drafts can be prepared; filing is Oleg's.
- (d) Re-run gemma3/dinov3 with an HF token if that coverage is wanted.
