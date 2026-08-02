# int8 (qwix) greedy-decode divergence, metal vs jax-CPU — certification

**Verdict: BENIGN.** The metal/CPU difference in the `maxtext-qwix-int8` row is
rounding noise at the *declared precision of the row itself* (bf16 activations +
per-tensor dynamic int8), not a wrong op. Every op-level check comes out exact or
1-ULP, the cross-backend delta is the same size as the quantization error the row
opts into, and the token that flips is decided by an **exact tie** on metal. No
correctness bug; nothing to fix for 0.11.2.

Date 2026-08-02, metaljax main (editable, post-0.11.0), jax/jaxlib 0.11.0,
mlx 0.32.0, MaxText @ `~/.cache/metaljax-bench/maxtext/repo`, Qwen3-0.6B.

---

## 1. What diverges

`maxtext-qwix-int8` (`use_qwix_quantization=true quantization=int8`,
`weight_dtype=bfloat16`, activations bf16), prompt `"The capital of France is"`
(5 real tokens, `PREFILL_LEN=64`), greedy, prefill token + 24 generate steps:

```
jax-CPU : ' Paris. The capital of France is also the capital of the French Republic. The capital of France is also the capital of the'
metal   : ' Paris. The capital of France is also the capital of the country of the country of the country of the country of the country'
```

First divergence: **output token index 12 = generate step 11**
(CPU `8585 ' French'`, metal `3146 ' country'`). Both continuations are degenerate
greedy loops of a 0.6B model; neither is "more correct" than the other.

## 2. Logits at the divergent step

MaxText's decode logits land on the **bf16 grid** (the final dense runs in bf16;
`cast_logits_to_fp32` only casts afterwards). 1 ULP = 0.0625 at |x| ∈ [8,16).

Natural runs, step 11 (each backend on its own state):

| rank | jax-CPU | metal |
|---|---|---|
| 1 | `8585 ' French'` **14.6875** | `3146 ' country'` **14.5** |
| 2 | `5429 ' Republic'` 14.25 | `8585 ' French'` **14.5**  ← exact tie |
| 3 | `7513 ' European'` 14.1875 | `7513 ' European'` 14.1875 |
| 4 | `3146 ' country'` 13.4375 | `5429 ' Republic'` 14.0625 |

* CPU top1–top2 gap **0.4375** (7 ULP); metal top1–top2 gap **0.000000**.
* metal's winner is CPU's rank 4; CPU's winner is metal's rank 2 — *tied for
  first*. `argmax` breaks the tie by lowest index (3146 < 8585). The flip is
  literally a tie-break, not a preference.
* Per-token delta (cpu − metal): `8585` +0.1875 (3 ULP), `3146` −1.0625 — both
  inside the step's own delta distribution (p99 = 2.03).

## 3. Noise scale (single-step replays: identical input state on both sides)

The decode state at steps 8–11 was pickled from the CPU run and **loaded onto
both backends**, so these numbers are pure one-step compute differences with zero
state drift. "centred" = mean removed (a constant logit offset is invisible to
softmax/argmax, so the centred figure is the decision-relevant one).

| comparison | rms | rms centred | max centred |
|---|---|---|---|
| **int8, metal vs CPU** | 0.26 – 0.43 | **0.246 – 0.336** | 1.23 – 1.79 |
| int8 vs unquantized, CPU only (= the quantization error this row opts into) | 0.30 – 0.34 | 0.220 – 0.294 | 1.09 – 1.44 |
| bf16-activations vs f32-activations, CPU only (= declared-precision floor) | 0.29 – 0.45 | — | — |
| **bf16 model, metal vs CPU** (known-benign baseline; tokens identical) | 0.045 – 0.091 | 0.044 – 0.059 | 0.22 – 0.29 |

The cross-backend int8 delta is **1.1× the CPU-only quantization error** and
**0.8× the bf16 rounding floor**. The step-11 gap of 0.4375 is **1.3 σ** of the
centred cross-backend noise — a flip there is the expected outcome, not an
anomaly.

Sensitivity probe on CPU: bumping the 1024 values of one layer-0 key vector by
**1 bf16 ULP each** moves the step-11 logits by rms **0.368** / max 1.75 (same
order as the cross-backend delta). Bumping a *single* element by 1 ULP, or one
weight element by 1 ULP, changes nothing at all (bitwise-identical logits) — the
model's response to minimal perturbations spans exactly this range.

## 4. Where the difference enters (per-layer, identical input state)

Relative RMS delta of the K/V vector written at step 11, per decoder layer:

| curve | L0 | L1 | L7 | L13 | L27 | mean |
|---|---|---|---|---|---|---|
| metal vs CPU int8 — K | 3.2e-3 | 1.4e-2 | 4.7e-2 | 8.0e-2 | 5.3e-2 | 6.7e-2 |
| metal vs CPU int8 — V | **0.0** | 1.0e-1 | 1.9e-1 | 4.0e-2 | 1.5e-1 | 1.5e-1 |
| int8 vs unquantized (CPU) — K / V | 8.5e-3 / 3.3e-2 | | | | | 6.0e-2 / 1.3e-1 |
| bf16act vs f32act (CPU) — K / V | 5.3e-3 / 1.8e-2 | | | | | 7.3e-2 / 1.6e-1 |
| metal vs CPU **bf16 model** — K / V | 0.0 / 0.0 | | | | | 9.1e-3 / 1.7e-2 |

* **No jump at any layer.** The metal-vs-CPU curve lies on top of the two
  known-benign reference curves (quantization error; bf16-vs-f32 activations)
  for all 28 layers, and the unquantized model's cross-backend curve is ~7×
  lower — i.e. the amplification is contributed by the *quantizer*, not by the
  backend.
* **Layer-0 value cache is bit-identical (0 / 1024 elements).** The value path is
  `embedding → s8×s8→s32 dot → dequant`, so the int8 dot and its dequantisation
  reproduce exactly on real data, confirming the earlier isolated-shape tests.
* **Layer-0 key differs in 244 / 1024 elements, median 1 bf16 ULP, p90 2 ULP**
  (elements above 2 ULP have |x| ≈ 0.12 vs the vector median 0.68 — cancellation,
  as expected). The key-only tail is QK-RMSNorm + RoPE, i.e. an f32 reduce/rsqrt
  and sin/cos multiply-add: a 1-ULP-class elementwise difference, which is the
  representation's own granularity and cannot be called wrong.

## 5. Why int8 amplifies 5× what bf16 does (mechanism, measured)

qwix quantizes **activations dynamically**: `scale = absmax/127` computed in the
array dtype (bf16), then `round(x/scale)` (round-half-to-even) → int8. That is a
*discontinuous* function of the input. Measured on CPU, bumping **one** element
(the row absmax) by 1 bf16 ULP:

```
scale relative change 3.98e-3 (= 1 bf16 ULP)
int8 codes changed    14.1 % of elements (mean over 5 trials), max |Δcode| = 2
```

So any 1-ULP upstream difference is converted, within one layer, into a
full quantization-noise-scale difference — exactly what the table in §3 shows
(int8 cross-backend noise ≈ the quantization error itself, while the unquantized
model's cross-backend noise stays 5–7× lower). This is inherent to per-tensor
dynamic quantization and would happen between any two implementations differing
in the last bit anywhere, including two XLA versions.

**The quantize path itself is bit-identical between backends.** qwix
`quantize` (bf16 divide → round-half-even → int8 clip) on 4.19M gaussian bf16
values *and* on 16 320 adversarial values sitting exactly on the `.5` rounding
boundary:

```
gauss/per-tensor  : 0 / 4194304 code mismatches, scale bit-identical
gauss/channelwise : 0 / 4194304
halfway/per-tensor: 0 / 16320    (exact-.5 inputs, both channelwise variants)
```

## 6. Determinism / the compile path is not involved

* jax-CPU rerun: logits **bitwise identical**, tokens identical.
* metal rerun: logits **bitwise identical**, tokens identical.
* metal `METALJAX_COMPILE=0` (eager) vs compiled: logits **bitwise identical** at
  all four replayed steps (max delta 0.0000) *and* over the whole 24-step natural
  run, same tokens.

So this is not the 0.11.1 MLX command-buffer bug and not an `mx.compile` issue —
the eager interpreter and the fused-graph path agree to the bit.

## 7. Context: step 11 is a genuine coin toss

Only 3 of the 24 CPU decision points have a top1–top2 gap ≤ 0.5 (steps 0: 0.25,
4: 0.25, 11: 0.4375). Step 11 is the first one deep enough into the sequence for
accumulated rounding to reach it. Four legitimate implementations of *the same
model* give three different answers there:

| run | step-11 pick |
|---|---|
| jax-CPU int8 | `8585 ' French'` |
| metal int8 | `3146 ' country'` (exact tie with `' French'`, index tie-break) |
| jax-CPU bf16 (unquantized) | `5429 ' Republic'` |
| metal bf16 (unquantized) | `5429 ' Republic'` |
| metal int8, replayed from the CPU state | `5429 ' Republic'` |

and the unquantized model replayed on the int8 decode state is itself **exactly
tied** between `' Republic'` and `' French'` (14.9375 vs 14.9375 on CPU, 14.875 vs
14.875 on metal). Quantization alone already flips this token on CPU: jax-CPU
int8 and jax-CPU bf16 diverge at the *same* output index 12.

## 8. Certification (for STATUS)

> The `maxtext-qwix-int8` greedy-decode divergence between metaljax-metal and
> jax-CPU (Qwen3-0.6B, identical for 12 tokens, then `' country'` vs `' French'`)
> is certified **benign numerics, not a correctness bug**. With the decode state
> held identical on both backends, the one-step logit difference is rms 0.25–0.34
> (mean-removed) — 1.1× the int8 quantization error the row opts into and 0.8× the
> bf16-vs-f32 activation rounding floor, versus rms 0.05 for the same model
> unquantized. The flipped token sits on an exact tie on metal (both candidates
> 14.5, argmax breaks by index) and on a 7-ULP gap on CPU, i.e. 1.3 σ of the noise;
> the unquantized model is itself exactly tied between the same two tokens. The
> earliest measurable difference is 1 bf16 ULP (median, 244/1024 elements) in the
> layer-0 key after QK-norm/RoPE, while the layer-0 value cache — the pure
> s8×s8→s32 dot plus dequant — is bit-identical, and qwix's quantizer produces
> bit-identical int8 codes and scales on both backends including exact `.5`
> boundaries. Per-layer growth is smooth and tracks the two known-benign
> perturbation curves with no jump. Both backends are bitwise deterministic and
> metal's eager and compiled paths agree bitwise. Cause: per-tensor dynamic int8
> quantization is discontinuous — a measured 1-ULP change of one activation
> element flips 14% of the int8 codes — so any last-bit difference is amplified to
> the quantization-noise scale, in any implementation pair.

## 9. Repro

Scratch harness (outside the repo, as agreed):
`~/.cache/metaljax-bench/logs/int8div/` —
`logitdump.py` (drives MaxEngine, dumps per-step logits / decode states / replays
a saved state through one `generate`), `analyze.py` (delta stats, per-layer cache
deltas, state diffs), `quantprobe.py` (qwix quantize bit-consistency).
Venv `~/.cache/metaljax-bench/maxtext/venv`.

```sh
cd ~/.cache/metaljax-bench/logs/int8div
PY=~/.cache/metaljax-bench/maxtext/venv/bin/python

# token streams + per-step logits + decode states at steps 8..11
JAX_PLATFORMS=cpu   $PY logitdump.py --bench-id maxtext-qwix-int8 --steps 24 \
    --out cpu_int8.pkl  --save-state-at 8,9,10,11
JAX_PLATFORMS=metal $PY logitdump.py --bench-id maxtext-qwix-int8 --steps 24 \
    --out metal_int8.pkl --save-state-at 8,9,10,11
$PY analyze.py cpu_int8.pkl metal_int8.pkl            # -> divergence at index 12

# single-step replays from the SAME state (no drift) + per-layer cache deltas
for P in cpu metal; do JAX_PLATFORMS=$P $PY logitdump.py --bench-id maxtext-qwix-int8 \
    --out ${P}_int8_replay.pkl \
    --replay cpu_int8.state8.pkl,cpu_int8.state9.pkl,cpu_int8.state10.pkl,cpu_int8.state11.pkl; done
$PY analyze.py cpu_int8_replay.pkl metal_int8_replay.pkl --replay

# reference noise scales
JAX_PLATFORMS=cpu $PY logitdump.py --bench-id qwen3-06b-maxtext  --out cpu_bf16_on_int8states_replay.pkl \
    --replay cpu_int8.state11.pkl                                   # quantization error
JAX_PLATFORMS=cpu $PY logitdump.py --bench-id maxtext-qwix-int8 --out cpu_int8_f32act_replay.pkl \
    --cast-state float32 --extra-overrides "dtype=float32" --replay cpu_int8.state11.pkl   # bf16 floor
#   bf16-model baseline: same two commands with --bench-id qwen3-06b-maxtext on cpu+metal

# quantizer bit-consistency (incl. exact-.5 boundary values)
JAX_PLATFORMS=cpu   $PY quantprobe.py --gen qdata.pkl
JAX_PLATFORMS=cpu   $PY quantprobe.py --run qdata.pkl --out cpu_q.pkl
JAX_PLATFORMS=metal $PY quantprobe.py --run qdata.pkl --out metal_q.pkl
$PY quantprobe.py --cmp cpu_q.pkl metal_q.pkl

# eager vs compiled on metal (bitwise)
JAX_PLATFORMS=metal METALJAX_COMPILE=0 $PY logitdump.py --bench-id maxtext-qwix-int8 \
    --out metal_int8_replay_eager.pkl --replay cpu_int8.state11.pkl
```

## 10. Note for whoever benchmarks int8 next

Because dynamic int8 quantization is discontinuous, **token-stream equality is
not a usable correctness criterion for quantized decode** — CPU-vs-CPU with a
different activation dtype already breaks it. Use the logit-delta ladder in §3
(cross-backend delta vs the model's own quantization error) or compare against
the unquantized reference. A real bug would show as a per-layer curve that
*jumps* at one layer, a delta far above the quantization error, or a systematic
one-sided bias — none of which is present here.
