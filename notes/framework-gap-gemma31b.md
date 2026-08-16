# The framework gap on gemma4-31B decode (2026-08-16)

*Part 2 of the P26 pass. Part 1 (the callee-scoped attention recognizer) closed
the gap between metaljax's two stacks; this document is about the gap between
metaljax and everybody else on the same machine and the same model. It is
MEASUREMENT AND RANKING ONLY — nothing here was fixed, by instruction: the
list at the end is for Oleg to pick from.*

Row 1 of the model suite: `google/gemma-4-31B-it`, bf16, 62.6 GB of weights,
the DeepMind gemma library's sampler, 1024-deep cache, 128 greedy tokens.
Artifacts and runners: `~/.cache/metaljax-bench/logs/p26b-callee-sdpa/`.

## What the machine can do, and what we do

A decode token must read every weight once. At the model's real dims (hidden
5376, 60 layers = 50 sliding + 10 global, ffw 21504, vocab 262144, tied
embeddings) that is:

| traffic | GiB |
|---|---:|
| 50 sliding layers x 0.892 | 44.6 |
| 10 global layers x 1.015 | 10.2 |
| embedding / head table | 2.6 |
| KV cache read (1024 deep, both stacks) | 0.86 |
| **mandatory per token** | **58.2 GiB = 62.5 GB** |

Divide that by each stack's measured ms/token and you get the bandwidth each
one achieves:

| stack | ms/tok | effective GB/s | of llama.cpp |
|---|---:|---:|---:|
| llama.cpp bf16 (2026-08-03) | 111.2 | 562 | 1.00 |
| **mlx-lm (measured today)** | **131.8** | **474** | 0.84 |
| torch-MPS (2026-08-03) | 148.7 | 421 | 0.75 |
| metaljax native, P26b (today) | 239.8 | 261 | 0.46 |
| metaljax Stage 1 (today) | 242.4 | 258 | 0.46 |

The mlx-lm cell is fresh (`run_bench.py gemma4-31b-bf16 mlx`, same prompt, same
128 greedy tokens, same guard, same hold): **131.8 ms/tok**, against the 137 the
table has carried since 2026-08-03. So the live gap is **1.82x**, and it is the
same 1.8x for both metaljax stacks — P26b closed the metaljax-vs-metaljax gap
without touching this one.

llama.cpp at 562 GB/s is the practical roofline of this machine, and it is a
useful one: it says the gap is not "Metal is slow", it is that **we move about
twice the bytes, or at half the rate, of a stack doing the same arithmetic**.

## Method

Three independent measurements, all under the machine lock, none of them
inferred:

1. **The same-day mlx-lm cell** — `run_bench.py gemma4-31b-bf16 mlx`, the
   comparison arm the harness has always had, re-run today so the ratio is
   not against a 2026-08-03 number.
2. **One layer, piece by piece, in both vocabularies**
   (`layer_profile.py`). The gemma library writes every projection as an
   `einsum` over a rank-3/rank-4 weight (`BTD,NDH->BTNH`,
   `...F,NHF->...NH`, `BTKGH,BSKH->BTKGS`); mlx-lm writes 2-D weights and
   plain matmuls. The same arithmetic is timed through this plugin and
   through `mlx.core` directly, at the real dims, at decode shapes.
3. **The layout arms** — inside the MLX process, the same gemma-shaped
   weights contracted (a) the way `dot_general` does it
   (`transpose` -> `reshape` -> `matmul`, `runtime/ops_linalg.cc`) and (b) as
   a batched matmul that never rewrites the weight. That pair separates
   "the model's tensor layout costs this" from "our lowering of it costs
   this", which no whole-model number can.

Plus one A/B on a knob that is ours alone: metaljax pins
`MLX_METAL_GPU_ARCH=applegpu_g16g` (M5 f32 GEMM goes through the neural
accelerators at ~4e-3 otherwise — `src/metaljax/__init__.py`,
`metal_client.cc`), and mlx-lm runs on the default arch. Row 1 is **bf16**, so
the pin is being paid on a row whose accuracy problem it does not solve.

## What was already attributed before this pass

P26 split the native stack's own 59.2 ms/token gap against Stage 1 (measured,
not inferred — one environment variable on one frozen binary):

| component | ms/tok | where it lives |
|---|---:|---|
| the compile cliff (body over the op budget) | 46.8 | our lowering — **closed by P26b** |
| the native stack's baseline overhead vs Stage 1 | 12.4 | runtime, shared with row 2 |

That accounting is about metaljax-vs-metaljax. What follows is about the
~130 ms/token that BOTH metaljax stacks are behind mlx-lm, which no compile
decision explains: Stage 1 compiles its body, replays one graph per token, and
still runs at 258 GB/s against mlx-lm's 456 and llama.cpp's 562.

## One layer, piece by piece

`layer_profile.py`, best of 20, machine lock held, gemma-4-31B's own dims at
decode shapes (T=1 against a 1024-deep cache), bf16. The jax arm runs through
the P26b native plugin; the mlx arm is the same arithmetic as mlx-lm writes it.
Raw: `notes/data/p26b-layer-profile-2026-08-16.jsonl`.

| piece | ours ms | mlx ms | x | excess x60 (ms/tok) |
|---|---:|---:|---:|---:|
| **layer whole** | **3.968** | **2.651** | **1.50** | **79.0** |
| kv_einsum `BSD,CKDH` (84 MB) | 0.768 | 0.294 | 2.61 | 28.5 |
| q_einsum `BTD,NDH` (84 MB) | 0.734 | 0.293 | 2.51 | 26.5 |
| kv cache update | 0.594 | 0.182 | 3.27 | 24.7 |
| attn_vec `BTNH,NHD` (84 MB) | 0.865 | 0.481 | 1.80 | 23.0 |
| attention itself (K=1024) | 0.283 | 0.164 | 1.73 | 7.2 |
| rmsnorm | 0.256 | 0.151 | 1.69 | 6.3 |
| rope | 0.245 | 0.147 | 1.67 | 5.9 |
| linear `...H,HF` (220 MB) | 1.021 | 1.019 | 1.00 | 0.1 |
| gating `...F,NHF` (441 MB) | 0.983 | 1.019 | **0.96** | -2.2 |

**The headline is the last two rows.** The two MLP matmuls are **74 % of every
layer's bytes**, and on them we are at parity with MLX — 0.96x and 1.00x, i.e.
the gating einsum runs at ~449 GB/s through this plugin, which is llama.cpp
territory. Whatever the framework gap is, it is *not* that our matmuls are
slow, and it is not bandwidth: on the bulk of the traffic we already achieve
what the machine can do.

The gap is the small work around them. And the per-layer total is not a guess:
**60 x 3.968 = 238.1 ms against the 239.8 ms/token actually measured** — this
layer is essentially the whole token. The synthetic layer pays one dispatch and
one host sync of its own (~0.2 ms, the floor `rmsnorm` and `rope` are measuring
almost entirely), so the strict reading is: 60 layers of real work account for
~226 ms of the 239.8, and whatever else a decode token does — the head, the
sampler, the loop's own sync — is the remaining ~14 ms. Either way there is no
large unexplained runtime cost hiding outside the layers, which is itself a
result: after P26b the decode loop replays one compiled graph per token and the
overhead of doing so is small.

(The individual pieces each pay that same floor, so the per-piece deltas sum to
more than the layer-whole delta. The 79.0 ms is the honest total; the per-piece
column ranks the causes.)

## The mechanism behind the biggest item, isolated

Every one of gemma's projections is an `einsum` over a rank-3/4 weight, and
`dot_general` lowers all of them the same way (`runtime/ops_linalg.cc`, and
`ops/linalg.py` identically):

```
r = transpose(rhs, rb + rc + rfree);  matmul(reshape(l,[B,M,K]), reshape(r,[B,K,N]))
```

That `reshape` of the transposed weight is a free VIEW only when the merge of
the weight's free axes is expressible in the transposed strides. `layout_probe.py`
measures the same contraction three ways **inside MLX**, so the number is the
algorithm's and not the plugin's:

| contraction | MB | ours (transpose+reshape) | batched matmul | mlx-lm's 2-D | ours vs best |
|---|---:|---:|---:|---:|---:|
| q `BTD,NDH` [32,5376,256] | 84 | 0.656 | 0.307 | 0.299 | **2.20x** |
| kv `BSD,CKDH` [2,16,5376,256] | 84 | 0.655 | 0.302 | 0.255 | **2.57x** |
| attn_vec `BTNH,NHD` [32,256,5376] | 84 | 0.484 | 0.465 | 0.476 | 1.02x |
| gating `...F,NHF` [2,21504,5376] | 441 | 0.995 | 0.940 | 1.021 | 0.97x |

The rule falls straight out of the strides:

* contracted axis **last** (`...F,NHF`) — the merge is a valid view and MLX's
  GEMM takes B transposed with no copy. Parity, on the biggest weight in the
  model.
* contracted axes **leading** (`BTNH,NHD`) — the transpose is the identity and
  the reshape is a view. Parity.
* contracted axis **in the middle** (`BTD,NDH`, `BSD,CKDH`) — merging the free
  axes across the contracted stride is not expressible, so **MLX materializes
  the whole weight, every token, in every layer**. 2.2-2.6x, and the 0.35 ms
  it costs on an 84 MB weight is exactly an 84 MB copy read-and-written at
  roofline.

The alternative that fixes it needs no checkpoint change and no new kernel: a
**batched matmul over the weight's leading free axes** — 0.307 ms against
mlx-lm's own pre-transposed 0.299 ms, i.e. all of the difference recovered by
reading the weight where it already lies.

## The A/B that came back negative, so nobody re-litigates it

metaljax pins `MLX_METAL_GPU_ARCH=applegpu_g16g` (the M5's f32 GEMM otherwise
goes through the neural accelerators at ~4e-3) and mlx-lm does not, so the pin
was a natural suspect on a bf16 row. It costs **nothing**:

| arm | layer whole |
|---|---:|
| jax, pinned g16g (shipped default) | 3.968 ms |
| jax, `METALJAX_MATMUL_PRECISION=default` | 3.987 ms |
| mlx, default arch | 2.651 ms |
| mlx, pinned g16g | 2.571 ms |

Within noise on our side, and if anything MLX is *faster* pinned. The accuracy
pin can stay.

## Ranked, measured, with an address for each

Each item's cost is what it is worth **per decode token on row 1** (60 layers).
Nothing below was implemented — that is Oleg's call.

| # | item | ms/tok | where it lives | evidence |
|---:|---|---:|---|---|
| 1 | `dot_general` copies the weight when the contracted axis is in the middle (q + kv projections) | **~42** | our lowering: `runtime/ops_linalg.cc` + `ops/linalg.py` (both stacks) | layout probe, 2.20x/2.57x vs two alternatives in MLX itself |
| 2 | KV-cache update: functional `dynamic_update_slice` vs an in-place write | **~25** | jax-lowering shape + our lowering (a donated/last-use case we could detect) | 0.594 vs 0.182 ms/layer |
| 3 | `attn_vec` projection, 1.80x with the layout ruled OUT | **~23** | unattributed — the next thing to bisect | layer profile vs layout probe (1.02x there) |
| 4 | norms and RoPE built from primitives instead of `mx.fast.rms_norm` / `mx.fast.rope` | **~12** | our lowering (a recognizer, like sdpa/qmm) | 1.69x / 1.67x |
| 5 | attention itself at T=1 | ~7 | our lowering / MLX | 1.73x |
| 6 | the prefill program misses the op budget by 839 units (20839 vs 20000) | **~300 ms per prefill** (not per token) | the budget policy, shared by both stacks | both stacks narrate `cost=20839 compile=0`; P26's budget-21000 probe read 2007.7 vs today's 2311.8 |
| — | the two MLP matmuls (74 % of the bytes) | **0** | — | 0.96x and 1.00x: nothing to win |
| — | the GPU-arch pin | **0** | — | A/B above |

**Sum of items 1-5: ~109 ms/token against a measured whole-token gap of 108 ms
(239.8 vs 131.8).** They do not add cleanly — the per-piece floors inflate them
and my hand-written MLX layer is itself slower than mlx-lm in situ (60 x 2.651 =
159 ms against mlx-lm's real 131.8) — but the ordering is measured and the two
biggest items are mechanism-level, not folklore.

## Caveats on these numbers

* The per-piece timings each pay one dispatch and one host sync (~0.15-0.25 ms);
  the layer-whole and the layout-probe rows are the load-bearing ones.
* The jax arm syncs by pulling a rank-0 reduction to the host, because
  `jax.block_until_ready` is a no-op on this backend and the host's `mlx.core`
  is a different runtime from the plugin's.
* The mlx arm uses `mx.fast.rms_norm` / `mx.fast.rope` /
  `mx.fast.scaled_dot_product_attention` and 2-D weights, i.e. what mlx-lm
  actually runs — the comparison is deliberately "our stack vs their stack",
  not "our kernels vs their kernels".
* The first draft of `layout_probe.py` allocated its 2-D weight INSIDE the
  timed lambda and read 87 ms for a 0.3 ms matmul. The table above is the
  corrected run; the bug is recorded because it is the obvious way to get this
  wrong.
* Token streams are not comparable across frameworks (mlx-lm's sampler is its
  own); the mlx-lm cell here is timing only.
