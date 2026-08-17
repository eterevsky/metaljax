# Row 15 (qwix-int8 Qwen3-8B) emits wrong text — evidence, and the ladder that placed it

**Status (2026-08-17, evening): MECHANISM ESTABLISHED — nondeterministic MLX
command-buffer corruption at 8B traffic, on BOTH engines, amplified into a
total logit collapse by qwix's per-tensor `absmax` scale. Not ours, not new,
not the quantized arithmetic. Section 8 is the measured ladder and is the
authority; sections 1–7 are the pre-measurement reasoning and are kept because
two of their leading hypotheses were refuted by their own predictions.**

Sections 1–7 were written under the release-gate machine lock (zero GPU work
available), so every claim in them is derived from artifacts already on disk,
from the source of both stacks, and from static arithmetic. **Section 8 is the
measured record**, and where the two disagree, section 8 wins — see §8.4, where
the ladder refutes §4's compile-independence argument, §5's H5, and the
chunk-boundary lead that §8.2's own localization first suggested.

**What must not be inherited:** the fresh row-15 line in
`notes/no-panic-governor.md` calls the wrong output *"the row's known
MLX-quantization bug"*. **There is no such prior finding.** The 2026-08-03
diagnosis (`7932b4d`) attributed the garbage to the MLX command-buffer split
and **explicitly exonerated the quantized dots** ("layer-0 KV bit-exact vs CPU
at K=4096; pre- and post-`fdc7cde` engines fail identically"). The label is
corrected in that note and in STATUS. The 2026-08-03 *command-buffer*
attribution was then put under test rather than inherited (sections 4–5) and
has now been **confirmed by measurement** (§8.5: the committed 8B bf16 canary
still corrupts at today's shipped budgets, identically on both engines) — while
the quantization stays exonerated as a cause and is instead identified as the
amplifier (§8.2).

---

## 1. The observation, decoded

| run (2026-08-17, `frozen-gov7`, native stack) | text | tokens |
|---|---|---|
| row 15 (shipped defaults) | `" fragment!!!!!!!"` | `[12289, 0, 0, 0, 0, 0, 0, 0]` |
| row 15f (`MAXTEXT_PREFILL_LEN=64 METALJAX_BODY_COMPILE=0`) | `"!!!!!!!!"` | `[0, 0, 0, 0, 0, 0, 0, 0]` |

`!` is **token id 0** of the Qwen3 vocabulary (verified against
`~/.cache/huggingface/.../Qwen--Qwen3-8B/.../vocab.json`: `vocab["!"] == 0`).
Greedy decode picks `argmax(logits)`; index 0 is what `argmax` returns for a
**constant** logit vector (all-equal, all-zero, or all-NaN under our
NaN-wins rule).

So the row is not emitting *plausible wrong text*. It is emitting **collapsed
logits** — the model's output distribution has gone flat. That is a different
signature from the 2026-08-03 command-buffer corruption on the 0.6B row, which
produced varied, plausible-magnitude garbage tokens
(`urancesuchos책international…`) that differed on every run. Recorded in
`notes/mlx-command-buffer-split.md` as: *"corrupted values are
plausible-magnitude data, not uninitialized garbage"*.

Raw: `~/.cache/metaljax-bench/logs/no-panic-governor/qwix-int8-qwen3-8b-row15-0817-115121.log`
(and `-row15f-`), `notes/data/no-panic-governor-rows-2026-08-17.json`
(the aggregate json carries the 10 rows of the governor sweep; the row-15
numbers — 369.7 ms/tok, load 80.5 s — are in the log's `RESULT` line and in
`notes/no-panic-governor.md` §5.1).

## 2. What row 15 is, against row 14 — the search space, narrowed statically

Both rows are the **same adapter, same code path, same overrides**
(`scripts/model_bench/adapter_maxtext.py`):

```python
"maxtext-qwix-int8":   model_name="qwen3-0.6b", ckpt="qwen3-0.6b",
                       overrides=["use_qwix_quantization=true", "quantization=int8"]
"qwix-int8-qwen3-8b":  model_name="qwen3-8b",   ckpt="qwen3-8b",
                       overrides=["use_qwix_quantization=true", "quantization=int8"]
```

There is **no different quantization path and no different harness adapter**.
The whole diff is the maxtext model config. Two entries of it matter:

| | qwen3-0.6b (row 14) | qwen3-8b (row 15) |
|---|---|---|
| `base_emb_dim` / `base_mlp_dim` | 1024 / 3072 | **4096 / 12288** |
| `base_num_query_heads` | 16 | 32 |
| `base_num_decoder_layers` | 28 | 36 |
| **`logits_via_embedding`** | **true** (tied) | **false** — a separate `logits_dense` kernel |

Consequences, computed rather than guessed:

* **Per scanned decoder layer**: 15.7 M weights (31 MB bf16) at 0.6B,
  **192.9 M** (386 MB bf16) at 8B — 12.3×.
* MLX's `MLX_MAX_MB_PER_BUFFER` counts **elements, not bytes**, deduped per
  distinct buffer (`notes/mlx-command-buffer-split.md`, 2026-08-03 addendum),
  so the shipped 512 is a budget of ~537 M elements per Metal command buffer.
* The engine replays the layer loop in **compiled chunks of `kmax=16`**
  (logged: `while gate: cost=684 bytes=3259MB … chunkable=1 kmax=16`,
  `compiles=2 compiled_calls=6` for 36 iterations = 2×16 + 4×1).
  One 16-layer chunk is ≈ **251 M elements at 0.6B** (fits in ONE command
  buffer) and ≈ **3.09 G elements at 8B** (**≥6 forced splits per chunk**,
  ~13 per 36-layer pass).
* The 8B's untied **`logits_dense` is 4096×151936 = 622 M elements — larger
  than the entire per-command-buffer budget by itself**, so that one dot
  forces a split on every call. The 0.6B's tied embedding is 155 M and never
  crosses it.

**So the row-14/row-15 difference is scale, not path** — and specifically
scale measured in the unit the known MLX bug is sensitive to. Row 14 cannot
draw a losing ticket in that lottery (its chunk fits in one buffer); row 15
draws several per replay. That is the strongest argument *for* inheriting the
2026-08-03 attribution.

**Measured, both on the release binary, native stack, 2026-08-17** — row 14
from the 0.11.5 gate-5 run `maxtext-qwix-int8-row14hi-0817-140634.log`
(32 GB budget, 31.995 ms/tok, coherent `" Paris. The capital of France is
also the capital of the European Union…"`), row 15 from
`no-panic-governor/qwix-int8-qwen3-8b-row15-0817-115121.log`:

| | row 14 (0.6B) | row 15 (8B) |
|---|---|---|
| prefill program | 90 entries, **15** args, 15 results, 5 output copies | 89 entries, **16** args, 15 results, 5 output copies |
| decode program | 68 entries, 24 args, 15 results, 11 donated | 67 entries, 25 args, 15 results, 11 donated |
| prefill `@main` | **compiled** (`compiled=1`, `compiles=1 compiled_calls=1 unrolls=1`) | eager (`cost=24729 bytes=143925.6MB compile=0`) |
| prefill layer-loop traffic / iteration | **323 MB** | **3259 MB** (10.1×) |
| decode `@main` | eager (`cost=22492 bytes=10141.6MB`) | eager (`cost=28891 bytes=140124.1MB`, 13.8×) |
| decode layer-loop traffic / iteration | **289 MB** | **3154 MB** (10.9×) |
| decode layer loop | `chunkable=1 kmax=16`, 13 compiled calls (28 layers) | `chunkable=1 kmax=16`, 6 compiled calls (36 layers) |
| recognizer emits | **1 sdpa**, 0 qmm, 0 moe | **1 sdpa**, 0 qmm, 0 moe |
| msl plans | 0 | 0 |
| text | **coherent** | **collapsed** |

The two rows take the *same* emits, the *same* int8 dot arm, the *same*
chunked-replay structure, and differ by ~10× in traffic per compiled unit —
plus the one structural asymmetry above (tied vs untied logits). That is the
whole search space.

Note how close the *programs* are: **89 vs 90 tape entries** at prefill,
**67 vs 68** at decode, with the 8B carrying exactly **one extra argument** —
the untied `logits_dense` kernel. Any hypothesis of the form "row 15 lowers
something row 14 does not" has one op's worth of room to live in, and that op
is the logits projection (H5).

## 3. What the fresh run says about our side of the stack

From the row-15 log (`METALJAX_DEBUG=1` was on):

| | prefill (`jit__prefill_jit`) | decode (`jit__generate_jit`) |
|---|---|---|
| lowered | 89 entries, 16 args, 15 results | 67 entries, 25 args, 11 donated |
| `@main` | `pure=0 cost=24729 bytes=143925.6MB compile=0` (**eager**) | `cost=28891 bytes=140124.1MB compile=0` (**eager**) |
| layer loop | `cost=684 bytes=3259MB … kmax=16 period=36`, `compiles=2 compiled_calls=6` | `cost=800 bytes=3154MB … kmax=16`, `compiles=2 compiled_calls=6` |
| recognizer emits | **0 fused quantized matmuls**, 0 expert gathers, **1 fused attention**, 0 packed arrays | 0 / 0 / — / 0 |
| msl_scan | 0 plans (`not eligible (dtype bf16)`) | 0 plans |

Reading: `qmm` does **not** fire on qwix int8 (the int8 dots take
`ops/linalg.py::_int_dot_via_f32`, the exact-f32 K-chunk path, in both
stacks — `plugin-native/runtime/ops_linalg.cc` `kind == 1`); `sdpa` **does**
fire, once, on the scanned attention; `msl_scan` is not involved at all.

Both int8 dot implementations were read line by line today and agree with each
other: `chunk = 1024` for i8×i8, every partial `|Σ| ≤ 2^24` (exactly f32's
exact-integer ceiling), accumulation in i32 (worst case at K=12288: 12 × 2^24 =
2.0e8, far inside i32). The arithmetic is sound *as written*; what is not
verified is that MLX's f32 GEMM is exact on these shapes at 8B widths, which is
the assumption the whole path rests on (probe A, section 6).

## 4. Three pieces of evidence that sit badly with the inherited attribution

1. **The failure is compile-independent.** Row 15 runs the layer body
   *compiled* (`body_compile=20`, `compiles=2`); row 15f runs it *uncompiled*
   (`body_compile=0`) — and 15f is **worse**: the collapse reaches the first
   token instead of the second. The 2026-08-03 command-buffer face is
   compile-dependent by construction ("correct under `METALJAX_COMPILE=0`",
   "first call clean, replays differ per process"). **15f is a clean
   single-variable arm**, which took checking: it also passes
   `MAXTEXT_PREFILL_LEN=64`, but that is a **no-op** — `adapter_maxtext.py`
   already rounds a 5-token prompt up to the next power of two ≥ 64, so both
   arms prefilled 64 and both decoded 8 tokens (`gov_rows.sh`, and the same
   no-op caught on row 10f). The only difference between them is
   `METALJAX_BODY_COMPILE`. Residual caveat: the eager path has a
   command-buffer face of its own (the ops-alignment one, STATUS footnote 11),
   so "not the compiled face" is not yet "not MLX".
2. **The signature is degenerate, not random.** Constant logits → index 0,
   twice, deterministically, rather than the per-process-random plausible
   garbage the split bug produced on the 0.6B.

And a third, found by reading `plugin-native/runtime/control.cc` rather than
the logs: **the two arms are two very different command-buffer layouts.**
`METALJAX_BODY_COMPILE=0` does not shrink the evals — `run_chunked`'s
`chunk()` still unrolls `K = min(trip, kmax) = 16` iterations, just op by op
instead of through `mx::compile` — but the eager flush then fires far more
often: **78 flushes per program call in 15f against 6 in row 15**. Thirteen
times the sync points, no fusion, the same eight collapsed tokens. A lottery
whose draw depends on where the cuts land should not survive that.

### The candidate that reconciles determinism with the MLX bug

There is one, and it is specific to this row. `MLX_MAX_MB_PER_BUFFER` counts
**elements**, so the shipped 512 is ~537 M elements per command buffer. Row
15's untied **`logits_dense` is 4096 × 151936 = 622 M elements — larger than
the whole budget** — so MLX must cut a command buffer *inside that single
matmul*, in the same place, on every call. Row 14's tied embedding is 155 M
and never crosses the line. That is simultaneously (a) a command-buffer
split, (b) **deterministic** rather than a draw, and (c) present in row 15 and
absent in row 14 — the exact shape of the evidence. It is also the one extra
argument the 8B program carries (§2). `row15_probe.py big` asks it directly,
in a minute, with an under-budget control and a budget sweep across the
crossing.

A fourth mechanism reconciles the *amplification* with the split bug and is
specific to this row: under **per-tensor dynamic** int8 quantization
(`scale = absmax/127`, measured in
`notes/int8-divergence-verdict.md` §5), a *single* corrupted outlier in an
activation sets `absmax`, and every other element of that tensor quantizes to
code 0 — the layer's output collapses to zeros, the residual stream follows,
RMSNorm(0)=0, and the logits go flat. One corrupted element is enough to
produce exactly the observed output. That is a *hypothesis with a mechanism*,
not a verdict: it predicts specific, cheap measurements (probes B and D).

## 4b. The amplifier — why the output cannot distinguish the hypotheses

Worth writing down before anyone reads the two identical collapses as proof of
a deterministic bug. The observed text is what this model emits when **one**
value anywhere in the stack goes non-finite:

* `argmax` on this backend is **NaN-wins, lowest index** (v0.4.0 semantics,
  `src/metaljax/ops/reduction.py` and `plugin-native/runtime/ops_reduce.cc`
  agree: `where(has_nan, first_nan, arg)`). All-NaN logits ⇒ index 0 ⇒ `!`.
* A single non-finite element in the final hidden state makes **every** logit
  NaN, because the logits are one dot away: `h @ W`. One bad element,
  151936 NaNs.
* And one `inf` in `h` makes the final RMSNorm produce `x/inf = 0` for every
  finite element and `inf/inf = NaN` for the bad one — the same destination
  from the other direction.
* Upstream of that, per-tensor dynamic int8 quantization is the strongest
  amplifier we have measured (`notes/int8-divergence-verdict.md` §5: a 1-ULP
  change in one element moves 14 % of the tensor's codes; an `inf` absmax
  sends *all* of them to zero).

**Therefore the identical collapsed text in the compiled and uncompiled arms
is NOT evidence that both took the same code path.** The amplifier destroys
exactly the information that would have distinguished a lottery draw from a
deterministic fault: any single corrupted element, wherever it enters, lands
on the same eight tokens. Section 4's objection #1 is weakened by this and
must be re-asked numerically.

The consequence for the ladder: **the criterion is not the text.** Every rung
below compares numbers on identical inputs (metal vs CPU, replay vs replay,
compiled vs op-by-op) and reports *where the first non-finite or divergent
value appears*, not what the model said.

## 5. The hypotheses, ranked, each with its discriminator

| # | hypothesis | discriminator | outcome if true |
|---|---|---|---|
| H1 | MLX command-buffer split corruption at 8B traffic, amplified to a collapse by dynamic int8 quantization | probe B (does the 8B canary still corrupt at today's 800/512?) + probe D's budget arm (512 vs 2048 flips it) | MLX; measure the knob, document, do **not** default it without a memory ladder (2048 is panic-#4 territory) |
| H2 | our int8 dot is not exact at 8B widths (MLX f32 GEMM not exact where `_int_dot_via_f32` assumes it is; or the M5 neural-accelerator arch pin not in force in this process) | probe A — pure int8 dot inventory vs numpy, seconds, no checkpoint | OURS, contained, fixable (widen the arm or narrow the chunk) |
| H3 | the `sdpa` emit is wrong on this attention shape (32 q-heads / 8 kv, 36 layers, prefill mask) | `METALJAX_SDPA=0` arm | OURS, contained |
| H4 | the chunked 16-layer replay is the trigger (a graph too big for one command buffer, or an unroll bug) | `METALJAX_CHUNK_MAX=1` arm | OURS or MLX, narrowed either way |
| **H5** (promoted to first after the §4 reading) | the collapse is in the logits stage — the untied **622 M-element** `logits_dense`, the one buffer that exceeds a whole command buffer by itself, so the split is forced in the same place every call: deterministic, and absent from row 14 | **probe A's `big` arm** — the over-budget dot vs an under-budget control, bf16 and int8, against CPU, swept 512→2048 in fresh processes | MLX, but *deterministic* — and then the workaround is a per-shape budget rule, not a global raise |
| H6 | harness/checkpoint (the 8B qwix conversion) | jax-CPU rerun of the same row | not ours — but note jax-CPU already produced **coherent** text on this row (`34f627c`: "2118 (maxtext; coherent)"), which is why this is ranked last |

## 6. The ladder (queued behind the release-gate lock; big-rows-first, guarded)

Ordered so that each rung is cheaper than the row itself and can kill a
hypothesis on its own. Every rung: one process, `mem_guard.sh`, machine
settled, artifacts written **as produced** under
`~/.cache/metaljax-bench/logs/row15-mechanism/`.

Rungs A and B are built and syntax-checked, and take the machine lock
themselves:

```
scripts/model_bench/row15_ladder.sh A      # ~5 min total, <30 GB, no checkpoint
scripts/model_bench/row15_ladder.sh B      # ~15 min, ~20-40 GB, guarded
ROW15_RAISED=1 scripts/model_bench/row15_ladder.sh B    # + the 2048 control
```

**How to read them.** Rung A's `dots` arm is pass/fail with no tolerance —
`exact: false` on any row is hypothesis H2 and the investigation ends there.
Its `qwix` arm reports per layer and flags `collapsed` when metal loses a
signal the CPU kept (`metal_std < 1 % of cpu_std`) or produces non-finite
values; the *first* flagged layer is the entry point. Its `big` arm is the
H5 test: `rel_rms` or `metal_nonfinite` nonzero on `over-budget-logits` while
`under-budget-logits` is clean, and the error disappearing as the budget is
swept past 622 M elements, is the whole hypothesis in one table. Rung B is
`status: ok`
plus a `--check` verdict against the CPU reference; a mismatch there with the
`nocompile` arm clean is the command-buffer face, a mismatch in **both** arms
is not.

**A — arithmetic and the over-budget dot** (~5 min total, <30 GB, no
checkpoint, both stacks). `row15_probe.py dots` — every (K, N) row 15 and row
14 actually contract, s8×s8→s32 against an exact numpy reference (kills or
confirms H2 without touching a model); `… qwix` — the dynamic-quantization
pattern per layer, reporting where a divergence *enters*; `… big` — **the
over-budget `logits_dense` shape against CPU, swept across the
512→2048 element budget**, which is the direct test of H5 and the cheapest
decisive measurement in this note.

**B — the 8B canary already in the repo** (~1 min, ~20 GB, guarded, both
stacks). `notes/data/qwen3_8b_prefill_36layer.mlir` is the 2026-08-03 asset:
real-shape 36-layer 8B **bf16** maxtext prefill, weights as arguments,
"corrupts in 0.3 s at 512, clean at 2048" — measured when the shipped kernel
budget was 400, which became **800** the next day (`52b90a2`). *Nobody has
re-run it since.* If it is clean at today's shipped budgets on both stacks,
the inherited attribution is stale and H1 needs a new witness.
`scripts/run_stablehlo_bench.py … --platform metal --save-out/--check`, plus
the compiled-vs-eager and call-to-call detectors of
`tests/test_command_buffer.py`.

**C — capture row 15's own programs** (one guarded run at its measured 79 GB
peak, ~5 min, native stack): `METALJAX_DUMP_MODULE=1`, save prefill + generate
to `notes/data/`. This is the one expensive rung, and it converts every rung
after it from a 79 GB model run into a ~25 GB replay.

**D — knob A/B on the captured program** (~1 min per arm, ~25 GB): single
variable each — `MLX_MAX_MB_PER_BUFFER` 512 vs 2048, `METALJAX_COMPILE=0`,
`METALJAX_VERIFY_COMPILE=1` (native: re-runs every executable op-by-op and
reports differing outputs — *the* discriminator between our lowering and the
compiled/fused path), `METALJAX_MLX_COMPILE_MODE=no_fuse`,
`METALJAX_RECOGNIZE=0`, `METALJAX_SDPA=0`, `METALJAX_CHUNK_MAX=1`,
`METALJAX_MSL=0`. Both stacks, plus the **0.11.2 `src/metaljax` on
`PYTHONPATH`** arm (P20's technique) for the pre-migration release question.

*Why the "pre-existing" question (rung D's Stage-1 and 0.11.2 arms) is worth
its cost*: everything new in 0.11.5 — the memory governor, the ingest
page-cache sweep (`METALJAX_INGEST_ADVISE_KB` hands consumed source ranges
back to the OS *while a 16 GB checkpoint streams*), P25/P27's flush policy,
and the whole native lowering — is excluded in one stroke if the same
collapse reproduces on 0.11.2's `src/metaljax` with the frozen Stage-1 dylib.
Conversely, wrong on ONE stack only would make it **ours** and recent, and
would move this from "an inherited MLX bug" to a migration defect. The
historical record says both (Stage 1 emitted garbage on 2026-08-03, native
does today) but those are two different trees two weeks apart, which is
exactly the kind of inheritance this session is not allowed to make.

**E — minimize**: shrink the captured module by layers until it stops failing;
then attempt the pure-JAX synthetic (quantize → int8 dot → dequant, scanned)
at the same widths. Deliverable: a committed test asset, or — if it lands on
MLX — the minimal MLX-level reproducer for the upstream pile.

**F — verify**: guarded row-15 rerun with the fix or the knob, token/text
checked against the jax-CPU reference (` Paris…`, coherent per `34f627c`),
plus `texmo_gate` + `execute_test` + `tests/` if any code changed.

**Machine-safety rule that governs rung D's 2048 arm** (TASKS.md, verbatim):
*"DO NOT run 8B-class maxtext on metal at raised budgets without a memory
watchdog and Oleg's sign-off."* The raised-budget arm on the **standalone
replay** (≈25 GB, no 80 s checkpoint load — the load is what panicked the
machine in #4/#5) is the safe way to ask that question; the raised-budget arm
on the **full row** needs Oleg's explicit go, even though 0.11.5's memory
governor now paces the load that killed it twice.

## 7. What is already settled (so nobody re-derives it)

* ~~jax-CPU runs this row **coherently** at 2118 ms/tok (`34f627c`,
  2026-08-02)~~ — **WITHDRAWN by §8.6: there is no artifact behind that cell,
  and it could not be re-measured tonight.** The conclusion it was used for
  (the checkpoint, the qwix conversion and the harness are not the cause) is
  still true, but it now rests on row 14's ten identical, correct draws
  through the same pipeline (§8.1), not on this.
* Row 14 (identical path, 0.6B) produces coherent text; its only divergence vs
  int8-CPU is a certified exact-tie flip (`notes/int8-divergence-verdict.md`,
  STATUS footnote 22).
* `qmm` and `msl_scan` are **not** in row 15's picture (0 emits, 0 plans).
  `sdpa` is (1 fused attention).
* Both stacks pin `MLX_MAX_OPS_PER_BUFFER=800`, `MLX_MAX_MB_PER_BUFFER=512`
  and `MLX_METAL_GPU_ARCH=applegpu_g16g` (`src/metaljax/__init__.py`,
  `plugin-native/metal/metal_client.cc`) — the budgets are the same on both
  tickets, the sync-point layouts are not.
* Row 15's memory behaviour is **not** the issue and is now good: 79 GB peak,
  flat page cache, 0 governor refusals, completes at exit 0 — the no-panic
  campaign's own result. The row's blocker moved from *memory* to *this*.

---

# 8. The measured ladder (2026-08-17 evening, machine lock held per phase)

Every arm below: one process, `scripts/model_bench/mem_guard.sh`, machine
settled first, artifacts written as produced under
`~/.cache/metaljax-bench/logs/row15-mechanism/` (`ladder.log` is the index).
Driver: `scripts/model_bench/row15_ladder.sh <rungs>`; rungs `A1,B,C,D,Dp,Ref,K,Rate`.
Stack selection is `METALJAX_PLUGIN_PATH` (frozen `0.11.5-ebe56e71.dylib` =
native release binary; unset = Stage 1's Python engine) plus `PYTHONPATH` for
the 0.11.2 arm — the same selection `g5_model.sh` uses.

## 8.1 The verdict, first

**Row 15 is nondeterministic value corruption on the metal side at 8B
traffic.** It is present on the native release engine AND on Stage 1's Python
engine, in the same session, on the same checkpoint. It is present in **bf16
with no quantization at all**. It is absent at 0.6B. qwix is not the cause; it
is the *amplifier* that turns one corrupted element into eight tokens of `!`.

The single measurement that settles it — ten prefills of the **same loaded
parameters, in one process, on identical inputs** (`Rate-native`,
`row15_forensics.py --prefill-reps 10`):

| row | draws | distinct first tokens | fully collapsed | first-bad layer per draw |
|---|---|---|---|---|
| **15** (8B), native | 10 | **8** | **2** | –, 6, –, –, 6, –, –, –, –, – |
| **15** (8B), Stage 1 | 10 | **10** | 0 | all – (garbage tokens, no collapse) |
| **14** (0.6B), native | 10 | **1** | 0 | all – |

Row 14 returns token 12095 ten times out of ten and decodes
`" Paris. The capital"`. Row 15 returns a different answer almost every time.
A correct implementation is deterministic; **this is self-proving, and it does
not depend on any external reference.** That matters, because the reference
could not be obtained (§8.6).

## 8.2 The localization — where the first non-finite value appears

`scripts/model_bench/row15_forensics.py` (new). MaxEngine's `prefill` returns
the whole prefix, and with `scan_layers=true` the per-layer KV cache is stacked
on axis 0, so the first layer whose K/V is non-finite is the entry point of the
fault — read off one array, with no model surgery and no second backend.

`C-row15-native`, shipped defaults:

| layers 0–32 | layers 33, 34, 35 |
|---|---|
| finite; `cached_prefill_key` absmax 7.3–50.8, std 1.7–3.7 | **65536 / 65536 NaN**, absmax 0, std 0 |

Prefill logits: **151936 / 151936 NaN**, zero infs, first token 0. Decode
`[0,0,0,0]` = `"!!!!"`. Row 14 under the same script: every layer finite,
logits std 3.00 / absmax 16.75 / 0 non-finite, top token 12095, `" Paris. The"`.
(For scale: the arms below whose prefill survived had logits std 3.37 and a
top-5 spread of 15.19…14.00 — healthy-looking numbers attached to a wrong
answer, which is why the criterion in §8.1 is a rate and not a plausibility
judgement.)

**The entry point is a draw, not a place.** Across arms it landed at layer
33, 6, 6, 11, or nowhere. Decode-side (`--probe-decode`, the AR half of the
cache) it landed at layer 27, then 1, then 1 within a single three-token
decode.

**Why it is always 100 % of elements and never an inf** — qwix's own
arithmetic, read line by line
(`qwix/_src/core/qarray.py::calibrate` and `compute_scale_zero_point`):

```python
absmax = jnp.max(jnp.abs(array), axis=reduce_axes, keepdims=...)   # per tensor
scale  = calibration['absmax'] / qmax
tiny_sqrt = jnp.sqrt(jnp.finfo(scale.dtype).tiny)
scale = jnp.where(scale < tiny_sqrt, jnp.ones_like(scale), scale)  # guards ZERO
```

The guard covers a zero scale. It does **not** cover a non-finite one, and
`NaN < tiny` is `False`, so a NaN passes straight through. One non-finite
element anywhere in an activation makes the whole tensor's scale non-finite,
and every element of the dequantized result is then NaN. That is why the
corruption is never a sprinkle: by the time it is observable it has been
laundered through a per-tensor scalar. §4b guessed at this amplifier; this is
it, in qwix's source, with the guard that does not close.

## 8.3 What it is NOT — the single-variable arms

All on the native release dylib, same checkpoint, same prompt, same script:

| arm | prefill | first token | decode |
|---|---|---|---|
| baseline (default knobs) | **collapse at layer 33** | 0 | `!!!!` |
| **baseline, repeated** | clean | 22852 | `LIST!!!` |
| `METALJAX_CHUNK_MAX=12` (36 = 3×12, no remainder) | clean | 114179 | `中关!!!` |
| `METALJAX_CHUNK_MAX=10` (remainder 6, from layer 30) | clean | 6623 | ` gold Estr!!` |
| `METALJAX_RECOGNIZE=0` (no fused sdpa) | clean | 99431 | `班!!!` |
| `METALJAX_COMPILE=0` (no `mx.compile` anywhere) | **collapse at layer 11** | 0 | `!!!!` |

Read down the `first token` column: six arms, six different answers, and the
second row is the **same configuration as the first**. Every "clean prefill"
above is a draw, not a cure.

* **Not our int8 arithmetic (H2 refuted).** `row15_probe.py dots`: every (K, N)
  row 14 and row 15 actually contract, s8×s8→s32 against an exact int64 numpy
  reference — `8b-qkv/kv/attn-out/mlp-wi/mlp-wo/logits` and the six 0.6B
  shapes — **bit-exact, 0 of 12 wrong, on BOTH stacks**. `row15_probe.py qwix`
  at 8B widths (4 layers, unrolled, vs CPU): no collapse, no non-finite,
  `rel_rms` 2.8e-3…1.5e-2, `metal_std` equal to `cpu_std` to five digits.
* **Not the recognizer (H3 refuted).** `METALJAX_RECOGNIZE=0` is still wrong.
* **Not the chunked replay (H4 refuted, by its own prediction).** The first
  localization looked decisive: 36 iterations at `kmax=16` replay as 16 + 16 +
  4×1 (`control.cc::run_chunked`), so layer 32 — where the residual went NaN —
  is exactly the first single-iteration remainder call. The falsifiable form of
  that is `CHUNK_MAX=10`, which moves the remainder to layer 30 and **predicts a
  break at 30**. It broke nowhere. And the baseline repeat, with the geometry
  unchanged, also broke nowhere. The layer-32 coincidence was a lottery draw.
* **Not our compiled path (§4's objection #1 inverted).** `METALJAX_COMPILE=0`
  is not a cure; it is *worse* (collapse at layer 11 rather than 33). §4 read
  the 15f arm as evidence against the command-buffer attribution because that
  face is "compile-dependent". It is not: TASKS.md records three faces, and the
  ops-boundary alignment face corrupts **eager** scans. §8.5 measures both.
* **Not H5 (already refuted at the gate, and again here).** The over-budget
  622 M-element `logits_dense` is clean in isolation
  (`notes/release-gates-0.11.5.md`), and the collapse appears at layer 6, 11,
  27 or 33 — nowhere near the logits stage.

## 8.4 Rung D — provenance

| engine, same day / checkpoint / harness | result |
|---|---|
| native release dylib (0.11.5) | wrong; 8 distinct answers in 10 draws, 2 collapses |
| **Stage 1 today** (current `src/metaljax` + `plugin/build/libmetal_pjrt.dylib`) | **wrong**; `'瞭那一天什麽训练'`, and **10 distinct answers in 10 draws** |
| 0.11.2 `src/metaljax` on `PYTHONPATH` | **BLOCKED** — guard kill at 94 GB footprint |

**Verdict: this is not a 0.11.3+ migration defect.** Stage 1's Python engine,
running today on the same checkpoint through the same adapter, is wrong on this
row too — and *more* nondeterministic than the native engine (10/10 distinct
against 8/10). Its wrongness has a different **shape**: plausible-magnitude
garbage tokens, never a full collapse in ten draws — which is precisely the
signature `notes/mlx-command-buffer-split.md` recorded on 2026-08-03
("corrupted values are plausible-magnitude data, not uninitialized garbage").
The native engine additionally reaches the fully-collapsed face, at 2/10.

The 0.11.2 arm is blocked by **memory, not by the question**: 0.11.2 predates
the 0.11.3 eager-flush-returns-cache fix (`4d34bff`) and the 0.11.5 memory
governor, so it exceeds row 15's measured 79 GB envelope and the guard fired at
94 GB against a 92 GB budget. Raising the budget puts a full 8B maxtext load
into the 67–109 GB band of the panic-#5 ledger, which needs Oleg's sign-off;
**not taken.** It is also no longer load-bearing: rung B (§8.5) puts the same
question to a *committed asset* rather than to an engine, and answers it for
both engines at once.

## 8.5 Rung B — the 8B canary, re-run for the first time since 2026-08-03

`notes/data/qwen3_8b_prefill_36layer.mlir`: real-shape 36-layer 8B **bf16**
maxtext prefill, weights as arguments. **No qwix. No int8. No checkpoint. No
maxtext at run time.** 16.4 GB of parameters, checked against a jax-CPU
reference computed in the same rung (`--rtol/--atol 2e-2`).

| arm | check | max_abs_err | max_norm_err | ms |
|---|---|---|---|---|
| native, shipped budgets | **FAIL(5)** | **1.085e+04** | **1.000e+00** | 329 |
| Stage 1, shipped budgets | **FAIL(5)** | **1.085e+04** | **1.000e+00** | 403 |
| native, `METALJAX_COMPILE=0` | FAIL(3) | 1.29e-02 | 7.35e-02 | 1131 |
| Stage 1, `METALJAX_COMPILE=0` | FAIL(3) | 1.06e-02 | 5.04e-02 | 2340 |
| native, `MLX_MAX_MB_PER_BUFFER=2048` | FAIL(3) | 9.28e-03 | 4.10e-02 | 335 |

Reproduced identically across two independent passes.

* **The inherited attribution is NOT stale.** The canary still corrupts at
  today's shipped `MLX_MAX_OPS_PER_BUFFER=800` / `MLX_MAX_MB_PER_BUFFER=512`.
  A `max_norm_err` of exactly 1.000 on 5 of 15 outputs is total loss of signal,
  not drift.
* **It is engine-independent by construction** — the two stacks produce
  *bit-identical* error figures. Whatever this is, it is below both of our
  interpreters. That is the provenance answer the blocked 0.11.2 arm was for.
* **qwix is exonerated as a cause.** This asset has no quantization anywhere,
  and it is the same architecture, the same 36 layers and the same widths as
  row 15. It is also a closer analogue than it first looks: `--scan-params`
  shows the qwix rows' stored weights are **bf16**, not int8 — qwix quantizes
  on the fly inside the jit — so the canary streams the same dtype through the
  same shaped layer loop. (That scan is also a control in its own right: every
  parameter leaf of row 14 is finite with sane magnitudes, so nothing is broken
  before any math runs.)
* **Both faces are visible.** Compiled is catastrophic (norm 1.000); eager
  (`METALJAX_COMPILE=0`) is 50–70× smaller but still fails — which is the
  ops-alignment face, and which is why row 15f (`BODY_COMPILE=0`) collapsed too.
* **The workaround, measured.** `MLX_MAX_MB_PER_BUFFER=2048` removes the
  catastrophic face (norm 1.000 → 4.10e-02) but does **not** make the asset
  pass. Raising the budget is necessary and not sufficient, and 2048 on a full
  8B *load* is panic-#4 territory. **Not a shipped default, and not a fix**;
  §8.7 lists what is left to attribute in the residual.

Harness fix that unblocked this rung: `scripts/run_stablehlo_bench.py` never
registered the `chlo`/`sdy`/`mpmd` dialects, so the asset failed to parse
(`Dialect 'sdy' not found for custom op 'sdy.mesh'`). That parse error — not
any decision — is why "nobody has re-run it since 2026-08-03".

## 8.6 The reference that could not be obtained — correct the record

`34f627c` (2026-08-02) records row 15's jax-CPU cell as
"2118 (maxtext; coherent)", and both this note (§7) and the gate document lean
on it to exonerate the checkpoint. **There is no artifact behind it**: the
commit is a one-line STATUS table edit, and no log on disk carries jax-CPU text
for this row (the only two recorded outputs are the two metal ones).

Re-measuring it tonight failed twice on memory: guard-killed at a 60 GB budget
(+16 GB/sample on the Orbax restore) and again at 92 GB, having reached an 89 GB
footprint with the system at 99.4 GB and the free list at 4.9 GB. Going higher
is the same sign-off that §8.4 needs. **Treat "jax-CPU is coherent on row 15"
as an unverified 2026-08-02 claim, not as evidence**, until it is re-measured.

The verdict does not need it. Row 14 — the same adapter, the same qwix int8
overrides, the same script, ten draws — is deterministic and correct, which
exonerates the harness, the tokenizer, the qwix path and the conversion
pipeline; and row 15's own nondeterminism proves corruption without reference
to what the right answer is.

## 8.7 What is left open

1. **The 0.11.2 arm** (§8.4) and the **jax-CPU reference** (§8.6) are both
   blocked by the same thing: a full 8B maxtext load above row 15's 79 GB
   envelope. Both need Oleg's explicit go; neither changes the verdict.
2. **The residual in every metal canary arm.** Even at `MB_PER_BUFFER=2048`,
   and in both eager arms, 3 of 15 outputs miss by 4–7 % norm against CPU.
   The tolerance is 2 %, and 36 layers of bf16 accumulation can legitimately
   drift — but it is *unattributed*, and it may be a second, smaller fault
   hiding under the catastrophic one. Cheapest next step: sweep the budget
   between 512 and 2048 on the standalone replay and see whether the residual
   moves at all; if it is flat, it is bf16 depth, not corruption.
3. **No fix is available at our level.** This is the upstream MLX report that
   TASKS.md already carries as URGENT
   (`notes/mlx-command-buffer-upstream-issue.md`,
   `notes/data/mlx-cbuf-repro/`). Row 15 now adds the strongest evidence the
   pile has: a whole-model failure with a measured rate (2/10 collapses,
   8/10 distinct answers), a clean 0.6B control, and a checkpoint-free bf16
   asset that fails identically on two independent engines.
4. **Row 15 ships ✗.** Nothing here changes the 0.11.5 cell; it changes the
   attribution behind it from "under test" to "measured", and from "possibly
   ours" to "not ours, on both engines".
