# Row 15 (qwix-int8 Qwen3-8B) emits wrong text — evidence, and the ladder that places it

**Status: OPEN, mechanism NOT established.** This note is the evidence file and
the queued experiment ladder. It was written under the release-gate machine
lock (zero GPU work available), so every claim below is derived from artifacts
already on disk, from the source of both stacks, and from static arithmetic.
Nothing here has been measured today by me; the measured ladder is section 6.

**What must not be inherited:** the fresh row-15 line in
`notes/no-panic-governor.md` calls the wrong output *"the row's known
MLX-quantization bug"*. **There is no such prior finding.** The 2026-08-03
diagnosis (`7932b4d`) attributed the garbage to the MLX command-buffer split
and **explicitly exonerated the quantized dots** ("layer-0 KV bit-exact vs CPU
at K=4096; pre- and post-`fdc7cde` engines fail identically"). The label is
corrected in that note and in STATUS; the attribution itself is now *under
test*, not assumed — see section 4, where three pieces of the fresh evidence
sit badly with it, and where a fourth candidate reconciles them.

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

* jax-CPU runs this row **coherently** at 2118 ms/tok (`34f627c`, 2026-08-02):
  the checkpoint, the qwix conversion and the harness are exonerated as the
  *cause of wrongness* — whatever this is, it is on the metal side.
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
