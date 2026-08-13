# Native plugin vs Stage 1 — the complete performance baseline (2026-08-12)

*Campaign: both texmo suites and every non-embargoed model row, measured on
the phase-2 native plugin (`plugin-native`) and on the frozen Stage 1
trampoline, on the tree at 845ab89. Sequential throughout, machine lock held
for every measured run, `guarded_run.sh` (precheck + `mem_guard.sh`) for every
model row.*

> **2026-08-12 update (P18, tree 8c61e72).** The exported-symbols relink is in
> the tree as the default build and its validation was re-run for real; the
> model rows were then re-measured with the P17 recognizer emits. Table 3 now
> carries both native columns, and "Where the performance phases must act" has a
> status section. Everything above the tables is the original P16 campaign and
> is left as written.

**The campaign was HALTED at 03:33 by kernel panic #8**, during the row-9
(R1-Distill-32B) *native* attempt — a watchdog wedge in the 65 GB streaming
load, the same signature as panic #7 (footprint 54 GB, RSS 54.6, system
64.5 GB, every guard sample "ok", flight log stops mid-line:
`r1-distill-32b-0812-033145-flight.log`). The native plugin has no equivalent
of the Stage 1 engine's load-phase cache-clear cadence; that fix is being
built separately. Everything below completed **before** the panic. Rows not
reached are marked *not run (campaign halted)*.

## What the two stacks are

| stack | selection | what it is |
|---|---|---|
| **Stage 1** | default env | `plugin/` trampoline → `metaljax.engine` (Python compile) → the C++ tape (`native/`, the M6 default engine) |
| **native** | `METALJAX_PLUGIN_PATH=…/libmetal_pjrt_native.dylib` | `plugin-native/`: C++ StableHLO parse + lowering + the same executor runtime. **No recognizer emits and no msl_scan in the lowering** — that is what this baseline measures |

## Protocol and route calibration

`scripts/texmo_topconfs.py` drives `metaljax.engine` directly, which is a
Stage-1-only route (the native plugin holds no Python engine), so a new runner
— `scripts/bench_texmo_pjrt.py` (untracked, written for this campaign) — runs
the same training chunks **through jax/PJRT**, where the environment's plugin
choice is the thing being measured. Results are forced through `np.asarray`
(`jax.block_until_ready` is a no-op on this backend, CLAUDE.md item 9).

Three routes were run on top_confs so the columns are comparable:

| comparison | geomean | reading |
|---|---:|---|
| anchor (0.11.3, engine route, 256-step chunks) / Stage 1 today (same route) | **1.046** | today is 4.6 % faster |
| jax-CPU control, anchor / today | 0.986 | machine 1.4 % slower — ambient |
| Stage 1 engine-route@256 / Stage 1 PJRT-route@64 | **1.009** | **the route and chunk-length change is worth 0.9 %** |

The third line is what licenses the tables: the PJRT-route, 64-step-chunk
numbers used for the native pairing sit within 1 % of the anchor's own route.

---

## Table 1 — texmo 106-config suite (`benchmarks/texmo-suite.csv`)

PJRT route, 64-step chunks, ms/step. No historical perf anchor exists for this
suite (it is the correctness gate's config list), so the Stage-1 column is the
reference. Full per-config data:
`~/.cache/metaljax-bench/logs/native-baseline/suite106-{stage1,native}.jsonl`.

| aggregate | n | Stage 1 anchor | Stage 1 today | native today | native/Stage 1 |
|---|---:|---|---|---|---:|
| whole suite (geomean) | 106 | *(none — new suite)* | 1.00× ref | — | **4.24×** |
| whole suite (median) | 106 | | | | 3.37× |
| `big*` (large cells: gru/lstm 512–1024, transformer) | 34 | | | | **1.34×** |
| `mid*` | 30 | | | | 3.22× |
| `db*` (small recurrent — msl_scan territory) | 40 | | | | **14.61×** |
| `synth*` | 2 | | | | 1.52× |

Distribution: 33 of 106 configs within 1.2×, 41 between 1.2× and 10×, **32 at
or above 10×**. Every config ran on both stacks (0 errors, 0 declines).

Worst rows (ms/step):

| config | Stage 1 | native | ratio |
|---|---:|---:|---:|
| `db02-b4l1024` | 0.790 | 72.7 | **92×** |
| `db03-b4l1024` | 0.818 | 74.1 | **91×** |
| `db13-b4l1024` | 1.877 | 109.5 | **58×** |
| `db14-b16l128` | 0.563 | 32.3 | **57×** |
| `db15-b16l128` | 0.627 | 33.4 | **53×** |
| `db17-b4l1024` | 5.747 | 305.7 | **53×** |
| `db15-b64l256` | 1.285 | 61.1 | **48×** |
| `db11-b64l256` | 0.647 | 30.2 | **47×** |

`db02-b4l1024` at 72.7 ms/step is the pre-msl number of CLAUDE.md item 8
(72.9 ms) to three digits: without msl_scan the native plugin computes exactly
what Stage 1 computed before that milestone.

Rows where **native is faster** — matmul-bound work with no msl plan, where the
C++ compile path wins on dispatch:

| config | Stage 1 | native | ratio |
|---|---:|---:|---:|
| `big09-b8l256` (`bytes.emb.1024\|rnn.1024.tanh`) | 38.59 | 26.07 | **0.68×** |
| `big14-b32l128` (transformer d512) | 21.89 | 17.91 | **0.82×** |
| `big14-b8l256` | 11.97 | 10.61 | 0.89× |
| `big13-b8l256` (`bytes\|lstm.1024`) | 95.47 | 85.43 | 0.89× |
| `mid13-b64l128` (`attn.512.8.64`) | 6.78 | 6.27 | 0.92× |

---

## Table 2 — top_confs (163 configurations)

Anchor: `notes/data/texmo-topconfs-2026-08-05.jsonl` (0.11.3 release gate).
Stage-1-today column is the **same runner and route as the anchor**
(`scripts/texmo_topconfs.py`, 256-step chunks, checks included: **163/163 ok,
0 FAIL**); the native pairing is the PJRT route at 64-step chunks.

| aggregate | n | Stage 1 anchor | Stage 1 today | native today | native/Stage 1 |
|---|---:|---:|---:|---:|---:|
| geomean (ratio vs anchor / vs Stage 1) | 163 | 1.000 | **1.046 faster** | 0.029 (34× slower than anchor) | **36.46×** |
| median | 163 | | | | 46.3× |
| weights 0–100 | 47 | | 1.023 | | 50.1× |
| weights 100–500 | 61 | | 1.058 | | 31.2× |
| weights 500–1500 | 29 | | 1.052 | | 27.6× |
| weights 1500+ | 26 | | 1.053 | | 40.4× |
| configs beating jax-CPU | 163 | 53 (recorded) | **54** | **0** | |

Distribution: 8 of 163 within 1.2×, 5 between 1.2× and 10×, **150 at or above
10×**. Stage 1 vs anchor: best +18.7 % (`tc003-w10`), worst −5.4 %
(`tc145-w1888`) — no regression outside noise.

Worst native rows (ms/step):

| config | spec | shape | Stage 1 | native | ratio |
|---|---|---|---:|---:|---:|
| `tc018-w27` | `bits.1+bp\|split.add(mingru.1, pass)-norm-…` | b1 l4096 | 1.585 | 277 | **175×** |
| `tc014-w23` | `…mingru.1…` | b1 l2048 | 0.868 | 143 | **165×** |
| `tc015-w24` | `…mingru.1…` | b1 l2048 | 0.886 | 142 | **160×** |
| `tc019-w28` | `…mingru.1…` | b1 l2048 | 0.908 | 142 | **156×** |
| `tc029-w47` | `…mingru.1-rglru.1…` | b1 l2048 | 1.677 | 259 | **155×** |

At parity (no msl plan in the loop): `tc000-w5` 0.98×, `tc001-w7` 1.00×,
`tc075-w229` 1.00× — i.e. where msl does not fire the two stacks are
indistinguishable, which is the P5 finding (`METALJAX_MSL=0` parity) confirmed
on the whole suite.

**The single sentence for this table**: the entire top_confs sweep is
msl_scan's home ground, so the native plugin loses the whole 0.3.0-era win —
and with it every one of the 54 configs where metaljax beats jax-CPU.

---

## Table 3 — model rows (STATUS.md numbering)

Headline metric per row as in STATUS.md (LLM = warm decode ms/token; vision =
forward ms; diffusion = ms/step; training = ms/step). "peak" = guard flight-log
footprint. Anchor column = `benchmarks/models.md` 0.11.3.

Two native columns: **P16** is the baseline this campaign measured (no
recognizer emits, no msl_scan, no exported-symbols relink) and **P18** is the
same rows re-measured on 2026-08-12 after the P17 emits landed and the relink
became the default build (`notes/data/p18-relink-models-2026-08-12.jsonl`;
artifacts under `~/.cache/metaljax-bench/logs/p18-relink/`). The ratio column is
**P18 / Stage 1 today**.

| # | model | metric | 0.11.3 anchor | Stage 1 today | native P16 | native P18 | P18/S1 | notes |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | gemma4-31B bf16 | ms/tok | 237.5 | **243.1** (peak 66 G) | 301.6 (67 G) | *not re-run* | — | dense decode; the sdpa emit now exists but this is a 66 G row |
| 2 | gemma4-12B bf16 | ms/tok | 92.5 | **93.9** (29 G) | 98.8 (30 G) | *not re-run* | — | |
| 3 | gemma4-26B-A4B (MoE) | ms/tok | 44.3 | **43.7** (53 G) | 300.5 (54 G) | **43.4** (53 G) | **0.99×** | the MoE expert-gather emit closes 6.88× to parity; greedy tokens 64/64 identical to Stage 1 |
| 4 | gemma4-E2B bf16 | ms/tok | 27.5 | **27.2** (12 G) | 27.2 (12 G) | *not re-run* | — | was already exact parity |
| 5 | Qwen3-8B bf16 | ms/tok | 57.8 | **58.2** (18 G) ᶜ | 61.5 (18 G) | **57.9** (18 G) | **0.99×** | ᶜ same-day control re-measure (P16 read 59.1 — the Stage 1 column is stable to 1.5 %) |
| 6 | Llama-3.1-8B bf16 | ms/tok | 54.2 | **55.5** (18 G) | 57.6 (18 G) | **54.7** (18 G) | **0.99×** | |
| 7 | gpt-oss-20b (MXFP4) | ms/tok | 22.2 | **21.9** (26 G) ᵖ | 9497.6 (21 G) | **guard-killed** → **25.3** (35 G) ᴾ¹⁹ | **1.16×** | P18 was killed at 46 G under 45 and 62 G under 60; **P19 UNBLOCKS IT** — 35 G peak at the historical 45 G budget, 128 tokens, four samples 25.3/25.5/25.3/25.3. ᵖ Stage 1 re-measured same-day and reproduced its anchor exactly |
| 8 | Qwen3.6-35B-A3B | — | ✗ | *not run* | *not run* | *not run* | — | PAUSED, kernel-panic embargo (TASKS.md) |
| 9 | R1-Distill-32B | ms/tok | 217.7 | **213.8** (67 G) | **PANIC #8** | *(main agent)* | — | native embargoed; the retry is the main agent's |
| 10 | DeepSeek-V2-Lite | — | ✗ | *not run* | *not run* | *not run* | — | embargoed (maxtext 8B class) |
| 11 | Qwen3-0.6B maxtext decode | ms/tok | 15.8 | **16.42** | 16.67 | *not re-run* | — | token stream diverges from Stage 1 at token ~3 (see incidents) |
| 12 | Mixtral 8×7B | — | ✗ | *not run* | *not run* | *not run* | — | PAUSED, same wedge class as row 8 |
| 13 | gemma4-E2B keras-int4 | ms/tok | 81.1 | **80.6** (peak 44 G) | 454.6 (44 G) | **249.0** (48 G) → **275.6** (46 G) ᴾ¹⁹ | **3.42×** | qmm fires on all 777 dots (group 64/128, regrouping engaged) and **prefill is already ahead of Stage 1** (218.3 vs 241.0); the residual is the decode loop running UNCOMPILED — see below. Greedy tokens 64/64 identical to Stage 1. **P19 is timing-NEUTRAL here** (the P19-off control on the same binary reads 271.7, and P18's own byte-cap control read 274.6 — 249.0 was the low end of this row's spread); what P19 changes is the **steady state, 4.2 → 3.2 GB**, and 518 of the 777 pack builds |
| 14 | maxtext qwix-int8 0.6B | ms/tok | 32.5 | **60.1** ⚠ | 62.1 | **56.8** | **0.95×** | ⚠ **Stage-1 regression vs anchor, 1.85×** (see below); same greedy text |
| 15 | qwix-int8 Qwen3-8B | — | ✗ | *not run* | *not run* | *not run* | — | embargoed (MLX command-buffer bug) |
| 16 | SigLIP 2 fwd b1 | ms | 82.9 | **85.8** (7.6 G) | 108.7 (11 G) | **96.9** (8.2 G) | **1.13×** | |
| 16b | SigLIP 2 fwd b32 | ms | — | **2501.6** | 4444.8 | **2400.5** | **0.96×** | the sdpa emit closes 1.78× and goes past Stage 1 |
| 17 | SD 3.5 Large @1024² | ms/step | 5141 | **5107** (23 G) | 16016 (27 G) | **5781.6** (24 G) | **1.13×** | sdpa emit: 3.14× → 1.13×; real image (pixel_std 68.0) |
| 17b | SD 3.5 Large @512² | ms/step | 1389 | **1520.7** (21 G) ᶜ | *not run* | **1234.8** (21 G) | **0.81×** | ᶜ measured today for this pairing. Native is 19 % **faster** than Stage 1 and 11 % faster than the 0.11.3 anchor; real images both (pixel_std 61.1 native / 77.5 Stage 1) |
| 18 | LoRA E2B train | ms/step | 407 | **402.1** (56 G) | 656.3 (49 G) | *not re-run* | — | |
| 19 | maxtext train 0.6B | ms/step | 440 | **956.5** ⚠ | 962.0 | *not re-run* | — | ⚠ **Stage-1 regression vs anchor, 2.17×**; losses bit-identical across stacks |
| 20 | 235B-A22B 3-bit | — | ✗ | — | — | — | — | mlx-only row |

**Unmeasured**: rows 8/10/12/15 (pre-existing embargoes), row 9 native
(embargoed; the retry is the main agent's), the P18 cells of rows 1/2/4/11/18/19
(not re-run — 1/2/18 are 30–67 GB rows and 4/11/19 were already at parity), and
CPU-column re-measurements (not in scope — anchors used).

### The one blocked row, and the one gap the emits did not close

**Row 7 (gpt-oss-20b) is blocked on memory, not compute.** The emits fire, but
building those packs without qmm's row-blocked `_Source` evaluation and without
its cross-executable build cache (P17 left both out deliberately) keeps a full
pack set per compiled shape live at once. Measured: a steady climb to 46 GB at
the row's historical 45 GB budget, and an oscillating 49–62 GB plateau at 60.
Stage 1 runs the same row at 25 GB. No further escalation was attempted — 62 GB
is panic #7/#8 territory.

> **UNBLOCKED 2026-08-13 (P19).** Both optimizations are ported
> (`notes/cpp-p19-packing.md`) and the row completes at the historical 45 GB
> budget: **35 GB peak, 25.3 ms/tok, 128 tokens** — 1.16× of Stage 1's 21.9,
> inside the ~1.5× target and better than P17's micro proxy (1.55×). Raw cells:
> `notes/data/p19-packing-models-2026-08-13.jsonl`.
>
> **Which of the two ports did it is not what was expected.** An ablation at the
> same budget, one knob at a time:
>
> | configuration | peak | outcome |
> |---|---:|---|
> | both (P19 default) | **35 G** | ok, 25.3 ms/tok |
> | cache off, blocking on | 46 G | **guard-killed** — P18's number to the gigabyte |
> | blocking off, cache on | 36 G | ok, 25.5 ms/tok |
> | neither (P17/P18) | 46 G / 62 G | guard-killed at both budgets |
>
> So the **build cache is the load-bearing fix** (46 → 36 GB: three executables
> were each building their own ~10 GB pack set) and row-blocking is worth a
> further gigabyte on top. P17's argument that "the tape already stages the
> evaluation op by op with last-use pruning" was substantially right about the
> per-weight TRANSIENT; what it did not cover was the same pack set built three
> times over. Both are in, and only the pair clears the 45 GB line.
>
> Mechanism, from the run's own log: 94 packs built and **188 reused** (three
> executables × 94 weights, a 100 % hit rate on the second and third), every one
> of the 94 blocked (47 in 16 row blocks, 47 in 32), and the pack-wave peak
> reported by the plugin's own libmlx is **33.9 GB for the first wave and 0.000
> GB for the other two** — the reuse waves allocate nothing at all. (That 33.9
> is a process-wide high-water mark, so it includes the resident model; the
> flight-log footprint is the figure to compare across rows.)
>
> Row 7 does **not** have row 13's compile problem: `METALJAX_TRACE_BUDGET=1e7`
> gives 25.3 ms/tok, the same number, with the compile decisions bit-identical
> (16 compiles / 354 compiled calls either way). Its decode loop was already
> compiling.
>
> Scrutiny: the greedy tokens diverge from Stage 1 at index 52 of 64. There is
> no prior native token record for this row (P18 never completed it), so this is
> a first observation rather than a change, and it is the same late-divergence
> ladder class already carried for rows 3, 5 and 11.

> **Row 13 (P19).** Peak barely moves — 48 → 46 GB — and the reason is worth
> recording: that peak is the **keras streaming LOAD transient**, not the packs.
> Stage 1's own is 44 GB on this row (see the regression section below), and the
> plugin's pack wave peaks at 6.64 GB against a 46 GB flight peak. What P19 does
> change here is the steady state, **4.2 → 3.2 GB**, and the work: **259 packs
> built, 518 reused** (three executables × 259 weights, 0 fingerprint declines),
> with the two reuse waves peaking at 0.000 GB. All 259 pack whole, on both
> stacks — a keras `Dense`'s `[K, N]` weight needs a transpose to reach the
> `[(B,) N, K]` matrix `quantized_matmul` wants, which is exactly the
> precondition `_blocking` tests.

**Row 13's residual 3.09× is the compile gate reading the UNFUSED IR.** The
fused decode program reports `compiles=0 compiled_calls=0 serial_loops=1`: the
decode while body never compiles, so the loop runs op by op. It is not the byte
budget (`METALJAX_COMPILE_BYTES_MB=1e8` alone: 274.6 ms/tok, no change) and it
is not the packs (`METALJAX_RECOGNIZE=0` also reports `compiles=0`). It is the
**cost** term: `body_compile_max = min(kTraceBudget/cost, …)`, and `BlockCost`
walks the StableHLO block, still charging every op the qmm emit ABSORBS — the
whole int4 dequant chain. With `METALJAX_TRACE_BUDGET=1e7` the same row measures
**85.5 ms/tok = 1.06× of Stage 1**. Making the fused lowering's cost and byte
estimates follow the rewrite plan is worth 2.9× here, and is a candidate
explanation for any other emit row that lands short of parity.

---

## Where the performance phases must act, ordered by measured gap

1. **`msl_scan` on the native lowering — 36× geomean (top_confs), 14.6× on the
   suite's `db` class, 175× worst case.** This is the largest *aggregate* gap
   and the one that decides whether metaljax beats jax-CPU at all on texmo:
   Stage 1 wins 54 of 163 top_confs, native wins 0. Parity is exact wherever no
   msl plan fires, so the port is additive — nothing else has to move.
2. **qmm (quantized matmul) emit — 434× on gpt-oss-20b, 5.6× on E2B-int4.**
   The largest single-row gap in the campaign. gpt-oss decodes at 9.5 s/token
   natively against 21.9 ms on Stage 1. Memory is *not* the blocker (21 GB peak,
   the packed weights survive) — the whole cost is in-graph MXFP4 dequant plus
   dense expert dispatch.
3. **MoE expert gather — 6.9× on gemma4-26B-A4B** (43.7 → 300.5 ms/tok, memory
   unchanged at 51.6 GB). This reproduces the pre-gather Stage 1 number (284)
   almost exactly.
4. **sdpa fusion — 3.1× on SD 3.5 @1024², 1.78× on SigLIP b32, 1.24× on the
   31B decode, 1.63× on LoRA training.** Also the memory term: SD 3.5's peak
   moves 23 → 27 GB without it.
5. **The load-phase memory cadence (correctness/safety, not throughput).**
   Panic #8 is the native plugin missing Stage 1's clear/flush discipline on a
   65 GB streaming load. Until it lands, no 60 GB+ row can be attempted
   natively.

Below the frontier, and worth recording because they are *already* good: dense
bf16 decode is within 4–5 % (rows 2/5/6), E2B is exact parity, maxtext decode
and training are within 1–2 %, and on large matmul-bound texmo configs the
native path is **up to 32 % faster** than Stage 1 (`big09-b8l256` 0.68×). The
dispatch argument for the C++ engine is settled — what is left is the emits.

### Status of that list after P17 + P18 (2026-08-12)

Items 2, 3 and 4 are **closed on the rows that could be re-run**: MoE 6.88× →
0.99×, sdpa 3.14× → 1.13× (@1024²), 1.78× → 0.96× (SigLIP b32), and at 512² SD
3.5 is now 0.81× of Stage 1. Item 1 (`msl_scan`) is untouched and remains the
whole texmo gap. Three new items join the list, all of them found by re-running
these rows:

6. **The fused lowering's compile decisions read the unfused IR** — `BlockCost` /
   `BlockBytes` charge the ops a recognizer absorbs, so a decode body that the
   emits made cheap can still exceed `METALJAX_TRACE_BUDGET` and run
   uncompiled. Worth **2.9×** on row 13 (249.0 → 85.5 ms/tok with the budget
   lifted). Suspected on any emit row that lands short of parity.
7. ~~**Pack building has no memory discipline**~~ — **CLOSED 2026-08-13 (P19)**.
   Both are ported; row 7 completes at its historical 45 GB budget (35 GB peak,
   25.3 ms/tok = 1.16× of Stage 1) and row 13's steady state drops 4.2 → 3.2 GB
   with 518 of its 777 pack builds eliminated. The ablation above says the
   **cache** was the load-bearing half. Row 13's 46 GB peak survives and is now
   attributed: it is the keras streaming load transient, which Stage 1 shares
   (44 GB), not the packs (6.6 GB).
8. **Row 5's greedy tokens now diverge from Stage 1 at token 61 of 64**
   (they agreed before the sdpa emit) — the footnote-21 tie-flip class, but it
   is a *new* divergence introduced by fusion and should be walked down the
   logit-delta ladder rather than assumed benign.

---

## Stage-1-vs-anchor regressions (flagged per the campaign's charter)

| row | anchor | today | ratio | attribution |
|---|---:|---:|---:|---|
| 14, maxtext qwix-int8 0.6B | 32.5 | 60.1 | **1.85× slower** | `METALJAX_ENGINE=py` gives 42.3 → ~1.30× predates the tape, ~1.42× is the tape |
| 19, maxtext train 0.6B | 440 | 956.5 | **2.17× slower** | `METALJAX_ENGINE=py` gives 1043 — **not** the tape; regression is older or environmental |

Everything else is within ±4.5 % of its anchor (rows 1–6, 11, 13, 16, 17, 18),
and the texmo top_confs sweep is 4.6 % *faster* than its anchor with a 0.986
CPU control. Two Stage-1 memory observations, both new on this tree and both
independent of the plugin choice: the **LoRA row's load transient peaks at
56 GB** and the **E2B-int4 row's at 44 GB** (steady states 10.2 and 3.1 GB) —
both were guard-killed at a 45 GB budget before completing at 70.

## Incidents and findings

* **PANIC #8 (row 9 native)** — see the header. Evidence:
  `r1-distill-32b-0812-033145-flight.log`.
* **The native plugin cannot share a process with TensorFlow, array_record, or
  any other statically-linked protobuf/LLVM carrier.** Symmetric, load-order
  dependent, and fatal at `dlopen`: TF first → `EXC_BAD_ACCESS` inside
  `google::protobuf::internal::AddDescriptors` (the plugin's descriptor
  registration executing TF's `MergeFromImpl` — dyld weak-definition
  coalescing); plugin first → `CommandLine Error: Option 'info-output-file'
  registered more than once` when TF's LLVM loads. It reached every model row:
  the keras rows import TF directly, the gemma-lib rows through
  `kauldron`, the maxtext rows through `array_record_module.so`.
  **Fix, verified**: link the dylib with an exported-symbols list holding only
  `_GetPjrtApi` and `_metaljax_native_set_callback_trampoline`
  (`-Wl,-exported_symbols_list`). The dylib drops 166 MB → 46 MB and coexists
  with TF. *Every native model number in Table 3 was measured with that
  relinked build*; `plugin-native/metal/BUILD` was restored immediately
  afterwards and the tree carries no trace of it (the experimental dylib lived
  in session scratch and was wiped by the reboot). It is a one-line build
  change and it belongs in the plugin.
  **LANDED 2026-08-12 (P18)** as `plugin-native/metal/exported_symbols.exp` +
  the `linkopts`/`additional_linker_inputs` pair in `plugin-native/metal/BUILD`,
  the DEFAULT build, with `plugin-native/coexist_test.py` as its standing
  contract. Re-verified from scratch, not from the transcript:
  `notes/data/p18-relink-battery-2026-08-12.txt`.
* **`mx.get_active_memory()` is unreliable under the native plugin** (0.0 on
  the gemma-venv rows): the Python `mlx` module and the plugin's linked libmlx
  can be two runtimes. Read memory from the guard flight log.
* **Row 11 token divergence**: Stage 1 and native produce different greedy
  continuations from token ~3 (`"The capital of Italy is Rome…"` vs `"The
  capital of France is also…"`). Consistent with the fused-vs-unfused attention
  rounding (footnote-22 ladder class), not established as benign — worth a
  logit-delta check when the emits land.
* **int4 run-to-run outlier**: one Stage-1 run of row 13 measured 439.0 ms/tok
  where three others give 80.4 / 80.6 (and the anchor 81.1). It was the run
  immediately following a five-model chain; treat sub-suite model timings taken
  straight after other big loads as contaminated (the suite-context trap of
  CLAUDE.md item 12, in the model harness).
* **gpt-oss native at 128 decode tokens never finished** (33 min, memory flat at
  21 GB, killed); the row was re-measured at 8 decode tokens.

## Provenance

Every number in the tables is sourced to a surviving artifact under
`~/.cache/metaljax-bench/logs/`:

| table | artifact |
|---|---|
| top_confs Stage 1 (engine, anchor-comparable) | `native-baseline/topconfs-stage1-engine.jsonl` |
| top_confs Stage 1 / native (PJRT) | `native-baseline/topconfs-{stage1,native}-pjrt.jsonl` |
| suite-106 both stacks | `native-baseline/suite106-{stage1,native}.jsonl` |
| model rows 1–7, 9, 13, 16, 17, 18 | `<bench-id>-0812-<stamp>.jsonl` + `-flight.log` |
| row 7 native (8 tokens) | `native-baseline/row7-native-0812-024351.jsonl` |
| model rows 11, 14, 19 (both stacks + py-engine controls) | `native-baseline/{qwen3-06b-maxtext,maxtext-qwix-int8,maxtext-train-06b}-{s1,native}-0812-*.log` |
| **P18 native cells + the Stage 1 controls** | `p18-relink/<bench-id>-<tag>-0812-<stamp>.{jsonl,log,flight.log}`, summarised in `notes/data/p18-relink-models-2026-08-12.jsonl` |
| **P18 relink evidence** (coexistence, size, perf neutrality, battery) | `p18-relink/{coexist-*,execute_test-relinked,texmo_gate-relinked,wheel_poc-relinked,perfneutral-*}.log`, summarised in `notes/data/p18-relink-battery-2026-08-12.txt` |

**The formerly UNVERIFIED relink validation is now verified**, from a rebuilt
dylib rather than a transcript, and it came out better than the transcript said:
`execute_test.py` **520 of 520** (the intermittent 8-thread
`There is no Stream(gpu, N) in current thread` row passed this run),
`texmo_gate.py` 106/106, `smoke_test.py`, `decline_census.py` 35/35,
`ingest_test.py` 8/8, `bazel test //...`, and the wheel run from a fresh 3.13
venv. Perf neutrality was re-measured A/B/A against a kept copy of the pristine
no-list dylib on the same five suite configs (relinked mean / pristine: 0.998,
0.984, 0.976, 0.984, 1.001 — every delta inside the relinked passes' own 2.4 %
spread). The wheel drops 42.2 MB → 11.8 MB with the dylib.
