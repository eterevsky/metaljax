# 0.11.5 final gate battery — 2026-08-17 (tree 6500370 → ff2cedd, docs-only)

**This document is the release gate.** Every gate is filled in as it lands;
nothing is pre-judged. Era rules: measurement only (harness-fix exceptions are
noted where used), machine lock held for every measured phase, strictly
sequential, big rows first on a fresh machine, durable artifacts, nothing
committed, pushed or uploaded.

**RELEASE RULE 1 — no stale numbers** (Oleg, 2026-08-16, after the 0.11.4
near-miss): *every number in a release table must come from the release binary.
Changes after the last benchmark run are acceptable only if they provably cannot
move a number; otherwise re-measure the affected rows before release.*

**RELEASE RULE 2 — never "PASS" over a regression**: *a significant regression
on any test suite or benchmark makes the gate verdict REGRESSION, not PASS.
Releasing over one requires (a) Oleg's explicit confirmation, (b) the regression
stated in the gate report itself.*

**THE NO-PANIC CONTRACT** (Oleg, 2026-08-17, after panic #9): *metaljax must
NEVER cause a kernel panic. … preferred = degrade performance under memory
pressure; acceptable = a clean OOM error (RESOURCE_EXHAUSTED at the PJRT
boundary); never a wedge. Applies to EVERY model row including the previously
embargoed ones (9/10/12/15/20) — they may OOM-error, they may not panic. A
0.11.5 release requirement alongside the test gates.*

## Provenance of this battery

This battery ran **twice**. A first pass on 2026-08-16 measured gates 1–4 and
6 on tree `bdaaa1c` (dylib `aa7bc0b6…`) and was interrupted by **kernel panic
#9** during its model phase. The memory governor then landed (`6500370`), which
under release rule 1 invalidates every earlier number, so **everything in this
document was re-measured on the governor build**. The first pass's artifacts are
kept at `~/.cache/metaljax-bench/logs/release-0.11.5/prepanic/` and its findings
that survive as *method* (the top_confs position effect, the `db08` finiteness
flake) are carried below with their evidence.

* **Tree**: `6500370` at build time; `ff2cedd` at write-up (three docs/probe
  commits — the `models.md`/`STATUS.md` restructure and the row-15
  investigation, none of which touches plugin code, so the frozen binary's hash
  was re-verified against `bazel-bin` before anything was banked).
* **Artifacts**: `~/.cache/metaljax-bench/logs/release-0.11.5/`.

## Checklist

**Checklist status as of 2026-08-18** (the table below it is the 2026-08-17
pre-vendoring record, kept as history). Two changes moved the binary after
that record — the vendored patched MLX and the P28 benefit gate — so under
release rule 1 the FINAL binary (`frozen-vendor-d651add3`) must re-attest
everything numeric. The consolidated re-gate is running; until it lands:

| # | gate | current status on the FINAL binary |
|---|---|---|
| 1 | Freeze | **PASS** — `frozen-vendor-d651add3`, wheel dylib byte-identical |
| 2 | Pinned jax suite | **RE-RUN IN PROGRESS** (the MLX substitution is numerics-relevant by design; the banked 99.54 % zero-new run was pre-vendoring) |
| 3 | `tests/` native leg (Stage 1 leg dropped — out of release scope per Oleg, 2026-08-18) | **RE-RUN IN PROGRESS** |
| 4 | texmo | **PASS** — `texmo_gate` 106/106 + suite-106 within noise of the recorded arms (0.9989 / 1.0026), both on the final binary |
| 5 | Model rows | **RE-MEASUREMENT IN PROGRESS**; row 15 **FIXED** (attested on the final binary); rows 11/14/19 re-spotted; **row 1 OPEN** — its 237.3 cell does not reproduce (256.8–292.3 on both libraries, vendoring-neutral by A/B; §re-attestation) |
| 5b | The no-panic contract | **PASS** (design unchanged; contract suites re-attested on the final binary) |
| 6 | Plugin contract suites | **PASS** on the final binary (incl. all 11 command-buffer detectors: 6 correctness PASS, 5 canaries can no longer find a corrupting budget) |
| 7 | Wheels — **native-only** (the trampoline/Stage-1 wheel is no longer a release artifact) | **PASS** — 12 files, zero Stage-1 modules, carries the gated dylib byte-identically, installs with no mlx in the venv |
| 8 | Finale | **PENDING** — flips when gates 2, 3 and 5 land on the final binary and row 1 is dispositioned |

### The 2026-08-17 pre-vendoring record (history — superseded above)

| # | gate | verdict |
|---|---|---|
| 1 | Freeze the release dylib (build, sha256, byte-identity) | **PASS** |
| 2 | Pinned jax suite, native, 164 files, `--jobs 1`, vs the 129 whitelist | **PASS** — 28,073 / 129, zero new |
| 3 | `tests/` both legs (Stage 1 + native) | **PASS** — 1258 / (1187+71 = 1053+205 deselected) |
| 4 | texmo: suite-106 + top_confs pairings, both stacks; both correctness gates | **PASS** |
| 5 | Model rows, every non-embargoed row, guarded | **PASS** — the one regression (P27's flush watermark, rows 11/14) is **RESOLVED** by P28's benefit gate, re-measured at the historical budgets |
| 5b | The no-panic contract | **PASS** |
| 6 | Plugin contract suites | **PASS** |
| 7 | Wheels: both variants, fresh-venv installs, `twine check` | **PASS** — nothing uploaded |
| 8 | Finale: verdicts, release notes, recommendation | **GO** — the one regression is resolved, not accepted (P28) |

---
## Gate 1 — the release binary, frozen — **PASS**

`g1_freeze.sh`, one lock hold, 2026-08-17 13:26:20.

| | |
|---|---|
| tree at build time | **`6500370`** — *memory governor: the no-panic contract, built and proven*; `git status` clean |
| `bazel build //metal:libmetal_pjrt_native.dylib` | rc=0 |
| **sha256** | **`ebe56e7168eff581906718dfb24153b2575a69a4b9698801c01aea0a89fb9f31`** |
| frozen copy (every native measurement below) | `~/.cache/metaljax-bench/frozen-0.11.5-ebe56e71.dylib` |
| byte-identity vs the tree's `bazel-bin` | **OK** |
| size | 47,406,600 B (46 MB, the P18 exported-symbols relink) |

**The release build reproduces the governor campaign's `frozen-gov7` byte for
byte** (`ebe56e71…` both). That is what licenses gate 5 to bank the campaign's
big-model rows: they were measured on *this* binary, not on a predecessor.

**Stage-1 stack, verified frozen** (the default wheel's stack): `src/metaljax/`
last changed 2026-08-10 (`27ec088`), `plugin/build/libmetal_pjrt.dylib` dated
2026-07-31. Neither is touched by any commit in this release, so Stage-1 numbers
are measured on the same code every prior campaign measured.

## Gate 6 — plugin contract suites on the frozen binary — **PASS**

Run in the same hold, **before** any long phase (P26b's policy: a broken binary
should cost minutes, not an hour).

| suite | result | wall |
|---|---|---:|
| `smoke_test.py` | all checkpoints passed | 1 s |
| `execute_test.py` | **all cases match the CPU backend**, **549 `ok` rows** (544 + the five governor contracts) | 38 s |
| `ingest_test.py` | **0 failed** — 13 checks (8 + the five page-cache / refusal checks) | 36 s |
| `decline_census.py` | **35 of 35** programs lower | 0 s |
| `coexist_test.py` (`.venv`) | passes, but **both carriers skipped** — the documented `.venv` trap | 0 s |
| `coexist_test.py` (**bench** venv) | **tensorflow, both load orders: PASS** | 6 s |
| `coexist_test.py` (**gemma** venv) | **all four**: tensorflow ×2 + array_record ×2 PASS | 8 s |
| `bazel test //... --nocache_test_results` | `//metal:runtime_gil_free_test` **PASSED** (uncached) | 1 s |

## Gate 5 — the model rows — **PASS** (was REGRESSION; the named one is RESOLVED, the soft second stands as named; **row 1's cell non-reproduction surfaced later and is OPEN** — see the re-attestation section and the top checklist)

One guarded process per row (`g5_model.sh`: recovery precheck, settle to a
recovered machine, `mem_guard.sh` at the row's historical budget, durable logs,
never chains), `METALJAX_DEBUG=1` on every headline run, machine lock held per
sub-phase. **Big rows first on a fresh machine**, with one deliberate exception:
row 19 ran first because its disputed cell needed the cleanest environment
available and it is a 25 GB row.

ᴳ = banked from the governor campaign, measured on this same binary
(`notes/no-panic-governor.md`); every other cell measured tonight.

| row | metric | anchor (0.11.3) | last cell | **0.11.5** | /cell | peak | tokens |
|---|---|---:|---:|---:|---:|---:|---|
| 1 gemma4-31B bf16 | ms/tok | 237.5 | 239.8 ᴾ²⁶ᵇ | **237.3** ᴳ | 0.990 | 67 GB (RSS 14) | ≡ Stage 1, ≡ P26b |
| 2 gemma4-12B bf16 | ms/tok | 92.5 | 92.9 ᴾ²⁷ | **94.3** | 1.015 | 30 GB | ≡ Stage 1, ≡ P27 |
| 3 gemma4-26B-A4B MoE | ms/tok | 44.3 | 43.4 ᴾ¹⁸ | **43.4** | 1.000 | 53 GB | ≡ P18 native |
| 4 gemma4-E2B bf16 | ms/tok | 27.5 | 27.2 ᴾ²⁶ᵇ | **27.1** ᴳ | 0.996 | 12 GB | ≡ Stage 1, 4/4 samples ≡ |
| 5 Qwen3-8B bf16 | ms/tok | 57.8 | 58.1 ᴿᶜ | **58.5** (r2 58.7) | 1.007 | 18 GB | run-to-run divergence at 51 |
| 6 Llama-3.1-8B bf16 | ms/tok | 54.2 | 54.7 ᴾ¹⁸ | **55.8** | 1.020 | 18 GB | diverges from P18 at 51 |
| 7 gpt-oss-20b MXFP4 | ms/tok | 22.2 | 22.0 ᴿᶜ | **21.9** ᶜ (spread 21.9–24.2) | 0.995 | 34–35 GB | run-to-run divergence at 51 |
| 8 Qwen3.6-35B-A3B | ms/tok | ✗ | ✗ (paused since panic #7) | **29.6** ᴳ | first ever | 73 GB | — |
| 9 R1-Distill-32B | ms/tok | 217.7 | 214.4 | **210.7** ᴳ | 0.985 | 67 GB (RSS 14) | — |
| 10 DeepSeek-V2-Lite | ms/tok | ✗ | ✗ (guard-killed @122 GB) | **1865** ᴳ | first ever | 88 GB | — |
| 11 Qwen3-0.6B maxtext decode | ms/tok | 15.8 | 16.63 ᴾ²⁴ | **16.92** ᶠ | 1.017 | **25 GB** (was 16) | — |
| 12 Mixtral 8×7B | ms/tok | ✗ | ✗ | ✗ blocked on a 93 GB download ᴳ | — | 87 GB (checkpoint streams) | — |
| 13 gemma4-E2B keras-int4 | ms/tok | 81.1 | 80.3 ᴾ²⁷ | **79.0** | 0.984 | 46 GB | — |
| 14 maxtext qwix-int8 0.6B | ms/tok | 32.5 | 35.0 ᴾ²⁰ | **32.0** ᶠ | 0.914 | **26 GB** (was ≤25) | — |
| 15 qwix-int8 Qwen3-8B | ms/tok | ✗ | ✗ | runs at 369.7 ᴳ, **wrong output** | — | 79 GB | logits collapse |
| 16 SigLIP 2 (fwd b1) | ms | 82.9 | 87.9 ᴾ²⁰ | **88.4** (b32 2374.7) | 1.006 | 16 GB | — |
| 17 SD 3.5 512² / 1024² | ms/step | 1389 / 5141 | 1234.8 / 5781.6 ᴾ¹⁸ | **1259.9 / 5707.9** | 1.020 / 0.987 | 23 / 33 GB | — |
| 18 LoRA E2B train | ms/step | 407 | 360.2 ᴾ²⁷ | **359.2** ᴳ | 0.997 | 57 GB | — |
| 19 maxtext train 0.6B | ms/step | 440 | 469.7 ᴾ²⁷ | **456.1** | 0.971 | 25 GB | loss ≡ P27 to 13 digits |
| 20 235B-A22B 3-bit | — | ✗ | ✗ | harness declines ᴳ (no packed sub-byte path) | — | — | — |

**Fifteen of seventeen measurable rows are inside ±2 % of their cells**, three
of them better; rows 8/10 produce their first metaljax numbers ever. `msl` is
verified absent, not assumed, on every row that narrates (`msl narration
lines: 0` on rows 1–7, 13, 16, 17; rows 11/14/19 narrate declines only).

### REGRESSION 1 — rows 11 and 14 guard-kill at their historical budgets, and it is P27's flush watermark

Both maxtext decode rows were **killed by the guard at the budgets every
previous campaign used**: row 11 at `footprint 22.00 GB > budget 20 GB`, row 14
at `26.00 GB > 25 GB`. Neither had ever exceeded them (P24 measured row 11 at a
**16 GB** peak under the same 20 GB budget).

One variable at a time, same binary, same hold:

| arm | row 11 | row 14 |
|---|---|---|
| shipped defaults, historical budget | **guard kill** (22 GB, then 25 GB on retry) | **guard kill** (26 GB) |
| `METALJAX_MEM_GOVERNOR=0`, historical budget | **guard kill** (25 GB) — *not the governor* | — |
| **P25 flush semantics** (`METALJAX_FLUSH_CLEAR_MB=2048 METALJAX_FLUSH_FOOTPRINT_MB=0`) | **completes, peak 7.6 GB**, 16.85 ms/tok | **completes, peak 15 GB**, 32.14 ms/tok |
| shipped defaults, raised budget | completes, **peak 25 GB**, 16.92 ms/tok | completes, **peak 26 GB**, 32.00 ms/tok |

**P27's watermark costs these two rows 17 GB and 11 GB of peak footprint and
buys nothing**: the speeds are identical either way (16.92 vs 16.85; 32.00 vs
32.14). P27's own battery measured rows 19/18/13/2 and never these two, so the
consequence is new information, not a re-litigation — the policy lets an eager
main claim pool up to a 48 GB footprint target, and on a program whose live set
is ~7 GB that is 18 GB of dead pool standing where a user's budget used to be.

It is **pre-existing in the 0.11.5 tree** (P27 landed in `00fba0f`, before the
version bump), **not caused by the governor**, and **opt-out with one variable**.
Under release rule 2 it is stated here rather than absorbed: the cells above are
the shipped-default runs at raised budgets, and the rows' *published* memory
characteristics have changed.

#### RESOLVED 2026-08-17 — P28's benefit gate (`notes/cpp-p28-benefit-gate.md`)

Oleg chose option (b): benefit-gate the watermark, so the decode rows keep
their historical footprint and row 19 keeps its fix. The two rows' flight logs
name the counter that separates them, and it is not the flush count — the load
takes **134 hard flushes in a single call**, so P27 reads it as an eager main —
but the **live set**: the training step swings 6.8 → 20.5 GB every flush, while
the load holds ~3 GB flat and simply has 14 GB of freed weights land in the
pool at its last flush, and the decode step holds 1,197 MB at all 71 of its
flushes. `flush_bound` gained a third rule, entering as one more `min` so it
can only ever LOWER a bound:

```
earned = min(METALJAX_FLUSH_EARN_MULT * (live_hi - live_lo), live_hi)
```

`METALJAX_FLUSH_EARN_MULT` defaults to **2**; `0` restores P27 exactly. What
each program earns: the load **3.6 GB** (was the 32 GB cap), the decode step
**the floor**, the training step **20.5 GB** — more than the 18.6 GB pool it
actually uses, so it pays nothing.

Re-measured on `frozen-p28b-37060770`, shipped defaults, at the rows' **own
historical budgets**, one guarded process per run:

| row | budget | 0.11.5 as gated | **P28** | P25 semantics (control) |
|---|---:|---|---|---|
| **11** | **20 GB** | **0 of 6 complete** (21–25 GB) | **9 of 9 complete**, 16.61–16.83 ms/tok, peaks 9.6–19 GB | 9 of 9, 16.52–17.07, 7.6–17 GB |
| **14** | **25 GB** | guard kill (26 GB) | **4 of 4 complete**, 31.82–32.13 ms/tok, peaks 9.1–17 GB | completes, 32.14, 7.7–15 GB |
| **19** | 48 GB | 456.1 ms/step | **459.2 / 458.4 / 462.5 ms/step**, 25 GB, `loss` ≡ P27 to 13 digits | 811–834 ms/step |

Speed is unchanged on all three (the decode rows were never paying for the
pool, which was the finding). Row 11's peak is a sub-second **live** transient
in the orbax restore — ~17 GB under every policy including P25's — with
whatever the pool holds standing beside it; at 2 Hz sampling the peak readings
scatter, so the completion counts are the statement.

**What the resolution cost elsewhere: nothing measurable.**

| check | result |
|---|---|
| **row 18** LoRA E2B train, 70 GB budget | **361.8 ms/step** (1.007× the 359.2 cell), exit 0; meter peak **57,478 MB** against P27's 56,712 / 57,480 / 57,479 MB for the same live-set spike — the same event to within 2 MB |
| **suite-106**, native, against the **recorded rc column** | **0.9963** geomean, median 0.9997, **0 of 106 rows past 1.1×**; one row outside ±10 % and it is faster (`big12-b8l256` 0.826). Against P27's native arm 1.0004 — flat |
| `texmo_gate` | **106 ok / 0 FAIL** of 106 |
| `execute_test` | 553 ok, all cases match the CPU backend |
| `ingest_test` / `smoke_test` / `bazel test //...` | 0 failed (13 checks) / all checkpoints / `runtime_gil_free_test` PASSED |

Two provenance notes, both under release rule 1:

* The battery was originally split across two builds — the contracts on the
  pre-clamp `frozen-p28-f7f3c708`, the rows on the shipped
  `frozen-p28b-37060770`. Re-running the contracts on the shipped binary
  **failed one of P28's four new cases**: it still asserted the rule's first
  draft (`earn == 2 × swing`) rather than the shipped
  `min(2 × swing, peak_live)`. The rule measured correct on every row; the
  *contract* was stale, is fixed, and now names which term bound each flush
  (`notes/cpp-p28-benefit-gate.md` §5.1). Nothing about the numbers above
  changed with it.
* The rows, the suite and row 18 are 2026-08-17 on pip's stock MLX; the
  contracts and `texmo_gate` were re-run 2026-08-18, by which time the
  vendoring work had staged our patched fork libmlx into the venv. Both
  re-runs are correctness gates and both pass on both libraries.

**The re-spot on the combined build.** Because the vendored MLX and the
benefit gate are two changes to one binary, the three cells were re-measured
on `frozen-vendor-d651add3` — the plugin linked against the private patched
`libmlx_metaljax.dylib` — at the same historical budgets, one guarded process
per row, 2026-08-18 11:40-11:43:

| row | budget | P28 on stock MLX (2026-08-17) | **P28 + vendored patched MLX** | peak |
|---|---:|---|---|---:|
| 11 | 20 GB | 16.61 / 16.81 / 16.75 (9 of 9 complete) | **16.60 ms/tok**, exit 0 | 16 GB |
| 14 | 25 GB | 31.95 / 32.13 / 31.82 (4 of 4) | **31.94 ms/tok**, exit 0 | 9.2 GB |
| 19 | 48 GB | 459.2 / 458.4 / 462.5 | **463.5 ms/step**, exit 0 | 25 GB |

Every row completes at its historical budget on the combined build. Row 14 is
inside its P28 spread, row 11 lands 0.01 ms under the bottom of its own, and
row 19's 463.5 sits 1.0 ms above the top of its trio and inside the 456–470
class it is required to hold — i.e. single runs against three-run spreads, all
within the run-to-run noise these rows have shown all week. Row 19's `loss`
**87.0428237915039** and
`loss_first` **228.39447021484375** are bit-identical across all eight runs of
this campaign — the seven on stock MLX and the one on the patched library. The
gate document's rows-11/14/19 cells can therefore be read on either build; the
vendoring milestone's own battery is the attestation for the rest of the
table.

The regression's own verdict line is therefore **RESOLVED**, and gate 5's
verdict is re-scored below.

### REGRESSION 2 — row 2 and row 6 sit 1.5–2.0 % above their cells

Row 2 reads **94.3** against 92.9 (P27) and row 6 **55.8** against 54.7 (P18).
Neither is a lottery: row 2's tokens are identical to Stage 1 and to P27's run,
and its emit narration is unchanged (48 + 8 fused attentions). The rows below
and above them moved the other way (rows 13 −1.6 %, 19 −2.9 %, 14 −8.6 %), so
this is not a global drift, and 1.5–2 % is inside the ±3 % tolerance the RC gate
used but outside the ±1 % these rows have held all week. Named for completeness
rather than escalated: no mechanism is identified, and both rows are within
their historical spread across campaigns (row 2 has read 92.9 / 93.9 / 94.3 /
98.6 on four native binaries).

### Row 19 — the 868 ms/step "regression" is a harness parameter, resolved

The governor campaign measured row 19 at **868 ms/step on all four arms
including the shipped 0.11.5 binary** and could only conclude "the environment".
It is `MAXTEXT_TRAIN_SEQ`:

* `adapter_maxtext.py` defaults it to **1024**; every published cell — and every
  P25/P27 runner, and `scripts/model_bench/final_run.sh`, the release harness —
  sets **256**. Across all logs on this machine: 17 runs at `seq_len 256`, 5 at
  `1024`, and the five are exactly the unprotocolled ones.
* Re-run with the canonical `MAXTEXT_TRAIN_STEPS=4 MAXTEXT_TRAIN_SEQ=256`:
  **456.1 ms/step**, peak 25 GB, and `loss` / `loss_first` **identical to P27's
  run to thirteen digits** (87.0428237915039 / 228.39447021484375) — the same
  workload, bit for bit.
* The 1024 cell is kept as data: **868.4 ms/step** at seq 1024 (4× the
  sequence), peak 28 GB.

So row 19 is **0.971 of its P27 cell** and there is no regression to carry. The
harness-fix exception is used here and noted: the runner was missing the two
canonical env vars, and `MAXTEXT_PREFILL_LEN=64` was added to rows 11/14 to
match `final_run.sh`.

### Row 7 — a bracketed cell, and why the first two samples were wrong

Row 7's first two samples read 24.2 and 23.9 against a 22.0 cell — a 9 %
apparent regression. A governor-off arm read **22.1**, and a fourth arm with the
governor back **ON** read **21.9**. Four arms, one hold:

| arm | ms/tok |
|---|---:|
| shipped, sample a | 24.2 |
| shipped, sample b | 23.9 |
| `METALJAX_MEM_GOVERNOR=0` | 22.1 |
| **shipped, sample c (the bracket)** | **21.9** |

The governor is exonerated by its own control *and* by the fourth arm; what the
first pair measured is the suite-context trap (CLAUDE.md item 12) with a 82 GB
ambient page cache behind it. The cell is 21.9; the spread is published.

### Row 15 — ✗ WRONG OUTPUT, and it is not a 0.11.5 regression

Row 15 (qwix-int8 Qwen3-8B, maxtext) **completes** under the governor — its
memory blocker is gone, 79 GB peak, 0 refusals — and that is the contract
result. It is **not published as a timing cell**, because the text it produces
is wrong: `" fragment!!!!!!!"`, token ids `[12289, 0, 0, 0, 0, 0, 0, 0]`, and
`!` is Qwen3's token 0 — the logits have collapsed to a constant and greedy
`argmax` returns index 0. Timing a program that computes the wrong answer
measures nothing.

**Verdict: ✗ WRONG OUTPUT, not a 0.11.5 regression — the row has never worked.**
It was embargoed from 2026-08-04 (the MLX command-buffer class) until this
campaign, so there is no earlier metaljax number for it to regress from. The
governor campaign's "known MLX-quantization bug" label was **withdrawn**: no
such bug exists — 2026-08-03 exonerated the quantized dots (`7932b4d`), and
row 14 is the same adapter, the same qwix int8 overrides and the same emits at
0.6B and is coherent (**32.0 ms/tok**, this gate's own run). Evidence,
hypotheses and the ladder: `notes/row15-wrong-output-2026-08-17.md`.

**The H5 probe, run as this battery's last measured item** —
`scripts/model_bench/row15_probe.py big` on the release dylib, which tests the
element-count hypothesis (the untied 622 M-element `logits_dense` is the one
buffer that exceeds a whole command buffer by itself, so a split would be forced
in the same place every call — deterministic, and absent from row 14):

| case | mode | elems | vs budget | rel_rms | non-finite | metal_std vs cpu_std |
|---|---|---:|---|---:|---:|---|
| over-budget-logits | bf16 | 622 M | **OVER** | 1.02e-06 | 0 | 0.04988192 vs 0.04988192 |
| over-budget-logits | int8 | 622 M | **OVER** | **0.0** | 0 | identical |
| under-budget-logits | bf16 | 268 M | under | 1.01e-06 | 0 | identical |
| under-budget-logits | int8 | 268 M | under | **0.0** | 0 | identical |
| tied-embed-06b (row 14's shape) | bf16 / int8 | 156 M | under | 4.2e-07 / **0.0** | 0 | identical |

**H5 is refuted at the level this probe tests.** The over-budget dot is clean —
bit-exact in int8, 1e-06 relative in bf16 (ordinary bf16 rounding, the same as
the under-budget control), no non-finite values, and metal's output standard
deviation equal to CPU's to every printed digit. An over-budget `logits_dense`
shape **alone** does not corrupt, so the collapse needs something the isolated
dot does not have.

**Post-gate follow-up, 2026-08-17 evening — the mechanism is now ESTABLISHED,
and the gate verdict is unchanged.** The remaining rungs were run on this same
release dylib (`notes/row15-wrong-output-2026-08-17.md` §8; raw
`~/.cache/metaljax-bench/logs/row15-mechanism/`). The row is **nondeterministic
MLX command-buffer corruption at 8B traffic, present on BOTH engines**,
amplified into the logit collapse by qwix's per-tensor `absmax` scale (which
guards a zero scale but not a NaN one, so one bad element NaNs a whole tensor):

* Ten prefills of the **same loaded params, one process, identical inputs**:
  row 15 native → **8 distinct first tokens, 2 full collapses**; row 15 on
  Stage 1 → **10 distinct**; **row 14 → the same token 10/10**, decoding
  `" Paris. The capital"`. Nondeterminism at fixed configuration is
  self-proving, and it is what makes the timing cell meaningless.
* Not ours: `METALJAX_COMPILE=0` fails *worse*, `METALJAX_RECOGNIZE=0` still
  fails, `METALJAX_CHUNK_MAX` 10/12/16 all fail, and all twelve row-14/row-15
  s8×s8→s32 contractions are bit-exact against an int64 numpy reference on
  both stacks.
* The committed 8B **bf16** canary (`notes/data/qwen3_8b_prefill_36layer.mlir`,
  no qwix, no checkpoint) still corrupts at today's shipped 800/512 —
  **FAIL(5), norm err 1.000e+00, bit-identical on native and Stage 1** — which
  is the fresh witness the inherited 2026-08-03 attribution was missing.
  (It had been unrunnable for a parse reason: `run_stablehlo_bench.py` never
  registered the `chlo`/`sdy`/`mpmd` dialects. Fixed.)

So H5's refutation above stands, and the cause is below both of our
interpreters — it is the upstream MLX command-buffer report.

### Row 15 — **FIXED by the vendored patch** (2026-08-18, native)

The "no fix at our level" clause expired the moment the level became ours.
`notes/mlx-patch-diagnosis.md` located the defect at `slicing.cpp:62` (a
donated dynamic-slice offset registered as a command-encoder temporary, which
`end_encoding()` then erases from the fence bookkeeping), and the vendoring
milestone shipped it: the release plugin now links our fork
(`libmlx_metaljax.dylib`, `vendor/0.32.0` @ `651c39cd` = upstream's own
unreleased `7e8b4ccc` + our `end_encoding` hardening).

Re-run on `frozen-vendor-d651add3` — the release binary — with the row's own
forensics, ten prefills of one loaded parameter set in one process plus a
greedy decode, 92 GB guard, peak 76 GB:

| build | engine | 10 draws | distinct first tokens | collapses | decode |
|---|---|---|---|---|---|
| pip wheel 0.32.0 | native | 10 | **8** | **2** | garbage |
| pip wheel 0.32.0 | Stage 1 | 10 | **10** | 0 | ` stack NIL�输出` (garbage) |
| patched fork | Stage 1 | 10 | 1 | 0 | `" Paris. The capital"` |
| **patched fork** | **native (ships)** | 10 | **1** — token 12095, ten times | **0** | **`" Paris. The capital"`** |

Token 12095 is the same first token row 14 (the 0.6B control) returns ten
times out of ten, and `logits_flat` is false on every rep (`logits_std`
2.292) where the collapse used to flatten them. **A row that has emitted
nothing but garbage since 2026-08-03, on every binary and both engines, is
now deterministic and coherent on the one that ships.**

Verdict: **✗ WRONG OUTPUT → ✅ FIXED**. The timing cell (369.7 ms/tok) was
always meaningless while the output was wrong; it now measures something,
but it is not re-published here — this battery attested correctness, and a
number would need its own measured run under rule 1.

Two follow-ups still need Oleg's go, both full 8B maxtext loads above row
15's 79 GB envelope: the 0.11.2-src provenance arm (guard-killed at 94 GB)
and a jax-CPU reference (guard-killed twice; note that `34f627c`'s
"coherent" CPU cell has **no artifact behind it** and should not be cited as
evidence until re-measured). Neither is load-bearing for the verdict above,
which rests on determinism, row-14 agreement, and coherent text.

### The recognizer nondeterminism — CHECKED on this binary, and it persists on rows 5 and 7

The RC gate's finding was that the fused-attention emits make decode run-to-run
nondeterministic. On the release binary, with two samples per row:

| row | same binary, run to run | vs Stage 1 |
|---|---|---|
| 1 gemma4-31B | — | **identical** (and identical to P26b) |
| 2 gemma4-12B | — | **identical** |
| 3 26B-A4B MoE | — | identical to P18 native |
| 4 gemma4-E2B | **identical**, 4 samples | **identical** |
| **5 Qwen3-8B** | **diverges at token 51** | diverges at 51 |
| **7 gpt-oss-20b** | **diverges at token 51** (a vs b, a vs c) | no CPU counterpart |

So P26b closed row 1's divergence (it now computes attention as Stage 1 does)
but **rows 5 and 7 remain nondeterministic**, exactly as the RC gate described:
opt-out via `METALJAX_RECOGNIZE=0`, native wheel only, timings reproducible to
±1 % regardless. The first pass of this battery saw row 5 agree across two
samples; that was luck, not determinism — the divergence index is a lottery
(50/51/52/53 across campaigns).

#### Re-measured on the vendored MLX — **it survives the fence fix** (2026-08-18)

The open question was whether the command-buffer fence drop was also the
source of this: the release-status note recorded rows 5/7 as "accepted as a
release-note item *if it survives the fence fix*". It survives. Measured on
`frozen-vendor-d651add3`, one guarded process per draw, token ids compared
pairwise (`~/.cache/metaljax-bench/logs/mlx-vendoring/tokens.py`):

| row | draws | outcome | divergence index |
|---|---:|---|---|
| **5 Qwen3-8B bf16** | **6** | 5 draws identical, **1 (draw 4) diverges from all five** | **51** |
| **7 gpt-oss-20b** | 3 | **all three pairs diverge** | 50 / 51 |

**Row 5's first three draws were identical, and that was the trap this
document already named.** Three more draws were run precisely because the
gate had been burned by a two-sample agreement before; the fourth draw broke
it. A 3-draw pass would have published "the fence fix retired row 5's
nondeterminism" — false. What changed is the *rate*, not the property: row 5
shows roughly one divergent draw in six here, row 7 diverges on every pair,
and both keep the token-50/51 signature they have had across campaigns.

This is consistent with the diagnosis' separation of concerns
(`notes/mlx-patch-diagnosis.md` §4): the fence drop is a command-buffer
synchronisation defect, cured and *proven* cured by budget-independence and
28 clean canary budgets; the recognizer nondeterminism is a different
mechanism (fused-attention emit ordering) and is untouched by it. **The
release-note item stands** — it is not retired by the vendored build.

## Gate 5b — the no-panic contract — **PASS**

The contract's own rows were proven by the governor campaign
(`notes/no-panic-governor.md`) on **this binary**; this gate records the
outcomes and adds tonight's evidence that nothing regressed under it.

| contract row | outcome | peak | governor |
|---|---|---:|---|
| oversized synthetic load, 1.5× physical, ×3 | **clean `RESOURCE_EXHAUSTED`**, process alive, identical message 3/3 | 84 GB | 11 sweeps, 5 stalls, 1 refusal |
| row 9's 61 GB checkpoint, load-only | page cache **6.354 → 6.355 GB** (control: → 59.7 GB), free list 47 GB (control: 0.055) | 61 GB | governed arm 10 % *faster* |
| hot-cache pair (58.3 GB then 61.0 GB back to back) | both complete; machine responsive at 110/128 GB in use | 58 / 61 GB | 76 sweeps, 70 paced, 0 refusals |
| **8** Qwen3.6-35B-A3B — panic #7's row | **COMPLETE**, 29.6 ms/tok | 73 GB | 30 lines, 0 refusals |
| **9** R1-Distill-32B — panic #9's row, run LAST after six 58–93 GB loads | **COMPLETE**, 210.7 ms/tok | 67 GB, RSS 14 | 23 lines, 0 refusals, 0 stalls |
| **10** DeepSeek-V2-Lite, original | **COMPLETE**, 1865 ms/tok | 88 GB | 14 lines, 0 refusals |
| **10f** with `MAXTEXT_PREFILL_LEN=64` (the minimal fix) | guard kill on the ramp, **no panic**; the fix is a no-op for this row | 95 GB | pacing, 0 refusals |
| **12** Mixtral 8×7B, original | cannot be attempted: the keras preset is a 93 GB KaggleHub download | — | — |
| **12** its 93.4 GB checkpoint through the transfer path | **COMPLETE**, 87 GB moved, page cache flat | 87 GB | 175 paced, **0 refusals** |
| **15** qwix-int8 Qwen3-8B, original | **COMPLETE**, 369.7 ms/tok — but the **output is wrong** (logits collapse; `notes/row15-wrong-output-2026-08-17.md`) | 79 GB | 13 lines, 0 refusals |
| **15f** `MAXTEXT_PREFILL_LEN=64 METALJAX_BODY_COMPILE=0` | **COMPLETE**, 1064 ms/tok (2.9× the mitigation's cost), same wrong output | 75 GB | 0 refusals |
| **20** 235B-A22B 3-bit, original | harness declines (`blocked-metaljax`) — no metaljax execution, so nothing to panic | — | — |
| **20** its 96 GB checkpoint through the transfer path | completes; only 13.7 GB is transferable (packed `U32` blobs) | 13.8 GB | 0 refusals |

**Tonight's 24 further guarded runs add to the record**: 21 model runs and 3
guard kills (rows 11 ×2, 14 ×1 — *budget* kills, on the guard's own rule, with
the process reporting and exiting), **no panic, no wedge, no jetsam**, on a
machine carrying up to 82 GB of ambient page cache. The one thing the contract
does not yet cover is unchanged and stated in the governor note's own scrutiny
list: a single `mx::eval` that allocates tens of GB inside one operation has no
gate in reach, which is why the bench guard stays on.

**Rows 10/12/15/20 original-vs-fixed, per the CLAUDE.md amendment**: rows 10 and
15 run as **originals** (no benchmark-code change needed); their fixed variants
(10f, 15f) are reported beside them and are worse or neutral — 10f is killed on
the ramp, 15f costs 2.9× — so **the originals are the cells**. Row 12's
"fix" is a 93 GB download and row 20's is a packed sub-byte quantized path (a
feature, identified, not attempted); both were tested at their own scale through
the transfer path instead.

## Gate 4 — texmo: the two pairings, then the two correctness gates — **PASS**

Protocol: each pairing gets a hold of its own with a settle and nothing heavy
before it, and the correctness gates run **after** the measurements (a
106-subprocess gate poisons the next few minutes in the same hold, and it
poisons the arm that runs first — P23). `g4f_texmo_final.sh`, 14:13 → 15:08.

### 4a — texmo suite-106 (64-step chunks), native / Stage 1

Hold 14:15 → 14:31, native 480 s then Stage 1 498 s, 106/106 rows both arms.

| aggregate | n | **0.11.5 final** | governor campaign | 0.11.5 first pass | P27 | P23 |
|---|---:|---:|---:|---:|---:|---:|
| whole suite, geomean | 106 | **0.9840** | 0.9876 | 0.9917 | 0.9893 | 1.0050 |
| whole suite, median | 106 | **0.9983** | 1.0002 | 0.9999 | 1.0001 | 1.0012 |
| `big` | 34 | 0.9630 | | 0.9732 | | 1.0107 |
| `mid` | 30 | 0.9917 | | 1.0035 | | 1.0033 |
| `db` (msl territory) | 40 | 0.9970 | | 1.0011 | | 1.0013 |
| `synth` | 2 | 0.9750 | | 0.9485 | | 1.0062 |
| **rows where native is faster** | 106 | **65** | | 55 | | 42 |
| rows within 1.2× **as swept** | 106 | 104 | 106 | 106 | 106 | 106 |
| **rows within 1.2× after the outliers are re-measured standalone** | 106 | **106** | | 106 | | 106 |

**Drift controls**: the native arm reads **1.0026** of the first pass's native
arm (a different binary — the governor costs 0.3 % here) and 1.0041 of P27's;
the Stage-1 arm reads 1.0104 of the first pass's. So both arms reproduce, and
the pair's movement 0.9917 → 0.9840 is mostly the Stage-1 arm being ~1 % slower
today; four measurements this week put the pair in 0.984–0.992.

**The two rows outside 1.2× in the sweep are in-suite artifacts, re-measured
standalone before being reported** (`g4g_anomalies.sh`, 3 reps per arm, arms
interleaved):

| config | in-suite | standalone Stage 1 | standalone native | standalone | verdict |
|---|---:|---:|---:|---:|---|
| `big14-b32l128` | **1.2895** | 17.4518 | 17.4730 | **1.0012** | in-suite artifact |
| `big12-b8l256` | **1.2098** | 5.1898 | 5.1804 | **0.9982** | in-suite artifact |
| `big09-b8l256` | 0.6688 | 38.3611 | 25.3802 | **0.6616** | **REAL** (P22's coop width cap) |

Both artifacts are the documented lottery rows (`big14-b32l128`: P23 saw 1.143
in-suite and 0.998 standalone; `big12-b8l256`: P22 saw 1.159 and 0.998).
Substituting the three standalone numbers: geomean **0.9798**, median 0.9983,
**106 of 106 within 1.2×**, native faster on 66, worst row `big14-b8l256` 1.142.

### 4b — top_confs (163 configurations), bracketed

The first pass established that arm **position** inside a hold is worth ~2 % on
this sub-ms suite — larger than the effect being measured — so the pairing is
run as Stage 1 → native → Stage 1 and the ratio taken against the mean of the
two Stage-1 arms. The two Stage-1 arms differ by **1.0011**: the hold is flat.

| aggregate | n | **0.11.5 final** | first pass | P23 | P22 |
|---|---:|---:|---:|---:|---:|
| **native / Stage 1 (same PJRT route, bracketed)** | 163 | **0.9970** (median 0.9991) | 1.0025 | 1.0016 | 1.001 |
| rows within 1.2× / outside ±10 % | 163 | **163** / **0** | 163 / 0 | 163 / 1 | 163 |
| rows where native is faster | 163 | **91** | 52 | 63 | — |
| biggest win / loss | | `tc136-w1488` **0.936** / `tc005-w12` 1.021 | | | |
| **native vs the 0.11.3 anchor** | 163 | **1.072× faster** | 1.071 | | 1.073 |
| **Stage 1 vs the 0.11.3 anchor** | 163 | **1.069× faster** | 1.074 | | 1.071 |
| **configurations beating jax-CPU** | 163 | native **59** · Stage 1 **59** · anchor 53 | 59 / 59 | 58 / 59 | 59 |
| native arm vs the first pass's native arm | 163 | **0.9987** | | | |

The jax-CPU column is the **2026-08-16 23:08 engine-route arm** (163 ok, 0 FAIL,
0 error), not re-run tonight: it measures frozen Stage-1 code and the CPU
reference, neither of which can come from the release binary. Everything else in
the table is tonight's, on the release dylib.

### 4c — the two correctness gates (own hold, after the measurements)

| gate | stack | result | wall |
|---|---|---|---:|
| `plugin-native/texmo_gate.py` | native | **106 ok** (18 via sensitivity scaling), **0 decline, 0 FAIL, 0 error** | 264 s |
| `scripts/texmo_check.py` | Stage 1 | **106 ok, 0 FAIL, 0 error** (tol 0.002) | 143 s |

**A note the first pass earned.** On 2026-08-16 the Stage-1 `texmo_check` came
back 105/106 once, on `db08-b4l1024`, with `worst=inf sens=1.9e+01 bad=19` —
nineteen of twenty outputs differing in **finiteness** on a draw whose 1-ULP
sensitivity was 15× the row's historical value; two immediate re-runs were
106/106 and tonight's is 106/106. `texmo_check` builds a fresh model and samples
fresh data per run, so an `lrnn.4.4` recurrence over 1024 timesteps can draw a
chunk that diverges, and the two backends then overflow at different steps. It
is the documented lottery-row class (P21 `big10`, P23/P25 `mid03`) with a new
face: those flaked on tolerance, this one on finiteness, which `nbad == 0`
cannot excuse — so an equally unlucky future draw will fail again. Property of
the ill-conditioned configuration, not of either stack.

## Gate 2 — pinned jax suite, natively, all 164 files — **PASS (zero new failures)**

`g2_suite_tests.sh`, own lock hold, 15:12:55 → 15:41:34. `scripts/release/jax_suite.sh`
with `METALJAX_PLUGIN_PATH` = the frozen release dylib, `--jobs 1` (load-bearing:
parallel runs under-report), `--tests jax-v0.11.0/tests` relative so node ids
match the whitelist. **28.7 min**, 164 of 164 files, no timeouts.

| | **0.11.5 final** | first pass (pre-governor) | the RC gate | the approved native run (2026-08-11) |
|---|---:|---:|---:|---:|
| passed | **28,073** | 28,073 | 28,068 | 28,068 |
| failed | **129** | 129 | 129 | 129 |
| skipped | 6,161 | 6,161 | 6,160 | 6,158 |
| collection errors | 35 | 35 | 35 | 35 |
| **pass rate** | **99.54 %** | 99.54 % | 99.54 % | 99.54 % |

**The gate: `failures − whitelist = ∅`**, diffed against the reviewed native
list `notes/data/p12-14-native-failures.txt` (the 142 ids Oleg signed off one by
one in `notes/parity-whitelist-report.md`):

* **NEW failures: 0** (`g2-new-failures.txt` is empty) — nothing outside the
  reviewed set, so no id needed a standalone rerun.
* **Newly passing: 13** — exactly the 13 the report says were fixed rather than
  whitelisted (`aot_test` 2, `api_test` 2, `async_collectives_test` 2,
  `export_test` 2, `lax_test` 2, `memories_test` 2, `lax_numpy_indexing_test` 1).
  142 − 13 = **129**.
* **Against the RC gate's own list the set is IDENTICAL, id for id** (`comm`
  empty both ways) — and identical again to the pre-governor first pass, so the
  memory governor moved nothing in the suite.

Composition: `export_harnesses_multi_platform` 44, `lobpcg` 27, `x64_context` 13,
`export_test` 5, `api_test` 5, `xla_transform` 4, `shape_poly` 4,
`profiler_session` 3, `async_collectives` 3, `sparse_bcoo_bcsr` 2 (MLX's fusion
bug #8), `logging` 2, `layout` 2, then singletons — the f64-policy /
export-allowlist / PJRT-surface / harness-skew classes, unchanged.

**The wrapper's own verdict line reads "FAIL, NEW failures: 12" and that is
expected**: `jax_suite.sh` diffs against the **Stage-1-era 130-id list**
(`notes/data/pinned-0.11.0-failures.txt`); the 12 are the known
native-vs-Stage-1 split (`x64_context` 6, `sparse_bcoo_bcsr` 2,
`dtypes`/`lax_numpy`/`layout`/`pickle` 1 each), every one inside the 142-id
native list with a review verdict. Measured against the list that governs this
stack the count is zero.

Artifacts: `jax-suite/{jax_suite.md,jax_suite.log,jaxtests/{failures.txt,summary.csv}}`,
`g2-new-failures.txt` (empty), `g2-newly-passing.txt` (13).

## Gate 3 — `tests/` on both legs — **PASS**, and the 1053-vs-1187 question resolved

Same script, second hold (15:42:34 → 15:45:57).

| leg | result | wall |
|---|---|---:|
| default (**Stage 1** trampoline) | **1258 passed, 0 failed** | 89 s |
| native (`METALJAX_PLUGIN_PATH` = frozen release dylib) | **1187 passed, 71 failed** | 61 s |
| native, **deselected** (the governor campaign's form) | **1053 passed, 0 failed, 205 deselected** | 53 s |

**The two native numbers are the same run seen two ways, and both are right.**
The governor campaign ran `pytest tests -q -x --deselect tests/test_moe.py
--deselect tests/test_qmm.py --deselect tests/test_qmm_mxfp4.py --deselect
tests/test_engine_gc.py` and reported 1053/0; this gate runs the plain form and
reports 1187/71. The arithmetic closes exactly:

| | tests | of which pass natively | of which fail natively |
|---|---:|---:|---:|
| the four Stage-1-only files | **205** | 134 | **71** |
| everything else | 1053 | 1053 | 0 |
| **total** | **1258** | 1187 | 71 |

The 71 are the documented composition — `test_moe` 28, `test_qmm` 26,
`test_qmm_mxfp4` 16, `test_engine_gc` 1 — i.e. 70 recognizer-family Python
counter assertions plus the Python engine's buffer-GC test, none of which a
plugin holding no Python interpreter can satisfy (P17). Deselecting the four
files removes all 205 of their tests, not just the 71 that fail, which is why
the two counts look unrelated and are not. No new file appears in either form,
and the Stage-1 leg — the release number — is exact at **1258**.

## Gate 7 — wheels, both variants — **PASS** (nothing uploaded) *(2026-08-17 record, SUPERSEDED: the release wheel is now native-only with the vendored MLX inside — see the re-attestation section; these two wheels are not release artifacts)*

`g7_wheels.sh`, own hold, 15:56:58 → 15:58:09. Built into **separate out-dirs**
because both variants produce the *same* filename (the identical-filename trap).

| variant | size | sha256 |
|---|---:|---|
| default (Stage 1 trampoline) | 291,083 B (288 K) | `0c0dd8df1b35a8fb60ef6d9c7d91b3a3781cff7d8559856891edf894de489f81` |
| native (phase-2 plugin) | 12,077,249 B (12 M) | `138a1ee6ef2462333b72f85b93a6fa4759f020e1a3aaf2a45a1f7fded642ffcf` |

Paths: `~/.cache/metaljax-bench/wheels-0.11.5/{default,native}/metaljax-0.11.5-py3-none-macosx_14_0_arm64.whl`.

* `twine check`: **PASSED** on both.
* **The native wheel carries the gated binary, verified by hash**: the dylib
  inside it is `ebe56e71…9fb9f31` — byte-identical to
  `frozen-0.11.5-ebe56e71.dylib` and to the tree's `bazel-bin` build. The thing
  that was measured, gated and would be published is one file.
* Tree restored after the builds (hatch parks the trampoline while building the
  native wheel): `src/metaljax/lib/libmetal_pjrt.dylib` is back in place and
  `git status` shows only this gate document.

Fresh **non-editable** installs, each into a clean `uv venv`:

| wheel | python | checks |
|---|---|---|
| native | **3.12 / 3.13 / 3.14** | `wheel_poc_test.py` all checkpoints; eigh reconstruction + a 128-step `lax.scan` matvec cell with `value_and_grad` (msl-family) vs numpy — **WHEEL SMOKE OK** on all three |
| default | **3.12 / 3.13 / 3.14** | `[MetalDevice(id=0)]`, `2 * [1,2,3] = [2 4 6]` on all three |

Two things the installs surface, neither new but both release-relevant:

1. **`metaljax.__version__` reports `0.11.3` while the distribution is
   `0.11.5`** — on both wheels, all six interpreters. `RELEASING.md` step 0
   requires the version to be bumped in **both** `pyproject.toml` and
   `src/metaljax/__init__.py`; `src/` has been frozen since 2026-08-10, so the
   0.11.4 and 0.11.5 bumps touched only `pyproject.toml`. The skew therefore
   already shipped in 0.11.4 (TestPyPI). It is cosmetic — nothing in the code
   reads `__version__` — but it is wrong in a published artifact, and the fix is
   one line **that unfreezes `src/` and so invalidates every Stage-1 number in
   this document**. Flagged, deliberately not fixed here.
2. **The default wheel runs the Python engine**, printing `[metaljax] native
   engine not built (native/build.sh); running on the Python engine`. Documented
   behaviour, measured at the RC gate: 1.00–1.02× on decode and training, 1.23×
   on prefill, +2 GB peak.

**Nothing was uploaded.** The TestPyPI upload is the main agent's, after review;
public PyPI and the git push are Oleg's.

---
# Gate 8 — the go/no-go

*(This whole gate-8 section, its verdict table, release table and GO
recommendation are the **2026-08-17 pre-vendoring record** on `ebe56e71` —
kept as history. The binary changed twice after it (vendored MLX, benefit
gate); the CURRENT status lives in the checklist at the top of this document
and in the vendored-MLX re-attestation section below, and the final go/no-go
is written by the consolidated re-gate on `frozen-vendor-d651add3`.)*

## Verdicts

| # | gate | verdict | headline |
|---|---|---|---|
| 1 | Release binary frozen | **PASS** | `ebe56e71…9fb9f31`, byte-identical to the tree's `bazel-bin` **and** to the governor campaign's `frozen-gov7` — which is what licenses banking its model rows |
| 2 | Pinned jax suite, native, 164 files, `--jobs 1` | **PASS** | **28,073 passed / 129 failed = 99.54 %**, **zero new failures**, the failing set **identical id-for-id** to the RC gate's |
| 3 | `tests/` both legs | **PASS** | Stage 1 **1258 / 0**; native **1187 / 71**, reconciled with the governor campaign's 1053 / 0 (same run, 205 deselected) |
| 4 | texmo pairings + both correctness gates | **PASS** | suite-106 **0.9840** (0.9798 with the two artifacts re-measured standalone, **106/106 within 1.2×**, native faster on 65); top_confs bracketed **0.9970**, **1.072× the 0.11.3 anchor**, **59 configurations beating jax-CPU**; `texmo_gate` **106/106**, `texmo_check` **106/106** |
| 5 | Model rows, all non-embargoed | **PASS** (was REGRESSION) | 15 of 17 measurable rows inside ±2 % of their cells, three better, rows 8/10 first-ever numbers; the rows-11/14 footprint regression is **RESOLVED** — P28's benefit gate returns both to their historical budgets (9 of 9 and 4 of 4 complete) at unchanged speed, with row 19 holding 459.2 ms/step. Cost of the resolution measured: row 18 361.8 ms/step, suite-106 native **0.9963** of the recorded rc column (0 of 106 rows past 1.1×), `texmo_gate` 106/106, `execute_test` 553 ok. Re-spotted on the combined (vendored-MLX) build: 16.60 / 31.94 ms/tok, 463.5 ms/step, all at the historical budgets |
| 5b | The no-panic contract | **PASS** | 21 governed model runs + 8 synthetic rungs + 24 further guarded runs tonight: **no panic, no wedge, no jetsam**; 1.5×-physical loads refuse cleanly 3/3; rows 8/9/10/15 complete |
| 6 | Plugin contract suites | **PASS** | `execute_test` **549 ok**, `ingest_test` 13 checks / 0 failed, `decline_census` 35/35, coexist all four carrier cases, `bazel test` uncached PASSED |
| 7 | Wheels, both variants | **PASS** | `twine check` ×2; native wheel's dylib **hash-identical to the gated binary**; fresh 3.12/3.13/3.14 installs of both; **nothing uploaded** |

## The release table

| claim | number | where |
|---|---:|---|
| texmo suite-106, native / Stage 1 (geomean) | **0.9840** (median 0.9983) | gate 4a |
| — with the two in-suite artifacts substituted standalone | **0.9798** | gate 4a |
| — rows within 1.2× / native faster | **106 of 106** / 65 | gate 4a |
| texmo top_confs (163), native / Stage 1, bracketed | **0.9970** (median 0.9991) | gate 4b |
| — rows within 1.2× / outside ±10 % / native faster | **163** / **0** / 91 | gate 4b |
| — vs the 0.11.3 anchor | native **1.072× faster**, Stage 1 1.069× | gate 4b |
| — configurations beating jax-CPU | native **59**, Stage 1 **59** (anchor 53) | gate 4b |
| jax pinned suite (native) | **28,073 / 129 = 99.54 %**, zero new | gate 2 |
| texmo correctness, both stacks | **106 / 106** each | gate 4c |
| `tests/`, Stage 1 | **1258 / 1258** | gate 3 |
| model rows, biggest movers | row 1 **237.3** ms/tok (−21 % vs 0.11.4's 301.6) · row 14 **32.0** (−8.6 %) · row 19 **456.1** (−2.9 %) · row 8 **29.6** and row 10 **1865**, first ever | gate 5 |
| the no-panic contract | **holds** on every row tested, including 9/10/15 | gate 5b |

## The regression, stated per release rule 2 — and RESOLVED

**As gated, P27's flush watermark cost the two maxtext decode rows 17 GB and
11 GB of peak footprint, bought them nothing, and guard-killed them at the
budgets every previous campaign used.**

| row | shipped defaults, as gated | with P25 semantics (`METALJAX_FLUSH_CLEAR_MB=2048 METALJAX_FLUSH_FOOTPRINT_MB=0`) |
|---|---|---|
| 11 Qwen3-0.6B maxtext decode | **25 GB** peak, 16.92 ms/tok — guard-killed at its historical 20 GB budget | **7.6 GB** peak, 16.85 ms/tok |
| 14 maxtext qwix-int8 0.6B | **26 GB** peak, 32.00 ms/tok — guard-killed at its historical 25 GB budget | **15 GB** peak, 32.14 ms/tok |

It was **pre-existing in the 0.11.5 tree** (P27 landed in `00fba0f`, before the
version bump), **not the governor** (row 11 dies at 20 GB with
`METALJAX_MEM_GOVERNOR=0` too), **opt-out with one variable**, and **new
information**: P27's battery measured rows 19/18/13/2 and never these two.

**RESOLVED the same evening, by Oleg's option (b) — a benefit gate on the
watermark** (P28, `notes/cpp-p28-benefit-gate.md`, and gate 5's
"RESOLVED 2026-08-17" section above). The rows' own flight logs name the
counter that separates them — the LIVE SET, not the flush count, which says
"eager main" for all three — and `flush_bound` gained a third rule bounding a
program's pool by the live set it has demonstrated it CYCLES
(`METALJAX_FLUSH_EARN_MULT`, default 2, `0` = P27 exactly). It enters as one
more `min`, so no bound it decides is above P27's or below P25's floor.

| row | budget | as gated | **with the benefit gate** |
|---|---:|---|---|
| 11 | **20 GB**, its historical | **0 of 6 complete** | **9 of 9 complete**, 16.61–16.83 ms/tok |
| 14 | **25 GB**, its historical | guard kill | **4 of 4 complete**, 31.82–32.13 ms/tok |
| 19 | 48 GB | 456.1 ms/step | **459.2 ms/step** (458.4 / 462.5), `loss` ≡ P27 to 13 digits |

Neither decode row loses speed, which was the finding all along: they were
paying footprint for a pool they never read. **The published memory
characteristics of rows 11/14 return to their pre-P27 class**, and the cells in
`benchmarks/models.md` are the P28 runs at the historical budgets.

A **soft second** is recorded rather than escalated: rows 2 and 6 read 1.5–2.0 %
above their cells (94.3 vs 92.9; 55.8 vs 54.7) with identical tokens and
identical emit narration, inside the ±3 % tolerance the RC gate used and inside
their own multi-binary spread, with no mechanism identified.

## Known and accepted (with the quotes that accept them)

1. **The 129 whitelisted jax-suite failures** — reviewed one per row by Oleg in
   `notes/parity-whitelist-report.md`, reproduced exactly, identical id-for-id
   to the RC gate's set.
2. **The 71 native `tests/` failures** — the Stage-1 Python-counter families.
   Accepted, and scheduled for deletion: *"Post-0.11.5 retirement (Oleg,
   2026-08-16): confirmed FIRST post-release milestone — full cleanup INCLUDING
   docs. Delete: … engine-API tests (71 counter rows, test_native_tape,
   extension buffer tests), texmo_check (texmo_gate is the gate)."*
3. **The frozen-Stage-1 backlog** — Stage 1 still dumps its buffer pool
   (`src/metaljax/interpreter.py:776`, `native/program.cc:38`), so any same-day
   native/Stage-1 ratio on an eager-main row carries that difference. Same
   quote: Stage 1 is deleted first thing post-release.
4. **The fused-attention nondeterminism** — **re-checked on this binary and it
   persists on rows 5 and 7** (two samples each diverge at token 51), while
   rows 1/2/3/4 are stable and Stage-1-identical. Native wheel only, opt-out
   `METALJAX_RECOGNIZE=0`, timings reproducible to ±1 %. **No verbatim Oleg
   acceptance exists**; the RC gate raised it as "the single most
   decision-relevant thing in this document" and the 0.11.4 TestPyPI upload
   proceeded, which is evidence of tolerance, not a quote. It belongs in the
   release notes.
5. **The sub-ms dispatch floor** — configurations under ~0.5 ms/step measure the
   harness; on `top_confs` arm *position* inside a hold is worth ~2 %, which is
   why that pairing is bracketed.
6. **Rows 12 and 20 are not runnable for non-memory reasons** (a 93 GB
   KaggleHub download; a packed sub-byte quantized path metaljax does not
   implement — a feature, identified, not attempted). Both were tested at their
   own scale through the transfer path under the contract.
7. **Row 15 emits wrong text** — ✗ in the table, never worked, mechanism open,
   H5 refuted by tonight's probe.
8. **`metaljax.__version__` says 0.11.3** in both 0.11.5 wheels (see gate 7).

## Release notes draft — metaljax 0.11.5

*(Changes since 0.11.3, which is the last version whose numbers were published;
0.11.4 was burned on TestPyPI before P24–P27 existed.)*

**Headline: the whole engine now exists in C++, and the library will not panic
your machine.**

* **A second wheel variant: the fully native PJRT plugin** (`plugin-native/`,
  built with `METALJAX_WHEEL_PLUGIN=native`). StableHLO is parsed and lowered in
  C++ and executed with no Python interpreter in the loop — a GIL-free execute
  path. It reaches parity with the Stage-1 trampoline across the board: the
  texmo 106-configuration suite is **0.984** of Stage 1 (106/106 within 1.2×),
  the 163-configuration `top_confs` suite **0.997**, both stacks **1.07× faster
  than 0.11.3**, and **59 of 163** configurations now beat jax-CPU (0.11.3: 53).
  Getting there: native control flow, gather/scatter, convolution, host-FFI
  linalg, the emulated dtypes, sort/RNG/FFT, the qmm / MoE-gather / sdpa
  recognizer emits, all three `msl_scan` modes, the compile decisions,
  row-blocked quantized packing with a cross-executable cache, donation-aware
  output copies, and an exported-symbols relink so the dylib can share a process
  with TensorFlow and array_record.
* **The no-panic contract, and the memory governor that keeps it.** metaljax
  will not wedge the machine: under memory pressure it degrades (page-cache
  sweeps, paced admissions), and when it cannot, it **refuses cleanly** with
  `RESOURCE_EXHAUSTED` naming the ceiling and the knob. A load 1.5× physical
  memory refuses reproducibly with the process alive; a 61 GB checkpoint now
  leaves **1 MB** of page cache behind instead of 41–53 GB, with the free list
  at 47 GB instead of 55 MB — and the governed load is 10 % *faster*. Four
  previously embargoed model rows run for the first time: **Qwen3.6-35B-A3B
  29.6 ms/tok**, **R1-Distill-32B 210.7**, **DeepSeek-V2-Lite completes**, and a
  93.4 GB Mixtral checkpoint streams end to end.
* **The eager flush stops dumping MLX's buffer pool.** It now trims to a
  watermark (P25), the watermark is decided per flush from the program's own
  memory pressure (P27), and a program only gets one at all if its own live set
  shows it is CYCLING memory rather than storing it (P28). Worth **1.85×** on
  the maxtext training row (868 → 456 ms/step measured at the canonical
  sequence length) and ~2 % on the `big` half of the texmo suite, with no
  footprint cost anywhere: the two maxtext *decode* rows, whose checkpoint load
  briefly frees 14 GB into the pool and never reads it again, keep their
  historical peaks. `METALJAX_FLUSH_EARN_MULT=0` restores P27's behaviour and
  `METALJAX_FLUSH_CLEAR_MB=2048 METALJAX_FLUSH_FOOTPRINT_MB=0` P25's.
* **Faster where it matters**: gemma4-31B decode **301.6 → 237.3 ms/tok** (the
  fused-attention recognizer now sees into `func.call` callees, so the decode
  body compiles), maxtext int8 0.6B **35.0 → 32.0**, E2B keras-int4 **80.3 →
  79.0**, LoRA training **407 → 359**.
* **Correctness**: 28,073 of the pinned jax suite pass (99.54 %) with zero new
  failures against the reviewed whitelist; all 106 texmo training
  configurations match jax-CPU on both stacks.
* **Known issues.** (a) With the native wheel the fused-attention path is
  **run-to-run nondeterministic** on some models — identical timings, different
  token streams after ~50 tokens; `METALJAX_RECOGNIZE=0` restores determinism
  and Stage-1-identical output at ~4 %. The default wheel is unaffected.
  (b) The default wheel still runs the Python engine (1.00–1.02× on decode and
  training, 1.23× on prefill). (c) `metaljax.__version__` reports 0.11.3.
  (d) maxtext int8 at 8B produces wrong output (investigation open). (e) The
  129 whitelisted jax-suite failures and the f64 policy are unchanged.

## Artifacts

| what | path / hash |
|---|---|
| gated dylib | `~/.cache/metaljax-bench/frozen-0.11.5-ebe56e71.dylib` · `ebe56e7168eff581906718dfb24153b2575a69a4b9698801c01aea0a89fb9f31` |
| default wheel | `~/.cache/metaljax-bench/wheels-0.11.5/default/metaljax-0.11.5-py3-none-macosx_14_0_arm64.whl` · `0c0dd8df1b35a8fb60ef6d9c7d91b3a3781cff7d8559856891edf894de489f81` |
| native wheel | `~/.cache/metaljax-bench/wheels-0.11.5/native/metaljax-0.11.5-py3-none-macosx_14_0_arm64.whl` · `138a1ee6ef2462333b72f85b93a6fa4759f020e1a3aaf2a45a1f7fded642ffcf` |
| every log, runner and jsonl | `~/.cache/metaljax-bench/logs/release-0.11.5/` (first pass under `prepanic/`) |
| texmo aggregates | `notes/data/release-0.11.5-texmo.json`, `release-0.11.5-{suite106,topconfs}.csv` |

## Recommendation

**GO.** *(Updated 2026-08-17 evening: the decision below was taken — Oleg's
option (b) — and the regression is resolved rather than accepted. See the
RESOLVED sections above and `notes/cpp-p28-benefit-gate.md`; the paragraph is
kept as the record of what was decided and why.)*

**GO — with one decision to make first.** Every gate passes on the tree at
`6500370` with the binary that would ship: the pinned jax suite reproduces its
approved 129 failures with *zero* new ones and an identical failing set, both
texmo gates are 106/106 on both stacks, `tests/` is 1258/1258 on Stage 1 and
exactly the documented native split, the contract battery is 549/549 with the
governor's five new contracts, fifteen of seventeen measurable model rows are
inside ±2 % of their cells with two rows producing first-ever numbers, the
no-panic contract holds across 45 guarded runs, and both wheels install clean
into 3.12/3.13/3.14 with the native one carrying a hash-verified copy of the
gated dylib.

The decision is **the P27 watermark's memory cost on rows 11 and 14**: 17 GB and
11 GB of extra peak footprint for no speed, which guard-kills both rows at their
historical budgets. It is pre-existing in this tree, not the governor, opt-out
with one environment variable, and invisible to every prior campaign because
none of them measured those two rows. Under release rule 2 this document cannot
score gate 5 as PASS over it. The options are (a) ship as-is and state it in the
release notes (drafted above), (b) change the default before release — a
one-line policy change in `runtime.cc` plus a re-gate of the affected rows, or
(c) ship with the floor as the default for programs whose live set is small.
That is Oleg's call, not this document's.

---

## Vendored-MLX re-attestation (2026-08-18)

The binary this document scores changed once more after the benefit gate: the
plugin now links **our own MLX**. Everything below was re-run on
`frozen-vendor-d651add3` — `plugin-native` at HEAD (a rebuild reproduces the
same sha256; the only source commit since is `execute_test.py` itself) linked
against `libmlx_metaljax.dylib`, our fork at `vendor/0.32.0`. Full battery and
artifacts: `notes/mlx-vendoring-plan.md` §6.4,
`~/.cache/metaljax-bench/logs/mlx-vendoring/`.

| gate | on the vendored binary |
|---|---|
| 4 texmo | `texmo_gate` **106/106** (0 decline, 0 FAIL, 0 error). Suite-106 vs the **recorded** native arms: geomean **0.9989** (gate's) and **1.0026** (p28's), 105–106 of 106 within 1.2× — inside the 0.9963 noise of those two recorded runs against each other. Live Stage-1 pairings dropped per Oleg's 2026-08-18 native-only scope change |
| 5 models | **row 15 FIXED** (above); row 19 **462.2 ms/step** with bit-identical losses (a second vendored-binary run beside the benefit gate's 463.5 re-spot — two distinct runs, both inside the 456–470 class); rows 5/7 at their cells; **row 1 OPEN — its 237.3 cell does not reproduce**: battery 258.3, standalone after a hard settle 256.8 and 275.6, and a same-tree A/B unconfounds the library — public 292.3 / vendored 291.0 / public 285.7, i.e. vendoring-neutral, drifting on BOTH libraries within one session (full narrative: `notes/mlx-vendoring-plan.md` §6.4). The consolidated re-gate re-measures it on a settled machine |
| 6 contracts | smoke / ingest / decline / coexist ×2 / `bazel test` all pass; **`execute_test` passes** on the committed file (sha256 `3ff58598…`, rc=0). `tests/test_command_buffer.py`: all 11 ran for the first time (the native and pipelined detectors were previously skipped by a version skew), 6 correctness PASS, and the 5 corruption canaries can no longer find a corrupting budget across 28 settings — the patched-build signature |
| 7 wheels | **native-only**, 65 MB, 12 files, no Stage 1 module, dylib byte-identical to the gated binary; installs and drives Metal on 3.12/3.13/3.14 **in venvs with no mlx at all**, and coexists with pip mlx in one process |

**Two things this changes for the release notes.** The row-15 line flips from
"✗ wrong output, no fix at our level" to fixed-and-shipping. The rows-5/7
nondeterminism line **stays**: it was conditional on surviving the fence fix,
and six draws of row 5 (five identical, one divergent at token 51) plus three
of row 7 (every pair divergent) say it survives.

**And one that does not.** The default/trampoline wheel is no longer a release
artifact, so "metaljax carries a patched MLX" is a statement about the native
wheel only. Nothing that installs the public `mlx` from PyPI gets the fence
fix — that still waits on ml-explore/mlx#4099 being released.
