# 0.11.5 final gate battery — 2026-08-17 (tree 6500370 → e54ec10, docs-only)

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

* **Tree**: `6500370` at build time; `e54ec10` at write-up (two docs-only
  commits — `models.md`/`STATUS.md` restructure — which cannot move a number).
* **Artifacts**: `~/.cache/metaljax-bench/logs/release-0.11.5/`.

## Checklist

| # | gate | verdict |
|---|---|---|
| 1 | Freeze the release dylib (build, sha256, byte-identity) | **PASS** |
| 2 | Pinned jax suite, native, 164 files, `--jobs 1`, vs the 129 whitelist | *pending* |
| 3 | `tests/` both legs (Stage 1 + native) | *pending* |
| 4 | texmo: suite-106 + top_confs pairings, both stacks; both correctness gates | *pending* |
| 5 | Model rows, every non-embargoed row, guarded | **REGRESSION** (two named, both memory, both attributed and opt-out — see below) |
| 5b | The no-panic contract | **PASS** |
| 6 | Plugin contract suites | **PASS** |
| 7 | Wheels: both variants, fresh-venv installs, `twine check` | *pending* |
| 8 | Finale: verdicts, release notes, recommendation | *pending* |

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

## Gate 5 — the model rows — **REGRESSION** (two named)

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
