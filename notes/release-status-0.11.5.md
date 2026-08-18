# 0.11.5 — Release Status (final, post-re-gate)

Written 2026-08-18, after the consolidated re-gate finished at 16:17. Every
number in this document was measured on the one binary that ships:
`frozen-vendor-d651add3.dylib` (sha256 `d651add3…`), the native plugin
linked against our own patched MLX. Nothing has been uploaded anywhere yet.

Companion documents: `notes/release-gates-0.11.5.md` (the full gate record),
`benchmarks/models.md` (the numbers table), `STATUS.md` footnote 36 (the
per-row detail), `notes/mlx-patch-diagnosis.md` (the MLX bug),
`notes/no-panic-governor.md` (the memory governor).

## What this release is

0.11.5 is the fully native plugin, plus two things we had to build to make
it trustworthy:

1. **The C++ plugin** (`plugin-native/`). The whole backend — StableHLO
   parsing, lowering, the fused-pattern recognizers, the msl_scan kernel
   generator, LAPACK host ops, callbacks — now runs in native code with no
   Python in the execute path. It passes 99.54 % of the pinned JAX test
   suite and performs at or above the old Python engine everywhere we
   measure.

2. **The no-panic contract.** After kernel panic #9 you set the rule:
   metaljax must never take the machine down — degrade performance if
   possible, fail with a clean out-of-memory error if not, never panic. The
   memory governor implements it. In this campaign every over-budget
   situation ended in a clean, named refusal; the previously impossible
   rows (8, 9, 10, 15) all run.

3. **Our own MLX.** The wheel no longer depends on pip's mlx. It carries a
   private build (`libmlx_metaljax.dylib`) from our fork at v0.32.0 plus
   the command-buffer fence fix — upstream wrote the fix but has never
   released it, so we ship it ourselves. This is what fixed row 15's wrong
   output. Provenance is recorded in a `VENDOR_STAMP` file inside the
   wheel; the fork branches are on `eterevsky/mlx`.

## Verification — what ran and what it showed

Everything below ran today, on the release binary, in three sequential
phases (all model rows → the JAX test suite → texmo and the plugin's own
tests).

| check | result |
|---|---|
| **JAX test suite** (the pinned jax 0.11.0 tests, 164 files, single-job) | **PASS** — 28,073 passed / 129 failed = 99.54 %. The 129 failures are exactly the reviewed whitelist, test-for-test: nothing new broke, nothing quietly changed. |
| **Model benchmarks** (19 of the 20 STATUS.md rows; row 12 stays blocked on its 93 GB download) | **PASS** — every row within noise of its previous number or better; details below. |
| **texmo correctness** (`texmo_gate`) | **PASS** — 106 of 106 configurations match jax-CPU. |
| **texmo performance** (the 106-config suite + the 163-config top_confs sweep) | **PASS** — within 0.3–0.5 % of the recorded runs, every row within 1.2×, none outside ±10 %. 59 configurations beat jax-CPU, same as before. |
| **Plugin test suites** (`tests/` + execute/ingest/decline/coexist/bazel) | **PASS** — one explained delta, item 2 below. |
| **The wheel** | **PASS** — built, verified, not yet uploaded; see "What ships". |
| **No-panic contract** | **PASS** — zero panics, zero wedges across the whole campaign; every memory refusal was the governor's clean error. |

## The model numbers

The full table is `benchmarks/models.md` (numbers only; per-row detail in
STATUS.md footnote 36). Headlines:

- **Row 15 (qwix-int8 Qwen3-8B) is fixed and has its first real number:
  401.4 ms/tok.** It had produced garbage on every binary and both engines
  since August 3. The cause was MLX's fence-tracking bug; with our patched
  MLX it returns the same first token 10 times out of 10 and decodes
  coherent text. (jax-CPU on this model: 2118 ms/tok.)
- **First-ever numbers**: row 8 (29.7 ms/tok), row 10 (1871.1 ms/tok, ~90 GB,
  right at the machine's edge — one attempt ended in the governor's clean
  refusal rather than a panic, which is the contract working).
- **Everything else is at parity or slightly better** than the previous
  measurements — e.g. 31B at 235.5, 12B at 92.3, MoE-26B at 43.5, gpt-oss
  at 21.7, maxtext train at 460.2 with the training loss bit-identical
  across all nine runs of the campaign (the strongest evidence the MLX
  swap changed no numerics).
- **One named drift**: LoRA (row 18) read 370.7 against a 359–362 cluster
  (+3.2 %), a single sample inside the row's usual spread.

## Three disclosures (in the release notes, not hidden)

1. **Row 1 (gemma4-31B) has real run-to-run variance on this machine.**
   The release cell (235.5) reproduced today, but standalone re-runs within
   the same hour read 261–297. We tested the obvious suspects: it is not
   the vendored MLX (same drift on pip's build, A/B'd back to back), not
   the memory guard, not background daemons; slow runs are uniformly ~1.26×
   slower across prefill and decode alike, which points at GPU
   thermal/frequency state. Historical readings span 237–302. Getting
   further needs root (`powermetrics`).
2. **Five corruption-canary tests now "fail" — because they can't find the
   bug any more.** These tests exist to detect the MLX command-buffer
   corruption; on the patched build they search all 28 budget settings,
   find nothing to detect, and raise an assertion whose own text says "MLX
   fixed the command-buffer split bug". All other 1,253 tests are
   unchanged. The canaries should be re-pointed at the patched behavior
   post-release; this needs your ack since it shows up as "5 failed" in the
   raw pytest count.
3. **Rows 5 and 7 keep their token-nondeterminism note.** Timing is
   reproducible but greedy token streams can differ run to run (row 5: one
   divergence in six draws; row 7: every pair). It survived the fence fix,
   so it is a separate mechanism (fused-attention emit ordering), and the
   release note stands.

## What ships

One wheel: `metaljax-0.11.5-py3-none-macosx_14_0_arm64.whl` — 65 MB,
sha256 `95d6f1f07d66c7aa781659276362358770260ff7c5c4893d5946fdf7086087be`.

- **Native-only**, per your ruling: 12 files, none of the old Python
  engine's modules. `metaljax.__version__` reports 0.11.5.
- **Self-contained Metal runtime**: the patched MLX rides inside the wheel
  under a private name. A venv with no mlx installed works; a venv WITH
  pip's mlx also works (the two libraries coexist — that's what the rename
  is for).
- The plugin dylib inside the wheel is **byte-identical** to the gated
  binary every number above was measured on.
- Verified installs on Python 3.12 / 3.13 / 3.14; `twine check` passes.
- The old Python engine stays in the repo as a frozen dev reference until
  the post-release retirement. Note it still uses pip's **unpatched** MLX —
  the fence fix exists only inside the native wheel until upstream ships
  their release.

## What's left, in order

1. **Me**: upload to TestPyPI → install from TestPyPI into a fresh venv and
   verify end-to-end → report here.
2. **You**: review this document and the gate record → publish to real
   PyPI → `git push` + tag. Upstream, whenever you like: the release
   request on ml-explore/mlx#4099 and our hardening PR (drafts in
   `notes/patches/`).
3. **Post-release queue**: retire the Python path + docs cleanup (your
   confirmed first item); the framework-gap fix list (we are ~1.8× behind
   mlx-lm on 31B decode; the decomposed plan is in
   `notes/framework-gap-gemma31b.md`); re-point the corruption canaries;
   re-measure the Stage-1-era command-buffer budget workarounds that no
   longer have failing evidence behind them; fusion bug #8 fork branch;
   row 12's download; row 20's packed sub-byte feature; row 8 performance
   (29.7 vs torch's 13.7).
