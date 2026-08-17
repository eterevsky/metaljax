# 0.11.5 final gate battery — 2026-08-16/17 (tree a89ad70 → bdaaa1c, notes-only)

**This document is the release gate.** Every gate is filled in as it lands;
nothing is pre-judged, nothing pre-written. Era rules: measurement only (one
harness-fix exception is allowed and must be noted where used), machine lock
held for every measured phase, strictly sequential, durable artifacts, nothing
committed, pushed or uploaded.

**RELEASE RULE 1 — no stale numbers** (Oleg, 2026-08-16, after the 0.11.4
near-miss): *every number in a release table must come from the release binary.
Changes after the last benchmark run are acceptable only if they provably cannot
move a number; otherwise re-measure the affected rows before release.*

**RELEASE RULE 2 — never "PASS" over a regression**: *a significant regression
on any test suite or benchmark makes the gate verdict REGRESSION, not PASS.
Releasing over one requires (a) Oleg's explicit confirmation, (b) the regression
stated in the gate report itself.*

* **Tree**: `a89ad70` — *version 0.11.5 (0.11.4 burned on TestPyPI, predates
  P24–P27)*. Working tree clean at the start of the battery.
* **Release binary (native wheel)**: frozen at gate 1 below.
* **Stage-1 stack (default wheel)**: `plugin/build/libmetal_pjrt.dylib` +
  `src/metaljax/`, both frozen (last `src/` commit 2026-08-10, dylib 2026-07-31).
* **Artifacts**: `~/.cache/metaljax-bench/logs/release-0.11.5/`.

## Checklist

| # | gate | verdict |
|---|---|---|
| 1 | Freeze the release dylib (build, sha256, byte-identity to the tree) | **PASS** |
| 2 | Pinned jax suite, native, 164 files, `--jobs 1`, vs the 129 whitelist | **PASS** — zero new |
| 3 | `tests/` both legs (Stage 1 + native) | **PASS** — 1258 / (1187+71) |
| 4 | texmo: suite-106 + top_confs pairings, both stacks; both correctness gates | **PASS** |
| 5 | Model rows, every non-embargoed row, guarded | *pending* |
| 6 | Plugin contract suites (execute/ingest/coexist/smoke/census/bazel/gil-free) | **PASS** |
| 7 | Wheels: both variants, fresh-venv installs, `twine check` | *pending* |
| 8 | Release document finale: verdicts, release notes draft, recommendation | *pending* |

*(Filled in as each gate lands. A gate is not scored until its artifacts are on
disk.)*

---
## Gate 1 — the release binary, frozen — **PASS**

`g1_freeze.sh`, one lock hold, 2026-08-16 21:55:47.

| | |
|---|---|
| tree at build time | **`bdaaa1c`** (see the note below) |
| `bazel build //metal:libmetal_pjrt_native.dylib` | rc=0 |
| tree dylib | `plugin-native/bazel-bin/metal/libmetal_pjrt_native.dylib` |
| **sha256** | **`aa7bc0b6fb50479de584534b62851f1dd900a83f8fd6f93c5d573e1114ac0ed4`** |
| frozen copy (all native measurements below) | `~/.cache/metaljax-bench/frozen-0.11.5-aa7bc0b6.dylib` |
| frozen sha256 | identical — **byte-identity OK** |
| size | 47,387,000 B (46 MB, the P18 exported-symbols relink) |

**The tree moved by one commit during the battery, and it is a notes-only
commit.** The brief named `a89ad70`; a concurrent session committed `bdaaa1c`
("plan: post-0.11.5 retirement confirmed") at 21:55:44, three seconds before the
build, touching `notes/cpp-migration-plan.md` and nothing else
(`git show --stat`: 1 file, +12 lines). Under release rule 1 that is the
"provably cannot move a number" case — no source, no build file, no data file —
and it is recorded here rather than assumed away. Everything measured in this
document is `bdaaa1c` = `a89ad70` + that one docs commit.

**The binary is P27's, and that is expected**: `plugin-native/` has not been
touched since `00fba0f` (P27), so the build reproduces `aa7bc0b6…`, the same
hash P27 froze twice (`frozen-p27b` / `frozen-p27c`) after its own
rebuild-and-compare. The 0.11.5 freeze is a third independent reproduction of
it.

**Stage-1 stack, verified frozen** (the default wheel's stack): `src/metaljax/`
last changed 2026-08-10 (`27ec088`), `plugin/build/libmetal_pjrt.dylib` dated
2026-07-31 — neither is in any commit after the RC gate, so every Stage-1 number
in this document is measured on the same code the RC gate and P24 measured.

## Gate 6 — plugin contract suites on the frozen binary — **PASS**

Run in the same hold, **before** any long phase (P26b's policy: a broken binary
should cost minutes, not an hour).

| suite | result | wall |
|---|---|---:|
| `smoke_test.py` | all checkpoints passed | 1 s |
| `execute_test.py` | **all cases match the CPU backend**, **544 `ok` rows** | 36 s |
| `ingest_test.py` | **0 failed** (8 checks; cache peak 0 MB, 0 clears with the cadence off) | 16 s |
| `decline_census.py` | **35 of 35** programs lower | 1 s |
| `coexist_test.py` (`.venv`) | passes, but **both carriers skipped** — the documented `.venv` skip trap | 0 s |
| `coexist_test.py` (**bench** venv) | **tensorflow, both load orders: PASS** | 6 s |
| `coexist_test.py` (**gemma** venv) | **all four**: tensorflow ×2 + array_record ×2 PASS | 9 s |
| `bazel test //...` | `//metal:runtime_gil_free_test` PASSED — **cached**, re-run uncached in gate 4's hold (below) | 0 s |

**`execute_test` verified by diff, not by eyeball.** A whitespace/number-
insensitive diff of tonight's log against P27's own
`final-execute_test.log` is **empty apart from the plugin path line** — 576
lines, 544 `ok` rows, same order, including P25's four flush contracts, P26b's
two callee-sdpa contracts and P27's four flush-pressure contracts. The
`.venv` coexist run is kept as the record of the trap: it exits 0 while
proving nothing, which is why the two carrier interpreters are run explicitly.

## Gate 2 — pinned jax suite, natively, all 164 files — **PASS (zero new failures)**

`g2_suite_tests.sh`, own lock hold, 21:57:39 → 22:26:21. `scripts/release/jax_suite.sh`
with `METALJAX_PLUGIN_PATH` = the frozen 0.11.5 dylib, `--jobs 1` (load-bearing:
parallel runs under-report), `--tests jax-v0.11.0/tests` relative so node ids
match the whitelist. **28.7 min**, 164 of 164 files, no timeouts.

| | 0.11.5 tonight | the RC gate (2026-08-16) | the approved native run (2026-08-11) |
|---|---:|---:|---:|
| passed | **28,073** | 28,068 | 28,068 |
| failed | **129** | 129 | 129 |
| skipped | 6,161 | 6,160 | 6,158 |
| collection errors | 35 | 35 | 35 |
| **pass rate** | **99.54 %** | 99.54 % | 99.54 % |

**The gate: `failures − whitelist = ∅`.** Diffed against the reviewed native
list `notes/data/p12-14-native-failures.txt` (the 142 ids Oleg signed off one by
one in `notes/parity-whitelist-report.md`):

* **NEW failures: 0.** Nothing outside the reviewed set, so no id needed a
  standalone rerun.
* **Newly passing: 13** — exactly the 13 the report says were fixed rather than
  whitelisted (`aot_test` 2, `api_test` 2, `async_collectives_test` 2,
  `export_test` 2, `lax_test` 2, `memories_test` 2, `lax_numpy_indexing_test` 1).
  142 − 13 = **129**.
* **Against the RC gate's own failure list the set is IDENTICAL** — `comm` both
  ways is empty, id for id. Not just the same count: the same 129 tests.

Composition (top files): `export_harnesses_multi_platform` 44, `lobpcg` 27,
`x64_context` 13, `export_test` 5, `api_test` 5, `xla_transform` 4,
`shape_poly` 4, `profiler_session` 3, `async_collectives` 3,
`sparse_bcoo_bcsr` 2 (MLX's fusion bug #8), `logging` 2, `layout` 2, then
singletons — the f64-policy / export-allowlist / PJRT-surface / harness-skew
classes of the whitelist report, unchanged.

Two things that are *not* silent:

* **The wrapper's own verdict line reads "FAIL, NEW failures: 12".**
  `jax_suite.sh` diffs against `notes/data/pinned-0.11.0-failures.txt`, the
  **Stage-1-era 130-id list**; the 12 are the known native-vs-Stage-1 split
  (`x64_context` 6, `sparse_bcoo_bcsr` 2, `dtypes`/`lax_numpy`/`layout`/`pickle`
  1 each), every one of them inside the 142-id native list with a review
  verdict, and the same 12 the RC gate reported. Measured against the list that
  governs this stack the count is zero.
* **+5 passed / +1 skipped against the RC gate, with the failure set identical.**
  Six node ids moved between passed and skipped across the two runs — parameter
  generation, not behaviour — and no id moved into or out of `failed`. Recorded
  rather than smoothed over.

Artifacts: `~/.cache/metaljax-bench/logs/release-0.11.5/jax-suite/`
(`jax_suite.md`, `jax_suite.log`, `jaxtests/{failures.txt,summary.csv}`),
diff output `g2-new-failures.txt` (empty) / `g2-newly-passing.txt` (13).

## Gate 3 — `tests/` on both legs — **PASS**

Same script, second hold (22:27:21 → 22:29:54), sequential.

| leg | result | wall |
|---|---|---:|
| default (**Stage 1** trampoline) | **1258 passed, 0 failed** | 91 s |
| native (`METALJAX_PLUGIN_PATH` = frozen 0.11.5 dylib) | **1187 passed, 71 failed** | 62 s |

The Stage-1 leg is the release number and it is exact: **1258**, no failures.

The native leg's 71 are the documented composition, and the file split proves it
is the same 71 rather than a coincidence of counts:

| file | rows | what they assert |
|---|---:|---|
| `tests/test_moe.py` | 28 | Stage 1 `moe.stats()` Python counters |
| `tests/test_qmm.py` | 26 | Stage 1 `qmm.stats()` Python counters |
| `tests/test_qmm_mxfp4.py` | 16 | same, plus packer / build-cache internals |
| `tests/test_engine_gc.py` | 1 | the Python engine's buffer GC |
| **total** | **71** | |

70 recognizer-family counter rows + `engine_gc`, matching P17's finding that
these cannot pass through a plugin that holds no Python interpreter (`src/` and
`tests/` are frozen; the same graphs run as `execute_test` differential rows
instead). **No new file appears**, and the count is identical to the RC gate's.

Logs: `g3-tests-stage1.log`, `g3-tests-native.log`.

## Gate 4 — texmo: the two pairings, then the two correctness gates — **PASS**

Protocol (P23's, learned the hard way): **each pairing gets a hold of its own
with a 120 s settle and nothing heavy before it, and the correctness gates run
AFTER the measurements** — a 106-subprocess gate poisons the next few minutes in
the same hold, and it poisons the arm that runs first. Arm order inside a hold
is native then Stage 1, as in P23's clean pair.

### 4a — texmo suite-106 (`benchmarks/texmo-suite.csv`, 64-step chunks)

Hold 22:32:14 → 22:48:19, native 476 s then Stage 1 489 s, 106/106 both arms.

| aggregate | n | **0.11.5** | P23 (the RC) | P27 | P25 |
|---|---:|---:|---:|---:|---:|
| whole suite, geomean | 106 | **0.9917** | 1.0050 | 0.9893 | 0.9685 |
| whole suite, median | 106 | **0.9999** | 1.0012 | 1.0001 | 0.9904 |
| `big` (gru/lstm 512–1024, transformer) | 34 | 0.9732 | 1.0107 | | |
| `mid` | 30 | 1.0035 | 1.0033 | | |
| `db` (small recurrent — msl territory) | 40 | 1.0011 | 1.0013 | | |
| `synth` | 2 | 0.9485 | 1.0062 | | |
| rows within 1.2× | 106 | **106** | 106 | 106 | 106 |
| rows at or above 10× | 106 | **0** | 0 | | |
| **rows where native is faster** | 106 | **55** | 42 | | |
| biggest win | | `big09-b8l256` **0.662** (38.42 → 25.44 ms) | | | |
| biggest loss | | `mid11-b64l128` 1.129 (8.74 → 9.87 ms) | | | |

**Drift controls — both arms reproduce the last campaign's**: native
**1.0015** of P27's native arm, Stage 1 **0.9991** of P27's Stage-1 arm (and
0.9907 / 1.0039 against P23's). So the pair's movement from P23's 1.0050 to
0.9917 is the native arm getting faster on `big` — P25's trim-instead-of-dump,
whose signature P25 recorded as concentrated in exactly those eager-main rows.

**Every row outside ±10 % re-measured standalone** (`g4c_controls.sh`, one
process per arm, arms interleaved, three repetitions each), because the suite
context is not trustworthy at that resolution:

| config | in-suite | standalone Stage 1 | standalone native | standalone | verdict |
|---|---:|---:|---:|---:|---|
| `mid11-b64l128` | 1.129 | 8.4412 (8.432/8.424/8.468) | 8.4547 (8.450/8.441/8.472) | **1.0016** | **in-suite artifact** |
| `big09-b8l256` | 0.662 | 38.3505 (38.352/38.347/38.354) | 25.3440 (25.414/25.318/25.300) | **0.6609** | **REAL** (P22's coop width cap) |

The other six outliers are all native-*faster* `big` rows (`big00` 0.897,
`big13-b8l256` 0.884, `big09-b32l128` 0.871, `big13-b32l128` 0.839,
`big15-b32l128` 0.820, `big10-b32l128` 0.800) — P25's named set, left in the
sweep. Substituting the two standalone numbers gives geomean **0.9906**,
median 0.9999, 55 of 106 native-faster, and moves the worst row to
`big16-b32l128` at 1.086.

### 4b — top_confs (163 configurations), and the position effect that had to be measured out

The first pairing (hold 22:51 → 23:08, native then Stage 1) read **1.0123** —
against P23's published 1.0016 — with *both* arms slower than their P23 arms
(native 1.0258, Stage 1 1.0150). That is the shape of an ambient shift, but it
could equally have been the code, so it was **not** reported until three
controls had been run.

| control | reads | says |
|---|---:|---|
| the **0.11.4/RC binary** (`ed355691…`) tonight, same route (`g4c`) | 0.9900 of Stage 1 | old binary, same night |
| … RC tonight / RC at P23 | **1.0032** | the machine is P23's machine |
| 0.11.5 r1 / RC r1 (across holds) | 1.0225 | *looked like* a 2 % code regression |
| **0.11.5 r2 / RC r2, interleaved in ONE hold** (`g4d`) | **1.0019** | it is not the code |
| 0.11.5 r2 / 0.11.5 r1 (same binary, two holds) | **0.9766** | it is the arm's POSITION, worth 2.3 % |
| 0.11.5 shipped / same binary with the P27 flush policy off | **0.9979** | not the flush policy either (shipped is marginally faster) |

So the pairing was re-run **bracketed** (`g4e`: Stage 1 → native → Stage 1, one
hold, 180 s settle), which cancels any monotone drift. The two Stage-1 arms
differ by **1.0008** — this hold is flat — and that is the release measurement:

| aggregate | n | **0.11.5** | P23 | P22 |
|---|---:|---:|---:|---:|
| **native / Stage 1 (same PJRT route, bracketed)** | 163 | **1.0025** (median 1.0012) | 1.0016 | 1.001 |
| rows within 1.2× | 163 | **163** | 163 | 163 |
| rows outside ±10 % | 163 | **0** | 1 | |
| rows where native is faster | 163 | **52** | 63 | |
| biggest win / biggest loss | | `tc009-w16` **0.908** / `tc002-w8` 1.030 | | |
| **native vs the 0.11.3 anchor** | 163 | **1.071× faster** | | 1.073 |
| **Stage 1 vs the 0.11.3 anchor** | 163 | **1.074× faster** | | 1.071 |
| jax-CPU control, anchor / today | 163 | 0.9867 (machine 1.3 % slower — ambient) | | 0.990 |
| **configurations beating jax-CPU** | 163 | native **59** · Stage 1 **59** · anchor 53 | native 58 / S1 59 | native 59 |
| route factor (Stage 1 engine / Stage 1 PJRT) | 163 | 0.9984 | | 1.002 |
| correctness on the engine route | 163 | **163 ok, 0 FAIL, 0 error** | | 163 ok |

Both anchor ratios land on P22's published pair (1.073 / 1.071) to the third
digit, and the beating-jax-CPU count lands on its 59 — three independent
reproductions of the release claim from a different night's binary.

**The lesson, recorded because it will bite again**: on a suite whose rows are
0.1–2 ms, *arm position inside a hold is worth ~2 %*, which is larger than
anything this pairing is trying to measure. A cross-hold A/B of two binaries on
`top_confs` is not evidence; an interleaved or bracketed one is. The
uncorrected first pair (1.0123) is kept in the artifacts.

### 4c — the two correctness gates (hold of their own, after the measurements)

| gate | stack | result | wall |
|---|---|---|---:|
| `plugin-native/texmo_gate.py` (frozen 0.11.5 dylib) | native | **106 ok** (26 via sensitivity scaling), **0 decline, 0 FAIL, 0 error**, of 106 | 266 s |
| `scripts/texmo_check.py benchmarks/texmo-suite.csv` (run 1) | Stage 1 | **105 ok, 1 FAIL, 0 error** | 144 s |
| `scripts/texmo_check.py` (run 2) | Stage 1 | **106 ok, 0 FAIL, 0 error** | 144 s |
| `scripts/texmo_check.py` (run 3) | Stage 1 | **106 ok, 0 FAIL, 0 error** | 140 s |
| `bazel test //... --nocache_test_results` | — | `//metal:runtime_gil_free_test` **PASSED** (uncached, 0.2 s) | 1 s |

**The one FAIL, diagnosed rather than re-rolled.** Run 1's failing row:

```
FAIL db08-b4l1024  tokens.32.fold.oh|lrnn.4.4  worst=inf sens=1.9e+01 tol=9.6e+03 out[19]/20 bad=19
```

* `bad=19` means nineteen of twenty output leaves differ in **finiteness**
  (`np.isfinite(ref) != np.isfinite(got)`), which `nbad == 0` rejects
  unconditionally — no sensitivity scaling can pass it, by design.
* `sens=1.9e+01` is the row's own conditioning on that draw: **+1 ULP on every
  weight moves the training chunk's outputs by 1,900 %**, about **15× this
  row's historical sensitivity** (its five previous gate lines read
  sens 0.39 / 1.1 / 1.2 / 1.2 / 1.3, all `ok~`). `texmo_check` builds a fresh
  `ManagerJax` and samples fresh data per run, so each run is a different draw;
  this one drew a chunk whose `lrnn.4.4` recurrence over 1024 timesteps
  diverges, and the two backends overflowed at different steps.
* **It does not reproduce**: two further full-suite runs are **106/106**, with
  this row drawing sens 0.069 and 0.44. The **native** gate passed the same
  configuration the same night (`ok~`, sens 1.5).
* Class: the documented lottery-row family (P21 `big10-b8l256`, P23/P25
  `mid03`), but with a new face worth writing down — those flaked on
  *tolerance*, this one on *finiteness*, which the gate cannot excuse and a
  future run will therefore fail again on an equally unlucky draw. It is a
  property of the ill-conditioned configuration, not of either stack.
