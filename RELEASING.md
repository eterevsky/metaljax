# Releasing metaljax

Two halves: the **release gates** (steps 0–5 — prove the tree is good, get
Oleg's greenlight) and the **upload mechanics** (steps 5.5–7 — build, smoke,
publish).

One wheel covers all supported Pythons: the PJRT dylib is built against
CPython's limited API (>=3.12), so artifacts are
`metaljax-X.Y.Z-py3-none-macosx_14_0_arm64.whl` + an sdist (which
compiles the plugin at install time; needs Xcode CLT).

---

## The canonical checklist

### 0. Preflight

- Working tree clean (`git status`), everything intended for the release
  committed — the sdist packages `src/`, `scripts/`, `tests/`.
- Version bumped in **both** places:
  - `pyproject.toml` → `[project] version`
  - `src/metaljax/__init__.py` → `__version__`
- Raw gate data from the *previous* release committed under `notes/data/`
  (topconfs JSONL, model-bench JSONL, pinned-failure list) so the new run
  has an anchor to diff against.

### 1. Local test suite

```bash
cd /Users/oleg/metaljax
.venv/bin/python -m pytest tests/
```

All green, no skips you did not expect.

### 2–4. Release gates (overnight, one command)

```bash
nohup scripts/release/run_gates.sh > ~/gates.log 2>&1 &
```

Runs steps 2, 3, 4 **strictly sequentially** (each wants the whole GPU),
continues past a failing gate so one overnight run yields the complete
picture, and finishes by consolidating everything into a single markdown
report. Everything lands in
`~/.cache/metaljax-bench/logs/release-gate/<date>/`; the final line is
`RELEASE_GATES_DONE failed_steps=N dir=…`. Budget **6–8 h**.

Cheap plumbing check before trusting an overnight run:

```bash
scripts/release/run_gates.sh --smoke      # ~15 min, filtered/skipped workloads
```

Individual steps (each is also fire-and-forget on its own, and each accepts
`--smoke`):

#### Step 2 — JAX pinned test suite (`scripts/release/jax_suite.sh`, ~3.5 h)

Runs `scripts/run_jax_tests.py` against the pinned `jax-v0.11.0/` checkout in
**exactly** the configuration that produced the approved 27,649 / 130 run:

```
.venv/bin/python scripts/run_jax_tests.py <outdir> --jobs 1 --tests jax-v0.11.0/tests
```

- `--jobs 1` is load-bearing. Parallel (3/4-job) runs **under-report**
  failures — the campaign lesson in CLAUDE.md item 20 (fft_test showed 3
  failures at 4 jobs, 5 sequentially). Never gate on a parallel run.
- The `--tests` path must stay **relative**: pytest node ids inherit it and
  the whitelist is keyed on `jax-v0.11.0/tests/<file>.py::…`.

Pass criterion: **no NEW failures** vs `notes/data/pinned-0.11.0-failures.txt`
(130 approved ids). `scripts/release/jax_suite_diff.py` does the set
comparison and prints new failures, newly-fixed tests (informational — remove
them from the whitelist on approval), totals, and any file that timed out.
New failures are not automatically fatal to the *release* — they are fatal to
the gate: discuss with Oleg and whitelist case by case, as with the 0.11.0
sign-off.

#### Step 3 — texmo (`scripts/release/texmo_gate.sh`, ~2 h)

1. `scripts/texmo_check.py ~/texmo/benchmarks/m5-metal.csv` — whole-model
   correctness vs jax-CPU over all 104 suite configurations. Baseline:
   **104 ok**. (It was 96 ok + 8 errors on the `tokens.32.raw_fold.*` specs
   until commit `9ef5f58` taught the driver to remap the renamed tokenset,
   `.raw_fold` → `.fold`.) The wrapper still classifies a missing-tokenset
   error on those specs as a warning; **any other error, or any FAIL = gate
   fail**.
2. `scripts/texmo_topconfs.py top_confs.jsonl --out notes/data/texmo-topconfs-<date>.jsonl`
   — 163-config perf + correctness sweep.
3. `scripts/texmo_topconfs_compare.py <anchor> <new>` against
   `notes/data/texmo-topconfs-final.jsonl` (the 0.11.2 anchor).
   Gate: geomean regression > 3 % (`TEXMO_GEOMEAN_TOL`) or any single config
   more than 1.3× slower (`TEXMO_CONFIG_TOL`).

Commit the new topconfs JSONL under `notes/data/`.

#### Step 4 — models (`scripts/release/model_gate.sh`, 2–4 h)

Wraps `scripts/model_bench/final_run.sh` unmodified, then audits:

- merges the `[maxtext] … RESULT {json}` lines from
  `final_run.jsonl.maxtext` into records shaped like `final_run.jsonl`;
- `scripts/model_bench/compare_tokens.py` — bf16 greedy streams must match
  CPU. Certified-benign tie-flip rows (`gemma4-e2b-bf16`, `llama31-8b-bf16`
  — STATUS.md footnote 21) are warnings; a divergence on any other row is a
  gate fail (`MODEL_TOKEN_KNOWN` to adjust);
- per-row timing table vs the newest column of `benchmarks/models.md`,
  flagging rows regressed > 10 % (`MODEL_REGRESS_TOL`) and rows newly
  failing.

**Locking:** `final_run.sh` grabs `/tmp/metaljax-bench.lock` per cell — the
wrapper deliberately adds no outer lock (it would deadlock). Nothing else
may touch the GPU while it runs.

Two `final_run.sh` quirks the wrapper works around without editing it:
its trailing texmo re-baseline **overwrites the perf anchor**
`notes/data/texmo-topconfs-final.jsonl` (the wrapper saves the fresh sweep
as `notes/data/texmo-topconfs-<date>-finalrun.jsonl` and restores the
anchor), and it **appends** to `final_run.jsonl.maxtext` instead of
truncating it (the wrapper rotates the old side file away first). Since
step 3 now owns the texmo sweep, dropping the trailing re-baseline from
`final_run.sh` would save ~1 h per gate.

### 4.5 Docs

Update, in this order:

- `STATUS.md` — the per-benchmark rows with the new numbers/footnotes.
- `benchmarks/texmo.md` — append a column/row for this run (geomean vs the
  previous anchor, checks, raw file).
- `benchmarks/models.md` — append the new run column (transposed ledger:
  benchmark rows, run columns). The gate's per-row table is the source.
- `README.md` — headline numbers.
- `notes/data/` — commit the raw JSONLs the gates produced.

### 5. Summary to Oleg

`~/.cache/metaljax-bench/logs/release-gate/<date>/summary.md` is the report:
per-step PASS/FAIL, wall time, new-failure lists, perf tables, links to every
raw log. Oleg greenlights TestPyPI from it.

---

## Upload mechanics

### 5.5 Build and local smoke

```bash
rm -rf dist
uv build
uvx twine check dist/*
```

The wheel tag must be `py3-none-macosx_14_0_arm64` (the hatch build hook
in `hatch_build.py` compiles the dylib and sets it).

Install the wheel **non-editable** into a fresh venv on a *different* Python
than the one that built it — do this on **3.12 and 3.14** (the limited-API
build has broken on exactly one version before: the `PyObject_CallMethod`
`'#'`-format ABI bug only showed on 3.12):

```bash
for PY in 3.12 3.14; do
  uv venv --python $PY /tmp/mj-check-$PY
  uv pip install -p /tmp/mj-check-$PY/bin/python dist/*.whl
  JAX_PLATFORMS=metal /tmp/mj-check-$PY/bin/python -c \
    "import jax, jax.numpy as jnp; print(jax.devices()); print(2 * jnp.array([1,2,3]))"
done
```

### 6. TestPyPI (only with Oleg's explicit approval)

```bash
uv publish --index testpypi --token <testpypi-token>
uv venv --python 3.13 /tmp/mj-testpypi
uv pip install -p /tmp/mj-testpypi/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ metaljax
JAX_PLATFORMS=metal /tmp/mj-testpypi/bin/python -c \
  "import jax, jax.numpy as jnp; print(jax.devices()); print(2 * jnp.array([1,2,3]))"
```

(`uv publish --index testpypi` needs `[[tool.uv.index]]` config; the
plain form is `uv publish --publish-url https://test.pypi.org/legacy/`.)

### 7. Public PyPI + git — **Oleg does this himself**

```bash
uv publish --token <pypi-token>   # or export UV_PUBLISH_TOKEN
git push
git tag v0.11.3 && git push origin v0.11.3
```

Tokens come from https://pypi.org/manage/account/token/ (create the
project-scoped token after the first upload).

---

## Gate knobs

| variable | default | meaning |
|---|---|---|
| `RELEASE_GATE_DIR` | `~/.cache/metaljax-bench/logs/release-gate/<date>` | where every gate artifact goes |
| `GATE_DATE` | today | just the directory name (use for re-runs: `2026-08-04-rerun`) |
| `METALJAX_GATE_LOCK` | 1 | take `/tmp/metaljax-bench.lock` around steps 2 and 3 |
| `JAX_SUITE_JOBS` | 1 | **do not raise for a real gate** — parallel runs under-report |
| `TEXMO_GEOMEAN_TOL` | 0.03 | max geomean regression vs the anchor |
| `TEXMO_CONFIG_TOL` | 1.3 | max single-config slowdown |
| `TEXMO_ANCHOR` | `notes/data/texmo-topconfs-final.jsonl` | perf anchor |
| `MODEL_REGRESS_TOL` | 0.10 | per-row model regression threshold |
| `MODEL_REGRESS_FAIL` | 1 | whether a regression fails the gate (0 = warn) |
| `MODEL_TOKEN_KNOWN` | `gemma4-e2b-bf16,llama31-8b-bf16` | certified-benign token divergences |
