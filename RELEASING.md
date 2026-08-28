# Releasing metaljax

Two halves: the **release gates** (steps 0–5 — prove the tree is good, get
Oleg's greenlight) and the **upload mechanics** (steps 5.5–7 — build, smoke,
publish).

One wheel covers all supported Pythons — `metaljax-X.Y.Z-py3-none-macosx_14_0_arm64.whl`
— because the PJRT plugin embeds no CPython at all. **Wheel only, no
sdist**: the plugin is built with bazel against a pinned XLA workspace and
links our vendored MLX runtime, none of which an sdist could compile at
install time.

---

## The canonical checklist

### 0. Preflight

- Working tree clean (`git status`), everything intended for the release
  committed.
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

All green, no skips you did not expect. Baseline: **484 tests** (483
passed + 1 xfail, the `stablehlo.dot` decline). Everything runs through
the plugin, so pin the binary under test with `METALJAX_PLUGIN_PATH` when
it matters.

Then the plugin's own suites (they are the differential ones):

```bash
.venv/bin/python plugin-native/execute_test.py
.venv/bin/python plugin-native/ingest_test.py
cd plugin-native && bazel test //... && cd ..
```

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

1. `plugin-native/texmo_gate.py benchmarks/texmo-suite.csv` — whole-model
   correctness vs jax-CPU over every suite configuration, through the
   plugin. Baseline: **106 ok, 0 decline**. A `decline` (the plugin
   refusing a program by name) is a coverage TODO → warning; **any FAIL or
   unexpected ERROR = gate fail**.
2. `scripts/bench_texmo_pjrt.py top_confs.jsonl --out notes/data/texmo-topconfs-<date>.jsonl`
   — 223-config perf sweep, same route.
3. `scripts/texmo_topconfs_compare.py <anchor> <new>` against
   `notes/data/topconfs16k-metal-2026-08-22.jsonl`.
   Gate: geomean regression > 3 % (`TEXMO_GEOMEAN_TOL`) or any single config
   more than 1.3× slower (`TEXMO_CONFIG_TOL`).

Both legs were re-pointed at the PJRT route in 0.11.6 (the Stage-1
drivers `texmo_check.py` / `texmo_topconfs.py` could not see a plugin at
all), and the anchors were re-baselined with them: the old
`texmo-topconfs-final.jsonl` is retired twice over — wrong route, and it
covers the superseded 163-config set (`top_confs.jsonl` became the
223-config 16k set at `112ae10`). The suite-CSV perf anchor, for runs that
bench the 106 suite configs rather than top_confs, is the 0.11.5 release
native arm (`TEXMO_SUITE_ANCHOR`).

Commit the new topconfs JSONL under `notes/data/`.

#### Step 4 — models (`scripts/release/model_gate.sh`, 2–4 h)

Wraps `scripts/model_bench/final_run.sh` unmodified, then audits:

- merges the `[maxtext] … RESULT {json}` lines from
  `final_run.jsonl.maxtext` into records shaped like `final_run.jsonl`;
- `scripts/model_bench/compare_tokens.py` — bf16 greedy streams must match
  CPU. Certified-benign tie-flip rows (`gemma4-e2b-bf16`, `llama31-8b-bf16`
  — STATUS.md footnote 21) are warnings; a divergence on any other row is a
  gate fail (`MODEL_TOKEN_KNOWN` to adjust);
- per-row timing table vs the newest column of `models.md`,
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
- `models.md` — append the new run column (transposed ledger:
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
scripts/build_native_wheel.sh          # --dylib <path> to pin another build
uvx twine check ~/.cache/metaljax-bench/wheels-vendored/native/*.whl
```

The script is a thin wrapper around `uv build` plus the checks that matter:
it defaults the plugin to the **frozen gated binary**
(`~/.cache/metaljax-bench/logs/mlx-vendoring/frozen-path.txt`, release rule
1), and afterwards asserts that no Stage-1 module reappeared in the wheel
and that the dylib inside it is bit-identical to that binary.
`hatch_build.py` refuses to build at all if `metaljax.__version__` and the
pyproject version disagree (they drifted once). Expect **12 files** and
~65 MB: the loader, `metaljax/__init__.py`, the plugin, and the vendored
MLX runtime (`libmlx_metaljax`, `libjaccl_metaljax`, `mlx.metallib`,
`VENDOR_STAMP`). The tag must be `py3-none-macosx_14_0_arm64`.

Install the wheel **non-editable** into a fresh venv on a *different* Python
than the one that built it — do this on **3.12 and 3.14**, and note that
neither venv should have `mlx` in it: the wheel carries its own:

```bash
W=$(ls ~/.cache/metaljax-bench/wheels-vendored/native/*.whl)
for PY in 3.12 3.14; do
  uv venv --python $PY /tmp/mj-check-$PY
  uv pip install -p /tmp/mj-check-$PY/bin/python "$W"
  JAX_PLATFORMS=metal /tmp/mj-check-$PY/bin/python -c \
    "import jax, jax.numpy as jnp; print(jax.devices()); print(2 * jnp.array([1,2,3]))"
done
```

### 6. TestPyPI (only with Oleg's explicit approval)

```bash
uv publish --index testpypi --token <testpypi-token> \
  ~/.cache/metaljax-bench/wheels-vendored/native/*.whl
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
| `METALJAX_PLUGIN_PATH` | the frozen gated dylib | which plugin every gate step measures (`gatelib.sh` pins it; export it yourself to override) |
| `TEXMO_ANCHOR` | `notes/data/topconfs16k-metal-2026-08-22.jsonl` | perf anchor for the 223-config top_confs sweep |
| `TEXMO_SUITE_ANCHOR` | 0.11.5 `suite106-native.jsonl` | perf anchor for the 106-config suite CSV |

## A/B-ing the MLX runtime

The Stage-1-era trick — swap the venv's `mlx` wheel and re-run — died with
the Python engine: the plugin links its MLX by private install name. The
replacement is a BUILD-level A/B, which is what the vendoring battery used
for its row-1 comparison: point the bazel workspace at another MLX tree
with `METALJAX_MLX_DIR` (a public pip layout works, as does a second
vendored build), rebuild the plugin, and pin each resulting dylib with
`METALJAX_PLUGIN_PATH` when measuring.

```bash
METALJAX_MLX_DIR=/path/to/other/mlx bazel build //metal:libmetal_pjrt_native.dylib
```
