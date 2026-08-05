# metaljax release gates — 2026-08-05

**Overall: ❌ FAIL** · version `0.11.3` · tree `0012813` (+1 dirty files) · total gate wall time **1h27m**

| step | status | wall | headline |
|---|---|---:|---|
| Step 2 — JAX pinned suite | ❌ FAIL | 45m06s | 28067 passed / 132 failed (99.53%), 2 new, 0 fixed (not re-run in this batch — record from an earlier run) |
| Step 3 — texmo correctness + perf | ✅ PASS | 21m10s | 104 ok / 0 FAIL / 0 unexpected err (0 env), geomean 1.0016x (not re-run in this batch — record from an earlier run) |
| Step 4 — model suite | ⚠️ WARN | 21m22s | tokens PASS (0 unexpected / 2 known divergences), 0 regressed, 0 newly failing |

## Blocking

- ❌ Step 2 — JAX pinned suite: 2 NEW test failures vs the whitelist

## Non-blocking

- ⚠️ Step 4 — model suite: token agreement: certified-benign divergence on gemma4-e2b-bf16 (token 51/64), llama31-8b-bf16 (token 51/64)

---

### Step 2 — JAX pinned suite (`jax-v0.11.0/tests`): **FAIL**

- run: `--jobs 1 --tests jax-v0.11.0/tests` (164 test files, 45.1 min), driver rc=0
- totals: **28,067 passed / 132 failed** / 6,158 skipped / 35 collection-errors → **99.53%**
- whitelist: 130 known failures (130 in scope, 0 in files not run)
- **NEW failures: 2** · fixed since whitelist: 0 · still failing: 130
- collection/setup ERROR nodes: 0 (environment imports — Pallas/Mosaic CUDA+TPU, optional `hypothesis`; identical on CPU-only, out of scope)

**NEW failures (gate-fail — discuss and whitelist case by case):**

- `jax-v0.11.0/tests/sparse_bcoo_bcsr_test.py` (2)
  - `BCOOTest::test_bcoo_spdot_general0`
  - `BCOOTest::test_bcoo_spdot_general6`

| file | pass | fail | skip | s |
|---|---:|---:|---:|---:|
| `export_harnesses_multi_platform_test.py` | 3162 | 44 | 2392 | 246 |
| `lobpcg_test.py` | 28 | 27 | 0 | 6 |
| `api_test.py` | 724 | 7 | 160 | 53 |
| `export_test.py` | 115 | 7 | 51 | 2 |
| `x64_context_test.py` | 15 | 7 | 2 | 1 |
| `async_collectives_test.py` | 0 | 5 | 7 | 0 |
| `shape_poly_test.py` | 2342 | 4 | 109 | 204 |
| `xla_transform_test.py` | 0 | 4 | 3 | 0 |

Raw: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/jaxtests` (per-file logs, summary.csv, failures.txt), driver log `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/jax_suite.log`


### Step 3 — texmo (correctness + perf): **PASS**

- wall: 21.2 min
- **correctness** (`texmo_check.py`): 104 ok (17 via sensitivity scaling), 0 FAIL, 0 unexpected error, 0 known-environmental error · script summary line: 104 ok / 0 FAIL / 0 error
- **perf sweep** (`texmo_topconfs.py`): 163 ok, 0 FAIL, 0 error → `/Users/oleg/metaljax/notes/data/texmo-topconfs-2026-08-05.jsonl`
- **geomean vs anchor** (`texmo-topconfs-final.jsonl`, old/new, >1 = faster): **1.0016x** over 163 matched configs (CPU control 0.9919x); improved >5%: 1, regressed >5%: 0
- thresholds: geomean ≥ 0.97x, no config > 1.3x slower (`TEXMO_GEOMEAN_TOL` / `TEXMO_CONFIG_TOL`)

| | config | anchor ms | new ms | ratio |
|---|---|---:|---:|---:|
| worst | `tokens.32.hexbpe.emb.16\|dense.8.tanh-norm-rnn.` fp32 b64 l128 | 0.8122 | 0.8388 | 0.968x |
| worst | `bits.4.oh+bp\|rnn.16.gelu-mgru.16-norm-dense.16` fp32 b64 l128 | 0.7251 | 0.7477 | 0.970x |
| worst | `bits.4.oh+bp\|rnn.32.gelu-dense.16.gelu` fp32 b64 l64 | 0.4307 | 0.4428 | 0.973x |
| worst | `bits.4.oh+bp\|dense.8.gelu-rnn.32.gelu-rnn.16.g` fp32 b64 l128 | 0.8048 | 0.8265 | 0.974x |
| worst | `tokens.32.shift.emb.4\|split.add(split.mul(spli` fp32 b32 l64 | 0.9482 | 0.9664 | 0.981x |
| worst | `bits.4.oh+bp\|mgru.8-norm-suffix.2-dense.8.gelu` fp32 b32 l128 | 0.649 | 0.6584 | 0.986x |
| worst | `tokens.32.fold.emb.4\|split.add(rnn.2.tanh-spli` fp32 b8 l256 | 0.4136 | 0.4191 | 0.987x |
| worst | `bits.4.oh+bp\|rnn.16.gelu-split.cat(rnn.16.gelu` fp32 b64 l128 | 0.7015 | 0.7105 | 0.987x |
| best | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-norm-spli` fp32 b32 l256 | 1.0111 | 0.9456 | 1.069x |
| best | `bits.4.emb.4\|split.add(conv.2-split.add(split.` fp32 b1 l256 | 0.5814 | 0.5616 | 1.035x |
| best | `bits.4.emb.4\|rglru.2-norm-conv.4-split.mul(pas` fp32 b4 l128 | 0.8269 | 0.8051 | 1.027x |

Raw: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/texmo_check.log`, `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/texmo_topconfs.log`, `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/texmo_compare.log`; data `/Users/oleg/metaljax/notes/data/texmo-topconfs-2026-08-05.jsonl` (commit under notes/data/)


### Step 4 — model suite: **WARN**

- wall: 21.4 min
- merged records: 17 from `final_run.jsonl` + 6 maxtext RESULT rows → `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/model_merged.jsonl`
- token agreement (`compare_tokens.py`): **PASS** — 2 divergent row(s), 2 certified-benign (`MODEL_TOKEN_KNOWN`), 0 unexpected; script verdict: GATE FAIL: 2 bf16 divergences
- timing vs ledger column **0.11.3-dev (2026-08-04)⁴** (`models.md`); regression threshold 10% (`MODEL_REGRESS_TOL`)

| # | benchmark | 0.11.3-dev (2026-08-04)⁴ | this run | Δ | |
|---|---|---:|---:|---:|---|
| 1 | gemma4-31B | — | 237.5 | — | NEW (was blocked) |
| 2 | gemma4-12B | — | 92.5 | — | NEW (was blocked) |
| 3 | gemma4-26B-A4B (MoE) | 44.3 | 43.9 | -0.9% |  |
| 4 | gemma4-E2B | — | 27.5 | — | NEW (was blocked) |
| 5 | Qwen3-8B | — | 57.8 | — | NEW (was blocked) |
| 6 | Llama-3.1-8B | — | 54.2 | — | NEW (was blocked) |
| 7 | gpt-oss-20b | 22.2 | 21.8 | -1.8% |  |
| 8 | Qwen3.6-35B-A3B | — | — | — | not in final_run.sh |
| 9 | R1-Distill-32B | 217.7 | — | — | not in final_run.sh |
| 10 | DeepSeek-V2-Lite | — | — | — | not in final_run.sh |
| 11 | Qwen3-0.6B maxtext decode | — | 15.78 | — | NEW (was blocked) |
| 12 | Mixtral 8×7B | — | — | — | not in final_run.sh |
| 13 | E2B keras-int4 | 85 | 81.1 | -4.6% |  |
| 14 | qwix-int8 0.6B | 31.8 | 32.53 | +2.3% |  |
| 15 | qwix-int8 8B | — | — | — | not in final_run.sh |
| 16 | SigLIP 2 (fwd ms) | — | 82.86 | — | NEW (was blocked) |
| 17 | SD3.5 (ms/diff-step) | — | — | — | not in final_run.sh |
| 18 | LoRA E2B (ms/step) | — | 535.1 | — | NEW (was blocked) |
| 19 | maxtext train 0.6B (ms/step) | — | 964.2 | — | NEW (was blocked) |
| 20 | 235B-A22B 3-bit (mlx-only) | — | — | — | not in final_run.sh |

**Newly measured (previously blocked):**

- row 1 gemma4-31B: 237.5
- row 2 gemma4-12B: 92.5
- row 4 gemma4-E2B: 27.5
- row 5 Qwen3-8B: 57.8
- row 6 Llama-3.1-8B: 54.2
- row 11 Qwen3-0.6B maxtext decode: 15.78
- row 16 SigLIP 2 (fwd ms): 82.86
- row 18 LoRA E2B (ms/step): 535.1
- row 19 maxtext train 0.6B (ms/step): 964.2

<details><summary>compare_tokens.py output</summary>

```
gemma4-12b-bf16          AGREE (64 tokens)
  gemma4-26b-a4b           single-backend (recorded only)
  gemma4-31b-bf16          single-backend (recorded only)
  gemma4-e2b-bf16          FAIL: diverges at token 51/64
  gemma4-e2b-int4          AGREE (64 tokens)
  gpt-oss-20b              single-backend (recorded only)
  llama31-8b-bf16          FAIL: diverges at token 51/64
  qwen3-8b-bf16            AGREE (64 tokens)

GATE FAIL: 2 bf16 divergences
```

</details>

Raw: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/model_final_run.log`, `/Users/oleg/.cache/metaljax-bench/logs/final_run.jsonl`, `/Users/oleg/.cache/metaljax-bench/logs/final_run.jsonl.maxtext`, merged `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/model_merged.jsonl`, tokens `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/model_tokens.log`


---

## Raw artifacts

- gate directory: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05`
- jax / run_dir: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/jaxtests`
- jax / driver_log: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/jax_suite.log`
- texmo / check: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/texmo_check.log`
- texmo / topconfs: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/texmo_topconfs.log`
- texmo / compare: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/texmo_compare.log`
- models / run: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/model_final_run.log`
- models / raw: `/Users/oleg/.cache/metaljax-bench/logs/final_run.jsonl`
- models / tokens: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/model_tokens.log`
- models / merged: `/Users/oleg/.cache/metaljax-bench/logs/release-gate/2026-08-05/model_merged.jsonl`

Per-step wall time (from `steps.tsv`): jax 45m06s, texmo 21m10s, models 21m22s
