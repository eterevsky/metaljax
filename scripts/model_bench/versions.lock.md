# Benchmark stack versions — lock file

Every number the model benchmark suite publishes is a claim about *these*
versions. Regenerate this file whenever any venv is rebuilt; a results file
without a matching lock entry is not publishable.

Recorded **2026-08-02**. Machine: Apple **M5 Max**, 137.4 GB unified memory
(128 GiB), macOS **26.5.2** (build 25F84), Xcode 26.6.

## Host / OS

| item | value |
| --- | --- |
| chip | Apple M5 Max |
| unified memory | 137.4 GB (128 GiB) |
| macOS | 26.5.2 (25F84) |
| metaljax git | `0c5b144` (2026-08-02) |

## `bench-venv` — metaljax / cpu / mlx legs (Python 3.14.4)

The suite venv did not exist yet when this file was written; the versions
below are what the repo currently pins and what PyPI served on 2026-08-02.
**Replace the "resolved" column with `uv pip freeze` output the moment
`bench-venv` is built** — an unconfirmed row must not back a published
number.

| package | pinned / resolved | source of truth |
| --- | --- | --- |
| python | 3.14.4 | existing `.venv` (texmo needs PEP-649) |
| jax | 0.11.0 | installed in `.venv`; metaljax pins `>=0.11,<0.12` |
| jaxlib | 0.11.0 | installed in `.venv` |
| mlx | 0.32.0 | installed in `.venv` (PyPI latest, 2026-07-07) |
| metaljax | 0.11.0 dist / 0.11.1 in `pyproject.toml` | editable install from `/Users/oleg/metaljax/src`; the installed dist metadata is stale — reinstall before the campaign so the recorded version is real |
| numpy | 2.5.1 | installed in `.venv` |
| mlx-lm | **0.31.3** *(unconfirmed — not installed anywhere yet)* | PyPI latest, 2026-04-22 |
| keras | **3.15.1** *(unconfirmed)* | PyPI latest, 2026-07-29 |
| keras-hub | **0.30.0** *(unconfirmed)* | PyPI latest, 2026-07-24 |
| gemma (DeepMind lib) | **4.0.1** *(unconfirmed)* | PyPI latest, 2026-05-20 |

Regenerate with:

```sh
bench-venv/bin/python -c "import importlib.metadata as m; \
  [print(p, m.version(p)) for p in \
   ('jax','jaxlib','mlx','mlx-lm','metaljax','keras','keras-hub','numpy','gemma')]"
```

## `torch-venv` — torch-MPS leg (Python 3.13.5) — CONFIRMED

Built and validated 2026-08-02 (`uv venv torch-venv --python 3.13`; see
`requirements-torch-mps.txt`). `torch.backends.mps.is_available()` → `True`.

| package | version |
| --- | --- |
| python | 3.13.5 (uv cpython-3.13.5-macos-aarch64-none) |
| torch | **2.13.0** (build `cf30153c4c13`, PyPI 2026-07-08) |
| transformers | **5.14.1** |
| accelerate | **1.14.0** |
| safetensors | **0.8.0** |
| huggingface-hub | **1.26.0** |
| tokenizers | 0.22.2 |
| numpy | 2.5.1 |
| hf-xet | 1.5.2 |
| jinja2 | 3.1.6 |

Full freeze: `uv pip freeze --python torch-venv/bin/python`.

Notes that matter for reproduction:

- Plain PyPI `torch` on macOS/arm64 carries MPS. No extra index.
- `PYTORCH_ENABLE_MPS_FALLBACK` must be **unset**; `adapter_torch_mps.py`
  raises at import if it is set. See `README_comparisons.md`.
- transformers 5.x renamed `torch_dtype` → `dtype` in `from_pretrained`.
- torch never shares a process with jax/mlx.

## Checkpoints

Shared HF cache: `~/.cache/huggingface` (366 GB as of 2026-08-02).
Benchmarks run with `HF_HUB_OFFLINE=1` where possible so a silent
re-resolution cannot swap a revision mid-campaign.

Revisions pinned for the quantized comparison rows are tabulated in
`README_comparisons.md`. Downloaded on 2026-08-02:

| repo | revision | on-disk |
| --- | --- | --- |
| `mlx-community/Qwen3-8B-4bit` | `545dc4251c05440727734bcd94334791f6ab0192` | 4.3 GB |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `241a666dad6cb93c8ff213d39a7f34a36bf26db4` | 4.2 GB |

## Chat templates

**mlx-community mirrors snapshot the upstream chat template at conversion
time and never re-sync it.** All `mlx-community` gemma-4 repos (bf16 *and*
4-bit) ship template sha256 `36e3a42e5cf14cd0` (17466 B) where
`google/gemma-4-*-it` now ships `ae53464bf3be2580` (18681 B) — Google's
carries the header *"Published: 2026-07-09 … Fixed tool-calling loops, turn
closures, and thinking content-ordering"*. `mlx-community/Qwen3-8B-4bit`
is likewise behind `Qwen/Qwen3-8B` (`87a2728cb8dc9fe4` vs
`a55ee1b1660128b7`). Llama-3.1-8B and Qwen3.6-35B-A3B mirrors match
upstream exactly.

Measured consequence for the current suite: **none**. For a single user
message with no system prompt, no tools and thinking off, upstream and
mirror templates render byte-identical text and identical ids (gemma-4-12B
63 tok, gemma-4-E2B 59 tok, Qwen3-8B 59 tok — all `same_text=true,
same_ids=true`). Every diff sits in a branch this prompt never enters.

That is a property of the prompt, so it is pinned rather than trusted. The
mechanism (proposed for `run_bench.py`; `adapter_torch_mps.py` already
implements the rendering half as `render_prompt(...)` +
`METALJAX_BENCH_CHAT_TEMPLATE`):

1. each manifest row names a `template_source` (the upstream `google/…`,
   `Qwen/…` repo);
2. the runner renders the prompt **once** with that template and the
   benchmarked repo's vocabulary, then passes **token ids** — never text —
   to every adapter (mlx-lm, transformers and the JAX/Keras paths all accept
   ids);
3. the template's sha256 goes into every result line, so a mirror that
   re-syncs mid-campaign shows up as a hash change rather than as an
   unexplained shift in prompt length.

Passing ids beats pinning `--chat-template` per stack: it takes the template
out of the runtime entirely, so no stack can disagree about
`add_generation_prompt`, BOS, or trailing whitespace. (`mlx_lm.generate`
suppresses a duplicate BOS only by the heuristic `not
prompt.startswith(tokenizer.bos_token)`; a template emitting BOS at any
other offset still double-counts a prompt token.)

Separately — and more urgently — the current `run_bench.py` legs disagree
about templating at all: `run_gemma_lib` templates via `sampler.chat`,
while `run_mlx` and `run_keras_lm` pass the raw manifest prompt. See
`README_comparisons.md`.
