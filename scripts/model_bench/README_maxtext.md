# MaxText rows of the public benchmark suite

Four manifest rows run AI-Hypercomputer/MaxText through metaljax:

| id | model | what it exercises |
|---|---|---|
| `qwen3-06b-maxtext` | Qwen/Qwen3-0.6B | MaxEngine decode (prefill + AR), dense transformer |
| `deepseek-v2-lite`  | deepseek-ai/DeepSeek-V2-Lite-Chat | MoE + MLA attention, 15.7B params |
| `maxtext-qwix-int8` | Qwen/Qwen3-0.6B | qwix int8: s8 x s8 -> s32 `dot_general` |
| `maxtext-train-06b` | Qwen/Qwen3-0.6B | one training step, AdamW, synthetic data |

All four go through `adapter_maxtext.py`:

```python
run_maxtext(bench, backend, prompt, n_decode)   # decode rows
run_maxtext_train(bench, backend)               # maxtext-train-06b
```

`run_bench.py` dispatches on the manifest `path` field (`maxtext` /
`maxtext_train`); wire it up with

```python
from adapter_maxtext import run_maxtext, run_maxtext_train
ADAPTERS["maxtext"] = run_maxtext
ADAPTERS["maxtext_train"] = run_maxtext_train   # ignores the prompt/n_decode args
```

(the module must be importable -- `sys.path.insert(0, str(HERE))`, as
`run_gemma_lib` already does for `gemma_loader`.)

## Install

```sh
scripts/model_bench/setup_maxtext.sh                 # venv + deps + maxtext
scripts/model_bench/convert_checkpoints.sh           # HF -> MaxText Orbax
```

Both are idempotent. Defaults:

* venv: `~/.cache/metaljax-bench/maxtext/venv` (override: first argument)
* MaxText checkout: `~/.cache/metaljax-bench/maxtext/repo` (`$MAXTEXT_REPO`)
* checkpoints: `~/.cache/metaljax-bench/maxtext/ckpt/<model>/0/items`
  (`$MAXTEXT_CKPT_ROOT`)

Nothing is written into the repo and nothing is re-downloaded from HF: the
converter reads the existing `~/.cache/huggingface` snapshots via
`--hf_model_path`. Checkpoint sizes: qwen3-0.6b 890 MB, deepseek2-16b 23 GB.

There is **no separate int8 checkpoint**: `maxtext-qwix-int8` reads the same
bf16 qwen3-0.6b checkpoint and qwix quantizes during the forward pass
(`checkpoint_is_quantized=false`). MaxEngine's `save_quantized_params_path`
route is AQT-only -- with `use_qwix_quantization=true`,
`quantizations.configure_quantization` returns `None`, so `self.model.quant`
is `None` and `quantize_params` never runs. Baking a `qrhs.frozen` checkpoint
would need a separate qwix conversion pass; it is not needed to measure the
int8 `dot_general` path, which is what the row is for.

Smoke test:

```sh
JAX_PLATFORMS=cpu   $VENV/bin/python scripts/model_bench/adapter_maxtext.py qwen3-06b-maxtext
JAX_PLATFORMS=metal $VENV/bin/python scripts/model_bench/adapter_maxtext.py qwen3-06b-maxtext
JAX_PLATFORMS=cpu   $VENV/bin/python scripts/model_bench/adapter_maxtext.py maxtext-train-06b --train
```

Debug knobs: `MAXTEXT_VERBOSE=1` (restore the ~900 config log lines),
`MAXTEXT_OVERRIDES="key=val ..."` (extra pyconfig overrides),
`MAXTEXT_PREFILL_LEN`, `MAXTEXT_TRAIN_STEPS`, `MAXTEXT_TRAIN_SEQ`.

## Config decisions that are load-bearing

* **`attention=dot_product`.** MaxText's `autoselected` only falls back to
  reference attention when `target_hardware == "cpu"`. Our platform string is
  `metal`, so prefill (length >= 128) would be routed into a Pallas/TPU splash
  kernel.
* **`DECOUPLE_GCLOUD=TRUE`** (set at adapter import; it is read at maxtext
  import time) stubs JetStream/GCS/Vertex/goodput.
* **`dataset_type=synthetic`, `skip_jax_distributed_system=True`.**
* **`weight_dtype=bfloat16`** for decode rows -- the default `float32` would
  upcast every checkpoint on load (deepseek-v2-lite: 31 GB -> 63 GB).
* **MoE (`deepseek-v2-lite`): `sparse_matmul=false megablox=false`.** megablox
  is a Pallas TPU kernel; the sparse path uses `lax.ragged_dot`.
* **MLA (`deepseek-v2-lite`): `mla_naive_kvcache=false`.** With the default
  naive cache, decode dies in `engine.insert`:
  `dynamic_update_slice update shape (1,64,16,1,192) for operand shape
  (1,64,16,1,128)` -- the prefill cache stores the full key
  (qk_nope 128 + qk_rope 64) while the AR cache is sized by `v_head_dim` 128.
  `mla_naive_kvcache=false` caches the compressed latent instead and is what
  MaxText's own `reshard_checkpoint.py` example uses for this model.

## Blockers hit, and how they were resolved

1. **No `cpu` extra on PyPI, and the accelerator extras pull libtpu/nvidia.**
   Hand-built `requirements_maxtext_macos.txt` from
   `src/dependencies/requirements/requirements_decoupled_jax_0_7.1.txt`, plus
   `constraints_maxtext.txt` pinning jax/jaxlib to 0.11.0 (MaxText's own file
   pins `jax==0.7.1`). Ten extra transitive imports had to be added by hand --
   each one is annotated in the requirements file.

2. **`typeguard` 4.x breaks tokamax/jaxtyping at import**: it `ast.parse`s
   jaxtyping's shape annotations (`"*B T H d"`) and raises `SyntaxError`.
   Pinned `typeguard==2.13.3`.

3. **TensorFlow and `grain` cannot coexist on macOS/arm64.** Both ship their
   own protobuf/absl C++ symbols; `python -c "import grain; import
   tensorflow"` SIGSEGVs, and so does the reverse order. `grain` is mandatory
   (`maxtext.common.checkpointing` imports it), TensorFlow is only needed by
   the TFDS input pipeline. TensorFlow is therefore *not* installed and
   `adapter_maxtext._stub_tensorflow()` provides the one attribute
   `maxtext.trainers.pre_train.train` touches at module scope.

4. **JetStream cannot be installed** (its `token_utils` imports `seqio` ->
   TensorFlow), so decode runs against MaxText's `DECOUPLE_GCLOUD` stubs. The
   stub `ResultTokens` is a plain object, and MaxEngine's `prefill`/`generate`
   are `jax.jit`'d functions that *return* one -- jit rejects it ("not a valid
   JAX type"). `adapter_maxtext._patch_jetstream_stub()` registers the stub
   class as a pytree (and adds `get_result_at_slot`) before maxengine is
   imported. Token ids are read straight from `ResultTokens.data`
   (col 0 = token, 1 = valid, 2 = length).

5. **`maxtext.inference.decode` is unusable in decoupled mode** -- it raises
   "decode requires the JetStream tokenizer". The adapter drives MaxEngine
   directly (`prefill` / `init_decode_state` / `insert` / `generate`) and uses
   `transformers.AutoTokenizer` for the HF tokenizer, which also gives exact
   control over the prefill/decode split the benchmark measures.

6. **`enable_checkpointing=false` + `load_parameters_path`** fails pydantic
   validation ("You must set enable_checkpointing=True to load a checkpoint"),
   so the decode rows leave checkpointing enabled and only the train row turns
   it off.

## Timing methodology

`jax.block_until_ready()` is a **no-op on metaljax** (PJRT events are born
ready), so every timed section is closed by pulling a value to the host with
`int(np.asarray(...))`, which is a real device sync. Both backends pay the
same per-token sync, so the numbers are comparable.

* `load_s` -- pyconfig + MaxEngine construction + `load_params`
* `warmup_s` -- first prefill + 8 generate steps (compiles everything)
* `prefill_ms` -- warm prefill alone
* `decode_ms_tok` -- `n_decode - 1` warm generate steps / count
  (the first token comes out of prefill)
* train: `warmup_s` = first step (trace + compile), `step_ms` = mean of the
  next `MAXTEXT_TRAIN_STEPS` (default 3)

## Known metaljax bug found by this row (NOT a MaxText problem)

**MaxText decode with `weight_dtype=bfloat16` produces garbage tokens on
metal whenever `mx.compile` is enabled, and the garbage is different on every
run.** See the report/notes for the full bisect. Reproduce:

```sh
JAX_PLATFORMS=metal $VENV/bin/python scripts/model_bench/adapter_maxtext.py qwen3-06b-maxtext --decode-tokens 8
#   -> ' interest菱?avelavel级-Disposition Goa'      (run 1)
#   -> '宵 pains完善的翱走了走了'                      (run 2)
JAX_PLATFORMS=metal METALJAX_COMPILE=0 $VENV/bin/python scripts/model_bench/adapter_maxtext.py qwen3-06b-maxtext --decode-tokens 8
#   -> ' Paris. The capital of France is also'      (correct, matches CPU)
JAX_PLATFORMS=metal MAXTEXT_OVERRIDES="weight_dtype=float32" $VENV/bin/python scripts/model_bench/adapter_maxtext.py qwen3-06b-maxtext --decode-tokens 8
#   -> ' Paris. The capital of France is also'      (correct)
```

Ruled out: the loaded parameters are bit-identical between backends; bf16
matmul/reduce/rsqrt accuracy matches XLA:CPU exactly; int8 `dot_general`
(s8 x s8 -> s32) is exact on metal for every shape tested, including the
batched ones qwix emits; float->int8 conversion clamps identically;
`lax.ragged_dot` is exact. Disabling `METALJAX_MSL` does *not* fix it (only
the garbage changes), so it is the `mx.compile` fused-graph path, not the MSL
kernels. It reproduces at `MAXTEXT_PREFILL_LEN=8 --decode-tokens 2`, so a
small repro is within reach.

With `METALJAX_COMPILE=0` metal is **token-identical to jax-CPU** on the
manifest prompt (32 tokens), for both the plain and the qwix-int8 row.

Because `maxtext-qwix-int8` also runs with bf16 weights, its metal output is
garbage for the same reason -- the int8 arithmetic itself is fine.

## Status of each row

| row | jax-CPU | metaljax |
|---|---|---|
| `qwen3-06b-maxtext` | coherent | coherent with `METALJAX_COMPILE=0`, garbage by default (bug above) |
| `maxtext-qwix-int8` | coherent, real s8xs8->s32 dots | same; ~16x slower prefill than bf16 (no OOM) |
| `maxtext-train-06b` | runs, loss decreases | runs, loss decreases; see the open question below |
| `deepseek-v2-lite` | **not completed** -- see below | not attempted |

### Open question on the train row (one sample each, not reproduced)

`MAXTEXT_TRAIN_STEPS=1 MAXTEXT_TRAIN_SEQ=256`, first-step loss:

| run | loss[0] | loss[1] |
|---|---|---|
| jax-CPU | 228.4169 | 135.9191 |
| metal, default (compiled) | 228.3945 | 155.3222 |
| metal, `METALJAX_COMPILE=0` | 191.2499 | 146.9232 |

The compiled metal run matches CPU on the first loss to 4 digits, the
uncompiled one does not -- the opposite of the decode result. That is one
sample per configuration; it could equally be the synthetic data iterator or
the train-state init differing per run. Needs a repeat run per configuration
(same config twice) before drawing any conclusion.

### deepseek-v2-lite: converted, configured, but memory-blocked here

The checkpoint converts cleanly (23 GB, ~2 min) and the config is worked out
(`mla_naive_kvcache=false`, `sparse_matmul=true megablox=false`), but a decode
run exceeds the 40 GB per-process budget this machine is under:

* `sparse_matmul=false` (dense MoE, all 64 experts per token): >105 GB RSS on
  jax-CPU during a 32-token prefill, killed.
* `sparse_matmul=true` (`lax.ragged_dot`): 50 GB after `load_params`, 83 GB
  during prefill lowering, killed by the 40 GB watchdog.

The 50 GB floor is the Orbax restore: 23 GB of bf16 weights plus a same-size
device copy plus tensorstore chunk buffers. The restored params are genuinely
bf16 (verified on qwen3-0.6b), so this is not a silent f32 upcast.

Leaving the sustained run to whoever owns the machine lock. Recommended first
attempt (with the exclusive lock held and nothing else running):

```sh
JAX_PLATFORMS=cpu MAXTEXT_PREFILL_LEN=64 $VENV/bin/python \
  scripts/model_bench/adapter_maxtext.py deepseek-v2-lite --decode-tokens 8
```
