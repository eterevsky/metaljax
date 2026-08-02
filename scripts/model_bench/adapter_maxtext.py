"""MaxText adapter for the public model-benchmark suite.

Exposes the two entry points `run_bench.py` dispatches to:

    run_maxtext(bench, backend, prompt, n_decode)   -> decode metrics
    run_maxtext_train(bench, backend)               -> train-step metrics

Both run AI-Hypercomputer/MaxText (the real thing: `maxtext.inference.maxengine`
for decode, `maxtext.trainers.train` for the training step) on whatever backend
`JAX_PLATFORMS` selects, so the metaljax and jax-CPU rows execute identical
model code.

Prerequisites (see README_maxtext.md / setup_maxtext.sh):
  * MaxText checked out at $MAXTEXT_REPO (default ~/.cache/metaljax-bench/maxtext/repo)
    and pip-installed with --no-deps into the same venv as metaljax.
  * MaxText-format Orbax checkpoints under $MAXTEXT_CKPT_ROOT
    (default ~/.cache/metaljax-bench/maxtext/ckpt), produced by
    `maxtext.checkpoint_conversion.to_maxtext` -- convert_checkpoints.sh does this.

Notes that matter for correctness on this backend:
  * attention=dot_product is mandatory. MaxText's `autoselected` branch only
    falls back to the reference dot-product attention when
    target_hardware == "cpu"; our platform string is "metal", so autoselect
    would route prefill (length >= 128) into a Pallas/TPU-splash kernel.
  * jax.block_until_ready() is a no-op on metaljax (PJRT events are born
    ready), so every timed section is closed by pulling a value to the host
    with int(...) -- that is a real device sync.
  * DECOUPLE_GCLOUD=TRUE stubs out JetStream/GCS/Vertex. The JetStream
    ResultTokens stub has no get_result_at_slot(), so tokens are read straight
    out of ResultTokens.data (col 0 = token, 1 = valid, 2 = length).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Must be set before maxtext is imported: the module-level gcloud_stub import
# decides tensorboardX/JetStream/GCS stubbing at import time.
os.environ.setdefault("DECOUPLE_GCLOUD", "TRUE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

BENCH_HOME = Path(os.environ.get("METALJAX_BENCH_HOME",
                                 Path.home() / ".cache" / "metaljax-bench"))
MAXTEXT_REPO = Path(os.environ.get("MAXTEXT_REPO", BENCH_HOME / "maxtext" / "repo"))
CKPT_ROOT = Path(os.environ.get("MAXTEXT_CKPT_ROOT", BENCH_HOME / "maxtext" / "ckpt"))
BASE_YML = str(MAXTEXT_REPO / "src" / "maxtext" / "configs" / "base.yml")


# ------------------------------------------------------------------ registry

# id -> (maxtext model_name, checkpoint subdir, extra config overrides)
SPECS = {
    "qwen3-06b-maxtext": dict(
        model_name="qwen3-0.6b",
        ckpt="qwen3-0.6b",
        overrides=[],
    ),
    "maxtext-qwix-int8": dict(
        model_name="qwen3-0.6b",
        ckpt="qwen3-0.6b",
        # qwix intercepts dot_general in the nnx model graph: s8 x s8 -> s32.
        overrides=["use_qwix_quantization=true", "quantization=int8"],
    ),
    "deepseek-v2-lite": dict(
        model_name="deepseek2-16b",
        ckpt="deepseek2-16b",
        # MoE: megablox is a Pallas TPU kernel, so megablox=false. That leaves
        # sparse_matmul=true -> jax.lax.ragged_dot (verified exact on metal,
        # f32 and bf16, even and uneven groups) vs sparse_matmul=false ->
        # dense_matmul, which evaluates all 64 experts for every token and
        # needs >105 GB of host RAM even for a 32-token prefill on jax-CPU.
        # capacity_factor=-1 keeps "no token dropping".
        # mla_naive_kvcache=false -> cache the compressed MLA latent. The naive
        # (materialized k/v) cache is prefill/AR-inconsistent for MLA: insert()
        # tries to write a [.., 192] prefill key into a [.., 128] AR cache
        # (qk_nope 128 + qk_rope 64 vs v_head_dim 128).
        overrides=["sparse_matmul=true", "megablox=false", "capacity_factor=-1.0",
                   "mla_naive_kvcache=false"],
    ),
    "maxtext-train-06b": dict(
        model_name="qwen3-0.6b",
        ckpt="qwen3-0.6b",
        overrides=[],
    ),
}

# manifest "model" (HF id) -> local tokenizer source. Falls back to the HF id,
# which resolves out of ~/.cache/huggingface when present.
TOKENIZERS = {
    "Qwen/Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-V2-Lite-Chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
}


def _spec(bench):
    if bench["id"] in SPECS:
        return SPECS[bench["id"]]
    raise KeyError(f"no MaxText spec for benchmark id {bench['id']!r}")


def _ensure_importable():
    if MAXTEXT_REPO.exists():
        src = str(MAXTEXT_REPO / "src")
        if src not in sys.path:
            sys.path.append(src)
    _patch_jetstream_stub()


_PATCHED = []


def _patch_jetstream_stub():
    """Make the decoupled-mode ResultTokens stub a JAX pytree.

    MaxEngine's prefill/generate are jax.jit'd and *return* a
    `engine_api.ResultTokens`. The real JetStream class is a registered pytree;
    the DECOUPLE_GCLOUD stub in maxtext/common/gcloud_stub.py is a plain object,
    so jit rejects it ("not a valid JAX type"). JetStream itself is not
    installable here (its token_utils imports seqio -> tensorflow, which has no
    macOS/arm64 wheel for tensorflow_text), so we register the stub instead.

    Must run before maxengine/decode/maxengine_config are imported: those do
    `... = jetstream()` at module scope.
    """
    if _PATCHED:
        return
    import jax
    from maxtext.common import gcloud_stub

    ns = gcloud_stub.jetstream()
    cfg, engine_api, tok_utils, tok_api, tok_params = ns
    if not getattr(engine_api, "_IS_STUB", False):
        _PATCHED.append(ns)  # real JetStream is installed; nothing to do
        return

    RT = engine_api.ResultTokens

    class _SlotData:
        __slots__ = ("tokens", "valid", "lengths")

        def __init__(self, tokens, valid, lengths):
            self.tokens, self.valid, self.lengths = tokens, valid, lengths

    def get_result_at_slot(self, slot):
        return _SlotData(
            tokens=self.data[slot, self.tokens_idx[0]: self.tokens_idx[1]],
            valid=self.data[slot, self.valid_idx[0]: self.valid_idx[1]],
            lengths=self.data[slot, self.length_idx[0]: self.length_idx[1]],
        )

    RT.get_result_at_slot = get_result_at_slot

    jax.tree_util.register_pytree_node(
        RT,
        lambda r: ((r.data, r.log_prob),
                   (r.tokens_idx, r.valid_idx, r.length_idx, r.samples_per_slot)),
        lambda aux, ch: RT(data=ch[0], log_prob=ch[1], tokens_idx=aux[0],
                           valid_idx=aux[1], length_idx=aux[2],
                           samples_per_slot=aux[3]),
    )
    gcloud_stub.jetstream = lambda: ns
    _PATCHED.append(ns)


def _tokenizer(bench):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZERS.get(bench["model"], bench["model"]))


def _quiet():
    """pyconfig logs ~900 'Config param ...' lines per run; absl logs orbax
    internals. Keep the bench output readable (MAXTEXT_VERBOSE=1 to restore)."""
    if os.environ.get("MAXTEXT_VERBOSE") == "1":
        return
    import logging
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.WARNING)
    for name in ("maxtext.configs.pyconfig", "maxtext.configs.types", "absl"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _config(bench, overrides):
    from maxtext.configs import pyconfig
    _quiet()
    spec = _spec(bench)
    argv = [
        "adapter_maxtext",
        BASE_YML,
        f"model_name={spec['model_name']}",
        "attention=dot_product",          # see module docstring
        "skip_jax_distributed_system=True",
        "dataset_type=synthetic",
        "scan_layers=true",
        "hardware=cpu",                   # only gates gpu-specific code paths
        *spec["overrides"],
        *overrides,
        # escape hatch for debugging: MAXTEXT_OVERRIDES="key=val key2=val2"
        *os.environ.get("MAXTEXT_OVERRIDES", "").split(),
    ]
    return pyconfig.initialize(argv)


def _sync(x):
    """Force completion of a device computation (block_until_ready is a no-op here)."""
    import numpy as np
    return np.asarray(x)


# ------------------------------------------------------------------- decode

def run_maxtext(bench, backend, prompt, n_decode):
    """MaxText MaxEngine decode: prefill + n_decode-1 autoregressive steps."""
    _ensure_importable()
    import jax
    import numpy as np

    spec = _spec(bench)
    ckpt = CKPT_ROOT / spec["ckpt"] / "0" / "items"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"MaxText checkpoint missing: {ckpt}. Run convert_checkpoints.sh first.")

    tok = _tokenizer(bench)
    prompt_ids = tok.encode(prompt)
    prefill_len = int(os.environ.get("MAXTEXT_PREFILL_LEN", 0)) or \
        1 << max(6, (len(prompt_ids) + 1).bit_length())  # next pow2, >= 64
    prompt_ids = prompt_ids[: prefill_len - 1]
    target_len = prefill_len + n_decode + 8

    t0 = time.monotonic()
    config = _config(bench, [
        f"load_parameters_path={ckpt}",
        "per_device_batch_size=1",
        # Weights stay in the checkpoint's bf16 (weight_dtype defaults to f32,
        # which would double every row's footprint: 31 GB -> 63 GB for
        # deepseek-v2-lite) -- the rest of the suite benchmarks bf16 too.
        "weight_dtype=bfloat16",
        f"max_prefill_predict_length={prefill_len}",
        f"max_target_length={target_len}",
    ])
    from maxtext.inference.maxengine import maxengine

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load = jax.random.split(rng)
    params = engine.load_params(rng_load)
    load_s = time.monotonic() - t0

    padded = np.zeros(prefill_len, dtype=np.int32)
    padded[: len(prompt_ids)] = prompt_ids
    true_length = len(prompt_ids)

    def _prefill(rng_key):
        return engine.prefill(params=params, padded_tokens=padded,
                              true_length=true_length, rng=rng_key, slot=0)

    def _first_token(result_tokens):
        # ResultTokens.data columns: [token, valid, length]
        return int(_sync(result_tokens.data)[0, 0])

    # -- warmup: compiles prefill, init_decode_state, insert and generate ----
    t0 = time.monotonic()
    rng, sub = jax.random.split(rng)
    prefill_result, first_token = _prefill(sub)
    _first_token(first_token)
    rng, sub = jax.random.split(rng)
    state = engine.init_decode_state(sub)
    state = engine.insert(prefill_result, state, slot=0)
    for _ in range(8):
        rng, sub = jax.random.split(rng)
        state, sampled = engine.generate(params, state, rng=sub)
    _first_token(sampled)
    warmup_s = time.monotonic() - t0
    del state, prefill_result

    # -- warm prefill -------------------------------------------------------
    t0 = time.monotonic()
    rng, sub = jax.random.split(rng)
    prefill_result, first_token = _prefill(sub)
    tid = _first_token(first_token)
    prefill_ms = 1000 * (time.monotonic() - t0)

    # -- decode -------------------------------------------------------------
    ids = [tid]
    rng, sub = jax.random.split(rng)
    state = engine.init_decode_state(sub)
    state = engine.insert(prefill_result, state, slot=0)
    t0 = time.monotonic()
    for _ in range(n_decode - 1):
        rng, sub = jax.random.split(rng)
        state, sampled = engine.generate(params, state, rng=sub)
        ids.append(_first_token(sampled))
    decode_s = time.monotonic() - t0
    decode_ms = 1000 * decode_s / max(len(ids) - 1, 1)

    text = tok.decode(ids)
    print(f"[maxtext] {bench['id']} / {backend} -> {text!r}", flush=True)
    return dict(load_s=load_s, warmup_s=warmup_s, prefill_ms=prefill_ms,
                decode_ms_tok=decode_ms, out_tokens=len(ids),
                token_ids=[int(t) for t in ids[:64]],
                prompt_tokens=true_length, text=text)


# -------------------------------------------------------------------- train

def run_maxtext_train(bench, backend, *_unused, steps=None):
    """MaxText training steps on synthetic data (AdamW, full fwd+bwd+update).

    `*_unused` swallows the (prompt, n_decode) that run_bench.py passes to every
    adapter, so this can be registered in ADAPTERS directly.

    This is MaxText's real `train_step` (maxtext.trainers.pre_train.train),
    wired up exactly as `train_loop` does -- only the metric logger, profiler
    and checkpointer are left out so the timing is the step itself.

    Metrics: warmup_s (first step -- traces + compiles the whole train step),
    step_ms (mean of the following `steps` steps).
    """
    _ensure_importable()
    _stub_tensorflow()
    import jax
    import numpy as np
    from flax import linen as nn, nnx
    from flax.linen import partitioning as nn_partitioning

    steps = int(os.environ.get("MAXTEXT_TRAIN_STEPS", steps or 3))
    seq = int(os.environ.get("MAXTEXT_TRAIN_SEQ", 1024))

    t0 = time.monotonic()
    config = _config(bench, [
        "per_device_batch_size=1",
        f"max_target_length={seq}",
        f"steps={steps + 2}",
        "run_name=metaljax_bench",
        f"base_output_directory={BENCH_HOME / 'maxtext' / 'train_out'}",
        "enable_checkpointing=false",
        "enable_goodput_recording=false",
        "monitor_goodput=false",
        "gcs_metrics=false",
        "enable_tensorboard=false",
        "enable_data_shuffling=false",
    ])

    from maxtext.trainers.pre_train import train as train_lib
    from maxtext.utils import maxtext_utils, sharding as sharding_utils, train_utils

    (init_rng, _ckpt_mgr, state_mesh_shardings, model, mesh,
     _lr_schedule, data_iterator, data_loader, rampup_manager,
     _eval_iter, state) = train_utils.setup_train_loop(config, None)

    if isinstance(model, nn.Module):
        jit_model = model
    else:
        jit_model, state = nnx.split(state)

    params_shardings, state_mesh_shardings = \
        sharding_utils.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

    p_train_step, _ = train_utils.jit_train_and_eval_step(
        config, jit_model, mesh, state, state_mesh_shardings,
        train_lib.train_step, train_lib.eval_step, None, params_shardings)
    load_s = time.monotonic() - t0

    linen = isinstance(model, nn.Module)

    def _step(state, step):
        batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        rng_args = (jax.random.fold_in(init_rng, step),) if linen else ()
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            return p_train_step(state, batch, *rng_args)

    with mesh:
        t0 = time.monotonic()
        state, metrics = _step(state, 0)
        loss0 = float(np.asarray(metrics["scalar"]["learning/loss"]))
        warmup_s = time.monotonic() - t0

        t0 = time.monotonic()
        for i in range(steps):
            state, metrics = _step(state, i + 1)
        loss = float(np.asarray(metrics["scalar"]["learning/loss"]))
        step_ms = 1000 * (time.monotonic() - t0) / steps

    print(f"[maxtext-train] {bench['id']} / {backend} step_ms={step_ms:.1f} "
          f"loss {loss0:.4f} -> {loss:.4f}", flush=True)
    return dict(load_s=load_s, warmup_s=warmup_s, step_ms=step_ms,
                steps=steps, loss=loss, loss_first=loss0,
                seq_len=seq, batch=1)


def _stub_tensorflow():
    """Stand in for TensorFlow, which cannot be co-installed here.

    `maxtext.trainers.pre_train.train` does `import tensorflow as tf` at module
    scope purely for `tf.config.set_visible_devices([], "GPU")` inside its own
    `initialize()` (which we never call), and multihost_dataloading imports it
    under try/except for tf.data datasets.

    TensorFlow (2.21.0, macOS/arm64) and `grain` SEGFAULT when both are
    imported into one process, in either order -- both ship their own
    protobuf/absl C++ symbols:

        python -c "import grain; import tensorflow"           -> SIGSEGV
        python -c "import tensorflow; import grain"           -> SIGSEGV

    grain is not optional (maxtext.common.checkpointing imports it), so
    TensorFlow stays out of the venv and this stub covers the one attribute
    the train module touches. Set MAXTEXT_REAL_TF=1 to opt out.
    """
    if os.environ.get("MAXTEXT_REAL_TF") == "1" or "tensorflow" in sys.modules:
        return
    import types
    tf = types.ModuleType("tensorflow")
    tf.config = types.SimpleNamespace(
        set_visible_devices=lambda *a, **k: None,
        list_physical_devices=lambda *a, **k: [],
    )
    tf.errors = types.SimpleNamespace(FailedPreconditionError=RuntimeError)
    tf.data = types.SimpleNamespace(Dataset=type("Dataset", (), {}))
    sys.modules["tensorflow"] = tf


# ---------------------------------------------------------------------- cli

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="smoke-test the MaxText adapter")
    ap.add_argument("bench_id")
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--train", action="store_true")
    ns = ap.parse_args()

    b = {"id": ns.bench_id,
         "model": {"deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite-Chat"}.get(
             ns.bench_id, "Qwen/Qwen3-0.6B")}
    backend = "metaljax" if "metal" in os.environ.get("JAX_PLATFORMS", "") else "cpu"
    out = (run_maxtext_train(b, backend) if ns.train
           else run_maxtext(b, backend, ns.prompt, ns.decode_tokens))
    print("RESULT " + json.dumps({k: v for k, v in out.items() if k != "token_ids"}))
