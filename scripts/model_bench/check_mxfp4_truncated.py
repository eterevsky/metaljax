"""Truncated gpt-oss-20b, packed vs dequantized-at-load, on the same backend.

    bench-venv/bin/python scripts/model_bench/check_mxfp4_truncated.py \
        --mode packed --layers 2 --out packed.npz
    bench-venv/bin/python scripts/model_bench/check_mxfp4_truncated.py \
        --mode stock  --layers 2 --out stock.npz
    bench-venv/bin/python scripts/model_bench/check_mxfp4_truncated.py \
        --compare packed.npz stock.npz

Builds a GptOssBackbone with only the first N transformer layers (keras-hub's
converter ports `range(backbone.num_layers)`, so a short backbone reads a
short prefix of the checkpoint), ports the REAL weights either way, then
greedy-decodes a few tokens and records the hidden states, the logits and
the token ids.

Both modes run on the SAME backend, so the comparison isolates the change:
anything other than the loader and the expert layer is identical. One mode
per process -- the patch is global, and two full backbones would double the
footprint for no reason.

N=2 fits in a few GB (the 201k-token embedding dominates); the full 24-layer
row is the main agent's job, not this script's.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

PRESET = "hf://openai/gpt-oss-20b"
# A fixed token prefix: no tokenizer needed, and the comparison is numeric.
PROMPT_IDS = [2, 1428, 3575, 382, 261, 1746, 4166, 328, 290, 8865, 2201]


def device_mem_gb():
    try:
        import mlx.core as mx
        return mx.get_active_memory() / 1e9
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


def build(num_layers, mode):
    """An N-layer backbone with real weights, loaded `mode`-ly."""
    import keras
    keras.config.set_dtype_policy("bfloat16")

    if mode == "packed":
        import mxfp4_gpt_oss
        if not mxfp4_gpt_oss.install():
            raise SystemExit("METALJAX_BENCH_NATIVE_QUANT=0 blocks --mode "
                             "packed")

    from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
    from keras_hub.src.utils.transformers import convert_gpt_oss
    from keras_hub.src.utils.transformers.safetensor_utils import (
        SafetensorLoader)
    from keras_hub.src.utils.preset_utils import get_file

    hf_config = json.load(open(get_file(PRESET, "config.json")))
    config = convert_gpt_oss.convert_backbone_config(hf_config)
    config["num_layers"] = num_layers
    t0 = time.monotonic()
    backbone = GptOssBackbone(**config)
    built = device_mem_gb()
    with SafetensorLoader(PRESET) as loader:
        convert_gpt_oss.convert_weights(backbone, loader, hf_config)
    return backbone, dict(load_s=time.monotonic() - t0,
                          mem_after_build_gb=built,
                          mem_after_load_gb=device_mem_gb())


def decode(backbone, ids, n_new):
    """Greedy decode, one JITTED backbone pass per step.

    The jit is the point: metaljax's recognizer works on compiled programs,
    so an eager keras call (every op dispatched on its own) would never show
    it a dequant-and-dot to fuse. No KV cache -- the sequence is a dozen
    tokens and clarity beats speed here.
    """
    import jax
    from keras import ops

    trainable = backbone.trainable_variables
    non_trainable = backbone.non_trainable_variables

    @jax.jit
    def forward(tv, ntv, token_ids, padding_mask):
        hidden, _ = backbone.stateless_call(
            tv, ntv, {"token_ids": token_ids, "padding_mask": padding_mask})
        return hidden

    tv = [v.value for v in trainable]
    ntv = [v.value for v in non_trainable]
    ids = list(ids)
    hidden = logits = None
    for _ in range(n_new):
        token_ids = np.array([ids], np.int32)
        hidden = forward(tv, ntv, ops.convert_to_tensor(token_ids),
                         ops.convert_to_tensor(np.ones_like(token_ids)))
        logits = backbone.token_embedding(hidden[:, -1, :], reverse=True)
        ids.append(int(np.asarray(ops.argmax(logits, axis=-1))[0]))
    # Sampled while `forward` is still referenced: dropping it would release
    # the executable, and with it the recognizer's packed weights -- which
    # in a real decode loop stay resident for the whole generation. That
    # retained pack is the honest steady-state cost of the rewrite.
    live_gb = device_mem_gb()
    return (np.asarray(hidden, np.float32), np.asarray(logits, np.float32),
            ids, live_gb)


def generate_path(backbone, n_new):
    """keras-hub's OWN generate: cached prefill + a jitted decode loop.

    This is the structure the real row runs, so it is what decides how many
    times each weight gets packed (the prefill and the decode loop are
    separate dots over the same weights) and what a decode step costs.
    """
    import numpy as np
    from keras_hub.src.models.gpt_oss.gpt_oss_causal_lm import GptOssCausalLM

    lm = GptOssCausalLM(backbone=backbone, preprocessor=None)
    length = len(PROMPT_IDS) + n_new
    token_ids = np.zeros((1, length), np.int32)
    token_ids[0, :len(PROMPT_IDS)] = PROMPT_IDS
    padding_mask = np.zeros((1, length), bool)
    padding_mask[0, :len(PROMPT_IDS)] = True
    inputs = {"token_ids": token_ids, "padding_mask": padding_mask}

    t0 = time.monotonic()
    out = lm.generate(inputs, max_length=length, stop_token_ids=None)
    warm_s = time.monotonic() - t0
    ids = np.asarray(out["token_ids"])[0]
    live_gb = device_mem_gb()

    t0 = time.monotonic()
    out = lm.generate(inputs, max_length=length, stop_token_ids=None)
    hot_s = time.monotonic() - t0
    return ids, dict(generate_warm_s=warm_s, generate_hot_s=hot_s,
                     ms_per_token=1000 * hot_s / n_new,
                     mem_live_gb=live_gb)


def qmm_stats():
    try:
        from metaljax import qmm
        return qmm.stats()
    except Exception:
        return {}


def run(ns):
    backbone, info = build(ns.layers, ns.mode)
    if ns.generate:
        ids, gen = generate_path(backbone, ns.decode_tokens)
        info.update(gen)
        info["qmm"] = qmm_stats()
        print(f"mode={ns.mode} layers={ns.layers} (keras generate)")
        for k, v in info.items():
            print(f"  {k}: {v}")
        np.savez(ns.out, hidden=np.zeros(1), logits=np.zeros(1),
                 ids=np.asarray(ids, np.int64),
                 info=np.array(json.dumps(info, default=str)))
        print(f"  -> {ns.out}")
        return 0
    t0 = time.monotonic()
    hidden, logits, ids, live_gb = decode(backbone, PROMPT_IDS,
                                          ns.decode_tokens)
    info["decode_s"] = time.monotonic() - t0
    info["mem_live_gb"] = live_gb        # weights + retained packs
    info["mem_after_release_gb"] = device_mem_gb()
    info["qmm"] = qmm_stats()
    print(f"mode={ns.mode} layers={ns.layers}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  tokens: {ids[len(PROMPT_IDS):]}")
    np.savez(ns.out, hidden=hidden, logits=logits,
             ids=np.array(ids, np.int64),
             info=np.array(json.dumps(info, default=str)))
    print(f"  -> {ns.out}")
    return 0


def compare(a_path, b_path):
    a, b = np.load(a_path, allow_pickle=True), np.load(b_path,
                                                       allow_pickle=True)
    print(f"A {a_path}: {json.loads(str(a['info']))}")
    print(f"B {b_path}: {json.loads(str(b['info']))}")
    ok = True
    for name in ("hidden", "logits"):
        x, y = a[name], b[name]
        scale = max(np.abs(y).max(), 1e-30)
        amax = np.abs(x - y).max()
        equal = np.array_equal(x, y)
        print(f"  {name:7s} shape={x.shape} identical={equal} "
              f"max|diff|={amax:.6g} rel={amax / scale:.3e}")
        ok = ok and amax / scale < 2e-2
    same = np.array_equal(a["ids"], b["ids"])
    print(f"  tokens identical={same}\n  A {list(a['ids'])}\n  B "
          f"{list(b['ids'])}")
    ok = ok and same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["packed", "stock"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=4)
    ap.add_argument("--out", default="mxfp4_truncated.npz")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--generate", action="store_true",
                    help="use keras-hub's own cached generate loop")
    ns = ap.parse_args()
    if ns.compare:
        return compare(*ns.compare)
    if not ns.mode:
        ap.error("--mode or --compare is required")
    os.environ.setdefault("KERAS_BACKEND", "jax")
    return run(ns)


if __name__ == "__main__":
    raise SystemExit(main())
