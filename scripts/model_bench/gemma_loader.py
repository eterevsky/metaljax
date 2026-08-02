"""HF safetensors -> DeepMind gemma-library params, for Gemma 4.

Extracted from the 0.11.0 release validation work. Supports the two
dense instruct models; dims come from the table below (the 12B is the
`gemma4_unified` text tower — same schema, different sizes; its class
is not in the gemma lib and is synthesized here).
"""

import glob
import json
import struct
import urllib.request
from pathlib import Path

import ml_dtypes
import numpy as np

# model -> (embed_dim, n_q_heads, n_layers, local_kv_heads, global_kv_heads)
DIMS = {
    "google/gemma-4-31B-it": (5376, 32, 60, 16, 4),
    "google/gemma-4-12B-it": (3840, 16, 48, 8, 1),
}

TOK_URL = ("https://storage.googleapis.com/gemma-data/tokenizers/"
           "tokenizer_gemma4.model")
TOK_PATH = Path.home() / ".cache" / "metaljax-bench" / "tokenizer_gemma4.model"


def _snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo)


def _headers(snap):
    headers, mms, bases = {}, {}, {}
    for fn in sorted(glob.glob(snap + "/model*.safetensors")):
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                headers[k] = (fn, v)
        mms[fn] = np.memmap(fn, dtype=np.uint8, mode="r")
        bases[fn] = 8 + n
    return headers, mms, bases


def load_gemma4(repo):
    """Returns (model, params, tokenizer) with params on the default
    jax device (bf16, bitcast transfer)."""
    import jax
    from gemma import gm
    from gemma.gm.nn.gemma4 import _config, _gemma4, _modules, _transformer

    D, NQ, NL, KVL, KVG = DIMS[repo]
    if repo.endswith("31B-it"):
        model = gm.nn.Gemma4_31B()
    else:
        pattern = (_modules.AttentionType.LOCAL_SLIDING,) * 5 + (
            _modules.AttentionType.GLOBAL,)
        cfg = _gemma4._gemma4_config(
            embed_dim=D, num_heads=NQ, num_kv_heads=KVL,
            num_global_kv_heads=KVG, per_layer_input_dim=0,
            frac_shared_layers=0.0,
            attention_types=_config.make_attention_layers_types(
                pattern, num_layers=NL),
            sliding_window_size=1024, k_eq_v_global=True,
            use_post_attn_norm=True, use_post_ffw_norm=True,
            override_ffw_hidden_for_kv_cache_sharing=False,
            vision_encoder=None, use_bidirectional_attention='vision')

        class _M(_gemma4._Gemma4Base):
            config: _config.TransformerConfig = cfg
            INFO = _transformer.ModelInfo(tokenizer_version=4)

        model = _M()

    headers, mms, bases = _headers(_snapshot(repo))

    def hf(name):
        fn, meta = headers[name]
        o0, o1 = meta["data_offsets"]
        return (mms[fn][bases[fn] + o0:bases[fn] + o1]
                .view(ml_dtypes.bfloat16).reshape(meta["shape"]))

    def put(a):
        return jax.device_put(np.ascontiguousarray(a))

    def setp(tree, path, val):
        keys = path.split("/")
        for k in keys[:-1]:
            tree = tree.setdefault(k, {})
        tree[keys[-1]] = val

    params = {}
    P = "model.language_model."
    setp(params, "embedder/input_embedding", put(hf(P + "embed_tokens.weight")))
    setp(params, "final_norm/scale", put(hf(P + "norm.weight")))
    for L in range(NL):
        is_global = (L + 1) % 6 == 0
        lp, g = f"{P}layers.{L}.", f"layer_{L}"
        H = 512 if is_global else 256
        q = hf(lp + "self_attn.q_proj.weight").reshape(NQ, H, D).transpose(0, 2, 1)
        o = hf(lp + "self_attn.o_proj.weight").reshape(D, NQ, H).transpose(1, 2, 0)
        setp(params, f"{g}/attn/q_einsum/w", put(q))
        setp(params, f"{g}/attn/attn_vec_einsum/w", put(o))
        if is_global:
            k = hf(lp + "self_attn.k_proj.weight").reshape(KVG, H, D).transpose(0, 2, 1)
            setp(params, f"{g}/attn/k_einsum/w", put(k))
        else:
            k = hf(lp + "self_attn.k_proj.weight").reshape(KVL, H, D).transpose(0, 2, 1)
            v = hf(lp + "self_attn.v_proj.weight").reshape(KVL, H, D).transpose(0, 2, 1)
            setp(params, f"{g}/attn/kv_einsum/w", put(np.stack([k, v])))
        setp(params, f"{g}/attn/query_norm/scale", put(hf(lp + "self_attn.q_norm.weight")))
        setp(params, f"{g}/attn/key_norm/scale", put(hf(lp + "self_attn.k_norm.weight")))
        setp(params, f"{g}/mlp/gating_einsum",
             put(np.stack([hf(lp + "mlp.gate_proj.weight"),
                           hf(lp + "mlp.up_proj.weight")])))
        setp(params, f"{g}/mlp/linear", put(hf(lp + "mlp.down_proj.weight").T))
        setp(params, f"{g}/pre_attention_norm/scale", put(hf(lp + "input_layernorm.weight")))
        setp(params, f"{g}/post_attention_norm/scale", put(hf(lp + "post_attention_layernorm.weight")))
        setp(params, f"{g}/pre_ffw_norm/scale", put(hf(lp + "pre_feedforward_layernorm.weight")))
        setp(params, f"{g}/post_ffw_norm/scale", put(hf(lp + "post_feedforward_layernorm.weight")))
        setp(params, f"{g}/skip_scale", put(hf(lp + "layer_scalar")))

    if not TOK_PATH.exists():
        TOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TOK_URL, TOK_PATH)
    tok = gm.text.Gemma4Tokenizer(path=str(TOK_PATH))
    return model, params, tok
