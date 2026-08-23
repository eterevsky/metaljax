"""Tiny end-to-end check of the packed 3-bit Qwen3-MoE loader.

Two models are built over the SAME random weights:

  * the patched one, holding the weights packed exactly as an mlx-community
    checkpoint ships them, dequantizing in the graph;
  * the stock keras-hub one, holding those weights DEQUANTIZED as ordinary
    floats, in keras' own layouts.

They must compute the same function.  That is what checks the parts a
numerical differential on a single dot cannot: the flipped einsums, the
`[out, in]` vs `[in, out]` layouts, the gate/up concat order, the reshape
after each attention projection, and the converter's key mapping.

Run it under the machine lock (it uses the GPU), with the bench venv.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GROUP, BITS = 64, 3


def quantize(rng, shape):
    """Random weights + their mlx-community packed form and exact dequant."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import ml_dtypes
    w = (rng.randn(*shape) * 0.05).astype(np.float32)
    wq, s, b = mx.quantize(mx.array(w), group_size=GROUP, bits=BITS)
    mx.eval(wq, s, b)
    # A real mlx-community checkpoint stores its maps in bf16, so round them
    # FIRST and dequantize from the rounded values.  Dequantizing from the f32
    # maps would make the reference a model the packed one is not trying to be,
    # and the ~0.5% gap that opens is bf16 rounding of the scales, not a bug.
    s16 = np.array(s).astype(ml_dtypes.bfloat16)
    b16 = np.array(b).astype(ml_dtypes.bfloat16)
    deq = mx.dequantize(wq, mx.array(s16.astype(np.float32)),
                        mx.array(b16.astype(np.float32)),
                        group_size=GROUP, bits=BITS)
    mx.eval(deq)
    return (np.array(wq), s16, b16, np.array(deq, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    os.environ.setdefault("KERAS_BACKEND", "jax")
    os.environ.setdefault("JAX_PLATFORMS", "metal,cpu")

    import jax
    import keras
    import qwen3_moe_mlx3bit as mod

    cfg = dict(vocabulary_size=128, num_layers=2, num_query_heads=4,
               num_key_value_heads=2, hidden_dim=128, intermediate_dim=256,
               moe_intermediate_dim=64, num_experts=4, head_dim=32, top_k=2,
               norm_top_k_prob=True, tie_word_embeddings=False,
               sliding_window_size=None, layer_norm_epsilon=1e-6,
               rope_max_wavelength=5000000, dtype=args.dtype)
    rng = np.random.RandomState(args.seed)
    V, H = cfg["vocabulary_size"], cfg["hidden_dim"]
    U, KV, HD = (cfg["num_query_heads"], cfg["num_key_value_heads"],
                 cfg["head_dim"])
    E, I = cfg["num_experts"], cfg["moe_intermediate_dim"]

    # ---- draw every weight once, in the checkpoint's [out, in] layout ----
    W = {}
    W["embed"] = quantize(rng, (V, H))
    W["head"] = quantize(rng, (V, H))
    for i in range(cfg["num_layers"]):
        W[f"{i}.q"] = quantize(rng, (U * HD, H))
        W[f"{i}.k"] = quantize(rng, (KV * HD, H))
        W[f"{i}.v"] = quantize(rng, (KV * HD, H))
        W[f"{i}.o"] = quantize(rng, (H, U * HD))
        W[f"{i}.router"] = quantize(rng, (E, H))
        W[f"{i}.gate_up"] = quantize(rng, (E, 2 * I, H))
        W[f"{i}.down"] = quantize(rng, (E, H, I))
    norms = {k: (rng.randn(n) * 0.1 + 1.0).astype(np.float32)
             for k, n in [("final", H)] +
             [(f"{i}.{w}", n) for i in range(cfg["num_layers"])
              for w, n in (("in", H), ("post", H), ("qn", HD), ("kn", HD))]}

    import keras_hub

    # ---- 1. the patched, packed model ------------------------------------
    assert mod.install(), "install() declined"
    packed = keras_hub.models.Qwen3MoeBackbone(**cfg)
    emb = packed.token_embedding
    for stem, key in (("embed", "embed"), ("head", "head")):
        c, s, b, _ = W[key]
        getattr(emb, f"{stem}_codes").assign(c)
        getattr(emb, f"{stem}_scales").assign(s)
        getattr(emb, f"{stem}_biases").assign(b)
    packed.get_layer("sequence_output_layernorm").scale.assign(norms["final"])
    for i, L in enumerate(packed.transformer_layers):
        a = L._self_attention_layer
        L._self_attention_layernorm.scale.assign(norms[f"{i}.in"])
        L._feedforward_layernorm.scale.assign(norms[f"{i}.post"])
        a._query_dense_layer_norm.scale.assign(norms[f"{i}.qn"])
        a._key_dense_layer_norm.scale.assign(norms[f"{i}.kn"])
        for nm, d in (("q", a._query_dense), ("k", a._key_dense),
                      ("v", a._value_dense), ("o", a._output_dense)):
            c, s, b, _ = W[f"{i}.{nm}"]
            d.codes.assign(c); d.scales.assign(s); d.biases.assign(b)
        r = L.mlp._sparse_feedforward_gate_dense
        c, s, b, _ = W[f"{i}.router"]
        r.codes.assign(c); r.scales.assign(s); r.biases.assign(b)
        ex = L.mlp.expert_bank
        c, s, b, _ = W[f"{i}.gate_up"]
        ex.gate_up_codes.assign(c); ex.gate_up_scales.assign(s)
        ex.gate_up_biases.assign(b)
        c, s, b, _ = W[f"{i}.down"]
        ex.down_codes.assign(c); ex.down_scales.assign(s)
        ex.down_biases.assign(b)

    # Run the packed model BEFORE uninstalling: keras resolves `call` on the
    # class at call time, so restoring the stock methods first would silently
    # run the stock bodies against packed variables.
    T = 6
    ids = rng.randint(0, V, size=(1, T)).astype("int32")
    inputs = {"token_ids": ids,
              "padding_mask": np.ones((1, T), dtype="int32")}
    results = {}
    for dev in ("metal", "cpu"):
        with jax.default_device(jax.devices(dev)[0]):
            results[("packed", dev)] = np.array(
                keras.ops.convert_to_numpy(packed(inputs)), dtype=np.float64)
    del packed
    mod.uninstall()

    # ---- 2. the stock model over the DEQUANTIZED weights ------------------
    stock = keras_hub.models.Qwen3MoeBackbone(**cfg)
    emb = stock.token_embedding
    emb.embeddings.assign(W["embed"][3])
    emb.reverse_embeddings.assign(W["head"][3].T)
    stock.get_layer("sequence_output_layernorm").scale.assign(norms["final"])
    for i, L in enumerate(stock.transformer_layers):
        a = L._self_attention_layer
        L._self_attention_layernorm.scale.assign(norms[f"{i}.in"])
        L._feedforward_layernorm.scale.assign(norms[f"{i}.post"])
        a._query_dense_layer_norm.scale.assign(norms[f"{i}.qn"])
        a._key_dense_layer_norm.scale.assign(norms[f"{i}.kn"])
        # keras stores [in, heads, head_dim]; the file stores [out, in].
        a._query_dense.kernel.assign(W[f"{i}.q"][3].T.reshape(H, U, HD))
        a._key_dense.kernel.assign(W[f"{i}.k"][3].T.reshape(H, KV, HD))
        a._value_dense.kernel.assign(W[f"{i}.v"][3].T.reshape(H, KV, HD))
        a._output_dense.kernel.assign(W[f"{i}.o"][3].T.reshape(U, HD, H))
        L.mlp._sparse_feedforward_gate_dense.kernel.assign(
            W[f"{i}.router"][3].T)
        ex = L.mlp.expert_bank
        # keras contracts the MIDDLE axis: [E, in, out].
        ex._expert_feedforward_gate_dense.assign(
            W[f"{i}.gate_up"][3].transpose(0, 2, 1))
        ex._expert_feedforward_output_dense.assign(
            W[f"{i}.down"][3].transpose(0, 2, 1))

    # ---- 3. compare ------------------------------------------------------
    for dev in ("metal", "cpu"):
        with jax.default_device(jax.devices(dev)[0]):
            results[("stock", dev)] = np.array(
                keras.ops.convert_to_numpy(stock(inputs)), dtype=np.float64)

    ref = results[("stock", "cpu")]
    scale = max(float(np.max(np.abs(ref))), 1e-3)
    fails = []
    for key, got in results.items():
        rel = float(np.max(np.abs(got - ref))) / scale
        # bf16 has ~3 decimal digits; two different orderings of a 2-layer
        # network over the same numbers land a few times its epsilon apart.
        tol = 0.02 if args.dtype == "bfloat16" else 2e-4
        ok = rel <= tol
        print(f"[{'PASS' if ok else 'FAIL'}] {key[0]:6s} on {key[1]:5s}: "
              f"max rel vs stock-cpu {rel:.4f} (tol {tol})", flush=True)
        if not ok:
            fails.append(key)

    print()
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("packed 3-bit Qwen3-MoE matches the stock dequantized model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
