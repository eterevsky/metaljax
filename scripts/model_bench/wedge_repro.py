"""Size-dialed repro of the streamed keras MoE load (kernel panic #7).

WHY.  Row 8 (`qwen36-35b-a3b`, arch `Qwen3_5MoeCausalLM`, 71.9 GB) hard-
wedged the machine 46 s into its streamed checkpoint load -- watchdogd
starvation, memory explicitly HEALTHY (TASKS.md, "PAUSED -- machine-wedge
class").  Per Oleg, no big model is retried until the SAME code path runs
green at a small size with a predicted peak.  This script is the size dial:
it writes a **synthetic HF-safetensors checkpoint** with the real
architecture's tensor NAMES, LAYOUT and COUNT but dimensions divided by
`div`, and loads it through the identical keras-hub path row 8 uses
(`Qwen3_5MoeCausalLM.from_preset` + `adapter_keras_extra`'s streaming
shim).  Nothing is downloaded: shapes come from a config we synthesise and
the tokenizer/preprocessor JSONs are copied from the checkpoint already in
the local HF cache.

RUNGS (`--rung`), all 40 text layers / 27 vision blocks / 256 experts, so
the tensor COUNT stays ~1045 like the real row and only the byte volume
moves:

    tiny   div=32   ~0.15 GB   shape-formula self-test (runs on CPU)
    small  div=4    ~4.7 GB    the rung this script was written for
    mid    div=2    ~17 GB     next rung -- human-approved step only
    full   div=1    ~66 GB     the real size; refuses to build

USAGE (from the repo root, bench venv, one run at a time, always guarded):

    B=~/.cache/metaljax-bench/venvs/bench/bin/python
    $B scripts/model_bench/wedge_repro.py check          # formulas vs real
    $B scripts/model_bench/wedge_repro.py build --rung small
    $B scripts/model_bench/wedge_repro.py profile --rung small
    $B scripts/model_bench/wedge_repro.py predict --rung small

    # the actual run: guard + machine lock (see mem_guard.sh)
    until mkdir /tmp/metaljax-bench.lock 2>/dev/null; do sleep 20; done
    BENCH_STREAM_MARK=100 bash scripts/model_bench/mem_guard.sh 12 \
        ~/.cache/metaljax-bench/logs/wedge-ladder/small-1-flight.log \
        $B scripts/model_bench/wedge_repro.py load --rung small
    rmdir /tmp/metaljax-bench.lock

`load` prints its PREDICTED peak before touching the checkpoint and its
measured peak after, so predicted-vs-actual is in the log itself.  Add
`--generate N` to run the whole row (`run_bench.run_keras_lm`, i.e. load +
compile + decode) instead of load-only; the wedge is a load-phase event, so
load-only is the default.

Levers worth sweeping at a small rung (all read by the shim):
    BENCH_STREAM_MARK=N       progress line every N assigns
    BENCH_STREAM_SYNC=N       mx.synchronize() every N assigns
    BENCH_STREAM_CLEAR_GB=X   gc+mx.clear_cache() every X GB (default 8)

DETERMINISM.  Every tensor's bytes are a function of its name only (a
seeded pool, rotated by a hash of the name), so a rebuilt checkpoint is
byte-identical and reruns are comparable.  Weights are random: this
measures the LOAD, not the model's outputs.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = Path(os.path.expanduser("~/.cache/metaljax-bench"))
DEFAULT_ROOT = CACHE / "synthetic"
# The real row-8 checkpoint: source of the tokenizer/preprocessor JSONs and
# of the `check` reference.  Already in the HF cache (nothing downloads).
REAL_REPO = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B"))

ARCH = "Qwen3_5MoeCausalLM"
SHARDS = 26                    # same shard count as the real checkpoint

RUNGS = {
    "tiny": dict(div=32, layers=40, vision_depth=27),
    "small": dict(div=4, layers=40, vision_depth=27),
    "mid": dict(div=2, layers=40, vision_depth=27),
    # `wide` trades layer count for FULL-SIZE tensors: 9 layers at div=1, so
    # every expert bank is the real 1.0 GB / 0.5 GB single assign the 35B row
    # makes, at a total that still fits a modest budget.  The div rungs scale
    # the tensor COUNT profile; this one scales the per-assign SIZE profile.
    "wide": dict(div=1, layers=9, vision_depth=27),
    "full": dict(div=1, layers=40, vision_depth=27),
}

# The real Qwen3.6-35B-A3B text/vision config, as the div=1 base.  Only the
# starred entries are divided by `div`; counts (layers, heads, experts,
# vocabulary, patch sizes) are held constant so the tensor-count profile is
# preserved at every rung.
BASE = dict(
    vocab_size=248320,
    hidden_size=2048,           # *
    head_dim=256,               # *
    num_attention_heads=16,
    num_key_value_heads=2,
    moe_intermediate_size=512,          # *
    shared_expert_intermediate_size=512,  # *
    num_experts=256,
    num_experts_per_tok=8,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,    # *
    linear_value_head_dim=128,  # *
    linear_conv_kernel_dim=4,
    partial_rotary_factor=0.25,
    rms_norm_eps=1e-06,
    rope_theta=10000000,
    full_attention_interval=4,
    attn_output_gate=True,
    vision_hidden_size=1152,    # *
    vision_intermediate_size=4304,  # *
    vision_num_heads=16,        # * (head dim 72 held constant)
    vision_patch_size=16,
    vision_temporal_patch_size=2,
    vision_in_channels=3,
    vision_spatial_merge_size=2,
    vision_num_position_embeddings=2304,
)


# ------------------------------------------------------------------- config

def _round8(x):
    return max(8, int(round(x / 8)) * 8)


def make_config(div=1, layers=40, vision_depth=27):
    """A transformers-style config.json for the scaled architecture."""
    b = BASE
    hidden = _round8(b["hidden_size"] / div)
    head_dim = max(32, _round8(b["head_dim"] / div))   # >=32: see mrope below
    vis_hidden_head = b["vision_hidden_size"] // b["vision_num_heads"]  # 72
    vis_heads = max(1, int(round(b["vision_num_heads"] / div)))
    vis_hidden = vis_hidden_head * vis_heads
    # M-RoPE splits half the rotary dims three ways, biggest first (real:
    # 64 rotary dims -> [11, 11, 10]).  Reproduces the real split exactly at
    # div=1 and stays positive down to head_dim 32 (-> [2, 1, 1]).
    half_rot = int(head_dim * b["partial_rotary_factor"]) // 2
    base, rem = divmod(half_rot, 3)
    mrope = [base + (1 if rem > 0 else 0), base + (1 if rem > 1 else 0), base]
    layer_types = [("linear_attention"
                    if (i + 1) % b["full_attention_interval"]
                    else "full_attention") for i in range(layers)]
    text = {
        "attn_output_gate": b["attn_output_gate"],
        "dtype": "bfloat16",
        "full_attention_interval": b["full_attention_interval"],
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "layer_types": layer_types,
        "linear_conv_kernel_dim": b["linear_conv_kernel_dim"],
        "linear_key_head_dim": _round8(b["linear_key_head_dim"] / div),
        "linear_num_key_heads": b["linear_num_key_heads"],
        "linear_num_value_heads": b["linear_num_value_heads"],
        "linear_value_head_dim": _round8(b["linear_value_head_dim"] / div),
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_moe_text",
        "moe_intermediate_size": _round8(b["moe_intermediate_size"] / div),
        "num_attention_heads": b["num_attention_heads"],
        "num_experts": b["num_experts"],
        "num_experts_per_tok": b["num_experts_per_tok"],
        "num_hidden_layers": layers,
        "num_key_value_heads": b["num_key_value_heads"],
        "rms_norm_eps": b["rms_norm_eps"],
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": mrope,
            "partial_rotary_factor": b["partial_rotary_factor"],
            "rope_theta": b["rope_theta"],
            "rope_type": "default",
        },
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": _round8(
            b["shared_expert_intermediate_size"] / div),
        "sliding_window_size": 32768,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": b["vocab_size"],
    }
    vision = {
        "deepstack_visual_indexes": [],
        "depth": vision_depth,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": vis_hidden,
        "in_channels": b["vision_in_channels"],
        "intermediate_size": _round8(b["vision_intermediate_size"] / div),
        "model_type": "qwen3_5_moe",
        "num_heads": vis_heads,
        "num_position_embeddings": b["vision_num_position_embeddings"],
        "out_hidden_size": hidden,
        "patch_size": b["vision_patch_size"],
        "spatial_merge_size": b["vision_spatial_merge_size"],
        "temporal_patch_size": b["vision_temporal_patch_size"],
    }
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe",
        "text_config": text,
        "tie_word_embeddings": False,
        "transformers_version": "4.57.1",
        "video_token_id": 248057,
        "vision_config": vision,
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
        "_metaljax_wedge_repro": True,
    }


# -------------------------------------------------------------- tensor spec

def tensor_spec(config):
    """[(hf_key, shape)] in checkpoint order, for a transformers config.

    Verified against the real 35B checkpoint by `check` -- every non-`mtp.`
    key and shape matches exactly.  (`mtp.*` is the speculative-decoding
    head; keras-hub's converter skips that prefix, so a repro that never
    writes it loads the same set of tensors.)
    """
    t = config["text_config"]
    v = config.get("vision_config")
    H = t["hidden_size"]
    hd = t["head_dim"]
    nq, nkv = t["num_attention_heads"], t["num_key_value_heads"]
    E = t["num_experts"]
    I = t["moe_intermediate_size"]
    S = t["shared_expert_intermediate_size"]
    lk, lv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    lkd, lvd = t["linear_key_head_dim"], t["linear_value_head_dim"]
    ck = t["linear_conv_kernel_dim"]
    qkv_dim = lk * lkd * 2 + lv * lvd
    z_dim = lv * lvd
    q_out = (2 if t.get("attn_output_gate") else 1) * nq * hd

    out = [("model.language_model.embed_tokens.weight",
            [t["vocab_size"], H])]
    for i, kind in enumerate(t["layer_types"]):
        p = f"model.language_model.layers.{i}"
        out.append((f"{p}.input_layernorm.weight", [H]))
        if kind == "full_attention":
            out += [
                (f"{p}.self_attn.q_proj.weight", [q_out, H]),
                (f"{p}.self_attn.q_norm.weight", [hd]),
                (f"{p}.self_attn.k_proj.weight", [nkv * hd, H]),
                (f"{p}.self_attn.k_norm.weight", [hd]),
                (f"{p}.self_attn.v_proj.weight", [nkv * hd, H]),
                (f"{p}.self_attn.o_proj.weight", [H, nq * hd]),
            ]
        else:
            out += [
                (f"{p}.linear_attn.in_proj_qkv.weight", [qkv_dim, H]),
                (f"{p}.linear_attn.in_proj_z.weight", [z_dim, H]),
                (f"{p}.linear_attn.in_proj_b.weight", [lv, H]),
                (f"{p}.linear_attn.in_proj_a.weight", [lv, H]),
                (f"{p}.linear_attn.conv1d.weight", [qkv_dim, 1, ck]),
                (f"{p}.linear_attn.dt_bias", [lv]),
                (f"{p}.linear_attn.A_log", [lv]),
                (f"{p}.linear_attn.norm.weight", [lvd]),
                (f"{p}.linear_attn.out_proj.weight", [H, z_dim]),
            ]
        out += [
            (f"{p}.mlp.gate.weight", [E, H]),
            (f"{p}.mlp.shared_expert.gate_proj.weight", [S, H]),
            (f"{p}.mlp.shared_expert.up_proj.weight", [S, H]),
            (f"{p}.mlp.shared_expert.down_proj.weight", [H, S]),
            (f"{p}.mlp.shared_expert_gate.weight", [1, H]),
            # The two tensors that dominate the load: the whole expert bank
            # of a layer arrives as ONE batched assign each.
            (f"{p}.mlp.experts.gate_up_proj", [E, 2 * I, H]),
            (f"{p}.mlp.experts.down_proj", [E, H, I]),
            (f"{p}.post_attention_layernorm.weight", [H]),
        ]
    out.append(("model.language_model.norm.weight", [H]))
    if not config.get("tie_word_embeddings"):
        out.append(("lm_head.weight", [t["vocab_size"], H]))

    if v:
        hv, inter, m = v["hidden_size"], v["intermediate_size"], \
            v["spatial_merge_size"]
        merged = hv * m * m
        out += [
            ("model.visual.patch_embed.proj.weight",
             [hv, v["in_channels"], v["temporal_patch_size"],
              v["patch_size"], v["patch_size"]]),
            ("model.visual.patch_embed.proj.bias", [hv]),
            ("model.visual.pos_embed.weight",
             [v["num_position_embeddings"], hv]),
        ]
        for i in range(v["depth"]):
            p = f"model.visual.blocks.{i}"
            out += [
                (f"{p}.norm1.weight", [hv]), (f"{p}.norm1.bias", [hv]),
                (f"{p}.attn.qkv.weight", [3 * hv, hv]),
                (f"{p}.attn.qkv.bias", [3 * hv]),
                (f"{p}.attn.proj.weight", [hv, hv]),
                (f"{p}.attn.proj.bias", [hv]),
                (f"{p}.norm2.weight", [hv]), (f"{p}.norm2.bias", [hv]),
                (f"{p}.mlp.linear_fc1.weight", [inter, hv]),
                (f"{p}.mlp.linear_fc1.bias", [inter]),
                (f"{p}.mlp.linear_fc2.weight", [hv, inter]),
                (f"{p}.mlp.linear_fc2.bias", [hv]),
            ]
        out += [
            ("model.visual.merger.norm.weight", [hv]),
            ("model.visual.merger.norm.bias", [hv]),
            ("model.visual.merger.linear_fc1.weight", [merged, merged]),
            ("model.visual.merger.linear_fc1.bias", [merged]),
            ("model.visual.merger.linear_fc2.weight",
             [v["out_hidden_size"], merged]),
            ("model.visual.merger.linear_fc2.bias", [v["out_hidden_size"]]),
        ]
    return out


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def spec_bytes(spec):
    return sum(_numel(s) for _, s in spec) * 2      # bf16


# ------------------------------------------------------------------- build

_POOL = None


def _pool():
    """One seeded pool of plausible weight values, tiled into every tensor.

    Drawing 2e9 independent normals would dominate the build time and buys
    nothing: the load is what is under test.  Tiling a rotated pool keeps
    the bytes deterministic per tensor name and the values finite (random
    bf16 BIT patterns would be full of NaN/Inf and poison `--generate`).
    """
    global _POOL
    if _POOL is None:
        import ml_dtypes
        rng = np.random.default_rng(0xC0FFEE)
        f = rng.standard_normal(1 << 20, dtype=np.float32) * 0.02
        _POOL = f.astype(ml_dtypes.bfloat16)
    return _POOL


def make_tensor(name, shape):
    pool = _pool()
    off = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4)
                         .digest(), "little") % len(pool)
    n = _numel(shape)
    if n <= len(pool) - off:
        buf = pool[off:off + n]
    else:
        rolled = np.roll(pool, -off)
        buf = np.tile(rolled, -(-n // len(rolled)))[:n]
    return np.ascontiguousarray(buf).reshape(shape)


def build(out_dir, config, shards=SHARDS, force=False):
    from safetensors.numpy import save_file

    spec = tensor_spec(config)
    total = spec_bytes(spec)
    out_dir = Path(out_dir)
    if out_dir.exists() and not force:
        idx = out_dir / "model.safetensors.index.json"
        if idx.exists() and json.load(open(idx))["metadata"][
                "total_size"] == total:
            print(f"[build] {out_dir} already holds {total / 2**30:.2f} GB "
                  f"({len(spec)} tensors) -- reusing (--force to rebuild)")
            return out_dir
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_shard = total / shards
    weight_map, cur, cur_bytes, shard = {}, {}, 0, 1
    names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
             for i in range(shards)]
    t0 = time.monotonic()
    for k, shp in spec:
        cur[k] = make_tensor(k, shp)
        weight_map[k] = names[min(shard - 1, shards - 1)]
        cur_bytes += _numel(shp) * 2
        if cur_bytes >= per_shard and shard < shards:
            save_file(cur, str(out_dir / names[shard - 1]))
            cur, cur_bytes, shard = {}, 0, shard + 1
    if cur:
        save_file(cur, str(out_dir / names[min(shard - 1, shards - 1)]))
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(out_dir / "model.safetensors.index.json", "w"))
    json.dump(config, open(out_dir / "config.json", "w"), indent=1)

    # Tokenizer/preprocessor metadata: copied from the checkpoint already in
    # the local HF cache (small JSON only -- no weights, no download).
    src = real_snapshot()
    copied = []
    for f in sorted(src.iterdir()):
        if f.name.startswith("model") and f.suffix in (".safetensors",):
            continue
        if f.name == "model.safetensors.index.json" or f.name == "config.json":
            continue
        shutil.copyfile(f, out_dir / f.name)
        copied.append(f.name)
    print(f"[build] {out_dir}: {len(spec)} tensors, {total / 2**30:.2f} GB, "
          f"{shards} shards in {time.monotonic() - t0:.1f}s; "
          f"metadata copied: {len(copied)} files")
    return out_dir


def real_snapshot():
    snaps = sorted((REAL_REPO / "snapshots").glob("*"))
    if not snaps:
        sys.exit(f"no local snapshot under {REAL_REPO} -- this script copies "
                 "the tokenizer/preprocessor JSONs from it and never "
                 "downloads; fetch the real preset first or point REAL_REPO "
                 "at another Qwen3_5Moe checkpoint")
    return snaps[-1]


# ------------------------------------------------------------------- checks

def check():
    """Prove the shape formulas against the real 35B checkpoint."""
    import struct

    snap = real_snapshot()
    real_cfg = json.load(open(snap / "config.json"))
    idx = json.load(open(snap / "model.safetensors.index.json"))
    real = {}
    for fname in sorted(set(idx["weight_map"].values())):
        with open(snap / fname, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, meta in hdr.items():
            if k != "__metadata__":
                real[k] = (meta["dtype"], meta["shape"])
    spec = tensor_spec(real_cfg)
    bad = []
    for k, shp in spec:
        if k not in real:
            bad.append((k, shp, "MISSING"))
        elif list(real[k][1]) != list(shp):
            bad.append((k, shp, real[k][1]))
    extra = sorted(set(real) - {k for k, _ in spec})
    non_mtp = [k for k in extra if not k.startswith("mtp.")]
    print(f"[check] real checkpoint: {len(real)} tensors, spec: {len(spec)}")
    print(f"[check] shape mismatches: {len(bad)}")
    for b in bad[:10]:
        print("   ", b)
    print(f"[check] in checkpoint but not in spec: {len(extra)} "
          f"({len(extra) - len(non_mtp)} mtp.*, {len(non_mtp)} other)")
    for k in non_mtp[:10]:
        print("    unexpected:", k)
    ok = not bad and not non_mtp
    print("[check]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def profile(spec, label):
    tot = spec_bytes(spec)
    big = sorted(spec, key=lambda kv: -_numel(kv[1]))
    over = [kv for kv in spec if _numel(kv[1]) * 2 >= 64 * 2**20]
    print(f"[profile] {label}: {len(spec)} tensors, {tot / 2**30:.2f} GB")
    print(f"[profile]   largest: {big[0][0]} {big[0][1]} "
          f"{_numel(big[0][1]) * 2 / 2**30:.3f} GB")
    print(f"[profile]   tensors >=64 MB: {len(over)} carrying "
          f"{sum(_numel(k[1]) for k in over) * 2 / 2**30:.2f} GB "
          f"({100 * sum(_numel(k[1]) for k in over) * 2 / tot:.0f}% of bytes)")
    med = sorted(_numel(s) for _, s in spec)[len(spec) // 2] * 2 / 2**20
    print(f"[profile]   median tensor: {med:.3f} MB")
    return tot


# ---------------------------------------------------------------- predict

# PEAK MODEL, calibrated on the R1-Distill-32B flight log of 2026-08-04 --
# the only clean big streamed load through this exact shim:
#
#     61 GB assigned + 1.2 baseline + 2 x 1.56 GB (its largest tensor)
#       = 65.3 GB   vs   65.0 GB measured peak footprint
#
# (`top`'s MEM column has 1 GB resolution at this size, so that is a hit.)
# The two transient copies are the host tensor `SafetensorLoader.get_tensor`
# returns and the contiguous copy the converter's `np.transpose` forces
# before the device write; MLX's freed-buffer cache contributes ~nothing
# because the shim clears it every BENCH_STREAM_CLEAR_GB and the freed
# staging buffers are reused in between.  `predicted_upper_gb` adds that
# cache back as a worst case -- use it, not the central value, to pick a
# guard budget.
BASELINE_GB = 1.2               # python + keras + jaxlib + plugin, measured


def predict_peak_gb(spec, clear_gb=None):
    if clear_gb is None:
        clear_gb = float(os.environ.get("BENCH_STREAM_CLEAR_GB", "8"))
    weights = spec_bytes(spec) / 2**30
    largest = max(_numel(s) for _, s in spec) * 2 / 2**30
    peak = weights + BASELINE_GB + 2 * largest
    return dict(weights_gb=round(weights, 2), baseline_gb=BASELINE_GB,
                largest_tensor_gb=round(largest, 3),
                predicted_peak_gb=round(peak, 2),
                predicted_upper_gb=round(peak + min(clear_gb, weights), 2))


# ------------------------------------------------------------------- load

def load(preset, backend="metaljax", generate=0, decode_tokens=8):
    os.environ.setdefault(
        "JAX_PLATFORMS", "metal" if backend == "metaljax" else "cpu")
    os.environ.setdefault("KERAS_BACKEND", "jax")
    sys.path.insert(0, str(HERE))

    config = json.load(open(Path(preset) / "config.json"))
    spec = tensor_spec(config)
    pred = predict_peak_gb(spec)
    print(f"[load] preset={preset} backend={backend}")
    print(f"[load] PREDICTED {json.dumps(pred)}", flush=True)
    profile(spec, Path(preset).name)

    t0 = time.monotonic()
    if generate:
        import run_bench
        bench = {"id": f"wedge-{Path(preset).name}", "path": "keras_lm",
                 "model": str(preset), "arch": ARCH}
        rec = run_bench.run_keras_lm(bench, backend, run_bench.MANIFEST[
            "prompt"], generate)
        report = rec.get("stream_load")
    else:
        # The load block of run_bench.run_keras_lm, minus generation: same
        # shim, same class, same from_preset call.  (Kept here rather than
        # calling run_keras_lm so a load-phase repro does not also pay a
        # multi-minute compile on a 40-layer hybrid graph.)
        import keras
        keras.config.set_dtype_policy("bfloat16")
        import adapter_keras_extra as extra
        extra.patch_sentencepiece_native()
        import keras_hub

        cls = getattr(keras_hub.models, ARCH)
        info = extra.install_streaming_load()
        print(f"[load] stream shim: {json.dumps(info)}", flush=True)
        lm = cls.from_preset(str(preset))
        report = extra.finalize_streaming_load(lm)
        extra.uninstall_streaming_load()
        rec = {}
        del lm
    load_s = time.monotonic() - t0

    peak = measured_peak_gb(backend)
    rec.update(load_s=round(load_s, 2), predicted=pred, measured_peak_gb=peak,
               stream_load=report, backend=backend, preset=str(preset))
    print(f"[load] done in {load_s:.1f}s  measured_peak_gb={peak}  "
          f"predicted={pred['predicted_peak_gb']}", flush=True)
    print("[load] stream report " + json.dumps(
        {k: v for k, v in (report or {}).items()
         if k not in ("build_reads", "completed")}), flush=True)
    print("RESULT " + json.dumps(
        {k: v for k, v in rec.items() if k not in ("token_ids",)}), flush=True)
    return rec


def measured_peak_gb(backend):
    """Peak of what the process actually holds.

    `mx.get_peak_memory` is the device (= unified) high-water mark; RSS
    includes the mapped checkpoint pages, which the guard reports
    separately.  Both are printed so a run can be compared against either
    column of a flight log.
    """
    import resource
    out = {"rss_gb": round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)}
    try:
        import mlx.core as mx
        out["mlx_peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
        out["mlx_active_gb"] = round(mx.get_active_memory() / 2**30, 2)
    except ImportError:
        pass
    return out


# -------------------------------------------------------------------- main

def rung_dir(root, rung):
    return Path(root) / f"qwen35moe-{rung}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["check", "build", "profile", "predict",
                                    "load"])
    ap.add_argument("--rung", default="small", choices=sorted(RUNGS))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--preset", default=None,
                    help="explicit preset dir (overrides --rung)")
    ap.add_argument("--shards", type=int, default=SHARDS)
    ap.add_argument("--backend", default="metaljax",
                    choices=["metaljax", "cpu"])
    ap.add_argument("--generate", type=int, default=0,
                    help="run the whole row (load+compile+decode N tokens) "
                         "through run_bench.run_keras_lm instead of the "
                         "load alone")
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()

    if ns.cmd == "check":
        return check()

    rung = RUNGS[ns.rung]
    config = make_config(**rung)
    spec = tensor_spec(config)
    preset = Path(ns.preset) if ns.preset else rung_dir(ns.root, ns.rung)

    if ns.cmd == "profile":
        profile(spec, f"{ns.rung} (div={rung['div']})")
        print("[profile] predicted " + json.dumps(predict_peak_gb(spec)))
        return 0
    if ns.cmd == "predict":
        print(json.dumps(predict_peak_gb(spec), indent=1))
        return 0
    if ns.cmd == "build":
        gb = spec_bytes(spec) / 2**30
        if ns.rung == "full" and not ns.force:
            sys.exit("refusing to build the 'full' rung (%.0f GB): that is "
                     "the paused row itself -- run the real preset instead, "
                     "and only with Oleg's approval" % gb)
        build(preset, config, shards=ns.shards, force=ns.force)
        return 0
    if ns.cmd == "load":
        if not (preset / "config.json").exists():
            sys.exit(f"{preset} not built yet -- run `build --rung {ns.rung}`")
        load(preset, backend=ns.backend, generate=ns.generate)
        return 0


if __name__ == "__main__":
    sys.exit(main())
