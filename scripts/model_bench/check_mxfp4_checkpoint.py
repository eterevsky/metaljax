"""Spot-check the native-MXFP4 path against the stock loader, on the real
gpt-oss-20b checkpoint.

    bench-venv/bin/python scripts/model_bench/check_mxfp4_checkpoint.py

Host-side only: a few real tensors are read from the safetensors shards and
run through BOTH dequantizations --

  * `keras_hub.src.utils.transformers.convert_gpt_oss`'s arithmetic E2M1
    decode (sign/exponent/mantissa masks), which is what the row does today;
  * `mxfp4_gpt_oss.np_dequant`'s table lookup, which is what the packed
    loader's in-graph chain computes, and what `metaljax.qmm` verifies
    against the E2M1 grid;

-- and compared BIT for bit, in float32 and in the bfloat16 the row actually
stores. Also reports the E8M0 byte range (the in-graph scale table has no
usable entry for 0 or 255) and the packed-vs-dequantized size ratio.

Slices a few experts out of each tensor so the peak stays a few hundred MB.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import safetensors

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from mxfp4_gpt_oss import np_dequant, GROUP  # noqa: E402

DEFAULT_SNAPSHOT = (Path.home() / ".cache/huggingface/hub"
                    / "models--openai--gpt-oss-20b/snapshots")


def stock_dequant(blocks, scales):
    """keras-hub's own `dequantize_mxfp4`, verbatim (convert_gpt_oss.py).

    Kept as a literal copy so the comparison is against what the row runs,
    not against a re-derivation of it.
    """
    scales = 2.0 ** (scales.astype(np.float32) - 127.0)
    num_experts, out_dim, num_blocks, block_size = blocks.shape

    blocks_uint8 = blocks.astype(np.uint8)
    high_nibble = (blocks_uint8 >> 4) & 0xF
    low_nibble = blocks_uint8 & 0xF
    blocks_4bit = np.stack([low_nibble, high_nibble], axis=-1)
    blocks_4bit = blocks_4bit.reshape(num_experts, out_dim, num_blocks,
                                      block_size * 2)

    s = (blocks_4bit >> 3) & 0x1
    e = (blocks_4bit >> 1) & 0x3
    m = blocks_4bit & 0x1

    bias = 1.0
    sign = 1.0 - 2.0 * s
    normal_mask = e != 0
    values = np.empty_like(blocks_4bit, dtype=np.float32)
    values[normal_mask] = (
        sign[normal_mask]
        * (2.0 ** (e[normal_mask].astype(np.float32) - bias))
        * (1.0 + m[normal_mask].astype(np.float32) / 2.0))
    values[~normal_mask] = (
        sign[~normal_mask]
        * (2.0 ** (1.0 - bias))
        * (m[~normal_mask].astype(np.float32) / 2.0))

    values = values.reshape(num_experts, out_dim, num_blocks * block_size * 2)
    scales_expanded = np.repeat(scales[..., np.newaxis], block_size * 2,
                               axis=3)
    scales_expanded = scales_expanded.reshape(
        num_experts, out_dim, num_blocks * block_size * 2)
    return values * scales_expanded


def find_snapshot(root):
    snaps = sorted(Path(root).glob("*"))
    if not snaps:
        raise SystemExit(f"no snapshot under {root}")
    return snaps[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 11, 23])
    ap.add_argument("--experts", type=int, default=3,
                    help="experts sliced out of each tensor (memory bound)")
    ns = ap.parse_args()

    snap = Path(ns.snapshot) if ns.snapshot else find_snapshot(
        DEFAULT_SNAPSHOT)
    index = json.load(open(snap / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    files = {}

    def tensor(key):
        fname = weight_map[key]
        if fname not in files:
            files[fname] = safetensors.safe_open(snap / fname, framework="np")
        return files[fname].get_tensor(key)

    print(f"checkpoint: {snap}")
    checked = 0
    bad = 0
    for layer in ns.layers:
        for which in ("gate_up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.experts.{which}"
            blocks = tensor(base + "_blocks")[:ns.experts]
            scales = tensor(base + "_scales")[:ns.experts]

            want = stock_dequant(blocks, scales)          # [E, out, k]
            flat = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
            got = np_dequant(flat, scales)

            exact32 = np.array_equal(got, want)
            # The row stores the dequantized weight in a bfloat16 variable,
            # so bit-equality THERE is what the change has to preserve.
            import ml_dtypes
            exact16 = np.array_equal(got.astype(ml_dtypes.bfloat16),
                                     want.astype(ml_dtypes.bfloat16))
            # ... and bf16 must itself be lossless for MXFP4 values: an E2M1
            # value carries one mantissa bit.
            round_trip = np.array_equal(
                want.astype(ml_dtypes.bfloat16).astype(np.float32), want)

            lo, hi = int(scales.min()), int(scales.max())
            ok = exact32 and exact16 and round_trip
            bad += not ok
            checked += 1
            print(f"  L{layer:<2d} {which:<13s} {str(blocks.shape):22s} "
                  f"f32={'EXACT' if exact32 else 'DIFFERS'} "
                  f"bf16={'EXACT' if exact16 else 'DIFFERS'} "
                  f"bf16-lossless={'yes' if round_trip else 'NO'} "
                  f"e8m0=[{lo},{hi}]")
            if not exact32:
                d = np.nonzero(got != want)
                print(f"      first mismatch at {[a[0] for a in d]}: "
                      f"{got[tuple(a[0] for a in d)]} vs "
                      f"{want[tuple(a[0] for a in d)]}")

    # whole-checkpoint size accounting
    codes = sum(int(np.prod(files.setdefault(
        weight_map[k], safetensors.safe_open(snap / weight_map[k],
                                             framework="np")
    ).get_slice(k).get_shape())) for k in weight_map if k.endswith("_blocks"))
    scale_bytes = sum(int(np.prod(files[weight_map[k]].get_slice(k)
                                  .get_shape()))
                      for k in weight_map if k.endswith("_scales"))
    other = sum(int(np.prod(files.setdefault(
        weight_map[k], safetensors.safe_open(snap / weight_map[k],
                                             framework="np")
    ).get_slice(k).get_shape())) for k in weight_map
        if not k.endswith(("_blocks", "_scales")))
    print(f"\n  expert codes   {codes / 1e9:7.2f} GB packed  "
          f"({codes * 2 * 2 / 1e9:6.2f} GB as bf16)")
    print(f"  expert scales  {scale_bytes / 1e9:7.3f} GB")
    print(f"  everything else{other * 2 / 1e9:7.2f} GB (bf16)")
    print(f"  resident: {(codes + scale_bytes + other * 2) / 1e9:.2f} GB "
          f"packed vs {(codes * 4 + other * 2) / 1e9:.2f} GB dequantized")

    print(f"\n{checked - bad}/{checked} tensors bit-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
