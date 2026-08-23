"""Host<->device bf16 transfers cross as a BITCAST, both directions.

The original Stage-1 path staged bf16 through float32 (MLX cannot ingest
ml_dtypes.bfloat16 directly).  That detour (a) uploaded 2x the bytes and
pinned a 2x-size f32 device buffer until the first eval -- loading a
62 GB model transiently held 123 GB -- and (b) went through MLX's
f32->bf16 astype, which canonicalizes every NaN to 0x7FC0, losing both
the payload and the SIGN bit.  The plugin's transfer path must preserve
bits exactly; these tests pin that through the real PJRT boundary.
(The Stage-1 to_mx/to_np unit tests and the device-memory accounting
test died with the Stage-1 retirement; the memory property is covered by
plugin-native/ingest_test.py.)
"""

import ml_dtypes
import numpy as np

import jax

# Bit patterns covering every interesting bf16 class: normals, +-0,
# +-inf, quiet/signaling NaNs with payloads (and a negative NaN),
# subnormals, and the extremes.
BITS = np.array(
    [0x3FC0, 0xBFC0, 0x0000, 0x8000, 0x7F80, 0xFF80,   # 1.5 -1.5 +-0 +-inf
     0x7FC0, 0x7FC1, 0xFFC0, 0x7F81, 0xFFFF,           # NaNs w/ payloads
     0x0001, 0x8001, 0x007F,                            # subnormals
     0x7F7F, 0xFF7F, 0x0080],                           # +-max, min normal
    np.uint16)
BF16 = BITS.view(ml_dtypes.bfloat16)


def bits(a):
    return np.asarray(a).view(np.uint16)


def _metal():
    return jax.devices("metal")[0]


def test_bf16_pjrt_transfer_preserves_nan_bits():
    # device_put + read back must not touch bits: payloads, signs,
    # subnormals and signaling NaNs all survive.
    dev = jax.device_put(BF16, _metal())
    assert dev.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(bits(dev), BITS)


def test_bf16_pjrt_transfer_strided_host_array():
    # A non-contiguous host view must upload its own elements (the plugin
    # receives base pointer + strides; a wrong offset walk shows here).
    dev = jax.device_put(BF16[::2], _metal())
    np.testing.assert_array_equal(bits(dev), BITS[::2])


def test_bf16_pjrt_transfer_rank0():
    scalar = BF16.reshape(-1)[4].reshape(())
    dev = jax.device_put(scalar, _metal())
    assert np.asarray(dev).shape == ()
    assert bits(np.asarray(dev).reshape(1))[0] == BITS[4]


def test_bf16_compute_output_bits_sane():
    """The EGRESS side, after real arithmetic: `x + 0` must preserve every
    NORMAL finite value bit-for-bit and keep NaN lanes NaN.

    Two classes are deliberately excluded, both documented GPU behavior
    rather than transfer bugs: subnormals flush to zero, and -0.0 + 0.0 is
    +0.0 by IEEE. NaN payload canonicalization through arithmetic is
    hardware behavior too, so the NaN lanes are checked for NaN-ness only.
    """
    dev = jax.device_put(BF16, _metal())
    out = jax.jit(lambda x: x + jax.numpy.bfloat16(0))(dev)
    f32 = BF16.astype(np.float32)
    nan = np.isnan(f32)
    normal = (~nan) & (np.abs(f32) >= float(np.finfo(np.float32).tiny)) \
        & (f32 != 0.0)
    np.testing.assert_array_equal(bits(out)[normal], BITS[normal])
    assert np.isnan(np.asarray(out).astype(np.float32)[nan]).all()
    # The zero/subnormal lanes must at least still be zero-magnitude.
    small = (~nan) & (~normal)
    assert np.all(np.asarray(out).astype(np.float32)[small] == 0.0)
