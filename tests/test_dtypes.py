"""Host<->device dtype conversion: bf16 crosses as a bitcast.

The original to_mx/to_np staged bf16 through float32 (MLX cannot ingest
ml_dtypes.bfloat16 directly). That detour (a) uploaded 2x the bytes and
pinned a 2x-size f32 device buffer until the first eval -- loading a
62 GB model transiently held 123 GB -- and (b) went through MLX's
f32->bf16 astype, which canonicalizes every NaN to 0x7FC0, losing both
the payload and the SIGN bit. The bitcast path preserves bits exactly
and allocates only the bf16-sized buffer; these tests pin both
properties.
"""

import ml_dtypes
import mlx.core as mx
import numpy as np

import jax

from metaljax import dtypes as mdt

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


def test_bf16_to_mx_roundtrip_bit_exact():
    m = mdt.to_mx(BF16)
    assert m.dtype == mx.bfloat16
    back = mdt.to_np(m)
    assert back.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(bits(back), BITS)


def test_bf16_to_mx_noncontiguous_and_0d():
    m = mdt.to_mx(BF16[::2])
    np.testing.assert_array_equal(bits(mdt.to_np(m)), BITS[::2])
    scalar = mdt.to_np(mdt.to_mx(BF16.reshape(-1)[4].reshape(())))
    assert scalar.shape == ()
    assert bits(scalar.reshape(1))[0] == BITS[4]


def test_bf16_to_np_noncontiguous_device_array():
    dev = mx.transpose(mx.array(np.arange(12, dtype=np.float32).reshape(3, 4))
                       .astype(mx.bfloat16))
    got = mdt.to_np(dev)
    want = np.arange(12, dtype=np.float32).reshape(3, 4).T
    np.testing.assert_array_equal(got.astype(np.float32), want)


def test_bf16_pjrt_transfer_preserves_nan_bits():
    # Through the real backend: device_put + read back must not touch bits.
    metal = jax.devices("metal")[0]
    dev = jax.device_put(BF16, metal)
    np.testing.assert_array_equal(bits(dev), BITS)


def test_bf16_upload_does_not_double_memory():
    # A bf16 upload must allocate ~its own size on device, not the 2x of
    # an f32 staging copy (3x transiently with the lazy cast's result).
    n = 32 * 1024 * 1024  # 64 MB of bf16
    host = np.zeros(n, ml_dtypes.bfloat16)
    mx.clear_cache()
    before = mx.get_active_memory()
    m = mdt.to_mx(host)
    mx.eval(m)
    grown = mx.get_active_memory() - before
    assert grown < 1.5 * host.nbytes, (
        f"bf16 upload grew device memory by {grown/1e6:.0f} MB "
        f"for a {host.nbytes/1e6:.0f} MB array (f32 staging is back?)")
