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
import pytest

import jax
import jax.numpy as jnp

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


# --------------------------------------------------------------------------
# jnp.int2 / jnp.uint2 (jax 0.11.2): the i4 pair's emulation at half the
# width, compared EXACTLY against jax-CPU (every answer is cast to int32
# before it leaves the program, so both backends hand back plain ints).

_INT2 = {"int2": (jnp.int2, np.arange(-2, 2)),
         "uint2": (jnp.uint2, np.arange(0, 4))}
_FVALS = np.array([-100.0, -2.5, -2.0, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 1.9,
                   2.0, 3.9, 4.0, 100.0, np.inf, -np.inf, np.nan], np.float32)


def _cpu_and_metal(f, *args):
    outs = []
    for dev in (jax.devices("cpu")[0], _metal()):
        with jax.default_device(dev):
            out = jax.jit(f)(*args)
        outs.append([np.asarray(v) for v in jax.tree.leaves(out)])
    return outs


def _exact(f, *args):
    cpu, metal = _cpu_and_metal(f, *args)
    for c, m in zip(cpu, metal):
        assert m.dtype == c.dtype and m.shape == c.shape
        np.testing.assert_array_equal(m.astype(np.float64),
                                      c.astype(np.float64))


@pytest.mark.parametrize("name", sorted(_INT2))
def test_int2_converts(name):
    dt, vals = _INT2[name]
    i32 = jnp.int32
    # float -> int2 SATURATES (XLA's fptosi.sat; NaN -> 0), int -> int2 wraps.
    for src in (np.float32, np.float16, ml_dtypes.bfloat16):
        _exact(lambda x: x.astype(dt).astype(i32), _FVALS.astype(src))
    _exact(lambda x: x.astype(dt).astype(i32),
           np.arange(-9, 9, dtype=np.int32))
    for tgt in (jnp.float32, jnp.bfloat16, jnp.int8, jnp.uint8):
        _exact(lambda x: x.astype(dt).astype(tgt), vals.astype(np.int32))


@pytest.mark.parametrize("name", sorted(_INT2))
def test_int2_arithmetic_wraps(name):
    dt, vals = _INT2[name]
    a = np.repeat(vals, len(vals)).astype(np.int32)
    b = np.tile(vals, len(vals)).astype(np.int32)
    nz = np.where(b == 0, 1, b).astype(np.int32)
    for op in (lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y,
               lambda x, y: -x, lambda x, y: ~x, jax.lax.shift_left,
               jnp.maximum, lambda x, y: x ^ y):
        _exact(lambda x, y: op(x.astype(dt), y.astype(dt)).astype(jnp.int32),
               a, b)
    for op in (jax.lax.div, jax.lax.rem):
        _exact(lambda x, y: op(x.astype(dt), y.astype(dt)).astype(jnp.int32),
               a, nz)


@pytest.mark.parametrize("name", sorted(_INT2))
def test_int2_transfer_and_bitcast(name):
    dt, vals = _INT2[name]
    host = np.resize(vals, (3, 4)).astype(np.int8).astype(np.dtype(dt))
    back = np.asarray(jax.device_put(host, _metal()))
    assert back.dtype == host.dtype
    np.testing.assert_array_equal(back.astype(np.int32), host.astype(np.int32))
    _exact(lambda x: x + x, host)                       # int2 in AND out
    bc = jax.lax.bitcast_convert_type
    # Four fields per byte, lowest bits first, both directions.
    _exact(lambda x: bc(x, dt).astype(jnp.int32),
           np.array([0b11100100, 0x01, 0x80, 0xFF], np.uint8))
    _exact(lambda x: bc(x, jnp.uint8), host)
    _exact(lambda x: bc(x, jnp.int16), host.reshape(3, 1, 4).repeat(2, 1)
           .reshape(3, 8))


def test_int2_overlap_program():
    # jax-v0.11.2 overlap_test::test_avoid_excess_precision's quantizer.
    def f(x):
        amax = jnp.abs(x).max(axis=-1, keepdims=True).astype(jnp.float32)
        scale = amax / jnp.iinfo(jnp.int2).max
        q = jnp.rint(x / scale).astype(jnp.int2)
        return q.astype(jnp.float32) * scale, q.astype(jnp.int32)
    x = np.asarray(jax.random.normal(jax.random.key(123), [16, 256],
                                     dtype=jnp.bfloat16))
    _exact(f, x)
