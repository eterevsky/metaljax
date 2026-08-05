"""Stage 2 M1: native buffer path, differential vs the Python path.

Every transfer rule that was ever a bug is pinned here BIT-EXACTLY
against the Python reference: bf16 NaN payload+sign survival (the
bitcast rule), f64/c128 narrow-store/widen-return, negative strides
with base offset, rank-0, 0-size, bool, and the data=None zeros path.
"""
import os
import sys

import ml_dtypes
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "native", "build"))

native = pytest.importorskip(
    "metaljax_native",
    reason="native engine not built (native/build.sh)")

import mlx.core as mx  # noqa: E402

from metaljax import engine  # noqa: E402

# type_enum values under test (name -> (enum, numpy wire dtype)).
WIRE = {
    "pred": (1, np.bool_), "s8": (2, np.int8), "s16": (3, np.int16),
    "s32": (4, np.int32), "s64": (5, np.int64), "u8": (6, np.uint8),
    "u16": (7, np.uint16), "u32": (8, np.uint32), "u64": (9, np.uint64),
    "f16": (10, np.float16), "f32": (11, np.float32),
    "f64": (12, np.float64), "bf16": (13, ml_dtypes.bfloat16),
    "c64": (14, np.complex64), "c128": (15, np.complex128),
}


def _wire_array(name, dtype, shape=(3, 5)):
    rng = np.random.default_rng(hash(name) % 2**32)
    if name == "pred":
        return rng.integers(0, 2, shape).astype(dtype)
    if name in ("c64", "c128"):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min // 2, info.max // 2, shape,
                            dtype=np.int64).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


def _roundtrip(data, enum, dims, strides=None, offset=0):
    """(python bytes, native bytes) for one host->device->host trip."""
    saved = engine.NATIVE
    try:
        engine.NATIVE = None
        pybuf = engine.buffer_from_host(data, enum, dims, strides, offset)
        py = engine.to_host(pybuf)
        engine.NATIVE = native
        nbuf = engine.buffer_from_host(data, enum, dims, strides, offset)
        nat = engine.to_host(nbuf)
    finally:
        engine.NATIVE = saved
    return py, nat


@pytest.mark.parametrize("name", sorted(WIRE))
def test_roundtrip_bit_exact(name):
    enum, dt = WIRE[name]
    arr = _wire_array(name, dt)
    py, nat = _roundtrip(arr.tobytes(), enum, list(arr.shape))
    assert py == nat


def test_bf16_nan_payloads_survive():
    """The reason the bitcast rule exists: astype canonicalizes NaNs."""
    bits = np.array([0x7FC1, 0xFFC0, 0x7F81, 0xFF95, 0x3F80], np.uint16)
    arr = bits.view(ml_dtypes.bfloat16)
    py, nat = _roundtrip(arr.tobytes(), 13, [bits.size])
    assert py == nat
    assert np.frombuffer(nat, np.uint16).tolist() == bits.tolist()


def test_f64_narrowing_matches_numpy():
    """double->float->double must round identically to the numpy path."""
    vals = np.array([1e-310, 1.0 + 2**-40, np.pi, -0.0, np.inf, 1e300],
                    np.float64)
    py, nat = _roundtrip(vals.tobytes(), 12, [vals.size])
    assert py == nat


def test_c128_roundtrip():
    vals = (np.array([1.5, -2.25]) + 1j * np.array([3.0, -0.0])
            ).astype(np.complex128)
    py, nat = _roundtrip(vals.tobytes(), 15, [2])
    assert py == nat


def test_negative_strides_with_offset():
    """A flipped view: base offset points at the logical [0, 0] element."""
    src = np.arange(24, dtype=np.float32).reshape(4, 6)
    flipped = src[::-1, ::-1]
    offset = (src.nbytes - src.itemsize)  # last element of the buffer
    py, nat = _roundtrip(src.tobytes(), 11, [4, 6],
                         strides=list(flipped.strides), offset=offset)
    assert py == nat
    assert np.frombuffer(nat, np.float32).reshape(4, 6).tolist() \
        == flipped.tolist()


def test_rank0_and_empty():
    scalar = np.float32(2.5)
    py, nat = _roundtrip(scalar.tobytes(), 11, [])
    assert py == nat
    empty = np.empty((0, 3), np.int32)
    py, nat = _roundtrip(empty.tobytes(), 4, [0, 3])
    assert py == nat == b""


def test_none_data_is_zeros():
    py, nat = _roundtrip(None, 11, [2, 2])
    assert py == nat == b"\x00" * 16


def test_native_declines_emulated_types():
    """i4/f8-class stay on the numpy path (dtypes.py owns their grids)."""
    for enum in (16, 17, 21, 22, 28, 29):  # f8e5m2, f8e4m3fn, s4, u4, f4, s1
        assert not native.native_type(enum)
