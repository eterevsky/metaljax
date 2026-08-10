"""Stage 2 M1: native buffer path, differential vs the Python path.

Every transfer rule that was ever a bug is pinned here BIT-EXACTLY
against the Python reference: bf16 NaN payload+sign survival (the
bitcast rule), f64/c128 narrow-store/widen-return, negative strides
with base offset, rank-0, 0-size, bool, and the data=None zeros path.
"""
import os
import sys
import time

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


def _to_host_both(arr, enum):
    """(python bytes, native bytes) for a DEVICE array of any layout."""
    dims = list(arr.shape)
    saved = engine.NATIVE
    try:
        engine.NATIVE = None
        py = engine.to_host(engine.MetalBuffer(arr, enum, dims))
        engine.NATIVE = native
        nat = engine.to_host(engine.MetalBuffer(arr, enum, dims))
    finally:
        engine.NATIVE = saved
    return py, nat


# One device array per layout an output can reach the wire in: the ones
# read straight out of their own buffer, and the ones that still have to
# be gathered first.
LAYOUTS = {
    "contiguous": lambda b: b,
    "transposed": lambda b: mx.transpose(b),
    "sliced_rows": lambda b: b[1:3],
    "strided": lambda b: b[:, ::2],
    "reversed": lambda b: b[::-1],
    "broadcast": lambda b: mx.broadcast_to(b[0:1], b.shape),
    "reshaped": lambda b: mx.reshape(b, (b.size,)),
    "unit_dims": lambda b: b[0:1, 0:1],
    "rank0": lambda b: b[0, 0],
    "unevaluated": lambda b: b * 2,
}


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_to_host_layouts_match_python(layout):
    """to_host reads the array's own buffer when the layout allows it.

    The fast path (skip `contiguous`, settle the array itself) is what
    keeps a per-output stream round trip out of every execute; a strided
    or broadcast result must still be gathered, and BIT-EXACTLY as the
    numpy path gathers it.
    """
    base = mx.arange(24, dtype=mx.float32).reshape(4, 6)
    mx.eval(base)
    py, nat = _to_host_both(LAYOUTS[layout](base), 11)
    assert py == nat


@pytest.mark.parametrize("name,enum", [("bf16", 13), ("f64", 12),
                                       ("c64", 14)])
def test_to_host_transposed_dtypes(name, enum):
    """Same, for the dtypes with a rule of their own (bitcast / widen)."""
    base = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    if name == "bf16":
        base = base.astype(mx.bfloat16)
    elif name == "c64":
        base = base.astype(mx.complex64)
    mx.eval(base)
    for arr in (base, mx.transpose(base)):
        py, nat = _to_host_both(arr, enum)
        assert py == nat


def test_to_host_settled_output_costs_no_round_trip():
    """A settled, contiguous output must not cost a stream round trip.

    The floor this pins: reading such an output is a memcpy, so it can
    only be as expensive as the numpy path's own memcpy. Wrapping it in
    a fresh `contiguous` node and evaluating that instead used to cost
    ~20us per output whatever its size — 0.5ms on a program handing back
    23 small tensors, which was the whole of the native engine's deficit
    against the Python one on texmo train chunks. Compared as a RATIO so
    machine load cancels; the real gap either side of the fix is ~300x,
    so 10x is a floor and not a stopwatch.
    """
    arr = mx.arange(16, dtype=mx.float32)
    mx.eval(arr)
    buf = engine.MetalBuffer(arr, 11, [16])
    saved = engine.NATIVE
    try:
        for warm in (True, False):
            engine.NATIVE = native
            t0 = time.perf_counter()
            for _ in range(400):
                engine.to_host(buf)
            t1 = time.perf_counter()
            engine.NATIVE = None
            for _ in range(400):
                engine.to_host(buf)
            t2 = time.perf_counter()
            if warm:
                continue
            nat, py = t1 - t0, t2 - t1
    finally:
        engine.NATIVE = saved
    assert nat < 10 * py, f"native to_host {nat:.4f}s vs numpy {py:.4f}s"


def test_native_declines_emulated_types():
    """i4/f8-class stay on the numpy path (dtypes.py owns their grids)."""
    for enum in (16, 17, 21, 22, 28, 29):  # f8e5m2, f8e4m3fn, s4, u4, f4, s1
        assert not native.native_type(enum)
