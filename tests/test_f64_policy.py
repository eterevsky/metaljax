"""Strict-f64 policy: pass-through is allowed, f64 arithmetic fails.

complex128 rides the same policy: it is float64 in both halves, so it is
stored as complex64 and any op that computes in it is refused.
"""

import io

import jax
import numpy as np
import pytest

from metaljax import Interpreter
from metaljax import dtypes as mdt
from metaljax.dtypes import UnsupportedDtypeError


def lower_x64(f, *args):
    with jax.enable_x64():
        lowered = jax.jit(f).lower(*args)
        buf = io.BytesIO()
        lowered.compiler_ir().operation.write_bytecode(buf)
        return buf.getvalue()


@pytest.mark.skipif(mdt.F64_DOWNCAST, reason="strict mode only")
def test_f64_passthrough_allowed():
    # x64-mode pattern: f64 scalar arg immediately converted to f32.
    def f(lo):
        return jax.numpy.float32(2.0) * lo.astype("float32")

    data = lower_x64(f, np.float64(1.5))
    interp = Interpreter(data)
    (out,) = interp(mdt.to_mx(np.float64(1.5)))
    assert float(out) == 3.0


@pytest.mark.skipif(mdt.F64_DOWNCAST, reason="strict mode only")
def test_f64_arithmetic_rejected_at_compile():
    def f(x):
        return x * x  # f64 multiply

    data = lower_x64(f, np.float64(1.5))
    with pytest.raises(UnsupportedDtypeError, match="float64"):
        Interpreter(data)


@pytest.mark.skipif(mdt.F64_DOWNCAST, reason="strict mode only")
def test_c128_passthrough_allowed():
    def f(z):
        return z.astype("complex64") * jax.numpy.complex64(2)

    data = lower_x64(f, np.complex128(1 + 2j))
    interp = Interpreter(data)
    (out,) = interp(mdt.to_mx(np.complex128(1 + 2j)))
    np.testing.assert_array_equal(mdt.to_np(out), np.complex64(2 + 4j))


@pytest.mark.skipif(mdt.F64_DOWNCAST, reason="strict mode only")
def test_c128_arithmetic_rejected_at_compile():
    def f(z):
        return z * z  # complex128 multiply

    data = lower_x64(f, np.complex128(1 + 2j))
    with pytest.raises(UnsupportedDtypeError, match="complex128"):
        Interpreter(data)


def test_c128_device_put_round_trip():
    """complex128 crosses the PJRT boundary and reports its own dtype
    (values narrowed to complex64 on the device, like f64 -> f32)."""
    want = np.array([1 + 2j, -3.5 - 4.25j], np.complex128)
    with jax.enable_x64():
        x = jax.device_put(want, jax.devices("metal")[0])
        assert x.dtype == np.complex128
        np.testing.assert_array_equal(np.asarray(x), want)