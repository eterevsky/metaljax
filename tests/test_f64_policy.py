"""Strict-f64 policy: pass-through is allowed, f64 arithmetic fails."""

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