"""Strict-f64 policy: buffers pass through, f64 PROGRAMS are declined.

Metal has no doubles.  The plugin's policy has two halves:

* **buffers** cross the PJRT boundary in their own dtype (stored as f32 /
  complex64 on the device, widened back on egress), so `device_put` +
  read-back of an f64 or complex128 array works and reports f64/c128;
* **programs** that so much as mention an f64 element type are declined at
  compile with `UNIMPLEMENTED: metaljax-native: element type f64` -- the
  tape has no dtype code for it.  That includes a *convert* out of f64, so
  the x64-mode "pass an f64 scalar and immediately cast it to f32" shape is
  declined too.

That last point is a deliberate narrowing versus the retired Stage-1
engine, which executed such converts (`notes/cpp-p11-dtypes.md` and the
decline censuses list `element type f64` as an intentional decline).  It
shipped with 0.11.5; the tests below pin the behavior as it is, and the
Stage-1 retirement report names the change.

NB the c128 decline currently spells the type `<unknown>` rather than
complex128 -- cosmetic, also named in that report.
"""

import io
import os

import jax
import numpy as np
import pytest

from helpers import run_module

_DOWNCAST = os.environ.get("METALJAX_F64", "") == "downcast"


def lower_x64(f, *args):
    with jax.enable_x64():
        lowered = jax.jit(f).lower(*args)
        buf = io.BytesIO()
        lowered.compiler_ir().operation.write_bytecode(buf)
        return buf.getvalue()


@pytest.mark.skipif(_DOWNCAST, reason="strict mode only")
def test_f64_convert_is_declined():
    # x64-mode pattern: f64 scalar arg immediately converted to f32. The
    # program still NAMES f64, so the tape declines it.
    def f(lo):
        return jax.numpy.float32(2.0) * lo.astype("float32")

    data = lower_x64(f, np.float64(1.5))
    with pytest.raises(Exception, match=r"element type f64"):
        run_module(data, (np.float64(1.5),))


@pytest.mark.skipif(_DOWNCAST, reason="strict mode only")
def test_f64_arithmetic_rejected_at_compile():
    def f(x):
        return x * x  # f64 multiply

    data = lower_x64(f, np.float64(1.5))
    with pytest.raises(Exception, match=r"element type f64"):
        run_module(data, (np.float64(1.5),))


@pytest.mark.skipif(_DOWNCAST, reason="strict mode only")
def test_f64_device_put_round_trip():
    """The buffer half of the policy: f64 crosses and reports f64 (values
    held as f32 on the device, so they round once)."""
    want = np.array([1.5, -0.25, 1e10], np.float64)
    with jax.enable_x64():
        x = jax.device_put(want, jax.devices("metal")[0])
        assert x.dtype == np.float64
        np.testing.assert_array_equal(np.asarray(x), want)


@pytest.mark.skipif(_DOWNCAST, reason="strict mode only")
def test_c128_arithmetic_rejected_at_compile():
    def f(z):
        return z * z  # complex128 multiply

    data = lower_x64(f, np.complex128(1 + 2j))
    with pytest.raises(Exception, match=r"element type"):
        run_module(data, (np.complex128(1 + 2j),))


def test_c128_device_put_round_trip():
    """complex128 crosses the PJRT boundary and reports its own dtype
    (values narrowed to complex64 on the device, like f64 -> f32)."""
    want = np.array([1 + 2j, -3.5 - 4.25j], np.complex128)
    with jax.enable_x64():
        x = jax.device_put(want, jax.devices("metal")[0])
        assert x.dtype == np.complex128
        np.testing.assert_array_equal(np.asarray(x), want)
