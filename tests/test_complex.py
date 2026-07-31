"""complex64 support, vs the CPU backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import check

rng = np.random.default_rng(9)
Z = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(np.complex64)


def test_complex_elementwise():
    check(lambda z: z * z + z, Z)
    check(lambda z: jnp.abs(z), Z, rtol=1e-5, atol=1e-6)
    check(lambda z: jnp.exp(z), Z, rtol=1e-5, atol=1e-5)
    check(lambda z: jnp.conj(z), Z)
    check(lambda z: jnp.angle(z), Z, rtol=1e-5, atol=1e-6)
    check(lambda z: jnp.where(jnp.real(z) > 0, z, -z), Z)


def test_complex_construct_and_parts():
    x = rng.standard_normal(6).astype(np.float32)
    check(lambda x: jax.lax.complex(x, 2 * x), x)
    check(lambda z: (jnp.real(z), jnp.imag(z)), Z)
    check(lambda z: z.astype(jnp.float32), Z)  # convert keeps real part


def test_complex_linalg_reduce():
    check(lambda z: z.reshape(2, 4) @ z.reshape(4, 2), Z,
          rtol=1e-5, atol=1e-5)
    check(lambda z: jnp.sum(z), Z, rtol=1e-5, atol=1e-6)
    check(lambda z: jnp.vdot(z, z), Z, rtol=1e-5, atol=1e-5)


def test_fft():
    x = rng.standard_normal(16).astype(np.float32)
    check(lambda x: jnp.fft.fft(x.astype(jnp.complex64)), x,
          rtol=1e-5, atol=1e-5)
    check(lambda x: jnp.fft.rfft(x), x, rtol=1e-5, atol=1e-5)
    check(lambda z: jnp.fft.irfft(z, n=8), np.fft.rfft(
        rng.standard_normal(8)).astype(np.complex64), rtol=1e-5, atol=1e-5)
    check(lambda z: jnp.fft.fft2(z.reshape(2, 4)), Z, rtol=1e-5, atol=1e-5)


def test_fft_unit_last_length():
    # MLX's rfftn/irfftn skip the transforms over the leading axes (and the
    # batching) when the last transform length is 1 — jax.numpy reaches that
    # via any s=(..., 1). The length-1 real transform is the identity on the
    # single DC bin, so only the leading axes actually transform.
    zr = (rng.standard_normal((4, 3, 1))
          + 1j * rng.standard_normal((4, 3, 1))).astype(np.complex64)
    check(lambda z: jnp.fft.irfftn(z, s=(3, 1), axes=(1, 2)), zr,
          rtol=1e-5, atol=1e-5)
    check(lambda z: jnp.fft.irfftn(z, s=(4, 3, 1), axes=(0, 1, 2)), zr,
          rtol=1e-5, atol=1e-5)
    check(lambda z: jnp.fft.irfftn(z, s=(1,), axes=(2,)), zr,
          rtol=1e-5, atol=1e-5)
    # jax issue #29325: outer axes must still transform.
    a = np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
                 dtype=np.complex64)
    check(lambda z: jnp.fft.irfftn(z, s=(5, 1), axes=(0, 1)), a,
          rtol=1e-5, atol=1e-5)
    xr = rng.standard_normal((4, 3, 1)).astype(np.float32)
    check(lambda x: jnp.fft.rfftn(x, s=(3, 1), axes=(1, 2)), xr,
          rtol=1e-5, atol=1e-5)
    check(lambda x: jnp.fft.rfftn(x, s=(4, 3, 1), axes=(0, 1, 2)), xr,
          rtol=1e-5, atol=1e-5)
    check(lambda x: jnp.fft.rfftn(x, s=(1,), axes=(2,)), xr,
          rtol=1e-5, atol=1e-5)


def test_fftn_many_axes_through_backend():
    """Transforms over >3 axes, through the REAL backend.

    jax lowers those to several stablehlo.fft ops separated by crops, pads
    and transposes. MLX's FFT kernels can start before a pending async
    evaluation of their input has landed, so the transform read the padded
    buffer mid-copy — wrong values, and only across execute boundaries, so
    the bare-Interpreter path in check() cannot see it. The eager call
    after a jit call is the one that used to break.
    """
    metal = jax.devices("metal")[0]
    shape = (2, 3, 4, 5, 6)
    x = (rng.standard_normal(shape)
         + 1j * rng.standard_normal(shape)).astype(np.complex64)
    cases = [
        lambda a: jnp.fft.ifftn(a, s=(1, 5, 6, 2, 7), axes=(0, 1, 2, 3, 4)),
        lambda a: jnp.fft.ifftn(a, s=(1, 3, 5, 11), axes=(0, 1, 2, 4),
                                norm="ortho"),
        lambda a: jnp.fft.fftn(a, axes=(0, 1, 2, 3, 4)),
        lambda a: jnp.fft.irfftn(a, s=(2, 2, 3, 1), axes=(0, 1, 3, 4)),
    ]
    for fn in cases:
        with jax.default_device(jax.devices("cpu")[0]):
            want = np.asarray(jax.jit(fn)(x))
        with jax.default_device(metal):
            cfn = jax.jit(fn)
            # interleave: a jit execute used to poison the next eager one
            for got in (fn(x), cfn(x), fn(x), fn(x), cfn(x), fn(x)):
                np.testing.assert_allclose(np.asarray(got), want,
                                           rtol=1e-4, atol=1e-4)


def test_fft_zero_size():
    # MLX rejects size-0 transforms; XLA returns the empty result.
    check(lambda z: jnp.fft.fft(z), np.zeros((0,), np.complex64))
    check(lambda z: jnp.fft.ifft(z), np.zeros((0,), np.complex64))
    check(lambda z: jnp.fft.fft2(z), np.zeros((0, 0), np.complex64))
    check(lambda z: jnp.fft.fftn(z), np.zeros((2, 0), np.complex64))
    check(lambda z: jnp.fft.irfft(z), np.zeros((0,), np.complex64))
    # zero-size batch dim with a non-empty transform axis
    check(lambda z: jnp.fft.fft(z), np.zeros((0, 4), np.complex64))
    check(lambda x: jnp.fft.rfft(x), np.zeros((0, 4), np.float32))
