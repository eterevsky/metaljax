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
