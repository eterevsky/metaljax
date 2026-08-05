"""Stage 2 M0: the native-engine boundary contract.

These tests only run when the extension has been built
(native/build.sh); they pin the three facts every later milestone
stands on: the version handshake refuses skew, mx.array crosses the
Python<->C++ boundary by shared handle (NB_DOMAIN=mlx — see
native/build.sh for the flag archaeology), and native compute runs on
the same GPU device the Python side uses.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "native", "build")

native = pytest.importorskip(
    "metaljax_native",
    reason="native engine not built (native/build.sh)")
sys.path.insert(0, BUILD)  # no-op when already importable

import mlx.core as mx  # noqa: E402


def test_version_handshake():
    """Compiled-against, linked, and imported mlx must all agree."""
    assert (native.compiled_mlx_version()
            == native.linked_mlx_version()
            == mx.__version__)


def test_array_crosses_by_handle():
    """mx.array passes through nanobind's shared mlx domain, both ways."""
    a = mx.array([1, 2, 3])
    out = native.roundtrip_double(a)
    assert isinstance(out, mx.array)
    np.testing.assert_array_equal(np.array(out), [2, 4, 6])


def test_native_default_device_is_gpu():
    assert native.default_device() == "gpu"


def test_engine_flag_import_contract():
    """METALJAX_ENGINE=native imports and handshakes in a fresh process.

    Subprocess: the flag is read at metaljax.engine import time, and this
    suite runs with the default py engine.
    """
    env = dict(os.environ, METALJAX_ENGINE="native",
               PYTHONPATH=os.pathsep.join(
                   [os.path.join(ROOT, "src"), BUILD,
                    os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run(
        [sys.executable, "-c",
         "from metaljax import engine; assert engine.NATIVE is not None; "
         "print('native engine loaded', engine.NATIVE.compiled_mlx_version())"],
        capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "native engine loaded" in r.stdout
