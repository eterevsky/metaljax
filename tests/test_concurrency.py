"""Executes driven from more than one thread.

jax hands a jitted call to whatever thread the caller is on (its own
pjit_test.test_concurrent_pjit uses a ThreadPoolExecutor), and buffers move
between threads freely: made on one, consumed on another, read back on a
third after the producer has exited.  MLX's default stream is thread-bound
and everything leaving execute is lazy, so the plugin must hand out values
that stay evaluable across threads (its own streams, settled results) and
its per-program caches must survive a compile storm.  The driver below is
deliberately small (8 threads, tiny arrays): what is under test is the
threading discipline, not the arithmetic.

The Stage-1 tests of `engine.bind_thread` and the Python
`compile_options` table died with the Stage-1 retirement; the plugin-side
equivalents are exercised here through the real PJRT route (and in
plugin-native's own runtime_gil_free_test).
"""

import concurrent.futures
import os
import subprocess
import sys

import numpy as np

import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTHREADS = 8


def _metal():
    return jax.devices("metal")[0]


def _run_threaded():
    """Exercise every thread-crossing shape a jit call has. Raises on any
    mismatch; returns the number of results checked."""
    with jax.default_device(_metal()):
        # Stage the inputs HERE, with device_put, and leave them alone.
        # That path is the sharp one: the staging program forwards its
        # argument, so its result is execute's no-alias copy -- which used
        # to be built after the run's own async_eval, and was then the one
        # value leaving unscheduled. A worker consuming it therefore had to
        # walk into this thread's graph, and failed deterministically with
        # "There is no Stream(gpu, 0) in current thread". The i == 0
        # element is empty on top of that: the native buffer path answers
        # an empty request with a lazy fill of its own.
        xs = [jnp.asarray(np.arange(i, dtype=np.float32))
              for i in range(NTHREADS)]

        shared = jax.jit(lambda x, k: x * k + 1.0, static_argnums=1)
        private = [jax.jit(lambda x, k: (x + k) * 0.5, static_argnums=1)
                   for _ in range(NTHREADS)]

        def work(i):
            # Same executable from every thread (the tape's caches are
            # per-program state a concurrent run must not corrupt), and
            # one nobody else touches. Neither is compiled yet: the compile
            # storm is itself part of what is under test.
            a = shared(xs[i], float(i))
            b = private[i](a, float(i))
            return b  # still lazy: the main thread reads it after we exit

        with concurrent.futures.ThreadPoolExecutor(NTHREADS) as pool:
            outs = list(pool.map(work, range(NTHREADS)))

        # Read the workers' results here, after the pool has shut down and
        # the threads that produced them are gone.
        for i, got in enumerate(outs):
            x = np.arange(i, dtype=np.float32)
            want = (x * i + 1.0 + i) * 0.5
            np.testing.assert_allclose(np.asarray(got), want,
                                       rtol=1e-6, atol=1e-6)
        return len(outs)


def test_threaded_executes():
    assert _run_threaded() == NTHREADS


def test_threaded_executes_repeated():
    """Repeat: the failures this guards are races, and a single pass of a
    race is not evidence. Cheap enough to mean something (8 threads x 5)."""
    for _ in range(5):
        assert _run_threaded() == NTHREADS


# --- subprocess ---------------------------------------------------------

_DRIVER = """
import sys
sys.path.insert(0, {root!r})
import tests.test_concurrency as t
for _ in range(3):
    assert t._run_threaded() == t.NTHREADS
print('threaded ok')
"""


def test_threaded_executes_subprocess():
    """Same driver in a fresh process.

    A crash is the failure mode that matters most here (an aborting
    interpreter cannot report anything), so this asserts on the return code
    as much as on the output.
    """
    env = dict(os.environ, JAX_PLATFORMS="metal,cpu",
               PYTHONPATH=os.pathsep.join(
                   [os.path.join(ROOT, "src"),
                    os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", _DRIVER.format(root=ROOT)],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, (
        f"exited {r.returncode}\n{r.stdout}\n{r.stderr}")
    assert "threaded ok" in r.stdout, r.stdout
