"""Executes driven from more than one thread.

jax hands a jitted call to whatever thread the caller is on (its own
pjit_test.test_concurrent_pjit uses a ThreadPoolExecutor), and buffers move
between threads freely: made on one, consumed on another, read back on a
third after the producer has exited. Two things in this backend make that
delicate.

* MLX's default stream is thread-BOUND, and everything leaving execute is
  LAZY, so a value carries the stream of the thread that built it. Evaluate
  it elsewhere and MLX raises "There is no Stream(gpu, 0) in current
  thread" -- or, once the owning thread is gone, throws where nothing
  catches and the process aborts with no traceback. engine.bind_thread
  gives each thread a cross-thread-evaluable stream of its own instead, and
  the engine settles everything it hands out, so a consumer waits on an
  event rather than encoding on another thread's stream (several of them
  doing that at once is a Metal abort of its own).
* Concurrent compiles walk jaxlib's DebugOptions table, whose owner must be
  held for the walk -- a dangling read that single-threaded got away with.

The driver below is deliberately small (8 threads, tiny arrays): what is
under test is the threading discipline, not the arithmetic. It runs both in
this process, under whatever engine the suite is configured for, and in
subprocesses pinned to each engine -- the native tape has its own buffer
path, its own run loop and its own per-program lock, and only a subprocess
can select it (METALJAX_ENGINE is read at import).
"""

import concurrent.futures
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

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
            # Same executable from every thread (the native tape's caches
            # are per-program state a concurrent run must not corrupt), and
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


def test_engine_stream_outlives_its_thread():
    """A value built on a worker stays evaluable after the worker dies.

    The direct statement of what bind_thread buys, with no async_eval to
    make it a race: MLX's own default stream is thread-bound, and this
    raises there -- from a destructor-ish context where nothing catches it,
    so the process aborts rather than failing.
    """
    import mlx.core as mx

    from metaljax import engine

    if not engine._HAVE_TU_STREAM:
        pytest.skip("mlx without new_thread_unsafe_stream")
    box = {}

    def produce():
        engine.bind_thread()
        box["v"] = mx.array([1.0, 2.0, 3.0]) * 3.0  # never evaluated here

    t = threading.Thread(target=produce)
    t.start()
    t.join()
    np.testing.assert_allclose(np.asarray(box["v"]), [3.0, 6.0, 9.0])


def test_engine_streams_are_one_per_thread():
    """Threads must NOT share a stream.

    "Thread unsafe" is about concurrent use, and Metal does not treat it
    gently: two threads encoding into one stream's command buffer trips
    `A command encoder is already encoding to this command buffer` and
    aborts. Distinct streams are what keeps concurrent executes apart.
    """
    import mlx.core as mx

    from metaljax import engine

    if not engine._HAVE_TU_STREAM:
        pytest.skip("mlx without new_thread_unsafe_stream")
    seen = []
    lock = threading.Lock()

    def record():
        engine.bind_thread()
        s = mx.default_stream(mx.default_device())
        with lock:
            seen.append(repr(s))

    ts = [threading.Thread(target=record) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(set(seen)) == len(seen), seen


def test_compile_option_table_does_not_read_freed_memory():
    """The DebugOptions walk must hold the CompileOptions that owns it.

    It used to read the table off a temporary that died at the end of the
    expression, so every value after that came out of freed memory --
    harmless until something else claimed the block, at which point it was
    a segfault inside jaxlib's string getters. That is how it showed up:
    ~25% of runs of jax's own pjit_test.test_concurrent_pjit, where ten
    threads compile at once, with the crash reported against a line that
    merely read an option.

    Waiting for a thread to lose that race is no way to test it. macOS's
    allocator will fill freed blocks on request (MallocScribble), which
    turns the dangling read into a certain crash -- and needs a subprocess,
    both for the environment variable and because the answer is a signal.
    """
    env = dict(os.environ, MallocScribble="1", MallocPreScribble="1",
               PYTHONPATH=os.pathsep.join(
                   [os.path.join(ROOT, "src"),
                    os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run(
        [sys.executable, "-c",
         "from metaljax import compile_options as c;"
         " t = c._types();"
         " assert t['xla_embed_ir_in_executable'] is bool, t;"
         " print('options', len(t))"],
        capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, (
        f"exited {r.returncode} (139 = the use-after-free)\n"
        f"{r.stdout}\n{r.stderr}")
    assert "options" in r.stdout, r.stdout


def test_compile_option_table_is_thread_safe():
    """...and building it from several threads at once must be clean."""
    from metaljax import compile_options as copts

    copts._option_types = None  # force the first build into the threads
    errors = []

    def build(_):
        try:
            assert "xla_embed_ir_in_executable" in copts._types()
        except BaseException as e:  # noqa: BLE001
            errors.append(repr(e))

    with concurrent.futures.ThreadPoolExecutor(NTHREADS) as pool:
        list(pool.map(build, range(NTHREADS * 4)))
    assert not errors, errors


# --- both engines ------------------------------------------------------

_DRIVER = """
import sys
sys.path.insert(0, {root!r})
import tests.test_concurrency as t
for _ in range(3):
    assert t._run_threaded() == t.NTHREADS
print('threaded ok')
"""


@pytest.mark.parametrize("engine", ["py", "native"])
def test_threaded_executes_both_engines(engine):
    """Same driver in a subprocess per engine.

    A crash is the failure mode that matters most here (an aborting
    interpreter cannot report anything), so this asserts on the return code
    as much as on the output.
    """
    if engine == "native":
        # The subprocess gets native/build on its PYTHONPATH either way;
        # look there for the extension too, so this runs whenever it has
        # been built rather than only under a native-configured suite.
        sys.path.insert(0, os.path.join(ROOT, "native", "build"))
        pytest.importorskip("metaljax_native",
                            reason="native engine not built (native/build.sh)")
    env = dict(os.environ, METALJAX_ENGINE=engine,
               JAX_PLATFORMS="metal,cpu",
               PYTHONPATH=os.pathsep.join(
                   [os.path.join(ROOT, "src"),
                    os.path.join(ROOT, "native", "build"),
                    os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", _DRIVER.format(root=ROOT)],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, (
        f"engine={engine} exited {r.returncode}\n{r.stdout}\n{r.stderr}")
    assert "threaded ok" in r.stdout, r.stdout
