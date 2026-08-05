"""Heap reclamation at compile boundaries must cost a bounded fraction.

`engine.reclaim` collects Python cycles before clearing MLX's buffer cache,
because a dead manager sitting in a reference cycle pins buffers that
`mx.clear_cache` cannot free (CLAUDE.md item 19). But `gc.collect` walks the
whole live heap, so it costs what the PROCESS holds -- and a compile boundary
is not evidence that anything died, it is just the cheapest place to notice.
Collecting at every boundary made programs-per-second workloads quadratic:
jax's sparse_bcoo_bcsr_test compiles one program per eager primitive (2,574
compiles in its first 60 tests) and spent 83% of its wall time inside
gc.collect, growing with the heap until the file blew the 3600 s release-gate
timeout.
"""

import types

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from metaljax import engine


class _FakeGC:
    """A gc module whose collect costs `cost` simulated seconds."""

    def __init__(self, clock, cost):
        self.clock = clock
        self.cost = cost
        self.calls = 0

    def collect(self, *args, **kwargs):
        self.calls += 1
        self.clock.t += self.cost
        return 0


class _Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


@pytest.fixture
def fake_heap(monkeypatch):
    """engine with a controlled clock and a collection of known cost."""
    clock = _Clock()
    gc = _FakeGC(clock, cost=0.1)
    monkeypatch.setattr(engine, "gc", gc)
    monkeypatch.setattr(engine, "time", types.SimpleNamespace(
        monotonic=clock.monotonic))
    monkeypatch.setattr(engine, "_gc_deadline", 0.0, raising=False)
    monkeypatch.setattr(engine, "GC_STATS",
                        {"collections": 0, "skipped": 0, "seconds": 0.0})
    return clock, gc


def test_reclaim_duty_cycle_bounds_the_cost(fake_heap):
    """Boundaries arriving faster than the duty cycle allows are deferred."""
    clock, gc = fake_heap
    work_per_call = 0.01          # what the caller does between boundaries
    for _ in range(1000):
        clock.t += work_per_call
        engine.reclaim()
    # 10 s of work + collections, one collection allowed per 50x its own
    # cost (5 s here): a handful, not a thousand.
    assert gc.calls <= 5, gc.calls
    assert engine.GC_STATS["collections"] == gc.calls
    assert engine.GC_STATS["skipped"] == 1000 - gc.calls
    # ...and that is the point: the time spent collecting stays a small
    # fraction of the time the process spent doing anything at all.
    assert engine.GC_STATS["seconds"] <= clock.t / 25, (
        engine.GC_STATS["seconds"], clock.t)


def test_reclaim_force_always_collects(fake_heap):
    """Recovery paths run because memory ran out: they never defer."""
    _clock, gc = fake_heap
    engine.reclaim()                      # arms the deadline
    assert gc.calls == 1
    for _ in range(10):
        engine.reclaim(force=True)
    assert gc.calls == 11
    assert engine.GC_STATS["skipped"] == 0


def test_reclaim_duty_zero_collects_every_time(fake_heap, monkeypatch):
    """METALJAX_COMPILE_GC_DUTY=0 restores the old every-boundary behaviour."""
    _clock, gc = fake_heap
    monkeypatch.setattr(engine, "_GC_DUTY", 0.0)
    for _ in range(10):
        engine.reclaim()
    assert gc.calls == 10
    assert engine.GC_STATS["skipped"] == 0


def test_compile_boundaries_do_not_collect_one_for_one():
    """End to end: N compiles in a row must not mean N full collections."""
    n = 40
    before = dict(engine.GC_STATS)
    metal = jax.devices("metal")[0]
    with jax.default_device(metal):
        for i in range(n):
            # A fresh callable of a fresh shape each time: no jit cache hit,
            # so each iteration reaches engine.compile_program.
            f = jax.jit(lambda x, k=i: x * (k + 1))
            out = np.asarray(f(jnp.arange(i + 1, dtype=jnp.float32)))
            assert out.shape == (i + 1,)
    calls = ((engine.GC_STATS["collections"] - before["collections"])
             + (engine.GC_STATS["skipped"] - before["skipped"]))
    collected = engine.GC_STATS["collections"] - before["collections"]
    assert calls >= n, f"{calls} reclamation points for {n} compiles"
    assert collected <= n // 4, f"{collected} collections for {n} compiles"
