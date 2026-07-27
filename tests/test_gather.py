import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from helpers import check

rng = np.random.default_rng(3)


def test_take_1d():
    x = rng.standard_normal(10).astype(np.float32)
    i = np.array([3, 0, 9, 3], np.int32)
    check(lambda x, i: x[i], x, i)


def test_take_rows():
    x = rng.standard_normal((7, 5)).astype(np.float32)
    i = np.array([[1, 2], [6, 0]], np.int32)
    check(lambda x, i: x[i], x, i)


def test_embedding_pattern():
    # texmo CE-loss pattern: (vocab, 1) table indexed by (B, T, 1) ids
    table = rng.standard_normal((2, 1)).astype(np.float32)
    ids = rng.integers(0, 2, (16, 63, 1)).astype(np.int32)
    check(lambda t, i: jnp.take(t, i[..., 0], axis=0), table, ids)


def test_take_along_axis():
    x = rng.standard_normal((4, 6)).astype(np.float32)
    i = rng.integers(0, 6, (4, 1)).astype(np.int32)
    check(lambda x, i: jnp.take_along_axis(x, i, axis=1), x, i)


def test_cross_entropy_integer_labels():
    logits = rng.standard_normal((8, 10)).astype(np.float32)
    labels = rng.integers(0, 10, (8,)).astype(np.int32)
    check(optax.softmax_cross_entropy_with_integer_labels, logits, labels,
          rtol=1e-5, atol=1e-6)


def test_vmapped_gather():
    x = rng.standard_normal((3, 7, 5)).astype(np.float32)
    i = rng.integers(0, 7, (3, 4)).astype(np.int32)
    check(jax.vmap(lambda xb, ib: xb[ib]), x, i)


def test_at_set_1d():
    x = rng.standard_normal(10).astype(np.float32)
    i = np.array([2, 7, 4], np.int32)
    v = np.array([1.0, 2.0, 3.0], np.float32)
    check(lambda x, i, v: x.at[i].set(v), x, i, v)


def test_at_set_rows():
    x = rng.standard_normal((6, 4)).astype(np.float32)
    i = np.array([5, 0], np.int32)
    v = rng.standard_normal((2, 4)).astype(np.float32)
    check(lambda x, i, v: x.at[i].set(v), x, i, v)


def test_at_set_scalar():
    x = rng.standard_normal(8).astype(np.float32)
    i = np.array([1, 6], np.int32)
    check(lambda x, i: x.at[i].set(0.0), x, i)


@pytest.mark.parametrize("method", ["add", "multiply", "max", "min"])
def test_at_methods(method):
    x = rng.standard_normal(10).astype(np.float32)
    i = np.array([3, 8, 0], np.int32)
    v = np.array([0.5, -1.5, 2.0], np.float32)
    check(lambda x, i, v: getattr(x.at[i], method)(v), x, i, v)


def test_at_set_inside_scan():
    # Exercises scatter-set under an mx.compile trace (compiled loop body).
    x = rng.standard_normal(8).astype(np.float32)
    idxs = np.array([1, 6, 3, 6], np.int32)

    def f(x, idxs):
        def body(c, i):
            return c.at[i].set(-1.0), c[i]
        return jax.lax.scan(body, x, idxs)
    check(f, x, idxs)


def test_one_hot_indexing_grad():
    table = rng.standard_normal((11, 4)).astype(np.float32)
    ids = rng.integers(0, 11, (6,)).astype(np.int32)

    def loss(t):
        return jnp.sum(t[ids] ** 2)
    check(jax.grad(loss), table, rtol=1e-5, atol=1e-6)

def test_scatter_oob_dropped():
    # XLA semantics: out-of-bounds scatter updates are DROPPED, not
    # clamped. jnp.nonzero/where pad with fill_value == size (clamps onto
    # the last real slot), bincount overflows, place fills.
    i_oob = np.array([2, 5], np.int32)
    v = np.array([8.0, 9.0], np.float32)
    check(lambda x, i, v: x.at[i].set(v, mode="drop"), np.zeros(3, np.float32),
          i_oob, v)
    check(lambda x, i, v: x.at[i].add(v, mode="drop"), np.zeros(3, np.float32),
          i_oob, v)
    check(lambda x, i, v: x.at[i].max(v, mode="drop"), np.zeros(3, np.float32),
          i_oob, v)
    check(lambda i: jnp.bincount(i, length=3), np.array([0, 1, 9], np.int32))
    check(lambda m: jnp.nonzero(m, size=3, fill_value=3)[0],
          np.array([0.0, 1.0, 0.0, 1.0, 1.0], np.float32))
    check(lambda a, m, v: jnp.place(a, m, v, inplace=False),
          np.zeros(3, np.float32), np.array([True, False, True]),
          np.array([7.0, 8.0], np.float32))


def test_scatter_windowed_indexed():
    # Windows on indexed dims (dynamic-update-slice-style scatters,
    # jnp.unique's mask scatter, polyadd's prefix write).
    check(lambda x, v: x.at[2:5].set(v), np.zeros(6, np.float32),
          np.array([1.0, 2.0, 3.0], np.float32))
    check(lambda x, v: x.at[1:4].add(v), np.zeros(6, np.float32),
          np.array([1.0, 1.0, 1.0], np.float32))
    check(lambda x, r: x.at[1].set(r), np.zeros((3, 4), np.float32),
          np.arange(4, dtype=np.float32))
    check(lambda x: jnp.unique(x, size=3), np.array([3, 1, 3, 2], np.int32))
    check(lambda a, b: jnp.polyadd(a, b), np.array([1.0, 2.0], np.float32),
          np.array([1.0, 2.0, 3.0], np.float32))
    check(lambda x: jnp.pad(x, 1, mode="linear_ramp"),
          np.array([1.0, 2.0], np.float32))
