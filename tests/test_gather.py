from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from helpers import check, run_metal

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


def test_gather_oob_modes():
    # jnp.take's DEFAULT out-of-bounds policy is "fill" (NaN for floats,
    # not a clamp) -- this is what turns a stale lookup table into NaN
    # rather than into the table's last entry, and CPU does exactly the
    # same. Diagnosing the SD3.5 all-zero image depended on it, so pin all
    # three modes against CPU.
    t = np.array([1.0, 0.75, 0.5, 0.25, 0.0], np.float32)
    i = np.array([0, 4, 5, 7, 20], np.int32)
    check(lambda t, i: jnp.take(t, i), t, i)                    # fill (NaN)
    check(lambda t, i: jnp.take(t, i, mode="clip"), t, i)
    check(lambda t, i: jnp.take(t, i, mode="fill", fill_value=-1.0), t, i)
    check(lambda t, i: jnp.take(t, i), t.astype(np.int32), i)   # int fill


def test_gather_oob_in_counted_loop():
    # The SD3.5 sampler shape, minimized: a counted loop whose body looks
    # up a schedule table that is SHORTER than the trip count (keras-hub
    # bakes `scheduler.sigmas` as a jit constant while `num_steps` stays
    # traced, so a sampler compiled for 4 steps and replayed for 20 reads
    # past the end). Every sigma from index 5 on is NaN, and the NaN has
    # to propagate through the carry exactly as it does on CPU -- our
    # counted-loop/msl paths must not "helpfully" clamp or drop it.
    t = np.array([1.0, 0.75, 0.5, 0.25, 0.0], np.float32)

    def sampler(t, n):
        def body(i, carry):
            sigma = jnp.take(t, jnp.expand_dims(i, 0))
            nxt = jnp.take(t, jnp.expand_dims(i + 1, 0))
            return carry + (nxt - sigma) * carry

        return jax.lax.fori_loop(0, n, body, jnp.ones((1,), np.float32))

    check(sampler, t, np.int32(4))    # in bounds -> finite
    check(sampler, t, np.int32(20))   # past the end -> NaN, like CPU


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


def test_scatter_apply_windowed_duplicates():
    # scatter_apply: computed body, unique_indices=false, and a window that
    # lands on an indexed dim. Duplicate starts apply the body repeatedly,
    # in update order; XLA CPU (via check) is the reference for that order.
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,))
    check(partial(jax.lax.scatter_apply, func=jnp.sin, update_shape=(3, 2),
                  dimension_numbers=dn),
          np.arange(10, dtype=np.float32), np.array([[0], [0], [4]], np.int32))
    # Overlapping (not identical) windows: slot 1 gets the body twice.
    check(partial(jax.lax.scatter_apply, func=jnp.sin, update_shape=(3, 2),
                  dimension_numbers=dn),
          np.arange(10, dtype=np.float32), np.array([[0], [1], [7]], np.int32))
    # Out-of-bounds start (limit = 10 - 2): the WHOLE window is dropped.
    check(partial(jax.lax.scatter_apply, func=jnp.sin, update_shape=(3, 2),
                  dimension_numbers=dn),
          np.arange(10, dtype=np.float32), np.array([[9], [0], [0]], np.int32))

    # Partial window on a free dim + an inserted (size-1 window) index dim.
    dn2 = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))
    check(partial(jax.lax.scatter_apply, func=jnp.sin, update_shape=(3, 3),
                  dimension_numbers=dn2),
          rng.standard_normal((10, 5)).astype(np.float32),
          np.array([[2], [2], [7]], np.int32))

    # E > 0 together with a non-empty free dim (full window on dim 0,
    # 2-wide indexed window on dim 1) -- the vmapped-scatter shape.
    dn3 = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(1,))
    check(partial(jax.lax.scatter_apply, func=jnp.cos, update_shape=(3, 5, 2),
                  dimension_numbers=dn3),
          rng.standard_normal((5, 10)).astype(np.float32),
          np.array([[0], [0], [8]], np.int32))

    # operand/indices batching dims + window, with a dropped OOB start.
    dn4 = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(2,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
        scatter_indices_batching_dims=(1, 0))
    check(partial(jax.lax.scatter_apply, func=jnp.sin, update_shape=(3, 2, 3),
                  dimension_numbers=dn4),
          rng.standard_normal((2, 3, 10)).astype(np.float32),
          np.array([[[0], [1]], [[2], [3]], [[8], [9]]], np.int32))


def test_scatter_complex_preserves_specials():
    # A complex scatter is run on the real/imag parts and recombined. The
    # recombination must be a bit layout, not arithmetic: re + im * 1j
    # turns a finite/inf pair into nan (inf * 1j == nan + infj) and loses
    # the sign of -0 -- corrupting elements the scatter never touched.
    x = np.array([complex(np.inf, 1.0),    # inf in the real part
                  complex(-0.0, -0.0),     # -0 in both parts
                  complex(1.0, np.inf),    # inf in the imag part
                  complex(np.nan, -0.0),
                  complex(3.0, 4.0)], np.complex64)
    i = np.array([4], np.int32)
    v = np.array([7.0 - 8.0j], np.complex64)

    def f(x, i, v):
        return x.at[i].set(v)

    got = run_metal(f, x, i, v)[0]
    want = np.asarray(jax.jit(f)(x, i, v))
    assert got.dtype == want.dtype
    for name, g, w in (("real", got.real, want.real),
                       ("imag", got.imag, want.imag)):
        np.testing.assert_array_equal(np.isnan(g), np.isnan(w), err_msg=name)
        m = ~np.isnan(w)
        np.testing.assert_array_equal(g[m], w[m], err_msg=name)
        np.testing.assert_array_equal(np.signbit(g), np.signbit(w),
                                      err_msg=name)
