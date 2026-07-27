import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import check

X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def test_broadcast_binop():
    check(jnp.add, X, np.float32(10.0))
    check(jnp.add, X, np.arange(4, dtype=np.float32))
    check(jnp.add, X, np.arange(3, dtype=np.float32).reshape(3, 1))


def test_broadcast_to():
    check(lambda x: jnp.broadcast_to(x, (5, 2, 3, 4)), X)


def test_reshape():
    check(lambda x: x.reshape(4, 6), X)
    check(lambda x: x.reshape(-1), X)


def test_transpose():
    check(lambda x: x.T, X)
    check(lambda x: jnp.transpose(x, (2, 0, 1)), X)


def test_expand_squeeze():
    check(lambda x: jnp.expand_dims(x, 1)[:, 0, ...], X)


def test_concatenate_stack():
    check(lambda a, b: jnp.concatenate([a, b], axis=1), X, X)
    check(lambda a, b: jnp.stack([a, b], axis=0), X, X)


def test_slice():
    check(lambda x: x[1, 0:3:2, 1:4], X)
    check(lambda x: x[:, -1, :], X)


def test_flip():
    check(lambda x: jnp.flip(x, axis=2), X)
    check(lambda x: jnp.flip(x), X)


def test_arange_iota():
    check(lambda: jnp.arange(10, dtype=jnp.float32))
    check(lambda: jnp.arange(5))


def test_pad_constant():
    check(lambda x: jnp.pad(x, ((1, 2), (0, 1), (2, 0)), constant_values=7.0), X)


def test_pad_interior():
    f = lambda x: jax.lax.pad(x, jnp.float32(-1.0), ((1, 1, 2), (0, 2, 1)))
    check(f, np.arange(6, dtype=np.float32).reshape(2, 3))


def test_dynamic_slice():
    f = lambda x, i: jax.lax.dynamic_slice(x, (i, 0, 1), (1, 2, 2))
    check(f, X, np.int32(1))


def test_dynamic_update_slice():
    f = lambda x, u, i: jax.lax.dynamic_update_slice(x, u, (0, i, 0))
    check(f, X, np.ones((2, 1, 4), np.float32), np.int32(2))


@pytest.mark.parametrize("src,dst", [
    (np.float32, np.int32), (np.int32, np.float32), (np.float32, np.bool_),
    (np.bool_, np.float32), (np.float32, np.float16), (np.int32, np.uint8),
])
def test_convert(src, dst):
    x = np.array([-2.7, -1.0, 0.0, 1.0, 2.7]).astype(src)
    check(lambda a: a.astype(dst), x)


def test_bitcast():
    x = np.array([1.0, -2.5, 3.14], np.float32)
    check(lambda a: jax.lax.bitcast_convert_type(a, jnp.uint32), x)


def test_one_hot():
    check(lambda i: jax.nn.one_hot(i, 8), np.array([0, 3, 7, 2], np.int32))


def test_bitcast_convert_size_changing():
    # Size-changing bitcasts add/remove a trailing ratio dim (StableHLO);
    # jnp.byteswap exercises narrow->reverse->widen incl. rank-0.
    check(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint8),
          np.arange(4, dtype=np.uint32))
    check(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32),
          np.arange(8, dtype=np.uint8).reshape(2, 4))
    check(lambda x: x.byteswap(), np.array([1, 256], np.int32))
    check(lambda x: x.byteswap(), np.int32(7))


def test_reverse_degenerate_dims():
    check(lambda x: jnp.flip(x, axis=1), np.zeros((2, 0), np.float32))
    check(lambda x: jnp.flip(x, axis=0), np.ones((1, 3), np.float32))


def test_empty_gather_result():
    # jnp folds empty takes at trace time; go through lax.gather directly
    # so the interpreter actually sees an empty-batch gather.
    dn = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,),
        start_index_map=(0,))
    check(lambda x, i: jax.lax.gather(x, i, dn, slice_sizes=(1, 2)),
          np.arange(8, dtype=np.float32).reshape(4, 2),
          np.zeros((0, 1), np.int32))
