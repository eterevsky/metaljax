"""stablehlo.sort via the comparator recognizer, vs the CPU backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import check

rng = np.random.default_rng(7)


def test_sort_float_total_order():
    x = np.array([3.0, -0.0, 1.0, np.nan, 0.0, -2.0, np.inf, -np.inf],
                 np.float32)
    check(lambda x: jnp.sort(x), x)
    check(lambda x: jnp.sort(x, descending=True), x)


@pytest.mark.parametrize("dt", [np.int32, np.uint32, np.int8, np.float16])
def test_sort_dtypes(dt):
    x = (rng.standard_normal(50) * 100).astype(dt)
    check(lambda x: jnp.sort(x), x)


def test_argsort_stable_ties():
    x = np.tile(np.array([3, 1, 2, 1], np.int32), 50)
    check(lambda x: jnp.argsort(x), x)


def test_sort_axis_and_kv():
    x = rng.standard_normal((4, 6)).astype(np.float32)
    check(lambda x: jnp.sort(x, axis=0), x)
    k = np.array([2, 1, 2], np.int32)
    v = np.array([10, 20, 30], np.int32)
    check(lambda k, v: jax.lax.sort((k, v), num_keys=1), k, v)


def test_sort_consumers():
    x = rng.standard_normal(9).astype(np.float32)
    check(lambda x: jnp.median(x), x)
    check(lambda x: jnp.partition(x, 3), x)
    check(lambda x: jnp.percentile(x, 50), x)
    check(lambda x: jax.lax.top_k(x, 3), x)


def test_searchsorted_nan_total_order():
    a = np.array([1.0, 2.0, np.nan], np.float32)
    v = np.array([2.0, np.nan, 3.0], np.float32)
    check(lambda a, v: jnp.searchsorted(a, v), a, v)


def test_sort_complex_lexicographic():
    z = np.array([3 + 1j, 1 + 5j, 1 + 2j, 3 - 1j], np.complex64)
    check(lambda z: jnp.sort(z), z)


def test_approx_top_k_no_aggregate():
    # aggregate_to_topk=False asks for a wider (unsorted, approximate)
    # result than backend_config's top_k; filling only top_k left the
    # rest of the output buffer uninitialised.
    x = rng.standard_normal((3, 8, 32)).astype(np.float32)
    check(lambda a: jax.lax.approx_max_k(a, k=4, reduction_dimension=1,
                                         aggregate_to_topk=False), x)
