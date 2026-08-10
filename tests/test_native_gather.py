"""Stage 2: stablehlo.gather / stablehlo.scatter on the native tape.

The two engines compute these by DIFFERENT compositions, which is the whole
point of the differential here. The Python handlers (ops/gather.py) reach
the values through numpy-style advanced indexing: every window on an indexed
dim expands into an explicit arange axis, free dims are pre-sliced, and the
operand is transposed so the indexed dims lead. The native tape goes
straight to `mx::gather` / `mx::scatter*`, whose slice starts at the index on
an indexed axis and at 0 elsewhere — XLA's window rule, natively — so no
window is ever expanded and no operand is ever transposed.

Two compositions, one set of bits. A gather MOVES data rather than computing
on it, so its differential is exact by construction; a scatter's is exact
wherever the updates land on distinct slots. Where they do NOT (duplicate
indices under an arithmetic combiner) MLX's scatter is atomic and its order
is a race on the GPU — the same race the Python engine runs, and run-to-run
nondeterministic on both — so those cases are pinned with integer updates,
where the order cannot change the sum.

The cases are jax-lowered rather than hand-written: the dimension numbers
that matter (batching dims, collapsed offsets, interleaved offset_dims,
windows on indexed dims) are the ones jax's indexing, vmap and AD actually
emit, and a hand-written module would only prove the tape reads its own
attrs back.
"""
import numpy as np
import pytest

from test_native_tape import (  # noqa: F401  (imports gate the native build)
    _f32,
    _mod,
    _native_engine,
    check,
    engine,
    native,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from helpers import lower_bytes  # noqa: E402

from metaljax import tape  # noqa: E402

OPS = native.opcodes()
GATHER = OPS["stablehlo.gather"]
SCATTER = OPS["stablehlo.scatter"]

RNG = np.random.default_rng(20260810)


def _diff(f, *args, lowered=True, compiled=False):
    """Both engines on a jitted function; returns the native executable."""
    mod = lower_bytes(f, *args)
    flat = [np.asarray(x) for x in jax.tree.leaves(args)]
    return check(mod, flat, lowered=lowered, compiled=compiled)


def _count(ex, opcode):
    return ex._native_prog.op_histogram().get(opcode, 0)


def _gathers(f, *args, n=1, **kw):
    """`_diff`, plus the assertion that the tape really emitted the op.

    Without it a test would pass just as happily on a program jax folded the
    gather out of — and the whole family under test would go unexercised.
    """
    ex = _diff(f, *args, **kw)
    assert _count(ex, GATHER) >= n, "no gather entry on the tape"
    return ex


def _scatters(f, *args, n=1, **kw):
    ex = _diff(f, *args, **kw)
    assert _count(ex, SCATTER) >= n, "no scatter entry on the tape"
    return ex


# --------------------------------------------------------------------------
# gather: the shapes jax's indexing emits
# --------------------------------------------------------------------------


def test_take_1d():
    x = _f32((10,))
    i = np.array([3, 0, 9, 3], np.int32)
    _gathers(lambda x, i: x[i], x, i)


def test_take_rows():
    # offset_dims on the trailing axis, one collapsed slice dim.
    x = _f32((7, 5))
    i = np.array([[1, 2], [6, 0]], np.int32)
    _gathers(lambda x, i: x[i], x, i)


def test_take_along_axis():
    # TWO mapped operand dims (index_vector_dim carries a pair).
    x = _f32((4, 6))
    i = RNG.integers(0, 6, (4, 1)).astype(np.int32)
    _gathers(lambda x, i: jnp.take_along_axis(x, i, axis=1), x, i)


def test_gather_middle_axis():
    # The batch dim lands BETWEEN two offset dims: the output transpose is a
    # real interleaving rather than a suffix.
    x = _f32((3, 7, 5))
    i = np.array([1, 6, 0, 6], np.int32)
    _gathers(lambda x, i: jnp.take(x, i, axis=1), x, i)


def test_embedding_and_cross_entropy():
    # The texmo cross-entropy shape — the gather that declined every texmo
    # train chunk before this op existed on the tape.
    table = _f32((11, 4))
    ids = RNG.integers(0, 11, (6, 3, 1)).astype(np.int32)
    _gathers(lambda t, i: jnp.take(t, i[..., 0], axis=0), table, ids)

    logits = _f32((8, 10))
    labels = RNG.integers(0, 10, (8,)).astype(np.int32)
    import optax
    _gathers(optax.softmax_cross_entropy_with_integer_labels, logits, labels)


def test_vmapped_gather_has_batching_dims():
    # operand_batching_dims / start_indices_batching_dims: the index arrays
    # the tape has to synthesize as iotas.
    x = _f32((3, 7, 5))
    i = RNG.integers(0, 7, (3, 4)).astype(np.int32)
    _gathers(jax.vmap(lambda xb, ib: xb[ib]), x, i)


def test_vmapped_gather_batch_dim_not_leading():
    x = _f32((7, 3, 5))
    i = RNG.integers(0, 7, (3, 4)).astype(np.int32)
    _gathers(jax.vmap(lambda xb, ib: xb[ib], in_axes=(1, 0)), x, i)


def test_gather_grad_is_a_scatter_add():
    # The gather VJP: a scatter-add whose updates are bigger than its
    # operand, which is what picks the dummy-pad drop strategy.
    table = _f32((11, 4))
    ids = np.array([0, 4, 10, 2, 7, 5], np.int32)   # distinct: see the module
    _scatters(jax.grad(lambda t: jnp.sum(t[ids] ** 2)), table)


def test_windowed_gather_under_vmap():
    # slice_sizes > 1 on an INDEXED dim: MLX starts the slice at the index,
    # where the Python handler expands the window into an arange axis.
    x = _f32((12,))
    i = np.array([0, 4, 9], np.int32)
    _gathers(jax.vmap(lambda s: jax.lax.dynamic_slice(x, (s,), (3,))), i)


def test_windowed_gather_with_a_free_dim():
    x = _f32((12, 5))
    i = np.array([0, 4, 9], np.int32)
    _gathers(jax.vmap(lambda s: jax.lax.dynamic_slice(x, (s, 0), (3, 5))), i)


def test_gather_partial_window_on_a_free_dim():
    # slice_sizes smaller than the extent on a dim nothing indexes: the
    # Python handler pre-slices it, MLX takes it from 0 on its own.
    dn = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))
    x = _f32((9, 6))
    i = np.array([[0], [3], [8]], np.int32)
    _gathers(lambda x, i: jax.lax.gather(x, i, dn, slice_sizes=(1, 3)), x, i)


def test_gather_interleaved_offset_dims():
    # offset_dims=(0, 2): the batch axis sits in the middle of the output.
    dn = jax.lax.GatherDimensionNumbers(
        offset_dims=(0, 2), collapsed_slice_dims=(), start_index_map=(1,))
    x = _f32((4, 6))
    i = np.array([[0], [2], [4]], np.int32)
    _gathers(lambda x, i: jax.lax.gather(x, i, dn, slice_sizes=(4, 2)), x, i)


def test_gather_out_of_bounds_starts_are_clamped():
    # PROMISE_IN_BOUNDS keeps jax from emitting its own clamp, so the raw
    # XLA rule is what runs: clamp the start to extent - slice_size. MLX
    # clamps NOTHING (a negative index wraps like `take`, a big one reads
    # past the end), so this is the case the tape's clamp exists for.
    dn = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,))
    x = _f32((10,))
    i = np.array([[-4], [-1], [0], [7], [8], [9], [10], [40]], np.int32)
    _gathers(lambda x, i: jax.lax.gather(
        x, i, dn, slice_sizes=(3,),
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS), x, i)


def test_gather_oob_modes():
    t = np.array([1.0, 0.75, 0.5, 0.25, 0.0], np.float32)
    i = np.array([0, 4, 5, 7, 20], np.int32)
    _gathers(lambda t, i: jnp.take(t, i), t, i)                  # fill (NaN)
    _gathers(lambda t, i: jnp.take(t, i, mode="clip"), t, i)
    _gathers(lambda t, i: jnp.take(t, i, mode="fill", fill_value=-1.0), t, i)


@pytest.mark.parametrize("idx_rows,slice1,out", [(0, 3, "0x3"), (3, 0, "3x0")])
def test_gather_zero_sized(idx_rows, slice1, out):
    # Both engines short-circuit an empty result to zeros: nothing is
    # gathered, and the index decomposition mis-shapes empty batches. Written
    # out as StableHLO because jax folds a zero-size gather away before it
    # ever reaches the plugin — there is no jitted program that carries one.
    it, ot = f"tensor<{idx_rows}x1xi32>", f"tensor<{out}xf32>"
    mod = _mod([("x", "tensor<6x3xf32>"), ("i", it)], [ot], f"""
    %0 = "stablehlo.gather"(%x, %i) <{{
      dimension_numbers = #stablehlo.gather<offset_dims = [1],
        collapsed_slice_dims = [0], start_index_map = [0],
        index_vector_dim = 1>,
      slice_sizes = array<i64: 1, {slice1}>, indices_are_sorted = false
    }}> : (tensor<6x3xf32>, tensor<{idx_rows}x1xi32>) -> tensor<{out}xf32>
    return %0 : tensor<{out}xf32>""")
    check(mod, [_f32((6, 3)), np.zeros((idx_rows, 1), np.int32)])


def test_gather_rank0_batch():
    # index_vector_dim == rank of a 1-element index array: the whole array
    # IS the component, and the batch shape is empty.
    x = _f32((8, 3))
    i = np.array([5], np.int32)
    _gathers(lambda x, i: jnp.take(x, i[0], axis=0), x, i)


@pytest.mark.parametrize("el,npdt", [("i32", np.int32), ("i8", np.int8),
                                     ("f16", np.float16), ("i1", np.bool_)])
def test_gather_dtypes(el, npdt):
    if npdt is np.bool_:
        x = RNG.integers(0, 2, (9,)).astype(np.bool_)
    elif np.issubdtype(npdt, np.integer):
        x = RNG.integers(-40, 40, (9,)).astype(npdt)
    else:
        x = _f32((9,)).astype(npdt)
    i = np.array([8, 0, 3, 3], np.int32)
    _gathers(lambda x, i: x[i], x, i)


def test_gather_uint_and_i64_indices():
    x = _f32((9,))
    _gathers(lambda x, i: x[i], x, np.array([1, 8, 0], np.uint32))
    _gathers(lambda x, i: x[i], x, np.array([1, 8, 0], np.int64))


def test_gather_from_a_transposed_operand():
    # A transpose is a lazy strided VIEW in MLX, and strided inputs are
    # where its kernels have bitten before (reductions, argsort). The Python
    # handler transposes the operand itself, so both engines feed gather a
    # view — but of different shapes, which is the point of checking.
    x = _f32((5, 9))
    i = np.array([7, 0, 3], np.int32)
    _gathers(lambda x, i: jnp.take(x.T, i, axis=0), x, i)
    _gathers(lambda x, i: jnp.take(x.T[::2], i, axis=1), x, np.array([4, 1],
                                                                    np.int32))


def test_gather_specials_move_bit_for_bit():
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan, 1e-45, 3.4e38],
                 np.float32)
    i = np.array([5, 4, 1, 0, 6, 3, 2, 7], np.int32)
    _gathers(lambda x, i: x[i], x, i)


def test_gather_inside_a_compiled_main():
    # Traced through mx::compile rather than dispatched eagerly.
    x = _f32((16, 4))
    i = RNG.integers(0, 16, (5,)).astype(np.int32)
    _gathers(lambda x, i: jnp.sum(x[i] * 2.0, axis=1), x, i, compiled=True)


def test_gather_inside_a_counted_loop():
    # The SD3.5 sampler shape: a table lookup in a loop body, which is where
    # a gather has to survive the loop's own compile/unroll machinery.
    t = np.array([1.0, 0.75, 0.5, 0.25, 0.0], np.float32)

    def sampler(t, n):
        def body(i, carry):
            sigma = jnp.take(t, jnp.expand_dims(i, 0))
            return carry + sigma
        return jax.lax.fori_loop(0, n, body, jnp.ones((1,), np.float32))

    _gathers(sampler, t, np.int32(4))
    _gathers(sampler, t, np.int32(20))   # past the end -> NaN, on both


# --------------------------------------------------------------------------
# scatter: combiners, windows, batching, and the two OOB-drop strategies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["set", "add", "multiply", "max", "min"])
def test_scatter_combiners(method):
    x = _f32((10,))
    i = np.array([3, 8, 0], np.int32)
    v = np.array([0.5, -1.5, 2.0], np.float32)
    _scatters(lambda x, i, v: getattr(x.at[i], method)(v), x, i, v)


def test_scatter_subtract():
    # No mx::scatter_subtract exists; it is add-of-negated on both engines.
    x = _f32((10,))
    i = np.array([3, 8, 0], np.int32)
    v = np.array([0.5, -1.5, 2.0], np.float32)
    _scatters(lambda x, i, v: jax.lax.scatter_sub(
        x, i[:, None], v, jax.lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))), x, i, v)


def test_scatter_rows_and_scalar_updates():
    x = _f32((6, 4))
    i = np.array([5, 0], np.int32)
    v = _f32((2, 4))
    _scatters(lambda x, i, v: x.at[i].set(v), x, i, v)
    _scatters(lambda x, i: x.at[i].set(0.0), x, i)


@pytest.mark.parametrize("method", ["add", "multiply"])
def test_scatter_duplicate_indices_integer(method):
    # Duplicates under an arithmetic combiner: MLX's scatter is atomic and
    # its order is a GPU race on BOTH engines, so this is pinned in an
    # integer dtype, where the order cannot change the answer.
    x = RNG.integers(-5, 5, (6,)).astype(np.int32)
    i = np.array([1, 1, 4, 1, 4], np.int32)
    v = RNG.integers(-3, 3, (5,)).astype(np.int32)
    _scatters(lambda x, i, v: getattr(x.at[i], method)(v), x, i, v)


@pytest.mark.parametrize("method", ["max", "min"])
def test_scatter_duplicate_indices_float(method):
    # max/min are idempotent: the race cannot be observed even in floats.
    x = _f32((6,))
    i = np.array([1, 1, 4, 1, 4], np.int32)
    v = _f32((5,))
    _scatters(lambda x, i, v: getattr(x.at[i], method)(v), x, i, v)


def test_scatter_windowed_on_an_indexed_dim():
    # The window lives on the dim the index maps to: MLX starts the update
    # slice there, where the Python handler expands it into an arange axis.
    # Non-overlapping starts on purpose: two windows that reach the same
    # slot are a duplicate under an arithmetic combiner, and MLX's scatter
    # is atomic (see the module docstring).
    i = np.array([0, 4, 8], np.int32)
    x = _f32((12,))
    v = _f32((3, 4))
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,))
    _scatters(lambda x, i, v: jax.lax.scatter_add(x, i[:, None], v, dn),
              x, i, v)


def test_scatter_window_and_free_dim():
    # A 2-wide window on dim 1 with dim 0 written in full: the vmapped-
    # scatter shape, and the one whose update transpose is not the identity.
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(1,))
    x = _f32((5, 10))
    i = np.array([[0], [4], [8]], np.int32)
    v = _f32((3, 5, 2))
    _scatters(lambda x, i, v: jax.lax.scatter_add(x, i, v, dn), x, i, v)


def test_scatter_partial_window_on_a_free_dim():
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))
    x = _f32((10, 5))
    i = np.array([[2], [7]], np.int32)
    v = _f32((2, 3))
    _scatters(lambda x, i, v: jax.lax.scatter_add(x, i, v, dn), x, i, v)


def test_scatter_batching_dims():
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(2,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
        scatter_indices_batching_dims=(1, 0))
    x = _f32((2, 3, 10))
    i = np.array([[[0], [1]], [[2], [3]], [[8], [9]]], np.int32)
    v = _f32((3, 2, 3))
    _scatters(lambda x, i, v: jax.lax.scatter_add(x, i, v, dn), x, i, v)


def test_scatter_vmapped_set():
    x = _f32((3, 7))
    i = np.array([[0, 6], [3, 1], [5, 2]], np.int32)
    v = _f32((3, 2))
    _scatters(jax.vmap(lambda xb, ib, vb: xb.at[ib].set(vb)), x, i, v)


@pytest.mark.parametrize("full", [True, False])
def test_scatter_with_no_index_components(full):
    # scatter_dims_to_operand_dims empty and no batching dims: every start
    # index is 0, so the update is a static write at the origin. Three of
    # these come out of shape-polymorphic LU; MLX wants an index array all
    # the same, so the tape synthesizes the constant zero. Hand-written,
    # because the index operand is an empty i32 vector no jitted signature
    # would hand us.
    ut = "tensor<5x7xf32>" if full else "tensor<2x7xf32>"
    mod = _mod([("x", "tensor<5x7xf32>"), ("i", "tensor<0xi32>"),
                ("u", ut)], ["tensor<5x7xf32>"], f"""
    %0 = "stablehlo.scatter"(%x, %i, %u) <{{
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0, 1], inserted_window_dims = [],
        scatter_dims_to_operand_dims = [], index_vector_dim = 0>,
      indices_are_sorted = false, unique_indices = true}}> ({{
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
      stablehlo.return %b : tensor<f32>
    }}) : (tensor<5x7xf32>, tensor<0xi32>, {ut}) -> tensor<5x7xf32>
    return %0 : tensor<5x7xf32>""")
    rows = 5 if full else 2
    check(mod, [_f32((5, 7)), np.zeros((0,), np.int32), _f32((rows, 7))])


def test_scatter_two_mapped_dims():
    x = _f32((5, 6))
    i = np.array([[1, 2], [4, 0], [0, 5]], np.int32)
    v = _f32((3,))
    _scatters(lambda x, i, v: x.at[i[:, 0], i[:, 1]].set(v), x, i, v)


# --- the OOB drop rules ---------------------------------------------------


def _strategy(f, *args, **kw):
    """The drop strategy the lowering picked (0 none, 1 neutral, 2 pad).

    Read off the attribute vector, where it rides second precisely so a test
    can see it: the two strategies compute the same values on the same
    inputs, so a differential alone would not say which one ran, and the
    choice is a decision the two engines have to make identically.
    """
    seen = []
    orig = tape._lower_scatter

    def spy(lo, o):
        attrs, payload = orig(lo, o)
        seen.append(attrs[1])
        return attrs, payload

    tape._lower_scatter = spy
    tape._HANDLERS["stablehlo.scatter"] = spy
    try:
        _scatters(f, *args, **kw)
    finally:
        tape._lower_scatter = orig
        tape._HANDLERS["stablehlo.scatter"] = orig
    assert seen, "the lowering never saw a scatter"
    return seen


def test_scatter_oob_set_uses_the_pad():
    # "set" always pads: neutralizing a dropped set would race a genuine
    # duplicate write at the clamped slot, and fill_value == size clamping
    # onto the last real slot is a systematic collision, not a rare one.
    x = np.zeros(3, np.float32)
    i = np.array([2, 5, 0, 99], np.int32)
    v = np.array([8.0, 9.0, 1.0, 2.0], np.float32)
    assert _strategy(lambda x, i, v: x.at[i].set(v, mode="drop"),
                     x, i, v) == [2]


def test_scatter_oob_add_small_updates_neutralizes():
    x = np.zeros(64, np.float32)
    i = np.array([2, 70, 0], np.int32)
    v = np.array([8.0, 9.0, 1.0], np.float32)
    assert _strategy(lambda x, i, v: x.at[i].add(v, mode="drop"),
                     x, i, v) == [1]


def test_scatter_oob_add_big_updates_pads():
    # Updates bigger than the operand (the embedding-grad shape): the pad
    # touches less data than rewriting every update would. Integer updates,
    # because 40 of them into 3 slots is duplicates by construction.
    x = np.zeros(3, np.int32)
    i = RNG.integers(-2, 6, (40,)).astype(np.int32)
    v = RNG.integers(-9, 9, (40,)).astype(np.int32)
    assert _strategy(lambda x, i, v: x.at[i].add(v, mode="drop"),
                     x, i, v) == [2]


@pytest.mark.parametrize("method", ["set", "add", "max", "min", "multiply"])
def test_scatter_oob_dropped(method):
    x = _f32((7,))
    i = np.array([-3, 0, 6, 7, 40], np.int32)
    v = _f32((5,))
    _scatters(lambda x, i, v: getattr(x.at[i], method)(v, mode="drop"),
              x, i, v)


def test_scatter_oob_windowed_drops_the_whole_window():
    # XLA drops an update whose START does not leave room for the window,
    # so index 9 with a 3-wide window on a 10-long operand writes nothing.
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,))
    x = _f32((10,))
    i = np.array([[9], [0], [8], [-1]], np.int32)
    v = _f32((4, 3))
    _scatters(lambda x, i, v: jax.lax.scatter_add(
        x, i, v, dn, mode=jax.lax.GatherScatterMode.FILL_OR_DROP), x, i, v)


def test_scatter_jnp_helpers_that_rely_on_dropping():
    # The library shapes v0.4.0's drop semantics were built for. (Their
    # siblings — jnp.nonzero, jnp.place, jnp.unique, linear-ramp padding —
    # all reach the same scatter but decline on something ELSE in the same
    # program: reduce_window for the cumsum, a key-computing sort
    # comparator, stablehlo.pad. Their scatters are covered above.)
    _scatters(lambda i: jnp.bincount(i, length=3),
              np.array([0, 1, 9, 1], np.int32))
    _scatters(lambda a, b: jnp.polyadd(a, b),
              np.array([1.0, 2.0], np.float32),
              np.array([1.0, 2.0, 3.0], np.float32))


def test_scatter_zero_sized_is_a_passthrough():
    # Empty updates apply nothing: ops/gather.py hands the operand array
    # back, so the tape aliases the slot — and the result still has to be a
    # buffer of its own, which is what the output-copy rule is for.
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,))
    x = _f32((5,))
    i = np.zeros((0, 1), np.int32)
    v = np.zeros((0,), np.float32)
    ex = _diff(lambda x, i, v: jax.lax.scatter_add(x, i, v, dn), x, i, v)
    assert _count(ex, SCATTER) == 0, "an empty scatter should be an alias"


def test_scatter_inside_a_compiled_main():
    # Traced through mx::compile, which is also where the neutral value
    # becomes a baked constant rather than an array built per call.
    x = _f32((16,))
    i = np.array([0, 15, 4, 9, 2], np.int32)
    v = _f32((5,))
    _scatters(lambda x, i, v: x.at[i].add(v) * 2.0, x, i, v, compiled=True)


def test_scatter_specials_move_bit_for_bit():
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan, 1.0],
                 np.float32)
    i = np.array([1, 3, 5], np.int32)
    v = np.array([-0.0, np.nan, -np.inf], np.float32)
    _scatters(lambda x, i, v: x.at[i].set(v), x, i, v)
    # ...and through an arithmetic combiner, where a neutralized drop would
    # turn -0.0 into +0.0 if the two engines disagreed about the strategy.
    xz = np.array([-0.0] * 8, np.float32)
    iz = np.array([0, 9, 3, -1], np.int32)
    vz = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    _scatters(lambda x, i, v: x.at[i].add(v, mode="drop"), xz, iz, vz)


@pytest.mark.parametrize("el,npdt", [("i32", np.int32), ("i16", np.int16),
                                     ("f16", np.float16), ("bf16", None)])
def test_scatter_dtypes(el, npdt):
    import ml_dtypes
    dt = ml_dtypes.bfloat16 if npdt is None else npdt
    if np.issubdtype(np.dtype(dt), np.integer):
        x = RNG.integers(-40, 40, (9,)).astype(dt)
        v = RNG.integers(-9, 9, (3,)).astype(dt)
    else:
        x = _f32((9,)).astype(dt)
        v = _f32((3,)).astype(dt)
    i = np.array([8, 0, 3], np.int32)
    _scatters(lambda x, i, v: x.at[i].add(v), x, i, v)


# --------------------------------------------------------------------------
# what still declines
# --------------------------------------------------------------------------


def test_decline_scatter_apply_body():
    # scatter_apply: an arbitrary elementwise body, evaluated on the
    # gathered current values (and, with duplicates, one update at a time).
    # Not portable to a primitive; the program runs in Python instead.
    from functools import partial
    dn = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,), inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,))
    _diff(partial(jax.lax.scatter_apply, func=jnp.sin, update_shape=(3, 2),
                  dimension_numbers=dn),
          np.arange(10, dtype=np.float32),
          np.array([[0], [1], [7]], np.int32), lowered=False)


def test_complex_gather():
    # complex64 joined the dtype table with the tail sweep, and a gather
    # MOVES bits rather than computing on them, so mx::gather carries it
    # (mx::scatter, which has no complex kernels, still declines below).
    x = np.array([1 + 2j, 3 - 4j, -5 + 6j], np.complex64)
    i = np.array([2, 0], np.int32)
    _diff(lambda x, i: x[i], x, i)


def test_decline_complex_scatter():
    # MLX has no complex GPU scatter kernels; ops/gather.py scatters the
    # two parts separately, which is not this entry's single primitive.
    x = np.array([1 + 2j, 3 - 4j, -5 + 6j], np.complex64)
    u = np.array([7 - 1j], np.complex64)
    _diff(lambda x, u: x.at[np.array([1])].set(u), x, u, lowered=False)
