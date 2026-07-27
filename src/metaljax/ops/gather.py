"""stablehlo.gather (the patterns JAX emits)."""

import re

import mlx.core as mx

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

import numpy as np

from metaljax import _ir, dtypes
from metaljax.interpreter import register, UnsupportedOpError


def _combiner_neutral(method, dtype):
    """Update value that makes the combiner a no-op (for dropped updates)."""
    if method in ("add", "subtract"):
        return mx.array(0, dtype=dtype)
    if method == "multiply":
        return mx.array(1, dtype=dtype)
    npdt = dtypes._MX_TO_NP[dtype]
    if method == "maximum":
        if dtype == mx.bool_:
            return mx.array(False)
        v = -np.inf if npdt.kind == "f" else np.iinfo(npdt).min
        return mx.array(v, dtype=dtype)
    if method == "minimum":
        if dtype == mx.bool_:
            return mx.array(True)
        v = np.inf if npdt.kind == "f" else np.iinfo(npdt).max
        return mx.array(v, dtype=dtype)
    raise UnsupportedOpError(f"no neutral for scatter method {method}")


def _gather_dims(op):
    attr = op.attributes["dimension_numbers"]
    try:
        dn = stablehlo.GatherDimensionNumbers(attr)
        return {
            "offset_dims": list(dn.offset_dims),
            "collapsed_slice_dims": list(dn.collapsed_slice_dims),
            "operand_batching_dims": list(
                getattr(dn, "operand_batching_dims", None)
                or getattr(dn, "input_batching_dims", None) or []),
            "start_indices_batching_dims": list(
                getattr(dn, "start_indices_batching_dims", [])),
            "start_index_map": list(dn.start_index_map),
            "index_vector_dim": dn.index_vector_dim,
        }
    except Exception:
        s = str(attr)

        def grab(name):
            m = re.search(rf"{name}\s*=\s*\[([^\]]*)\]", s)
            return [int(x) for x in m.group(1).split(",") if x.strip()] if m else []

        m = re.search(r"index_vector_dim\s*=\s*(\d+)", s)
        return {
            "offset_dims": grab("offset_dims"),
            "collapsed_slice_dims": grab("collapsed_slice_dims"),
            "operand_batching_dims": grab("operand_batching_dims"),
            "start_indices_batching_dims": grab("start_indices_batching_dims"),
            "start_index_map": grab("start_index_map"),
            "index_vector_dim": int(m.group(1)) if m else 0,
        }


@register("stablehlo.gather")
def _gather(interp, op, ins, env):
    operand, start_indices = ins
    d = _gather_dims(op)
    slice_sizes = _ir.i64_list(op, "slice_sizes")
    op_shape = list(operand.shape)
    idx_shape = list(start_indices.shape)
    ivd = d["index_vector_dim"]
    out_t = ir.RankedTensorType(op.results[0].type)
    out_rank = len(out_t.shape)
    if any(s == 0 for s in out_t.shape):
        # Zero-size result: nothing to gather (and the decomposition below
        # mis-shapes empty batches). Semantically exact short-circuit.
        return mx.zeros(list(out_t.shape), dtype=operand.dtype)

    # Batch shape of the indices (everything except the index vector dim).
    batch_dims_idx = [i for i in range(len(idx_shape)) if i != ivd]
    batch_shape = [idx_shape[i] for i in batch_dims_idx]

    # Split the index vector into one integer array per mapped operand dim.
    if ivd == len(idx_shape):
        index_arrays = [start_indices]
        k = 1
    else:
        k = idx_shape[ivd]
        index_arrays = []
        for j in range(k):
            sl = [slice(None)] * len(idx_shape)
            sl[ivd] = j
            index_arrays.append(start_indices[tuple(sl)])

    mapped = {}  # operand dim -> index array of batch shape
    for j, dim in enumerate(d["start_index_map"]):
        limit = op_shape[dim] - slice_sizes[dim]
        idx = mx.clip(index_arrays[j].astype(mx.int32), 0, max(limit, 0))
        mapped[dim] = idx

    # Batching dims: implicitly indexed by their own coordinate.
    for op_dim, idx_dim in zip(d["operand_batching_dims"],
                               d["start_indices_batching_dims"]):
        pos = batch_dims_idx.index(idx_dim)
        ramp = mx.arange(idx_shape[idx_dim], dtype=mx.int32)
        view = [1] * len(batch_shape)
        view[pos] = idx_shape[idx_dim]
        mapped[op_dim] = mx.broadcast_to(mx.reshape(ramp, view), batch_shape)

    indexed_dims = sorted(mapped)
    collapsed = set(d["collapsed_slice_dims"]) | set(d["operand_batching_dims"])
    free_dims = [i for i in range(len(op_shape)) if i not in indexed_dims]

    # Pre-slice free dims whose slice size is smaller than the dim.
    pre = [slice(None)] * len(op_shape)
    needs_pre = False
    for i in free_dims:
        if slice_sizes[i] != op_shape[i]:
            pre[i] = slice(0, slice_sizes[i])
            needs_pre = True
    if needs_pre:
        operand = operand[tuple(pre)]

    # Indexed dims with slice_size > 1 gather a window: give each its own
    # broadcast axis and add arange(w) to the (clamped) start index.
    nb = len(batch_shape)
    windowed = [m for m in indexed_dims if slice_sizes[m] > 1]
    W = len(windowed)
    idx_arrays = []
    for m in indexed_dims:
        base = mx.reshape(mapped[m], batch_shape + [1] * W)
        w = slice_sizes[m]
        if w > 1:
            k = windowed.index(m)
            view = [1] * (nb + W)
            view[nb + k] = w
            base = base + mx.reshape(mx.arange(w, dtype=mx.int32), view)
        idx_arrays.append(base)

    # Move indexed dims to the front, then advanced-index them all at once:
    # result = batch_shape + window axes (windowed order) + free dim sizes.
    operand_t = mx.transpose(operand, indexed_dims + free_dims)
    gathered = operand_t[tuple(idx_arrays)] if idx_arrays else operand_t

    # Rearrange trailing axes into operand-dim order, then reshape to the
    # uncollapsed slice shape (inserting size-1 axes for uncollapsed indexed
    # dims, dropping collapsed ones — all extent-1, so reshape is enough).
    post_dims = windowed + free_dims
    order = sorted(range(len(post_dims)), key=lambda i: post_dims[i])
    gathered = mx.transpose(gathered, list(range(nb)) + [nb + i for i in order])
    uncollapsed = [i for i in range(len(op_shape)) if i not in collapsed]
    slice_shape = [slice_sizes[i] for i in uncollapsed]
    gathered = mx.reshape(gathered, batch_shape + slice_shape)

    # Place batch dims / offset dims where the output wants them.
    offset_dims = d["offset_dims"]
    out_batch_positions = [i for i in range(out_rank) if i not in offset_dims]
    perm = [0] * out_rank
    for cur, outpos in enumerate(out_batch_positions):
        perm[outpos] = cur
    for cur, outpos in enumerate(offset_dims):
        perm[outpos] = len(batch_shape) + cur
    return mx.transpose(gathered, perm)


def _scatter_dims(op):
    attr = op.attributes["scatter_dimension_numbers"]
    try:
        dn = stablehlo.ScatterDimensionNumbers(attr)
        return {
            "update_window_dims": list(dn.update_window_dims),
            "inserted_window_dims": list(dn.inserted_window_dims),
            "operand_batching_dims": list(
                getattr(dn, "input_batching_dims", None)
                or getattr(dn, "operand_batching_dims", None) or []),
            "scatter_indices_batching_dims": list(
                getattr(dn, "scatter_indices_batching_dims", [])),
            "scatter_dims_to_operand_dims": list(dn.scattered_dims_to_operand_dims)
            if hasattr(dn, "scattered_dims_to_operand_dims")
            else list(dn.scatter_dims_to_operand_dims),
            "index_vector_dim": dn.index_vector_dim,
        }
    except Exception:
        s = str(attr)

        def grab(name):
            m = re.search(rf"{name}\s*=\s*\[([^\]]*)\]", s)
            return [int(x) for x in m.group(1).split(",") if x.strip()] if m else []

        m = re.search(r"index_vector_dim\s*=\s*(\d+)", s)
        return {
            "update_window_dims": grab("update_window_dims"),
            "inserted_window_dims": grab("inserted_window_dims"),
            "operand_batching_dims": grab("input_batching_dims") or
                grab("operand_batching_dims"),
            "scatter_indices_batching_dims": grab("scatter_indices_batching_dims"),
            "scatter_dims_to_operand_dims": grab("scatter_dims_to_operand_dims"),
            "index_vector_dim": int(m.group(1)) if m else 0,
        }


def _scatter_combiner(op):
    """Return the .at[] method name for the update-computation region."""
    block = op.regions[0].blocks[0]
    body = [o.operation for o in block.operations]
    if len(body) == 1 and body[0].name == "stablehlo.return":
        ret = body[0].operands[0]
        args = list(block.arguments)
        if len(args) == 2 and ret == args[1]:
            return "set"
        raise UnsupportedOpError("scatter body returns non-update value")
    if len(body) == 2 and body[1].name == "stablehlo.return":
        name = body[0].name
        table = {
            "stablehlo.add": "add",
            "stablehlo.multiply": "multiply",
            "stablehlo.maximum": "maximum",
            "stablehlo.minimum": "minimum",
            "stablehlo.subtract": "subtract",
        }
        if name in table:
            return table[name]
    # Arbitrary elementwise bodies (jax scatter_apply): evaluate the body
    # on gathered current values + updates, then scatter-set. Only sound
    # when indices are unique (scatter_apply guarantees it).
    if all(o.name.startswith(("stablehlo.", "chlo.")) for o in body):
        return "apply"
    raise UnsupportedOpError(
        f"scatter body {[o.name for o in body]} not implemented")


@register("stablehlo.scatter")
def _scatter(interp, op, ins, env):
    if len(ins) != 3:
        raise UnsupportedOpError("variadic scatter not implemented")
    operand, scatter_indices, updates = ins
    if any(s == 0 for s in updates.shape) or any(s == 0 for s in operand.shape):
        # No updates to apply (empty updates), or every update is
        # necessarily out of bounds (zero-size operand) and thus dropped.
        return operand
    d = _scatter_dims(op)
    method = _scatter_combiner(op)
    if operand.dtype == mx.complex64:
        # MLX GPU scatter has no complex64 kernels; set/add/subtract are
        # componentwise, so run the scatter on the parts.
        if method not in ("set", "add", "subtract"):
            raise UnsupportedOpError(f"complex scatter {method}")
        re = _scatter(interp, op,
                      [mx.real(operand), scatter_indices, mx.real(updates)],
                      env)
        im = _scatter(interp, op,
                      [mx.imag(operand), scatter_indices, mx.imag(updates)],
                      env)
        return re.astype(mx.complex64) + im.astype(mx.complex64) * 1j
    op_shape = list(operand.shape)
    idx_shape = list(scatter_indices.shape)
    upd_shape = list(updates.shape)
    ivd = d["index_vector_dim"]

    batch_dims_idx = [i for i in range(len(idx_shape)) if i != ivd]
    batch_shape = [idx_shape[i] for i in batch_dims_idx]

    if ivd == len(idx_shape):
        index_arrays = [scatter_indices]
    else:
        index_arrays = []
        for j in range(idx_shape[ivd]):
            sl = [slice(None)] * len(idx_shape)
            sl[ivd] = j
            index_arrays.append(scatter_indices[tuple(sl)])

    mapped = {}
    oob = None
    inserted = set(d["inserted_window_dims"]) | set(d["operand_batching_dims"])
    uwd = d["update_window_dims"]
    # Operand dims not inserted carry an update window dim, in order.
    window_dims = [i for i in range(len(op_shape)) if i not in inserted]
    if len(uwd) != len(window_dims):
        raise UnsupportedOpError(
            f"scatter window rank mismatch: dims={d}, op_shape={op_shape}, "
            f"upd_shape={upd_shape}")
    wsize = {dim: upd_shape[w] for w, dim in zip(uwd, window_dims)}
    uwd_of = {dim: w for w, dim in zip(uwd, window_dims)}

    # Start position per operand dim. XLA drops an update if the whole
    # window doesn't fit: 0 <= idx <= extent - window.
    for j, dim in enumerate(d["scatter_dims_to_operand_dims"]):
        raw = index_arrays[j].astype(mx.int32)
        limit = op_shape[dim] - wsize.get(dim, 1)
        bad = (raw < 0) | (raw > limit)
        oob = bad if oob is None else (oob | bad)
        mapped[dim] = mx.clip(raw, 0, max(limit, 0))
    for op_dim, idx_dim in zip(d["operand_batching_dims"],
                               d["scatter_indices_batching_dims"]):
        pos = batch_dims_idx.index(idx_dim)
        ramp = mx.arange(idx_shape[idx_dim], dtype=mx.int32)
        view = [1] * len(batch_shape)
        view[pos] = idx_shape[idx_dim]
        mapped[op_dim] = mx.broadcast_to(mx.reshape(ramp, view), batch_shape)
    # Unindexed dims whose window doesn't span the full extent start at 0
    # (inserted-but-unindexed dims have an implicit size-1 window; partial
    # windows on free dims write to [0:w]).
    zero = mx.zeros(batch_shape, dtype=mx.int32)
    for dim in range(len(op_shape)):
        if dim not in mapped and (dim in inserted
                                  or wsize[dim] != op_shape[dim]):
            mapped[dim] = zero

    # Windowed dims that carry an index expand into explicit per-element
    # indices: one new batch axis per such dim, index = start + arange(w).
    # Size-1 windows on indexed dims must be included too — they still
    # own an update-window axis that the transpose below has to place.
    expand = sorted(dim for dim in mapped if dim in uwd_of)
    E = len(expand)
    nb = len(batch_shape)
    exp_sizes = [wsize[dim] for dim in expand]

    indexed_dims = sorted(mapped)
    free_dims = [i for i in range(len(op_shape)) if i not in indexed_dims]

    # updates axes: batch positions, then the expanded windows (sorted by
    # operand dim, matching the new index axes), then the full free windows.
    upd_batch_positions = [i for i in range(len(upd_shape)) if i not in uwd]
    updates_t = mx.transpose(
        updates, upd_batch_positions + [uwd_of[dim] for dim in expand]
        + [uwd_of[f] for f in free_dims])
    full_shape = batch_shape + exp_sizes + [op_shape[f] for f in free_dims]
    updates_t = mx.reshape(updates_t, full_shape)

    operand_t = mx.transpose(operand, indexed_dims + free_dims)
    idx_list = []
    for dim in indexed_dims:
        base = mx.reshape(mapped[dim], batch_shape + [1] * E)
        if dim in expand:
            k = expand.index(dim)
            view = [1] * (nb + E)
            view[nb + k] = wsize[dim]
            base = base + mx.reshape(
                mx.arange(wsize[dim], dtype=mx.int32), view)
        idx_list.append(base)
    idx_tuple = tuple(idx_list)
    updates_t = updates_t.astype(operand_t.dtype)
    if method == "apply":
        unique = ("unique_indices" in op.attributes
                  and "true" in str(op.attributes["unique_indices"]))
        if not unique:
            raise UnsupportedOpError(
                "scatter with computed body needs unique_indices")
        from metaljax.ops.reduction import _eval_body
        block = op.regions[0].blocks[0]
        bargs = list(block.arguments)
        old = operand_t[idx_tuple] if idx_tuple else operand_t
        res = _eval_body(interp, block,
                         {bargs[0]: old, bargs[1]: updates_t}, env)
        updates_t = res[0].astype(operand_t.dtype)
        method = "set"
    if oob is not None:
        oob = mx.reshape(oob, batch_shape + [1] * E)
    # XLA semantics: an update whose index has any out-of-bounds component
    # is DROPPED. Clamping alone clobbers real data at the boundary
    # (jnp.nonzero/where pad with fill_value == size, bincount overflow
    # slots, sparse fill indices). Two drop strategies:
    # - dummy slot: append one row along the first indexed axis and
    #   redirect dropped updates there (touches 2x the operand). Required
    #   for "set" — neutralizing a set-update would race a genuine
    #   duplicate write at the clamped slot (fill_value == size clamps
    #   onto the last real slot, a systematic collision).
    # - neutral value: rewrite dropped updates to the combiner's identity
    #   (touches 2x the updates; order/duplicate-safe for arithmetic).
    # Pick by data touched: embedding-grad scatters (big updates, small
    # table) measured ~10% slower per texmo step under the neutral path.
    use_dummy = oob is not None and idx_tuple and (
        method == "set" or operand_t.size < updates_t.size)
    if oob is not None and method != "set" and not use_dummy:
        mask = mx.reshape(oob, batch_shape + [1] * (E + len(free_dims)))
        updates_t = mx.where(
            mask, _combiner_neutral(method, operand_t.dtype), updates_t)
    if use_dummy:
        ext = mx.concatenate(
            [operand_t,
             mx.zeros((1,) + operand_t.shape[1:], dtype=operand_t.dtype)],
            axis=0)
        idx0 = mx.where(
            oob, mx.array(operand_t.shape[0], dtype=mx.int32),
            idx_tuple[0])
        if method == "set":
            # __setitem__ is MLX's scatter-assign (rebinds only this
            # local array object, so no aliasing concerns).
            ext[(idx0,) + idx_tuple[1:]] = updates_t
        else:
            ext = getattr(ext.at[(idx0,) + idx_tuple[1:]], method)(updates_t)
        res_t = ext[: operand_t.shape[0]]
    elif method == "set":
        if idx_tuple:
            res_t = operand_t
            res_t[idx_tuple] = updates_t
        else:
            res_t = updates_t
    else:
        res_t = getattr(operand_t.at[idx_tuple], method)(updates_t)

    inv = [0] * len(op_shape)
    for pos, dim in enumerate(indexed_dims + free_dims):
        inv[dim] = pos
    return mx.transpose(res_t, inv)