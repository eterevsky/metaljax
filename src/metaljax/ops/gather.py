"""stablehlo.gather (the patterns JAX emits)."""

import re

import mlx.core as mx

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from metaljax import _ir
from metaljax.interpreter import register, UnsupportedOpError


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
    out_rank = len(ir.RankedTensorType(op.results[0].type).shape)

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
    raise UnsupportedOpError(
        f"scatter body {[o.name for o in body]} not implemented")


@register("stablehlo.scatter")
def _scatter(interp, op, ins, env):
    if len(ins) != 3:
        raise UnsupportedOpError("variadic scatter not implemented")
    operand, scatter_indices, updates = ins
    d = _scatter_dims(op)
    method = _scatter_combiner(op)
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
    for j, dim in enumerate(d["scatter_dims_to_operand_dims"]):
        limit = op_shape[dim] - 1
        mapped[dim] = mx.clip(index_arrays[j].astype(mx.int32), 0, max(limit, 0))
    for op_dim, idx_dim in zip(d["operand_batching_dims"],
                               d["scatter_indices_batching_dims"]):
        pos = batch_dims_idx.index(idx_dim)
        ramp = mx.arange(idx_shape[idx_dim], dtype=mx.int32)
        view = [1] * len(batch_shape)
        view[pos] = idx_shape[idx_dim]
        mapped[op_dim] = mx.broadcast_to(mx.reshape(ramp, view), batch_shape)

    indexed_dims = sorted(mapped)
    inserted = set(d["inserted_window_dims"]) | set(d["operand_batching_dims"])
    if not set(indexed_dims) <= inserted:
        raise UnsupportedOpError(
            f"scatter with window on indexed dims (indexed={indexed_dims}, "
            f"inserted={sorted(inserted)}) not implemented")
    free_dims = [i for i in range(len(op_shape)) if i not in indexed_dims]
    windowed_free = [f for f in free_dims if f not in inserted]
    inserted_free = [f for f in free_dims if f in inserted]
    for f in inserted_free:
        # Window of size 1 at offset 0 on an unindexed dim: only correct when
        # the dim itself has extent 1.
        if op_shape[f] != 1:
            raise UnsupportedOpError(
                f"scatter inserted window on dim {f} of size {op_shape[f]}")

    uwd = d["update_window_dims"]
    if len(uwd) != len(windowed_free):
        raise UnsupportedOpError(
            f"scatter window rank mismatch: dims={d}, op_shape={op_shape}, "
            f"upd_shape={upd_shape}, windowed_free={windowed_free}")
    for w, f in zip(uwd, windowed_free):
        if upd_shape[w] != op_shape[f]:
            raise UnsupportedOpError(
                f"scatter partial window {upd_shape[w]} != {op_shape[f]}")

    upd_batch_positions = [i for i in range(len(upd_shape)) if i not in uwd]
    updates_t = mx.transpose(updates, upd_batch_positions + uwd)
    # Re-insert the size-1 free dims so updates rank matches batch + free dims.
    full_shape = batch_shape + [op_shape[f] for f in free_dims]
    updates_t = mx.reshape(updates_t, full_shape)

    operand_t = mx.transpose(operand, indexed_dims + free_dims)
    idx_tuple = tuple(mapped[dim] for dim in indexed_dims)
    updates_t = updates_t.astype(operand_t.dtype)
    if method == "set":
        res_t = operand_t.at[idx_tuple].set(updates_t) if idx_tuple else updates_t
    else:
        res_t = getattr(operand_t.at[idx_tuple], method)(updates_t)

    inv = [0] * len(op_shape)
    for pos, dim in enumerate(indexed_dims + free_dims):
        inv[dim] = pos
    return mx.transpose(res_t, inv)