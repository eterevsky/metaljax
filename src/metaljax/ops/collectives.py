"""Cross-replica collectives on a single device.

metaljax exposes exactly one device, so every replica group has size 1
and each collective degenerates to an identity (or a trivial constant):
psum/pmax over one replica is the value itself, all_gather concatenates
one shard, permutes have nowhere to go. This makes jax.pmap and
shard_map programs on a single device work unchanged. Programs compiled
for >1 replica can't reach us — jax checks the device count first.
"""

import numpy as np

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir
from metaljax.interpreter import register, UnsupportedOpError


def _group_size(op):
    """Max replica-group size declared on the op (0 if absent)."""
    if "replica_groups" not in op.attributes:
        return 1
    attr = ir.DenseIntElementsAttr(op.attributes["replica_groups"])
    t = ir.RankedTensorType(attr.type)
    shape = list(t.shape)
    return shape[-1] if shape else 1


def _require_singleton(op):
    g = _group_size(op)
    if g > 1:
        raise UnsupportedOpError(
            f"{op.name} with replica groups of size {g} "
            "(metaljax is single-device)")


@register("stablehlo.all_reduce", "stablehlo.reduce_scatter")
def _all_reduce(interp, op, ins, env):
    # Reduction over a single replica: the value itself (for
    # reduce_scatter the scattered shard of one replica is also the
    # whole value — output type equals input type when the group is 1).
    _require_singleton(op)
    return list(ins) if len(ins) > 1 else ins[0]


@register("stablehlo.all_gather")
def _all_gather(interp, op, ins, env):
    _require_singleton(op)
    return list(ins) if len(ins) > 1 else ins[0]


@register("stablehlo.all_to_all")
def _all_to_all(interp, op, ins, env):
    _require_singleton(op)
    return list(ins) if len(ins) > 1 else ins[0]


@register("stablehlo.collective_permute")
def _collective_permute(interp, op, ins, env):
    # With one device the only legal pair is (0, 0); an empty pair list
    # means the replica receives nothing -> zeros per XLA semantics.
    pairs = []
    if "source_target_pairs" in op.attributes:
        arr = np.array(
            ir.DenseIntElementsAttr(op.attributes["source_target_pairs"]))
        pairs = arr.reshape(-1, 2).tolist()
    (x,) = ins
    if any(p != [0, 0] for p in pairs):
        raise UnsupportedOpError(
            f"collective_permute pairs {pairs} (single-device)")
    if not pairs:
        return mx.zeros_like(x)
    return x


@register("stablehlo.collective_broadcast")
def _collective_broadcast(interp, op, ins, env):
    return ins[0]


@register("stablehlo.replica_id", "stablehlo.partition_id")
def _replica_id(interp, op, ins, env):
    return mx.array(0, dtype=mx.uint32)
