"""Recognizer that maps softmax-attention subgraphs onto
``mx.fast.scaled_dot_product_attention``.

XLA fuses attention into a flash-attention kernel; interpreting the lowering
literally materializes the whole ``[batch, heads, q, k]`` logits tensor
**five times over** (the dot's result, the max-subtracted copy, its
exponential, the normalized probabilities, and the broadcast denominators).
At Stable Diffusion 3.5 MMDiT resolutions that is ~12 GB per attention block
and ~90 GB live across one command buffer -- which is how this project's
flight recorder caught the machine panicking, and why `MLX_MAX_MB_PER_BUFFER`
has a ceiling at all (see `metaljax/__init__.py`). Attention is also most of
the prefill gap on every LLM row: those logits dominate both bandwidth and
allocation.

MLX ships the fused kernel, so the fix is a pure structural rewrite. Unlike
`metaljax.qmm` there is no packed state and no first-execute prologue: a
matched subgraph is rewritten at analysis time and needs nothing from the
concrete buffers.

WHAT IS MATCHED
---------------
The canonical lowering of ``softmax(Q@K^T * s + mask) @ V``::

    L = dot_general(Q, K)                  # contracting on the head dim
    L = multiply(L, splat)                 # optional scale (either order)
    L = add(L, mask) | select(pred, L, C)  # optional mask
    m = reduce(L, -inf, maximum) over k    # softmax, canonical decomposition
    e = exp(L - broadcast(m))
    d = reduce(e, 0, add) over k
    O = dot_general(e / broadcast(d), V)   # "normalize first"

plus the shape reassociation jax sprays between every stage (transpose,
reshape, convert, `sdy.sharding_constraint`, and `func.call` wrappers such as
maxtext's ``@_where``). The **deferred-normalization** form that real LLM
lowerings emit is matched too, since the denominator is constant along V's
feature axis::

    O = dot_general(e, V) / broadcast(d)   # "normalize last", root = divide

Rather than pattern-matching layouts, every axis of the first dot is given an
ATOM -- batch, contracting (head dim), or free -- and atoms are propagated
through the reassociation ops (`_roles_*`). The softmax's reduction axis is
what identifies the `k` atom, hence which operand of the first dot is K and
which is Q; everything else follows. That is what lets one code path handle
the `[B,T,H,D]` and `[B,H,T,D]` conventions, `jax.nn.dot_product_attention`'s
rank-5 unit-dim spelling, and maxtext's reshaped GQA without a special case
for any of them.

GQA falls out of the same machinery. Query-side free axes other than the
query sequence (maxtext reshapes Q to ``[B, T, H_kv, g, D]`` and makes
``H_kv`` a batching dim) become the MINOR part of MLX's query-head axis, so
MLX's ``h // g`` head pairing is right by construction. K/V that arrive
already broadcast to the full head count (the `jnp.repeat` spelling) fuse as
plain MHA -- correct, just without GQA's bandwidth saving.

Two splits are free for correctness and are searched rather than fixed:

* which query-side axis is "the query sequence" and which are extra head
  axes -- each (head, group, query) triple is an independent softmax row, so
  any choice gives the same numbers;
* where the batching axes divide into MLX's `B` and `N` -- the kernel is
  independent per `(b, n)` pair, and `h // g` holds for any division because
  the kv-head axes stay major within `N`.

Both are decided by the MASK, which is the only thing that can tell them
apart: MLX broadcasts a mask against `[B, N, Tq, Tk]`, so a mask has to vary
along the axis chosen as `Tq` and has to cover a whole slot or none of it.
A mask varying along the MIDDLE of three batching axes satisfies no division
and is rejected.

NUMERICS
--------
Measured against an exact float64 reference: the fused kernel accumulates
softmax in float32 whatever the input dtype, so at f16/bf16 it is 3-5x MORE
accurate than the chain it replaces (B2 H4 L128-512 D64, and B1 H8 L128 D64
where `tests/test_sdpa.py` pins >2x), and at f32 it ties -- 1.0-1.2x, both
sitting at the f32 reduction's own error, pinned at 1.5x. No accuracy is
traded for speed, which is why this is on by default.

MASKS ARE ALWAYS PASSED ADDITIVELY, never as MLX's boolean mask. A boolean
mask makes MLX give a fully-masked row a finite value where the literal chain
produces NaN. An additive mask reproduces the literal chain exactly: `-inf`
gives the same NaN, and jax's `finfo.min` sentinel gives the same uniform
row, because ``L + C == C`` to the last bit whenever `|C|` is within a few
orders of the dtype's maximum. A `select` mask is therefore rewritten as
``where(pred, 0, C)`` with the SAME `C` the select used -- and only when `C`
is mask-like (`_MASK_FRACTION`), since a small constant would make select and
add genuinely different functions.

WHAT IS NOT MATCHED
-------------------
**Training graphs.** Autodiff keeps the softmax output as a residual for the
backward pass, so the probabilities are consumed twice and the use-count
discipline below refuses. Replacing them would mean recognizing the forward
AND the backward together and emitting a fused attention VJP, which MLX does
not expose at this level. Inference graphs -- LLM prefill and decode,
diffusion sampling -- are exactly the ones that fuse, and they are where the
logits blow up. texmo, a pure training workload, is therefore unaffected
(measured: within noise on `attn.512.8.64`, whose forward pass on its own
does fuse).

Also left alone: a softmax with no max subtraction (MLX always subtracts one,
which differs on overflow), a non-splat scale, a `select` whose constant is
too small to be a mask sentinel, and `jax.nn.softmax(..., where=mask)` --
that one sums `select(mask, exp, 0)` rather than the exponentials, so the
denominator is not the row sum the fused kernel computes and `_check_reduce`
refuses it.

METALJAX_SDPA=0 disables the whole recognizer.
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes

ENABLED = os.environ.get("METALJAX_SDPA", "1") != "0"
_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"

# Chains between stages are a handful of ops; these bounds only stop a
# pathological graph from making analysis quadratic.
_MAX_DEPTH = 32
_MAX_FANOUT = 32
# A `select` constant counts as a mask sentinel when it is at least this
# fraction of the dtype's largest finite value, negated. jax uses `finfo.min`
# (1.0) and maxtext -0.35 / -0.7 of `finfo.max`.
_MASK_FRACTION = 0.1

STATS = {"recognized": 0, "fused": 0}

_FINITE_MAX = {"f32": 3.4028234663852886e38,
               "bf16": 3.3895313892515355e38,
               "f16": 65504.0}
_FLOAT_ELS = frozenset(_FINITE_MAX)


def stats() -> dict:
    return dict(STATS)


def reset_stats():
    for k in STATS:
        STATS[k] = 0


class _Reject(Exception):
    """This subgraph is not (provably) attention: run it literally."""


# Ops that pass a tensor through elementwise-identically.
_IDENTITY = ("stablehlo.convert", "sdy.sharding_constraint")
_CALL = ("func.call", "stablehlo.composite")
# Pure reassociation: shape-only, and invertible in both directions.
_FORWARD = _IDENTITY + ("stablehlo.reshape", "stablehlo.transpose")
# Ops a value can be followed down through while its axes stay trackable.
# Only downwards: a broadcast is not invertible walking forward.
_TRANSPARENT = _FORWARD + ("stablehlo.broadcast_in_dim",)
# `jnp.max(..., initial=-inf)` lowers to `maximum(broadcast(-inf), reduce)`.
# Dropping a splat identity operand is exact for every input, NaN included
# (both forms propagate it), so these are followed through like a convert.
_CLAMP = {"stablehlo.maximum": float("-inf"), "stablehlo.minimum": float("inf")}
# Shape-preserving ops, for the role propagators.
_SAME_SHAPE = _IDENTITY + _CALL + tuple(_CLAMP)


# --------------------------------------------------------------------------
# IR helpers
# --------------------------------------------------------------------------


def _owner(v):
    return v.owner.operation if isinstance(v, ir.OpResult) else None


def _okey(op):
    """Identity key for an operation: its first result.

    `ir.Value` hashes by the underlying MLIR value. The python wrappers around
    *operations* are transient, and this project has twice shipped bugs from
    keying sets on `id()` of one (msl kernel names in 0.2.0, sort-comparator
    dependency walks in 0.4.1).
    """
    res = op.results
    return res[0] if len(res) else None


def _shape(v) -> list[int]:
    return list(_ir.tensor_type(v).shape)


def _el(v) -> str:
    return str(_ir.tensor_type(v).element_type)


def _walk_blocks(block):
    yield block
    for op in block.operations:
        for region in op.operation.regions:
            for b in region.blocks:
                yield from _walk_blocks(b)


def _deref(v, frame):
    """Resolve callee block arguments outward through the call frames.

    Returns `(value, frame)`; `frame is None` means the value lives in the
    block being interpreted, which is the only place `emit` can read it from.
    A block argument with no frame left to resolve through is such a value --
    `run_block` seeds the environment with them -- so it is returned as a
    leaf, and callers that need an operation reject it themselves.
    """
    for _ in range(_MAX_DEPTH):
        if not isinstance(v, ir.BlockArgument):
            return v, frame
        if frame is None:
            return v, frame
        binding = frame[0].get(v)
        if binding is None:
            raise _Reject("unbound block argument")
        v, frame = binding
    raise _Reject("call frames nested too deep")


def _callee(interp, op, frame):
    """`(returned_value, inner_frame)` for a call-like op."""
    attr = "callee" if op.name == "func.call" else "decomposition"
    try:
        name = ir.FlatSymbolRefAttr(op.attributes[attr]).value
    except Exception:
        raise _Reject(f"{op.name} without a resolvable callee")
    fn = interp.funcs.get(name)
    if fn is None or len(op.results) != 1:
        raise _Reject("unresolvable or multi-result call")
    blocks = fn.regions[0].blocks
    if len(blocks) != 1:
        raise _Reject("multi-block callee")
    body = blocks[0]
    term = list(body.operations)[-1].operation
    if len(term.operands) != 1:
        raise _Reject("callee returns several values")
    return term.operands[0], ({a: (v, frame) for a, v
                               in zip(body.arguments, op.operands)}, frame)


def _splat_float(interp, v, frame):
    """Value of a splat float constant reached through converts, else None."""
    for _ in range(_MAX_DEPTH):
        try:
            v, frame = _deref(v, frame)
        except _Reject:
            return None
        o = _owner(v)
        if o is None:
            return None
        if o.name in _IDENTITY or o.name == "stablehlo.broadcast_in_dim":
            v = o.operands[0]
            continue
        if o.name in _CALL:
            try:
                v, frame = _callee(interp, o, frame)
            except _Reject:
                return None
            continue
        if o.name != "stablehlo.constant":
            return None
        t = _ir.tensor_type(o.results[0])
        try:
            x = _ir.splat_scalar_np(o.attributes["value"], t)
            if x is None and not list(t.shape):
                x = _ir.dense_to_np(o.attributes["value"], t)
            return None if x is None else float(
                np.asarray(x, np.float64).reshape(()))
        except Exception:
            return None
    return None


def _reduce_kind(op) -> str | None:
    """"add" / "maximum" for a single-input reduce with that body, else None."""
    if op.name != "stablehlo.reduce" or len(op.operands) != 2:
        return None
    body = [o.operation for o in op.regions[0].blocks[0].operations]
    if len(body) != 2:
        return None
    if body[0].name in ("stablehlo.add", "stablehlo.maximum"):
        return body[0].name.split(".")[1]
    return None


# --------------------------------------------------------------------------
# axis roles
# --------------------------------------------------------------------------
#
# Every axis of the first dot gets an ATOM; a tensor's ROLE VECTOR gives, per
# dimension, the tuple of atoms that dimension carries -- empty for a unit
# dimension, several once a reshape has merged axes. Atoms:
#
#   ("b", i)  batching dim i of the first dot   (shared by Q and K)
#   ("d", i)  contracting dim i                 (the head dim; not in logits)
#   ("l", i)  free dim i of the first dot's lhs
#   ("r", i)  free dim i of the first dot's rhs
#   ("v", i)  free dim i of V in the second dot (the output feature dim)
#   ("*", n)  a broadcast axis: the value is REPLICATED along it


class _Atoms:
    __slots__ = ("size", "n")

    def __init__(self):
        self.size = {}
        self.n = 0

    def bcast(self, size):
        self.n += 1
        a = ("*", self.n)
        self.size[a] = size
        return a


def _is_bcast(roles) -> bool:
    return bool(roles) and all(a[0] == "*" for a in roles)


def _roles_transpose(roles, perm):
    return [roles[p] for p in perm]


def _roles_untranspose(roles, perm):
    out = [None] * len(perm)
    for i, p in enumerate(perm):
        out[p] = roles[i]
    return out


def _roles_reshape(in_shape, roles, out_shape, atoms):
    """Carry roles through a reshape by matching the atom stream.

    A reshape regroups one ordered stream of atoms, so merges
    (``[H, g] -> [H*g]``) and splits both work; splitting an ATOM does not.
    This is exact in both directions, so the inverse reshape uses it too.
    """
    stream = []
    for size, r in zip(in_shape, roles):
        stream.extend(r)
        if not r and size != 1:
            raise _Reject("a role-less axis is not unit-sized")
    out, i = [], 0
    for size in out_shape:
        if size == 1:
            out.append(())
            continue
        got, take = 1, []
        while got < size:
            if i >= len(stream):
                raise _Reject("reshape does not line up with the axis roles")
            got *= atoms.size[stream[i]]
            take.append(stream[i])
            i += 1
        if got != size:
            raise _Reject("reshape splits an axis")
        out.append(tuple(take))
    while i < len(stream):
        if atoms.size[stream[i]] != 1:
            raise _Reject("reshape drops a non-unit axis")
        i += 1
    return out


def _roles_broadcast(in_shape, roles, out_shape, dims, atoms):
    out = [None] * len(out_shape)
    for i, d in enumerate(dims):
        if in_shape[i] == out_shape[d]:
            out[d] = roles[i]
        elif in_shape[i] == 1:
            out[d] = (atoms.bcast(out_shape[d]),)
        else:
            raise _Reject("broadcast_in_dim is not a pure expansion")
    for d, r in enumerate(out):
        if r is None:
            out[d] = () if out_shape[d] == 1 else (atoms.bcast(out_shape[d]),)
    return out


def _roles_unbroadcast(out_roles, in_shape, out_shape, dims):
    """Roles of a broadcast's INPUT: axes it expanded carry nothing."""
    inn = []
    for i, d in enumerate(dims):
        if in_shape[i] == out_shape[d]:
            inn.append(out_roles[d])
        elif in_shape[i] == 1:
            inn.append(())
        else:
            raise _Reject("broadcast_in_dim is not a pure expansion")
    return inn


def _recipe(roles, groups, atoms):
    """`(perm, shape)` reshaping a tensor with `roles` into `groups`.

    `groups` is a list of atom lists, one per target dimension. Every atom of
    every group must appear, and the target order must be reachable by a
    permutation -- atoms merged into one source dimension have to stay
    adjacent.
    """
    flat = [a for g in groups for a in g]
    pos = {a: i for i, a in enumerate(flat)}
    if len(pos) != len(flat):
        raise _Reject("duplicate axis role")
    order, units = [], []
    for d, r in enumerate(roles):
        if not r:
            units.append(d)
            continue
        for a in r:
            if a not in pos:
                raise _Reject("axis is not part of the attention")
        order.append((pos[r[0]], d, r))
    order.sort()
    if [a for _, _, r in order for a in r] != flat:
        raise _Reject("axes cannot be permuted into the fused layout")
    perm = [d for _, d, _ in order] + units
    shape = [math.prod(atoms.size[a] for a in g) for g in groups]
    # Everything here is static, so decide NOW whether either step is a
    # no-op: `emit` runs per attention per execute, and an identity
    # `mx.transpose` still builds a graph node.
    got = [math.prod(atoms.size[a] for a in roles[d]) for d in perm]
    return (None if perm == list(range(len(perm))) else perm,
            None if got == shape else shape)


# --------------------------------------------------------------------------
# walking
# --------------------------------------------------------------------------


class _Cand:
    """Scratch state for one candidate match."""

    __slots__ = ("interp", "atoms", "ops", "k_atom", "dot1", "lroles",
                 "rroles", "scale", "mask", "mask_mul", "exp_val",
                 "exp_roles", "chain")

    def __init__(self, interp):
        self.interp = interp
        self.atoms = _Atoms()
        self.ops = {}
        # Every value on the logits chain, with its roles: the softmax's max
        # is not always reduced from the very value the subtract reads
        # (jax.nn.dot_product_attention transposes in between), so any of
        # them is a legitimate anchor.
        self.chain = {}
        self.k_atom = None
        self.dot1 = None
        self.lroles = None
        self.rroles = None
        self.scale = 1.0
        self.mask = None
        self.mask_mul = 1.0
        self.exp_val = None
        self.exp_roles = None

    def absorb(self, op, frame):
        # Ops inside a callee are never absorbed: the callee is shared with
        # every other call site, and skipping the CALL is what keeps this
        # one's copy from running.
        if frame is None:
            self.ops[_okey(op)] = op


def _down(m, v, frame, stop=None, enter_calls=True):
    """Walk down through transparent ops. Returns `(steps, value, frame)`.

    `steps` is outermost-first; each entry is `(op, frame_at_that_op)`.
    `stop` halts the walk at a specific value: without it a `func.call`
    wrapping the value (maxtext's ``@_where``) would be descended straight
    past, and the caller would compare against the callee's innards.
    """
    steps = []
    cur, cframe = _deref(v, frame)
    for _ in range(_MAX_DEPTH):
        if stop is not None and cframe is None and cur in stop:
            break
        o = _owner(cur)
        if o is None:
            break
        if o.name in _TRANSPARENT:
            steps.append((o, cframe))
            cur, cframe = _deref(o.operands[0], cframe)
            continue
        if o.name in _CALL:
            if not enter_calls:
                break
            inner, iframe = _callee(m.interp, o, cframe)
            steps.append((o, cframe))
            cur, cframe = _deref(inner, iframe)
            continue
        if o.name in _CLAMP:
            ident = _CLAMP[o.name]
            keep = None
            for i in (0, 1):
                s = _splat_float(m.interp, o.operands[1 - i], cframe)
                if s is not None and s == ident:
                    keep = i
                    break
            if keep is None:
                break
            steps.append((o, cframe))
            cur, cframe = _deref(o.operands[keep], cframe)
            continue
        break
    return steps, cur, cframe


def _up(m, roles, shape, steps):
    """Propagate roles up through `steps` (outermost-first)."""
    for op, _ in reversed(steps):
        n = op.name
        if n in _SAME_SHAPE:
            continue
        out = _shape(op.results[0])
        if n == "stablehlo.transpose":
            roles = _roles_transpose(roles, _ir.i64_list(op, "permutation"))
        elif n == "stablehlo.reshape":
            roles = _roles_reshape(shape, roles, out, m.atoms)
        else:
            roles = _roles_broadcast(shape, roles, out,
                                     _ir.i64_list(op, "broadcast_dimensions"),
                                     m.atoms)
        shape = out
    return roles


def _down_roles(m, roles, shape, steps):
    """Propagate roles DOWN through `steps` (outermost-first)."""
    for op, _ in steps:
        n = op.name
        if n in _SAME_SHAPE:
            continue
        inn = _shape(op.operands[0])
        if n == "stablehlo.transpose":
            roles = _roles_untranspose(roles, _ir.i64_list(op, "permutation"))
        elif n == "stablehlo.reshape":
            roles = _roles_reshape(shape, roles, inn, m.atoms)
        else:
            roles = _roles_unbroadcast(
                roles, inn, shape, _ir.i64_list(op, "broadcast_dimensions"))
        shape = inn
    return roles


def _absorb_steps(m, steps):
    for op, frame in steps:
        m.absorb(op, frame)


def _aligned(got, want, allowed_missing):
    """`got` may only differ from `want` by replicating `allowed_missing`."""
    if len(got) != len(want):
        raise _Reject("rank mismatch in the softmax alignment")
    for a, b in zip(got, want):
        if a == b:
            continue
        if _is_bcast(a) and b and all(x in allowed_missing for x in b):
            continue
        raise _Reject("softmax term is not aligned with the logits")


# --------------------------------------------------------------------------
# the logits path (first dot -> softmax input)
# --------------------------------------------------------------------------


def _logits(m, v, frame, depth=0):
    """Role vector of a value on the first-dot -> softmax path.

    Records every outer value it passes through, so the softmax's max
    reduction can be anchored to whichever of them it actually reads.
    """
    if depth > _MAX_DEPTH:
        raise _Reject("logits chain too deep")
    v, frame = _deref(v, frame)
    roles = _logits_inner(m, v, frame, depth)
    if frame is None:
        m.chain[v] = roles
    return roles


def _logits_inner(m, v, frame, depth):
    o = _owner(v)
    if o is None:
        raise _Reject("logits path reaches a block argument")
    name = o.name

    if name == "stablehlo.dot_general":
        return _at_dot1(m, o, frame)

    if name in _CALL:
        inner, iframe = _callee(m.interp, o, frame)
        roles = _logits(m, inner, iframe, depth + 1)
        m.absorb(o, frame)
        return roles

    if name in _IDENTITY:
        roles = _logits(m, o.operands[0], frame, depth + 1)
        m.absorb(o, frame)
        return roles

    if name == "stablehlo.transpose":
        roles = _logits(m, o.operands[0], frame, depth + 1)
        m.absorb(o, frame)
        return _roles_transpose(roles, _ir.i64_list(o, "permutation"))

    if name == "stablehlo.reshape":
        roles = _logits(m, o.operands[0], frame, depth + 1)
        m.absorb(o, frame)
        return _roles_reshape(_shape(o.operands[0]), roles,
                              _shape(o.results[0]), m.atoms)

    if name == "stablehlo.multiply":
        for i in (0, 1):
            s = _splat_float(m.interp, o.operands[1 - i], frame)
            if s is None:
                continue
            saved = _snapshot(m)
            try:
                roles = _logits(m, o.operands[i], frame, depth + 1)
            except _Reject:
                # A failed branch must leave nothing behind: absorbing an op
                # the rewrite then does not replace would delete it.
                _restore(m, saved)
                continue
            m.scale *= s
            if m.mask is not None:
                # This factor multiplies an already-applied mask as well:
                # (L + mask) * s == L*s + mask*s.
                m.mask_mul *= s
            m.absorb(o, frame)
            return roles
        raise _Reject("neither multiply operand is a splat scale")

    if name == "stablehlo.divide":
        # `logits / sqrt(D)`. MLX only takes a multiplicative scale, so the
        # reciprocal is formed here -- exact whenever the divisor is a power
        # of two (every D in {4, 16, 64, 256}), and otherwise off by at most
        # one ULP of the logits, far inside the softmax's own error.
        s = _splat_float(m.interp, o.operands[1], frame)
        if s is None or s == 0.0:
            raise _Reject("divide by a non-splat on the logits path")
        roles = _logits(m, o.operands[0], frame, depth + 1)
        f = 1.0 / s
        m.scale *= f
        if m.mask is not None:
            m.mask_mul *= f
        m.absorb(o, frame)
        return roles

    if name == "stablehlo.add":
        for i in (0, 1):
            saved = _snapshot(m)
            try:
                roles = _logits(m, o.operands[i], frame, depth + 1)
            except _Reject:
                _restore(m, saved)
                continue
            if m.mask is not None:
                raise _Reject("two masks on one logits path")
            m.mask = ("add", o.operands[1 - i], frame, list(roles), 0.0)
            m.absorb(o, frame)
            return roles
        raise _Reject("neither add operand leads to the first dot")

    if name == "stablehlo.select":
        try:
            roles = _logits(m, o.operands[1], frame, depth + 1)
        except _Reject:
            # select(pred, C, L) would need the predicate negated; jax does
            # not emit it and guessing is not worth a silent sign error.
            raise _Reject("select's true branch does not lead to the dot")
        if m.mask is not None:
            raise _Reject("two masks on one logits path")
        c = _splat_float(m.interp, o.operands[2], frame)
        if c is None:
            raise _Reject("select's false branch is not a splat")
        el = _el(o.results[0])
        if el not in _FLOAT_ELS:
            raise _Reject(f"select on {el}")
        if not (c < 0 and (math.isinf(c)
                           or -c >= _MASK_FRACTION * _FINITE_MAX[el])):
            # Not a mask sentinel: `select` and `add` are then genuinely
            # different functions, so this has to run literally.
            raise _Reject(f"select constant {c} is not a mask sentinel")
        m.mask = ("select", o.operands[0], frame, list(roles), c)
        m.absorb(o, frame)
        return roles

    raise _Reject(f"{name} on the logits path")


def _at_dot1(m, dot, frame):
    if frame is not None:
        raise _Reject("the first dot is inside a callee")
    if m.dot1 is not None:
        raise _Reject("two dots on the logits path")
    from metaljax.ops.linalg import _dot_dims
    lb, rb, lc, rc = _dot_dims(dot)
    lhs, rhs = dot.operands
    lshape, rshape = _shape(lhs), _shape(rhs)
    if not lc or len(lc) != len(rc) or len(lb) != len(rb):
        raise _Reject("degenerate dot dimensions")
    if any(lshape[a] != rshape[b] for a, b in zip(lc, rc)):
        raise _Reject("contracting dimensions disagree")
    if any(lshape[a] != rshape[b] for a, b in zip(lb, rb)):
        raise _Reject("batching dimensions disagree")
    if _el(lhs) not in _FLOAT_ELS or _el(rhs) not in _FLOAT_ELS:
        raise _Reject("non-float attention")
    if any(s == 0 for s in lshape + rshape):
        raise _Reject("empty attention")

    A = m.atoms
    lfree = [d for d in range(len(lshape)) if d not in lb and d not in lc]
    rfree = [d for d in range(len(rshape)) if d not in rb and d not in rc]
    for i, d in enumerate(lb):
        A.size[("b", i)] = lshape[d]
    for i, d in enumerate(lc):
        A.size[("d", i)] = lshape[d]
    for i, d in enumerate(lfree):
        A.size[("l", i)] = lshape[d]
    for i, d in enumerate(rfree):
        A.size[("r", i)] = rshape[d]

    lroles = [None] * len(lshape)
    rroles = [None] * len(rshape)
    for i, d in enumerate(lb):
        lroles[d] = (("b", i),)
    for i, d in enumerate(rb):
        rroles[d] = (("b", i),)
    for i, d in enumerate(lc):
        lroles[d] = (("d", i),)
    for i, d in enumerate(rc):
        rroles[d] = (("d", i),)
    # Free axes of size 1 carry no atom: jax inserts them freely (rank-5
    # `jax.nn.dot_product_attention`) and they must not become heads.
    for i, d in enumerate(lfree):
        lroles[d] = () if lshape[d] == 1 else (("l", i),)
    for i, d in enumerate(rfree):
        rroles[d] = () if rshape[d] == 1 else (("r", i),)

    m.dot1 = dot
    m.lroles, m.rroles = lroles, rroles
    m.absorb(dot, frame)
    return ([(("b", i),) for i in range(len(lb))]
            + [lroles[d] for d in lfree] + [rroles[d] for d in rfree])


# --------------------------------------------------------------------------
# the softmax reductions
# --------------------------------------------------------------------------


def _check_reduce(m, v, frame, kind, anchors, want_roles, missing=None):
    """Verify `v` is `broadcast(reduce(anchor, kind))` reduced over `k`.

    Returns the roles of the reduce's result. `anchors` maps the values the
    reduce is allowed to be computed from to their roles -- the logits chain
    for the max, the exponential alone for the sum. Keeping the two sets
    apart is load-bearing: a sum reduced over the LOGITS would otherwise
    validate as the softmax denominator and silently change the result.
    `missing` is the set of atoms the broadcast back up may replicate (the
    key axis for the max and for a pre-dot normalization; V's feature axes
    when the normalization is deferred past the second dot); None means "the
    key axis", which the max reduction is what discovers.
    """
    steps, base, bframe = _down(m, v, frame)
    if bframe is not None:
        raise _Reject("the softmax reduction lives inside a callee")
    red = _owner(base)
    if red is None or _reduce_kind(red) != kind:
        raise _Reject(f"softmax has no {kind} reduction")
    init = _splat_float(m.interp, red.operands[1], None)
    if kind == "add":
        # A non-zero init would change the denominator outright.
        if init != 0.0:
            raise _Reject("sum reduction has a non-zero init")
    elif init is None or not (math.isinf(init) and init < 0):
        raise _Reject("max reduction does not start at -inf")

    isteps, ibase, iframe = _down(m, red.operands[0], None, stop=anchors)
    if iframe is not None or ibase not in anchors:
        raise _Reject(f"{kind} reduction is not computed from the softmax input")
    in_shape = _shape(red.operands[0])
    in_roles = _up(m, anchors[ibase], _shape(ibase), isteps)
    if len(in_roles) != len(in_shape):
        raise _Reject("reduce input roles do not match its rank")

    dims = _ir.i64_list(red, "dimensions")
    flat = [a for d in dims for a in in_roles[d]]
    if m.k_atom is None:
        if len(flat) != 1:
            raise _Reject("softmax reduces more than one axis")
        m.k_atom = flat[0]
    if flat != [m.k_atom]:
        raise _Reject("softmax does not reduce the key axis")

    out_roles = [r for d, r in enumerate(in_roles) if d not in dims]
    out_shape = [s for d, s in enumerate(in_shape) if d not in dims]
    roles = _up(m, out_roles, out_shape, steps)
    _aligned(roles, want_roles,
             {m.k_atom} if missing is None else missing)
    _absorb_steps(m, steps)
    _absorb_steps(m, isteps)
    m.absorb(red, None)
    return roles


# --------------------------------------------------------------------------
# the probability path (softmax output -> second dot)
# --------------------------------------------------------------------------


def _probs(m, v, frame, depth=0):
    """Walk a second-dot operand down to the `exp`.

    Returns `(roles, normalized)`; `normalized` says the softmax division
    already happened, so the second dot's result is the final output.
    """
    if depth > _MAX_DEPTH:
        raise _Reject("probability chain too deep")
    v, frame = _deref(v, frame)
    o = _owner(v)
    if o is None:
        raise _Reject("probability path reaches a block argument")
    name = o.name

    if name == "stablehlo.exponential":
        steps, sub, sframe = _down(m, o.operands[0], frame)
        so = _owner(sub)
        if so is None or so.name != "stablehlo.subtract":
            raise _Reject("exp's operand is not a max subtraction")
        if sframe is not None or frame is not None:
            raise _Reject("the softmax lives inside a callee")
        roles = _logits(m, so.operands[0], sframe)
        # The max reduction is what discovers the key axis (hence which
        # operand of the first dot is K); everything downstream verifies
        # against it.
        _check_reduce(m, so.operands[1], sframe, "maximum", m.chain, roles)
        _absorb_steps(m, steps)
        m.absorb(so, sframe)
        m.absorb(o, frame)
        m.exp_val = o.results[0]
        m.exp_roles = _up(m, roles, _shape(so.operands[0]), steps)
        return list(m.exp_roles), False

    if name in _IDENTITY:
        roles, norm = _probs(m, o.operands[0], frame, depth + 1)
        m.absorb(o, frame)
        return roles, norm

    if name in _CALL:
        inner, iframe = _callee(m.interp, o, frame)
        roles, norm = _probs(m, inner, iframe, depth + 1)
        m.absorb(o, frame)
        return roles, norm

    if name == "stablehlo.transpose":
        roles, norm = _probs(m, o.operands[0], frame, depth + 1)
        m.absorb(o, frame)
        return _roles_transpose(roles, _ir.i64_list(o, "permutation")), norm

    if name == "stablehlo.reshape":
        roles, norm = _probs(m, o.operands[0], frame, depth + 1)
        m.absorb(o, frame)
        return (_roles_reshape(_shape(o.operands[0]), roles,
                               _shape(o.results[0]), m.atoms), norm)

    if name == "stablehlo.divide":
        roles, norm = _probs(m, o.operands[0], frame, depth + 1)
        if norm:
            raise _Reject("two normalizations")
        _check_reduce(m, o.operands[1], frame, "add",
                      {m.exp_val: m.exp_roles}, roles)
        m.absorb(o, frame)
        return roles, True

    raise _Reject(f"{name} on the probability path")


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


class Match:
    """One recognized fused attention."""

    __slots__ = ("key", "q", "k", "v", "q_rec", "k_rec", "v_rec",
                 "scale", "mask", "mask_rec", "out_rec", "dtype", "out_dtype",
                 "ops", "name")


def _snapshot(m):
    return (dict(m.ops), m.k_atom, m.dot1, m.lroles, m.rroles, m.scale,
            m.mask, m.mask_mul, m.exp_val, m.exp_roles, m.atoms.n,
            dict(m.atoms.size), dict(m.chain))


def _restore(m, s):
    (m.ops, m.k_atom, m.dot1, m.lroles, m.rroles, m.scale, m.mask,
     m.mask_mul, m.exp_val, m.exp_roles, m.atoms.n, m.atoms.size,
     m.chain) = s


def _try(interp, dot2) -> Match:
    """Read `dot2` as attention's probabilities-times-values dot, or reject.

    Either operand can be the probabilities: jax puts them on the right in
    `jax.nn.dot_product_attention` and maxtext, on the left in hand-rolled
    einsum attention.
    """
    for side in (0, 1):
        try:
            return _try_side(interp, dot2, side)
        except _Reject:
            continue
    raise _Reject("not attention")


def _dot2_roles(p_roles, v_roles, pb, pc, vb, vc, pside, pshape, vshape):
    """Role vector of the second dot's result."""
    p_free = [p_roles[d] for d in range(len(pshape))
              if d not in pb and d not in pc]
    v_free = [v_roles[d] for d in range(len(vshape))
              if d not in vb and d not in vc]
    lhs_free, rhs_free = (p_free, v_free) if pside == 0 else (v_free, p_free)
    return [p_roles[d] for d in pb] + lhs_free + rhs_free


def _try_side(interp, dot2, pside) -> Match:
    from metaljax.ops.linalg import _dot_dims
    m = _Cand(interp)
    p_roles, normalized = _probs(m, dot2.operands[pside], None)
    if m.k_atom is None:
        raise _Reject("no key axis")

    lb, rb, lc, rc = _dot_dims(dot2)
    pb, pc = (lb, lc) if pside == 0 else (rb, rc)
    vb, vc = (rb, rc) if pside == 0 else (lb, lc)
    vval = dot2.operands[1 - pside]
    vshape, pshape = _shape(vval), _shape(dot2.operands[pside])
    if len(p_roles) != len(pshape):
        raise _Reject("probability roles do not match the second dot")
    if _el(vval) not in _FLOAT_ELS or any(s == 0 for s in vshape):
        raise _Reject("empty or non-float values")
    if len(pb) != len(vb) or len(pc) != len(vc) or not pc:
        raise _Reject("degenerate second dot")

    # The probabilities contract with V over the key axis...
    if [a for d in pc for a in p_roles[d]] != [m.k_atom]:
        raise _Reject("the second dot does not contract the key axis")
    if math.prod(vshape[d] for d in vc) != m.atoms.size[m.k_atom]:
        raise _Reject("the second dot contracts the wrong extent")

    A = m.atoms
    vfree = [d for d in range(len(vshape)) if d not in vb and d not in vc]
    vroles = [None] * len(vshape)
    # ...and share every batch/head axis with them.
    shared = []
    for i, d in enumerate(vb):
        vroles[d] = p_roles[pb[i]]
        shared.extend(vroles[d])
        if vshape[d] != math.prod(A.size[a] for a in vroles[d]):
            raise _Reject("second dot batching sizes disagree")
    for i, d in enumerate(vc):
        # V's contracting axes carry the key atom: the first takes it and the
        # rest have to be unit, so that one recipe can place it.
        if i == 0:
            vroles[d] = (m.k_atom,)
        elif vshape[d] != 1:
            raise _Reject("V contracts several non-unit axes")
        else:
            vroles[d] = ()
    for i, d in enumerate(vfree):
        A.size[("v", i)] = vshape[d]
        vroles[d] = () if vshape[d] == 1 else (("v", i),)
    v_atoms = [a for d in vfree for a in vroles[d]]

    lb1, _rb1, lc1, _rc1 = _dot_dims(m.dot1)
    b_atoms = [("b", i) for i in range(len(lb1))]
    d_atoms = [("d", i) for i in range(len(lc1))]
    if sorted(shared) != sorted(b_atoms):
        raise _Reject("batch axes are not shared with the values")

    if m.k_atom[0] == "l":
        q_roles, k_roles = m.rroles, m.lroles
        qval, kval = m.dot1.operands[1], m.dot1.operands[0]
    else:
        q_roles, k_roles = m.lroles, m.rroles
        qval, kval = m.dot1.operands[0], m.dot1.operands[1]
    if [a for r in k_roles for a in r if a[0] in ("l", "r")] != [m.k_atom]:
        raise _Reject("the key operand has extra free axes")
    q_free = [a for r in q_roles for a in r if a[0] in ("l", "r")]
    p_free = [a for d in range(len(pshape)) if d not in pb and d not in pc
              for a in p_roles[d]]
    if sorted(p_free) != sorted(q_free):
        raise _Reject("query axes are not free in the second dot")
    if _el(qval) != _el(vval):
        raise _Reject("mixed operand dtypes")

    q_atom, mask = _pick_query_axis(m, q_free)
    extra = [a for a in q_free if a != q_atom]
    qslot = [q_atom] if q_atom else []

    d2_roles = _dot2_roles(p_roles, vroles, pb, pc, vb, vc, pside,
                           pshape, vshape)
    if normalized:
        root, root_roles = dot2, d2_roles
    else:
        root, root_roles = _find_norm(m, dot2, d2_roles, set(v_atoms))
    m.absorb(dot2, None)
    out_shape = _shape(root.results[0])

    mm = Match()
    mm.q, mm.k, mm.v = qval, kval, vval
    mm.scale = m.scale
    mm.key = _okey(root)
    mm.dtype = dtypes.mx_result_dtype(qval)
    mm.out_dtype = dtypes.mx_result_dtype(root.results[0])
    mm.ops = dict(m.ops)
    mm.ops.pop(mm.key, None)

    # Splitting the batching axes between MLX's B and N is FREE for
    # correctness -- the kernel is independent per (b, n) pair, and the
    # `h // g` head pairing holds for any split because the kv-head axes
    # stay MAJOR within N. It is NOT free for expressiveness: a mask must
    # cover a whole slot or none of it, so a bias varying along an outer
    # batching axis needs that axis in B. Try the conventional split
    # (last batching axis = the head) first, then the rest.
    natural = max(len(b_atoms) - 1, 0)
    last = _Reject("no batch/head split expresses this attention")
    for s in [natural] + [x for x in range(len(b_atoms) + 1) if x != natural]:
        batch_atoms, head_atoms = b_atoms[:s], b_atoms[s:]
        # The query heads are (kv head, group) with the kv head MAJOR, which
        # is exactly MLX's `h // g` pairing for grouped-query attention.
        n_atoms = head_atoms + extra
        try:
            mm.q_rec = _recipe(q_roles, [batch_atoms, n_atoms, qslot,
                                         d_atoms], A)
            mm.k_rec = _recipe(k_roles, [batch_atoms, head_atoms,
                                         [m.k_atom], d_atoms], A)
            mm.v_rec = _recipe(vroles, [batch_atoms, head_atoms,
                                        [m.k_atom], v_atoms], A)
            mm.mask, mm.mask_rec = _mask_recipe(m, mask, batch_atoms,
                                                n_atoms, qslot, A)
            mm.out_rec = _out_recipe(
                root_roles, batch_atoms + n_atoms + qslot + v_atoms, A,
                out_shape,
                [math.prod(A.size[a] for a in g)
                 for g in (batch_atoms, n_atoms, qslot, v_atoms)])
        except _Reject as e:
            last = e
            continue
        mm.name = (f"B{math.prod(A.size[a] for a in batch_atoms)}"
                   f"N{math.prod(A.size[a] for a in n_atoms)}"
                   f"Q{A.size[q_atom] if q_atom else 1}"
                   f"K{A.size[m.k_atom]}D{math.prod(A.size[a] for a in d_atoms)}")
        return mm
    raise last


def _pick_query_axis(m, q_free):
    """Choose the query-sequence atom, and resolve the mask to its base.

    Every query-side axis indexes an independent softmax row, so with no mask
    the split between "the query axis" and "extra head axes" is free, and the
    largest axis is taken (the one MLX's kernel wants to tile over). A mask
    forces the choice: MLX broadcasts it against `[B, N, Tq, Tk]`, so the
    axis the mask varies along has to be `Tq`.
    """
    if m.mask is None:
        return (max(q_free, key=lambda a: m.atoms.size[a], default=None),
                None)
    kind, mv, mframe, mroles, const = m.mask
    # Never descend into a callee here: the mask only has to be reduced to a
    # value `emit` can read and to the axes it varies along, and jax wraps
    # the mask's own construction in calls (`@tril`) whose insides are not
    # ours to skip.
    steps, base, bframe = _down(m, mv, mframe, enter_calls=False)
    if bframe is not None:
        raise _Reject("the mask is computed inside a callee")
    base_roles = _down_roles(m, list(mroles), _shape(mv), steps)
    if len(base_roles) != len(_shape(base)):
        raise _Reject("mask roles do not match its rank")
    have = {a for r in base_roles for a in r}
    varying = [a for a in q_free if a in have]
    if len(varying) > 1:
        raise _Reject("the mask varies along several query axes")
    q_atom = (varying[0] if varying else
              max(q_free, key=lambda a: m.atoms.size[a], default=None))
    _absorb_steps(m, steps)
    return q_atom, (kind, base, base_roles, const, m.mask_mul)


def _mask_recipe(m, mask, batch_atoms, n_atoms, qslot, atoms):
    """Lay the mask out as the rank-4 tensor MLX broadcasts.

    Each of the four slots must be either the whole slot or absent (size 1).
    A mask varying along only PART of a slot -- one head axis of a GQA pair,
    say -- could only be expressed by materializing it to the full slot,
    which is the allocation this recognizer exists to avoid.
    """
    if mask is None:
        return None, None
    kind, base, roles, const, mul = mask
    have = {a for r in roles for a in r}
    groups = []
    for slot in (batch_atoms, n_atoms, qslot, [m.k_atom]):
        present = [a for a in slot if a in have]
        if present and len(present) != len(slot):
            raise _Reject("the mask varies along part of a broadcast axis")
        groups.append(present)
    return (kind, base, const, mul), _recipe(roles, groups, atoms)


def _find_norm(m, dot2, roles, v_atoms):
    """Forward from the second dot's result to the deferred normalization.

    Real LLM lowerings divide by the softmax denominator AFTER the values
    dot (`O = (e @ V) / broadcast(sum e)`) -- legal because the denominator
    is constant along V's feature axis, and cheaper when there are more keys
    than features.
    """
    queue = [(dot2.results[0], roles, _shape(dot2.results[0]), [])]
    seen = 0
    while queue:
        v, r, shp, path = queue.pop(0)
        seen += 1
        if seen > _MAX_FANOUT:
            raise _Reject("normalization search too wide")
        for use in v.uses:
            u = use.owner.operation
            if u.name == "stablehlo.divide" and u.operands[0] == v:
                saved = _snapshot(m)
                try:
                    _check_reduce(m, u.operands[1], None, "add",
                                  {m.exp_val: m.exp_roles}, r, v_atoms)
                except _Reject:
                    _restore(m, saved)
                    continue
                for op in path:
                    m.absorb(op, None)
                return u, r
            # Forward through pure reassociation only: a broadcast on this
            # path would replicate the output, which is not a normalization.
            if u.name not in _FORWARD or u.operands[0] != v:
                continue
            queue.append((u.results[0], _up(m, r, shp, [(u, None)]),
                          _shape(u.results[0]), path + [u]))
    raise _Reject("no softmax normalization after the second dot")


def _out_recipe(roles, order, atoms, out_shape, mlx_shape):
    """`(pre_shape, perm, post_shape)` turning the fused `[B, N, Tq, Dv]`
    result -- whose atoms, expanded, are `order` -- into the root's layout.
    Any step that is a no-op comes back as None."""
    pre = [atoms.size[a] for a in order]
    idx = {a: i for i, a in enumerate(order)}
    perm = []
    for r in roles:
        for a in r:
            if a not in idx:
                raise _Reject("a result axis is not produced by the kernel")
            perm.append(idx[a])
    if sorted(perm) != list(range(len(order))):
        raise _Reject("result axes do not cover the fused output")
    # As in `_recipe`: all static, so the no-op steps are dropped here
    # rather than re-tested on every execute.
    permuted = [pre[p] for p in perm]
    return (None if pre == list(mlx_shape) else pre,
            None if perm == list(range(len(perm))) else perm,
            None if permuted == list(out_shape) else list(out_shape))



# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


class State:
    __slots__ = ("matches", "skip", "roots", "active")

    def __init__(self):
        self.matches = []
        self.skip = frozenset()
        self.roots = {}
        self.active = False

    def rebuild(self):
        self.roots = {m.key: m for m in self.matches}
        skip = set()
        for m in self.matches:
            skip.update(m.ops)
        self.skip = frozenset(skip)
        self.active = bool(self.roots)


def _all_blocks(interp):
    """Every block of every function in the module.

    Attention does not have to sit in @main: maxtext puts a whole decode
    step inside a `@closed_call`. Rewriting inside a callee is safe because
    the rewrite is a pure function of the IR -- it carries no per-call-site
    state -- and `run_block` seeds the callee's environment with its own
    block arguments, which is all `emit` ever reads.
    """
    for fn in interp.funcs.values():
        for region in fn.regions:
            for block in region.blocks:
                yield from _walk_blocks(block)


def _claimed(interp):
    """Op keys the quantized-matmul recognizer has already spoken for.

    The two rewrites compose (a dequantized weight feeding attention's
    projections is fine), but neither may swallow an op the other emits, so
    an overlap simply drops the sdpa candidate.

    Deliberately reads every qmm CANDIDATE, including ones a later packing
    prologue may disable: qmm decides that against concrete buffers, long
    after this runs. The conservative cost is that an attention overlapping
    a quantized matmul that then falls back stays unfused for the life of
    the executable; the alternative is two rewrites claiming one op.
    """
    from metaljax import qmm
    try:
        st = qmm.analyze(interp)
    except Exception:
        return set()
    keys = set()
    for mm in st.matches:
        keys.update(mm.ops)
        keys.add(mm.key)
    return keys


def _escapes(m, keys) -> bool:
    """True if anything `m` absorbs is still read from outside the chain."""
    return any(_okey(use.owner.operation) not in keys
               for op in m.ops.values()
               for r in op.results
               for use in r.uses)


def _exclusive(matches):
    """Drop candidates whose absorbed ops are needed elsewhere.

    Every absorbed op must be consumed only by the chain itself: attention
    probabilities that also feed a dropout mask, or are returned as an
    auxiliary output, have to be materialized anyway.

    Iterated to a fixpoint, exactly as `qmm._prune` is. Dropping one
    candidate turns its root back into an ordinary op, which can be the
    outside consumer that disqualifies another; settling for a single pass
    would leave that other candidate skipping an op the now-literal root
    still reads, which is a KeyError in `run_block`, at execute time.
    """
    live = list(matches)
    while True:
        keys = set()
        for m in live:
            keys.update(m.ops)
            keys.add(m.key)
        kept = [m for m in live if not _escapes(m, keys)]
        if len(kept) == len(live):
            return kept
        if _DEBUG:
            gone = set(id(m) for m in kept)
            for m in live:
                if id(m) not in gone:
                    print(f"[metaljax] sdpa: dropped {m.name} (an "
                          f"intermediate is used outside the attention)",
                          flush=True)
        live = kept


def analyze(interp) -> State:
    """Structural analysis of a program, once."""
    st = interp._sdpa
    if st is not None:
        return st
    st = State()
    interp._sdpa = st
    if not ENABLED:
        return st
    try:
        with interp.context:
            claimed = _claimed(interp)
            cands = []
            for block in _all_blocks(interp):
                for op in block.operations:
                    o = op.operation
                    if o.name != "stablehlo.dot_general":
                        continue
                    try:
                        mm = _try(interp, o)
                    except _Reject:
                        continue
                    except Exception as e:      # never break a program
                        if _DEBUG:
                            print(f"[metaljax] sdpa: candidate failed ({e})",
                                  flush=True)
                        continue
                    if claimed & (set(mm.ops) | {mm.key}):
                        continue
                    cands.append(mm)
            # A candidate absorbed by another one cannot also be a root.
            absorbed = set()
            for mm in cands:
                absorbed.update(mm.ops)
            cands = [mm for mm in cands if mm.key not in absorbed]
            seen = set()
            uniq = []
            for mm in cands:
                if mm.key in seen:
                    continue
                seen.add(mm.key)
                uniq.append(mm)
            st.matches = _exclusive(uniq)
            STATS["recognized"] += len(st.matches)
            if _DEBUG and st.matches:
                names = ", ".join(mm.name for mm in st.matches[:4])
                print(f"[metaljax] sdpa: {len(st.matches)} fused attention(s)"
                      f" [{names}{'...' if len(st.matches) > 4 else ''}]",
                      flush=True)
    except Exception as e:
        if _DEBUG:
            print(f"[metaljax] sdpa: analysis failed ({e})", flush=True)
        st.matches = []
    st.rebuild()
    return st


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------


def _apply(x, rec):
    perm, shape = rec
    if perm is not None:
        x = mx.transpose(x, perm)
    if shape is not None:
        x = mx.reshape(x, shape)
    return x


def _mask_array(m, env):
    """The additive mask, built once per distinct mask per block.

    Every layer of a transformer shares one causal mask, and turning it into
    the additive form costs a full `[.., .., Tq, Tk]` tensor -- 8 MB at
    T=2048 in bf16, so 60 layers would allocate half a gigabyte per token to
    recompute the same thing. `env` is per-`run_block` and is only ever
    indexed, never iterated, so it is the natural place to keep it.
    """
    kind, base, const, mul = m.mask
    key = ("metaljax.sdpa.mask", base, kind, const, mul, m.dtype)
    mask = env.get(key)
    if mask is None:
        mask = env[base]
        if kind == "select":
            # Additive, never boolean: MLX's boolean mask gives a
            # fully-masked row a finite value where the literal chain gives
            # NaN. `L + C == C` for a sentinel C, so this is exact.
            mask = mx.where(mask, mx.array(0.0, m.dtype),
                            mx.array(const * mul, m.dtype))
        else:
            if mask.dtype != m.dtype:
                mask = mask.astype(m.dtype)
            if mul != 1.0:
                mask = mask * mul
        env[key] = mask
    return _apply(mask, m.mask_rec)


def emit(interp, m, env):
    """One `mx.fast.scaled_dot_product_attention` in place of the chain."""
    q = _apply(env[m.q], m.q_rec)
    k = _apply(env[m.k], m.k_rec)
    v = _apply(env[m.v], m.v_rec)
    if q.dtype != m.dtype:
        q = q.astype(m.dtype)
    if k.dtype != m.dtype:
        k = k.astype(m.dtype)
    if v.dtype != m.dtype:
        v = v.astype(m.dtype)
    mask = None if m.mask is None else _mask_array(m, env)

    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=m.scale,
                                               mask=mask)
    pre, perm, post = m.out_rec
    if pre is not None:
        out = mx.reshape(out, pre)
    if perm is not None:
        out = mx.transpose(out, perm)
    if post is not None:
        out = mx.reshape(out, post)
    if out.dtype != m.out_dtype:
        out = out.astype(m.out_dtype)
    STATS["fused"] += 1
    return [out]
