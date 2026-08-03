"""Recognizer that maps a DENSE mixture-of-experts dispatch onto gather_mm.

Every MoE layer written against `jax.numpy` (keras-hub's, maxtext's, the HF
ports) computes the expert block like this:

    scores = router(x)                      # [T, E], zero off the top-k
    y_e    = expert_e(x)   for EVERY e      # [E, T, H]  <- the dense part
    out    = sum_e scores[t, e] * y_e[t]    # unrouted terms are 0 * y

XLA lowers that literally: all `E` experts run for all `T` tokens and the
routing weights then null the `E - K` contributions that were never selected.
At decode (T = 1, K = 4, E = 32) that is `E/K` times the arithmetic and, far
worse, `E/K` times the weight traffic -- the whole expert set is streamed
from memory to produce values that get multiplied by zero. gemma4-26B-A4B
measured 51.6 GB streamed per token this way, and a 4B-active MoE decoded
slower on this backend than a dense 31B model.

This module recognizes the pattern structurally at compile time and replaces
the dense region with a gathered one: only the `K` selected experts per token
are ever touched, through `mx.gather_mm` (float weights) or `mx.gather_qmm`
(weights already packed by `metaljax.qmm`).

WHAT IS MATCHED. The root is the weighted sum over the expert axis,

    reduce(add, dims=[e])( multiply(Y, broadcast(S)) )

where `S` is *provably* zero off a top-k selection because it is built the
only way a top-k router can build it:

    values, indices = top_k(logits, K)              # [T, K]
    w               = f(values)                     # softmax, sigmoid, ...
    S               = reduce(add, dims=[k])( one_hot(indices, E) * w )

`one_hot` is matched down to its `compare EQ (indices, iota)` form, so

    S[t, e] = sum_k [indices[t, k] == e] * w[t, k]

is an identity rather than a guess: every position outside the selection is
an EXACT zero for every input, and the dense sum is term for term the
gathered sum plus `E - K` zero products.

`Y` is then re-planned in PAIR SPACE: the dense `(expert, token)` grid
collapses onto the `P = T * K` selected pairs. Each region value of dense
shape `S` becomes `[P] + S-without-its-expert-and-token-axes`; per-expert
data is indexed by `indices`, per-token data by the pair's token, and each
per-expert dot becomes one `gather_mm` / `gather_qmm`. Everything between the
dots (bias adds, gate splits, clamps, activations) is ordinary elementwise
work that then runs on `P` rows instead of `E * T`.

The traversal boundary is exactly "does this value carry the expert axis":
anything that does not (the token activations, a norm, a router output) is
computed ONCE, densely, outside the region and gathered on the way in --
never duplicated `K` times.

NUMERICS. The gathered sum runs over `K` terms in a different order than the
dense sum over `E`, so results differ at f32 reduction-order scale. One
semantic difference is deliberate: `0 * inf` and `0 * nan` are NaN, so the
dense path turns an UNROUTED expert's overflow into a NaN in the routed
output, while the gathered path -- which never evaluates that expert -- does
not. That divergence is towards correctness and is documented rather than
emulated (emulating it would mean evaluating the experts we are skipping).

VERIFICATION. The structure is the proof, but a misread axis would be
silent, so every match is also checked numerically before it is ever used:
in the eager prologue, on SYNTHETIC logits. The router tail from `top_k`
down to `S` depends on nothing but the logits, so it can be evaluated on a
random `[T, E]` draw with the interpreter's own handlers and compared
against `w` scattered at `indices`. That costs microseconds, needs no real
activations, and fails the match closed if any axis was read wrong.
METALJAX_MOE_VERIFY=0 skips it.

Anything not understood rejects the whole match and the dense region runs
exactly as before: capacity-factor routing, a region value consumed outside
the region, a dot whose weight is not per-expert, an op the pair-space
planner does not model.

METALJAX_MOE=0 disables the recognizer.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.qmm import _okey, _owner, _prod, _shape, _users, _walk_blocks

ENABLED = os.environ.get("METALJAX_MOE", "1") != "0"
_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"
# Numeric check of the router tail on synthetic logits, once per match.
_VERIFY = os.environ.get("METALJAX_MOE_VERIFY", "1") != "0"
_VERIFY_DRAWS = int(os.environ.get("METALJAX_MOE_VERIFY_DRAWS", "3"))

# Diagnostics (tests assert on these; nothing else reads them).
STATS = {"recognized": 0, "verified": 0, "fallbacks": 0,
         "gather_qmm": 0, "gather_mm": 0}


def stats() -> dict:
    return dict(STATS)


def reset_stats():
    for k in STATS:
        STATS[k] = 0


class _Reject(Exception):
    """This is not a gatherable MoE dispatch: run the dense region."""


# Shape-preserving ops whose metaljax handlers depend on nothing but the
# operand arrays, the result dtype and their own attributes -- so they can be
# replayed verbatim on pair-space arrays.
_ELEMENTWISE = frozenset("""
stablehlo.abs stablehlo.add stablehlo.and stablehlo.atan2 stablehlo.cbrt
stablehlo.ceil stablehlo.clamp stablehlo.compare stablehlo.convert
stablehlo.cosine stablehlo.count_leading_zeros stablehlo.divide
stablehlo.exponential stablehlo.exponential_minus_one stablehlo.floor
stablehlo.is_finite stablehlo.log stablehlo.log_plus_one stablehlo.logistic
stablehlo.maximum stablehlo.minimum stablehlo.multiply stablehlo.negate
stablehlo.not stablehlo.or stablehlo.popcnt stablehlo.power
stablehlo.remainder stablehlo.round_nearest_afz
stablehlo.round_nearest_even stablehlo.rsqrt stablehlo.select
stablehlo.shift_left stablehlo.shift_right_arithmetic
stablehlo.shift_right_logical stablehlo.sign stablehlo.sine stablehlo.sqrt
stablehlo.subtract stablehlo.tan stablehlo.tanh stablehlo.xor
chlo.erf chlo.erf_inv
""".split())

_CALLS = ("func.call", "stablehlo.composite")
# A top-k arrives as `chlo.top_k` from a direct lowering but as a
# `stablehlo.composite` wrapping the same op once the program has been
# through a portable artifact -- which is how every program reaches the
# plugin. Both are opaque here: the recognizer wants the op, not its
# decomposition.
_TOPK_COMPOSITES = ("chlo.top_k",)


def _composite_name(op) -> str:
    try:
        return _ir.str_attr(op, "name")
    except Exception:
        return ""


def _is_topk(op) -> bool:
    if op is None:
        return False
    if op.name == "chlo.top_k":
        return True
    return (op.name == "stablehlo.composite"
            and _composite_name(op) in _TOPK_COMPOSITES)


def _topk_k(op) -> int:
    if op.name == "chlo.top_k":
        return _ir.int_attr(op, "k")
    attrs = ir.DictAttr(op.attributes["composite_attributes"])
    return ir.IntegerAttr(attrs["k"]).value


# --------------------------------------------------------------------------
# call frames: everything below sees through func.call / composite
# --------------------------------------------------------------------------


class _Frame:
    """A callee scope: its block arguments bound to the caller's values."""

    __slots__ = ("uid", "bind")

    def __init__(self, uid, bind):
        self.uid = uid
        self.bind = bind


def _fid(frame):
    return 0 if frame is None else frame.uid


class _Scope:
    """Frame factory plus `deref`, the one way values are followed."""

    def __init__(self, interp):
        self.interp = interp
        self._next = 0
        self.calls = []          # root-frame call ops entered (to be skipped)

    def deref(self, v, frame):
        """`(value, frame, owner)` with block args and calls followed out.

        `owner` is None when the value is a block argument nothing in scope
        defines -- an @main argument, or a loop carry.
        """
        for _ in range(64):
            while isinstance(v, ir.BlockArgument) and frame is not None:
                b = frame.bind.get(v)
                if b is None:
                    return v, frame, None
                v, frame = b
            if isinstance(v, ir.BlockArgument):
                return v, frame, None
            o = _owner(v)
            if o is None or o.name not in _CALLS or _is_topk(o):
                return v, frame, o
            attr = "callee" if o.name == "func.call" else "decomposition"
            name = ir.FlatSymbolRefAttr(o.attributes[attr]).value
            fn = self.interp.funcs.get(name)
            if fn is None:
                return v, frame, o
            blk = fn.regions[0].blocks[0]
            term = list(blk.operations)[-1].operation
            i = list(o.results).index(v)
            if frame is None:
                self.calls.append(o)
            self._next += 1
            frame = _Frame(self._next,
                           {a: (x, frame)
                            for a, x in zip(blk.arguments, o.operands)})
            v = term.operands[i]
        raise _Reject("call nesting too deep")


# --------------------------------------------------------------------------
# structural helpers
# --------------------------------------------------------------------------


def _splat(op):
    """The scalar value of a rank-0 / splat constant op, else None."""
    if op is None or op.name != "stablehlo.constant":
        return None
    try:
        t = ir.RankedTensorType(op.results[0].type)
        arr = np.asarray(_ir.dense_to_np(op.attributes["value"], t))
        if arr.size != 1:
            return None
        return arr.reshape(-1)[0]
    except Exception:
        return None


def _reduce_info(op):
    """`(dimensions, body-op-name)` for a single-operand `stablehlo.reduce`."""
    if op is None or op.name != "stablehlo.reduce":
        raise _Reject("not a reduce")
    if len(op.operands) != 2 or len(op.results) != 1:
        raise _Reject("variadic reduce")
    body = [o.operation for o in op.regions[0].blocks[0].operations]
    if len(body) != 2 or body[1].name != "stablehlo.return":
        raise _Reject("reduce body is not a single op")
    return _ir.i64_list(op, "dimensions"), body[0].name


def _zero_sum(op):
    """The reduced axis of a `reduce(add, init=0)`; `_Reject` otherwise."""
    dims, body = _reduce_info(op)
    if body != "stablehlo.add":
        raise _Reject(f"reduce body is {body}, not add")
    if len(dims) != 1:
        raise _Reject("reduce spans more than one dimension")
    init = _splat(_owner(op.operands[1]))
    if init is None or float(init) != 0.0:
        raise _Reject("the sum does not start at zero")
    return dims[0]


def _peel(scope, v, frame, stop=None):
    """Peel broadcast / transpose / unit-reshape / convert, tracking axes.

    Returns `(base, frame, dims)` where `dims[j]` is the axis of the ORIGINAL
    value that base axis `j` lands in (None for a unit axis the chain
    dropped). Nothing is required of the chain beyond being one of those four
    kinds; the caller checks where the axes ended up.

    `stop` halts the walk at the first value of that shape. Callers that will
    READ the value at run time must pass it: a half-precision router casts
    its weights to bf16 before broadcasting them, and peeling past that
    convert would hand the rewrite the unrounded f32 tensor while the dense
    graph multiplies by the rounded one.
    """
    dims = list(range(len(_shape(v))))
    while True:
        v, frame, o = scope.deref(v, frame)
        if stop is not None and list(_shape(v)) == stop:
            return v, frame, dims
        if o is None:
            return v, frame, dims
        if o.name == "stablehlo.convert":
            v = o.operands[0]
            continue
        if o.name == "stablehlo.broadcast_in_dim":
            bd = _ir.i64_list(o, "broadcast_dimensions")
            dims = [dims[d] for d in bd]
            v = o.operands[0]
            continue
        if o.name == "stablehlo.transpose":
            perm = _ir.i64_list(o, "permutation")
            inv = [0] * len(perm)
            for i, p in enumerate(perm):
                inv[p] = i
            dims = [dims[inv[j]] for j in range(len(perm))]
            v = o.operands[0]
            continue
        if o.name == "stablehlo.reshape":
            src = _shape(o.operands[0])
            dst = _shape(o.results[0])
            if [d for d in src if d != 1] != [d for d in dst if d != 1]:
                return v, frame, dims
            keep = [i for i, d in enumerate(dst) if d != 1]
            out, n = [], 0
            for d in src:
                if d == 1:
                    out.append(None)
                else:
                    out.append(dims[keep[n]])
                    n += 1
            dims = out
            v = o.operands[0]
            continue
        return v, frame, dims


# --------------------------------------------------------------------------
# the router: proving S is zero off a top-k selection
# --------------------------------------------------------------------------


class _Router:
    __slots__ = ("indices", "weights", "topk", "tframe", "score", "sframe",
                 "T", "K", "E", "e_in_score", "t_in_score")


def _is_iota(scope, v, frame, axis, length):
    """True when `v` is an iota of `length` running along `axis`."""
    base, bframe, dims = _peel(scope, v, frame)
    _b, _f, o = scope.deref(base, bframe)
    if o is None or o.name != "stablehlo.iota":
        return False
    d = _ir.int_attr(o, "iota_dimension")
    shp = _shape(o.results[0])
    return (0 <= d < len(dims) and dims[d] == axis
            and shp[d] == length and _prod(shp) == length)


def _cmp_dir(op):
    from metaljax.ops.elementwise import _comparison_direction
    return _comparison_direction(op)


def _match_router(scope, score, sframe, se, st):
    """Prove `score` is `w` scattered at a top-k's indices.

    `se` / `st` are the axes of `score`'s OWN shape that carry the expert and
    the token dimension.
    """
    r = _Router()
    r.score, r.sframe = score, sframe
    r.e_in_score, r.t_in_score = se, st
    sshape = _shape(score)
    r.E, r.T = sshape[se], sshape[st]

    _v, frame, red = scope.deref(score, sframe)
    kd = _zero_sum(red)

    _mv, mframe, mul = scope.deref(red.operands[0], frame)
    while mul is not None and mul.name == "stablehlo.convert":
        # A half-precision router sums the one-hot product in f32. Nothing
        # here depends on the dtype: only `indices` and `weights` are read.
        _mv, mframe, mul = scope.deref(mul.operands[0], mframe)
    if mul is None or mul.name != "stablehlo.multiply":
        raise _Reject("router scores do not reduce a product")
    mshape = _shape(mul.results[0])
    if len(mshape) != 3:
        raise _Reject("the one-hot product is not rank 3")
    # The reduce drops `kd`; the surviving axes keep their relative order.
    surv = [i for i in range(3) if i != kd]
    e_ax, t_ax = surv[se], surv[st]
    r.K = mshape[kd]
    if mshape[e_ax] != r.E or mshape[t_ax] != r.T:
        raise _Reject("the one-hot product does not match the score shape")

    hot = None
    tk_shape = [r.T, r.K]
    for i in (0, 1):
        base, bframe, bdims = _peel(scope, mul.operands[i], mframe,
                                    stop=tk_shape)
        if bdims == [t_ax, kd] and list(_shape(base)) == tk_shape:
            if bframe is not None:
                raise _Reject("the routing weights live inside a callee")
            r.weights = base
            hot = 1 - i
            break
    if hot is None:
        raise _Reject("neither product operand is a [tokens, k] weight")

    _hv, hframe, cmp_op = scope.deref(mul.operands[hot], mframe)
    # keras' one_hot returns f32 and the caller converts it: in bf16 there
    # are two converts here (one inside the callee, one outside).
    while cmp_op is not None and cmp_op.name == "stablehlo.convert":
        _hv, hframe, cmp_op = scope.deref(cmp_op.operands[0], hframe)
    if cmp_op is None or cmp_op.name != "stablehlo.compare":
        raise _Reject("the one-hot is not a comparison")
    if _cmp_dir(cmp_op) != "EQ":
        raise _Reject("the one-hot comparison is not EQ")
    if list(_shape(cmp_op.results[0])) != list(mshape):
        raise _Reject("the one-hot is not the shape of the product")
    idx = None
    for i in (0, 1):
        if _is_iota(scope, cmp_op.operands[i], hframe, e_ax, r.E):
            idx = 1 - i
            break
    if idx is None:
        raise _Reject("the one-hot compares against no expert iota")
    base, bframe, bdims = _peel(scope, cmp_op.operands[idx], hframe,
                                stop=tk_shape)
    if bdims != [t_ax, kd] or list(_shape(base)) != tk_shape:
        raise _Reject("the one-hot indices are not laid out [tokens, k]")
    if bframe is not None:
        raise _Reject("the routing indices live inside a callee")
    r.indices = base

    # The indices must come from a top-k. Downstream nothing needs this (a
    # scatter of ANY index set is just as gatherable), but a different index
    # source means the pattern is something else -- a capacity-factor
    # dispatch, say -- and this recognizer has not established what.
    _v2, tframe, tk = scope.deref(base, bframe)
    if not _is_topk(tk):
        raise _Reject(f"routing indices do not come from a top-k "
                      f"({'block argument' if tk is None else tk.name})")
    if _topk_k(tk) != r.K:
        raise _Reject("top-k width disagrees with the one-hot axis")
    if list(_shape(tk.operands[0])) != [r.T, r.E]:
        raise _Reject("top-k input is not [tokens, experts]")
    r.topk, r.tframe = tk, tframe
    return r


# --------------------------------------------------------------------------
# pair-space plan nodes
# --------------------------------------------------------------------------


class _Node:
    __slots__ = ("ea", "ta", "shape")

    def paired(self):
        return self.ea is not None or self.ta is not None


class _Ext(_Node):
    """A value from outside the region: read from `env` and take()n."""

    __slots__ = ("value", "op")

    def __init__(self, value, op, ea, ta, shape):
        self.value, self.op = value, op
        self.ea, self.ta, self.shape = ea, ta, shape


class _Elem(_Node):
    __slots__ = ("op", "srcs")

    def __init__(self, op, srcs, ea, ta, shape):
        self.op, self.srcs = op, srcs
        self.ea, self.ta, self.shape = ea, ta, shape


class _View(_Node):
    """squeeze / reorder / reshape / broadcast of the trailing axes."""

    __slots__ = ("src", "kind", "arg")

    def __init__(self, src, kind, arg, ea, ta, shape):
        self.src, self.kind, self.arg = src, kind, arg
        self.ea, self.ta, self.shape = ea, ta, shape


class _Dot(_Node):
    """One per-expert matmul: `gather_qmm` if packed, else `gather_mm`."""

    __slots__ = ("data", "weight", "pack", "M", "N", "K", "mshape", "nshape",
                 "out_dtype", "n_first")

    def __init__(self, ea, ta, shape):
        self.ea, self.ta, self.shape = ea, ta, shape
        self.pack = None
        self.weight = None


def _is_key(op, key) -> bool:
    """`op`'s identity key equals `key`.

    Terminators have no results and hash to None, and the MLIR bindings
    raise rather than return NotImplemented when a Value is compared
    against one -- so the None case has to be taken first.
    """
    k = _okey(op)
    return k is not None and k == key


def _trailing(shape, ea, ta):
    return [s for i, s in enumerate(shape) if i != ea and i != ta]


def _tpos(shape, ea, ta, axis):
    """Position of dense `axis` within the trailing shape."""
    return len([i for i in range(axis) if i != ea and i != ta])


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------


class _Planner:
    def __init__(self, scope, qst, E, T):
        self.scope = scope
        self.qst = qst
        self.E, self.T = E, T
        self.nodes = {}        # (frame, value, ea, ta) -> node
        self.order = []        # nodes in dependency order
        self.region = {}       # _okey(op) -> op   (root-frame ops to skip)
        self.reads = []        # values `emit` reads out of `env`

    def plan(self, v, frame, ea, ta):
        v, frame, o = self.scope.deref(v, frame)
        key = (_fid(frame), v, ea, ta)
        hit = self.nodes.get(key)
        if hit is not None:
            return hit
        node = self._build(v, frame, o, ea, ta)
        self.nodes[key] = node
        self.order.append(node)
        return node

    # --- boundary ---

    def _external(self, v, frame, ea, ta):
        """A value computed OUTSIDE the region and gathered on the way in."""
        if frame is not None:
            # Nothing in `env` holds a callee's internal values, so there is
            # no dense tensor to gather from.
            raise _Reject("region value is bound inside a callee")
        self.reads.append(v)
        return _Ext(v, None, ea, ta, _shape(v))

    def _build(self, v, frame, o, ea, ta):
        shape = _shape(v)
        if ea is not None and shape[ea] != self.E:
            raise _Reject("expert axis size disagrees")
        if ta is not None and shape[ta] != self.T:
            raise _Reject("token axis size disagrees")
        # THE BOUNDARY: in the enclosing scope, only values that carry the
        # expert axis are region work. Anything else is computed once,
        # densely, and gathered here -- traversing it would duplicate
        # token-only work K times. Inside a callee there is no such choice:
        # its values are not in `env`, so they are all region work (and a
        # value with neither axis stays unduplicated anyway).
        if o is None or (frame is None and ea is None):
            return self._external(v, frame, ea, ta)
        name = o.name
        if name in ("stablehlo.constant", "stablehlo.iota"):
            if ea is not None or ta is not None:
                raise _Reject(f"per-expert {name} inside the region")
            if frame is None:
                return self._external(v, frame, ea, ta)
            return _Ext(v, o, ea, ta, shape)   # re-run the handler at emit
        if frame is None:
            self.region[_okey(o)] = o
        if name in _ELEMENTWISE:
            return self._elem(o, frame, ea, ta, shape)
        if name == "stablehlo.concatenate":
            return self._concat(o, frame, ea, ta, shape)
        if name == "stablehlo.broadcast_in_dim":
            return self._bcast(o, frame, ea, ta, shape)
        if name == "stablehlo.transpose":
            return self._transpose(o, frame, ea, ta, shape)
        if name == "stablehlo.reshape":
            return self._reshape(o, frame, ea, ta, shape)
        if name == "stablehlo.slice":
            return self._slice(o, frame, ea, ta, shape)
        if name == "stablehlo.dot_general":
            return self._dot(o, frame, ea, ta, shape)
        raise _Reject(f"{name} is not modelled in pair space")

    # --- shape-preserving ---

    def _elem(self, o, frame, ea, ta, shape):
        srcs = []
        for x in o.operands:
            xs = _shape(x)
            if list(xs) == list(shape):
                srcs.append(self.plan(x, frame, ea, ta))
            elif not xs:
                srcs.append(self.plan(x, frame, None, None))
            else:
                raise _Reject(f"{o.name} operand shape {xs} != {shape}")
        return _Elem(o, srcs, ea, ta, shape)

    def _concat(self, o, frame, ea, ta, shape):
        d = _ir.int_attr(o, "dimension")
        if d in (ea, ta):
            raise _Reject("concatenate along the expert or token axis")
        srcs = [self.plan(x, frame, ea, ta) for x in o.operands]
        return _Elem(o, srcs, ea, ta, shape)

    # --- shape ops ---

    def _bcast(self, o, frame, ea, ta, shape):
        bd = _ir.i64_list(o, "broadcast_dimensions")
        if bd != sorted(bd):
            raise _Reject("broadcast_dimensions are not increasing")
        src = o.operands[0]
        sshape = _shape(src)
        bea = bta = None
        drop = set()
        for j, d in enumerate(bd):
            if d not in (ea, ta):
                continue
            full = self.E if d == ea else self.T
            if sshape[j] == full:
                if d == ea:
                    bea = j
                else:
                    bta = j
            elif sshape[j] == 1:
                drop.add(j)
            else:
                raise _Reject("expert/token axis broadcast from a part")
        node = self.plan(src, frame, bea, bta)
        keep = []
        for j in range(len(sshape)):
            if j in (bea, bta):
                continue
            keep.append(None if j in drop else _tpos(shape, ea, ta, bd[j]))
        dest = [k for k in keep if k is not None]
        if dest != sorted(dest):
            raise _Reject("broadcast reorders the trailing axes")
        return _View(node, "bcast", (keep, _trailing(shape, ea, ta)),
                     ea, ta, shape)

    def _transpose(self, o, frame, ea, ta, shape):
        perm = _ir.i64_list(o, "permutation")
        src = o.operands[0]
        bea = perm[ea] if ea is not None else None
        bta = perm[ta] if ta is not None else None
        node = self.plan(src, frame, bea, bta)
        sshape = _shape(src)
        order = [_tpos(sshape, bea, bta, perm[i])
                 for i in range(len(shape)) if i not in (ea, ta)]
        return _View(node, "perm", order, ea, ta, shape)

    def _reshape(self, o, frame, ea, ta, shape):
        src = o.operands[0]
        sshape = _shape(src)
        m = max(a for a in (ea, ta) if a is not None) + 1
        if len(sshape) < m or list(sshape[:m]) != list(shape[:m]):
            raise _Reject("reshape crosses the expert/token axes")
        node = self.plan(src, frame, ea, ta)
        return _View(node, "reshape", _trailing(shape, ea, ta), ea, ta, shape)

    def _slice(self, o, frame, ea, ta, shape):
        start = _ir.i64_list(o, "start_indices")
        limit = _ir.i64_list(o, "limit_indices")
        stride = _ir.i64_list(o, "strides")
        src = o.operands[0]
        sshape = _shape(src)
        for a in (ea, ta):
            if a is not None and (start[a], limit[a], stride[a]) != (
                    0, sshape[a], 1):
                raise _Reject("slice cuts the expert or token axis")
        node = self.plan(src, frame, ea, ta)
        spec = [(start[i], limit[i], stride[i])
                for i in range(len(sshape)) if i not in (ea, ta)]
        return _View(node, "slice", spec, ea, ta, shape)

    # --- the per-expert dots ---

    def _dot(self, o, frame, ea, ta, shape):
        from metaljax.ops.linalg import _dot_dims
        lb, rb, lc, rc = _dot_dims(o)
        lhs, rhs = o.operands
        lsh, rsh = _shape(lhs), _shape(rhs)
        lfree = [d for d in range(len(lsh)) if d not in lb and d not in lc]
        rfree = [d for d in range(len(rsh)) if d not in rb and d not in rc]
        nb = len(lb)
        if ta is None:
            raise _Reject("per-expert dot without a token axis")

        def where(axis):
            """`(side, dim)`; side 2 means a batching dim (both sides)."""
            if axis < nb:
                return 2, axis
            if axis < nb + len(lfree):
                return 0, lfree[axis - nb]
            return 1, rfree[axis - nb - len(lfree)]

        eside, edim = where(ea)
        tside, tdim = where(ta)
        if tside == 2:
            raise _Reject("the token axis is a dot batching dimension")
        dside, wside = tside, 1 - tside
        if eside == 2:
            wdim = (lb, rb)[wside][edim]
            ddim = (lb, rb)[dside][edim]
        elif eside == wside:
            wdim, ddim = edim, None
        else:
            raise _Reject("the expert axis belongs to the data operand only")

        wv, dv = (lhs, rhs)[wside], (lhs, rhs)[dside]
        wsh, dsh = _shape(wv), _shape(dv)
        wb, wc = (lb, rb)[wside], (lc, rc)[wside]
        db, dc = (lb, rb)[dside], (lc, rc)[dside]
        wfree = [d for d in range(len(wsh)) if d not in wb and d not in wc]
        dfree = [d for d in range(len(dsh)) if d not in db and d not in dc]
        if len(wb) > 1 or (wb and list(wb) != [wdim]):
            raise _Reject("the dot batches over more than the experts")
        if len(db) > 1:
            raise _Reject("the data operand has extra batching dims")
        wn = [d for d in wfree if d != wdim]
        wk = list(wc)
        # gather_* wants the weight as [E, ., .]: the expert axis outermost
        # and the two matrix axes contiguous, so the 3-D view stays free.
        if wdim != 0:
            raise _Reject("the expert axis is not the weight's leading axis")
        if wn + wk == list(range(1, len(wsh))):
            n_first = True
        elif wk + wn == list(range(1, len(wsh))):
            n_first = False
        else:
            raise _Reject("the weight's matrix dims are interleaved")

        node = _Dot(ea, ta, shape)
        node.mshape = [dsh[d] for d in dfree if d != tdim]
        node.nshape = [wsh[d] for d in wn]
        node.M, node.N = _prod(node.mshape), _prod(node.nshape)
        node.K = _prod(wsh[d] for d in wk)
        if node.K != _prod(dsh[d] for d in dc):
            raise _Reject("contraction sizes disagree")
        if node.M == 0 or node.N == 0 or node.K == 0:
            raise _Reject("empty per-expert dot")
        node.n_first = n_first
        node.out_dtype = dtypes.mx_result_dtype(o.results[0])

        # The dense result lists batching, then lhs free, then rhs free;
        # gather_* returns [P] + M + N, so record the reordering.
        mdims = [d for d in dfree if d != tdim]
        perm = []
        for i in range(len(shape)):
            if i in (ea, ta):
                continue
            side, dim = where(i)
            if side == 2:
                raise _Reject("unmodelled batching dim in a per-expert dot")
            perm.append(mdims.index(dim) if side == dside
                        else len(mdims) + wn.index(dim))

        pack = self._packed(o, wside)
        if pack is not None:
            node.pack = pack
        else:
            # The weight stays outside the region: read dense from `env` and
            # let gather_mm do the indexing (no [P, N, K] copy is ever made).
            wbase, wframe, _wo = self.scope.deref(wv, frame)
            if wframe is not None:
                raise _Reject("per-expert weight is bound inside a callee")
            if list(_shape(wbase)) != list(wsh):
                raise _Reject("per-expert weight changed shape under deref")
            node.weight = wbase
            self.reads.append(wbase)

        node.data = self.plan(dv, frame, ddim, tdim)
        self.order.append(node)
        return _View(node, "dotperm", (perm, _trailing(shape, ea, ta)),
                     ea, ta, shape)

    def _packed(self, dot, wside):
        """The qmm Match that already fused this dot, if there is one."""
        if self.qst is None:
            return None
        key = _okey(dot)
        for m in self.qst.matches:
            if m.key != key or m.disabled:      # `!=`: wrappers are transient
                continue
            if m.swapped != (wside == 0):
                raise _Reject("qmm packed the other dot operand")
            if len(m.bshape) > 1:
                raise _Reject("packed weight has several batching dims")
            if m.bshape and m.bshape[0] != self.E:
                raise _Reject("packed batching dim is not the expert axis")
            if not m.bshape and m.N % self.E:
                raise _Reject("packed N is not a multiple of the experts")
            return m
        return None


# --------------------------------------------------------------------------
# matches
# --------------------------------------------------------------------------


class Match:
    __slots__ = ("root", "key", "skip_keys", "order", "out", "router",
                 "e_axis", "t_axis", "E", "T", "K", "P", "out_axis",
                 "out_shape", "out_dtype", "sum_dtype", "disabled",
                 "verified", "packs", "name", "reads")

    def __init__(self):
        self.disabled = False
        self.verified = not _VERIFY
        self.packs = []


def _protect_closure(values):
    """`values` plus everything their computation depends on."""
    out, stack = set(), list(values)
    while stack:
        v = stack.pop()
        if v in out:
            continue
        out.add(v)
        o = _owner(v)
        if o is None:
            continue
        stack.extend(o.operands)
        for region in o.regions:
            for blk in region.blocks:
                for inner in blk.operations:
                    stack.extend(inner.operation.results)
    return out


_NEVER_SWEEP = ("stablehlo.custom_call", "stablehlo.while", "stablehlo.if",
                "stablehlo.case", "stablehlo.optimization_barrier",
                "stablehlo.outfeed", "stablehlo.send", "stablehlo.recv",
                "stablehlo.infeed")


def _dead_sweep(block, skip, protect, root_key):
    """Add ops whose every use is inside the rewritten region.

    The dense router scores, the one-hot and their broadcasts are only ever
    read by the sum being replaced; leaving them behind would keep computing
    a [tokens, experts] tensor nobody looks at.
    """
    changed = True
    while changed:
        changed = False
        for op in block.operations:
            o = op.operation
            k = _okey(o)
            if (k is None or k in skip or k == root_key
                    or o.name in _NEVER_SWEEP):
                continue
            if any(r in protect for r in o.results):
                continue
            uses = list(_users(o))
            if not uses:
                continue
            if all(_okey(u) in skip or _is_key(u, root_key) for u in uses):
                skip.add(k)
                changed = True


def _analyze_root(interp, qst, block, root):
    scope = _Scope(interp)
    e_axis = _zero_sum(root)
    _mv, mframe, mul = scope.deref(root.operands[0], None)
    # A bf16/f16 model sums in f32, so jax puts a convert between the
    # product and the sum. It stays in the region: the gathered products are
    # converted the same way before being summed (see `Match.sum_dtype`).
    casts = []
    while mul is not None and mul.name == "stablehlo.convert":
        casts.append(mul)
        _mv, mframe, mul = scope.deref(mul.operands[0], mframe)
    if mul is None or mul.name != "stablehlo.multiply" or mframe is not None:
        raise _Reject("the expert sum does not reduce a product")
    mshape = _shape(mul.results[0])
    if len(mshape) < 2:
        raise _Reject("the expert product is rank < 2")

    router = t_axis = yv = None
    for i in (0, 1):
        base, bframe, bdims = _peel(scope, mul.operands[i], mframe)
        if len(bdims) != 2 or e_axis not in bdims:
            continue
        se = bdims.index(e_axis)
        st = 1 - se
        if bdims[st] is None or bdims[st] == e_axis:
            continue
        router = _match_router(scope, base, bframe, se, st)
        t_axis = bdims[st]
        yv = mul.operands[1 - i]
        break
    if router is None:
        raise _Reject("neither product operand is a routing-score tensor")
    if mshape[e_axis] != router.E or mshape[t_axis] != router.T:
        raise _Reject("routing scores do not match the expert-output shape")

    m = Match()
    m.root, m.key = root, _okey(root)
    m.router = router
    m.e_axis, m.t_axis = e_axis, t_axis
    m.E, m.T, m.K = router.E, router.T, router.K
    m.P = router.T * router.K
    m.out_shape = [s for i, s in enumerate(mshape) if i != e_axis]
    m.out_axis = _tpos(mshape, e_axis, None, t_axis)
    m.out_dtype = dtypes.mx_result_dtype(root.results[0])
    m.sum_dtype = dtypes.mx_result_dtype(root.operands[0])

    planner = _Planner(scope, qst, router.E, router.T)
    planner.region[_okey(mul)] = mul
    for c in casts:
        planner.region[_okey(c)] = c
    # Only the calls entered from HERE on belong to the region; the router
    # match above may have walked through calls of its own (jax emits
    # `one_hot` as one), and those stay in the dense graph.
    mark = len(scope.calls)
    out = planner.plan(yv, mframe, e_axis, t_axis)
    if isinstance(out, _Ext):
        raise _Reject("the expert output is not computed in the region")
    if not any(isinstance(n, _Dot) for n in planner.order):
        raise _Reject("the region contains no per-expert dot")
    if list(out.shape) != list(mshape):
        raise _Reject("the expert output is not the product's shape")

    for o in scope.calls[mark:]:
        planner.region.setdefault(_okey(o), o)
    # USE-COUNT DISCIPLINE: a region op may only be skipped when nothing
    # outside reads it. Anything shared with an aux-loss head, a residual, or
    # a second consumer of the expert outputs fails the match closed.
    for _k, o in planner.region.items():
        for u in _users(o):
            if _okey(u) not in planner.region and not _is_key(u, m.key):
                raise _Reject(f"{o.name} feeds {u.name} outside the region")

    # Invariant: everything `emit` reads out of `env` must still be computed.
    # (A value reached both as region work and as an outside read would be
    # skipped and then looked up -- a KeyError at execute time, not a wrong
    # answer, but this says so at analysis time instead.)
    for v in planner.reads:
        o = _owner(v)
        if o is not None and _okey(o) in planner.region:
            raise _Reject(f"{o.name} is both region work and a run-time read")

    m.order = planner.order
    m.out = out
    m.reads = planner.reads
    skip = set(planner.region)
    protect = set(planner.reads)
    protect.add(router.indices)
    protect.add(router.weights)
    _dead_sweep(block, skip, _protect_closure(protect), m.key)
    m.skip_keys = frozenset(skip)
    m.packs = [n.pack for n in planner.order
               if isinstance(n, _Dot) and n.pack is not None]
    m.name = (f"E{m.E}/K{m.K}/T{m.T}->"
              f"{'x'.join(str(s) for s in m.out_shape)}")
    return m


def analyze(interp, qst):
    """Find gatherable MoE dispatches and extend `qst` with them."""
    if not ENABLED:
        return
    found = []
    try:
        with interp.context:
            for block in _walk_blocks(interp._main_block()):
                for op in block.operations:
                    o = op.operation
                    if o.name != "stablehlo.reduce":
                        continue
                    try:
                        found.append(_analyze_root(interp, qst, block, o))
                    except _Reject as e:
                        if _DEBUG and _looks_like_moe(o):
                            print(f"[metaljax] moe: rejected a candidate "
                                  f"({e})", flush=True)
                    except Exception as e:   # analysis must never break a run
                        if _DEBUG:
                            print(f"[metaljax] moe: analysis error ({e})",
                                  flush=True)
    except Exception as e:
        if _DEBUG:
            print(f"[metaljax] moe: analysis failed ({e})", flush=True)
        return
    keep, seen = [], set()
    for m in found:
        if m.key in seen or m.skip_keys & seen:
            continue
        seen |= set(m.skip_keys)
        seen.add(m.key)
        keep.append(m)
    if not keep:
        return
    for m in keep:
        for pk in m.packs:
            pk.absorbed = True
    qst.moe = keep
    STATS["recognized"] += len(keep)
    if _DEBUG:
        print(f"[metaljax] moe: {len(keep)} expert dispatch(es) gathered "
              f"({', '.join(m.name for m in keep)})", flush=True)


def _looks_like_moe(op):
    try:
        return len(_shape(op.operands[0])) >= 3
    except Exception:
        return False


# --------------------------------------------------------------------------
# verification (eager, on synthetic logits)
# --------------------------------------------------------------------------


def _eval_sub(interp, value, frame, scope, env):
    """Evaluate `value` with the interpreter's own handlers, on `env`."""
    from metaljax.interpreter import REGISTRY

    v, frame, o = scope.deref(value, frame)
    key = (_fid(frame), v)
    if key in env:
        return env[key]
    if o is None:
        raise _Reject("unbound value while verifying the router")
    if o.regions and o.name != "stablehlo.reduce":
        raise _Reject(f"{o.name} in the router tail")
    ins = [_eval_sub(interp, x, frame, scope, env) for x in o.operands]
    handler = REGISTRY.get(o.name)
    if handler is None:
        raise _Reject(f"no handler for {o.name}")
    out = handler(interp, o, ins, {})
    if isinstance(out, mx.array):
        out = [out]
    for r, val in zip(o.results, out):
        env[(_fid(frame), r)] = val
    return env[key]


def _verify(interp, m):
    """Check `S == scatter(w at indices)` on random logits, a few draws.

    The router tail below `top_k` depends on nothing but the logits, so this
    is a complete functional check of the structural reading -- which axis is
    which, and that the weights really land at the matched indices -- and it
    needs no real activations at all.
    """
    r = m.router
    scope = _Scope(interp)
    lv, lframe, _o = scope.deref(r.topk.operands[0], r.tframe)
    dt = dtypes.mx_result_dtype(r.topk.results[0])
    rng = np.random.default_rng(20260803)
    rows = mx.repeat(mx.arange(m.T, dtype=mx.int32), m.K)
    for draw in range(max(1, _VERIFY_DRAWS)):
        raw = rng.standard_normal((m.T, m.E)).astype(np.float32)
        if draw:
            # Coarse values on purpose: exact ties are where a top-k and a
            # hand-rolled scatter would most easily disagree.
            raw = np.round(raw * 2.0) / 2.0
        env = {(_fid(lframe), lv): mx.array(raw).astype(dt)}
        score = _eval_sub(interp, r.score, r.sframe, scope, env)
        idx = _eval_sub(interp, r.indices, None, scope, env)
        wgt = _eval_sub(interp, r.weights, None, scope, env)
        if r.e_in_score == 0:
            score = mx.swapaxes(score, 0, 1)
        if list(score.shape) != [m.T, m.E]:
            raise _Reject(f"router scores evaluated to {list(score.shape)}")
        ref = mx.zeros((m.T, m.E), score.dtype)
        ref = ref.at[rows, mx.reshape(idx, (-1,)).astype(mx.int32)].add(
            mx.reshape(wgt, (-1,)).astype(score.dtype))
        ok = mx.all(score == ref)
        nz = mx.max(mx.sum((score != mx.zeros((), score.dtype)).astype(
            mx.int32), axis=1))
        mx.eval(ok, nz)
        if not bool(ok.item()):
            raise _Reject("router scores are not the top-k weights scattered "
                          "at the matched indices")
        if int(nz.item()) > m.K:
            raise _Reject(f"a token has more than {m.K} nonzero routing "
                          f"weights")
    STATS["verified"] += 1


def prologue(interp, qst):
    """Verify pending matches (eagerly). True when the rewrite set changed."""
    matches = getattr(qst, "moe", None)
    if not matches:
        return False
    changed = False
    for m in matches:
        if m.disabled:
            continue
        if any(pk.disabled for pk in m.packs):
            m.disabled = True
            changed = True
            STATS["fallbacks"] += 1
            if _DEBUG:
                print(f"[metaljax] moe: {m.name} falls back (its quantized "
                      f"weight did not pack)", flush=True)
            continue
        if m.verified:
            continue
        try:
            with interp.context:
                _verify(interp, m)
            m.verified = True
        except Exception as e:
            m.disabled = True
            changed = True
            STATS["fallbacks"] += 1
            if _DEBUG:
                print(f"[metaljax] moe: {m.name} falls back to the dense "
                      f"dispatch ({e})", flush=True)
    if changed:
        for m in matches:
            if m.disabled:
                for pk in m.packs:
                    pk.absorbed = False
    return changed


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------


class _Ctx:
    __slots__ = ("interp", "env", "eidx", "tidx", "vals", "m")


def _pair_lead(x, node):
    """`x` with a leading pair axis (size P or 1)."""
    if node.paired():
        return x
    return mx.reshape(x, [1] + list(x.shape))


def _emit_ext(ctx, node):
    if node.op is not None:                      # nullary op inside a callee
        from metaljax.interpreter import REGISTRY
        return REGISTRY[node.op.name](ctx.interp, node.op, [], {})
    x = ctx.env[node.value]
    ea, ta = node.ea, node.ta
    if ea is None and ta is None:
        return x
    if ea is not None and ta is not None:
        x = mx.moveaxis(x, ea, 0)
        x = mx.moveaxis(x, ta + 1 if ta < ea else ta, 1)
        return x[ctx.eidx, ctx.tidx]
    axis = ea if ea is not None else ta
    idx = ctx.eidx if ea is not None else ctx.tidx
    return mx.moveaxis(mx.take(x, idx, axis=axis), axis, 0)


def _emit_view(ctx, node):
    x = _pair_lead(ctx.vals[id(node.src)], node.src)
    kind, arg = node.kind, node.arg
    if kind == "bcast":
        keep, out_trailing = arg
        view = [1] * len(out_trailing)
        take = [slice(None)]
        squeeze = False
        for j, dest in enumerate(keep):
            if dest is None:
                take.append(0)
                squeeze = True
            else:
                take.append(slice(None))
                view[dest] = x.shape[1 + j]
        if squeeze:
            x = x[tuple(take)]
        x = mx.reshape(x, [x.shape[0]] + view)
        return mx.broadcast_to(x, [x.shape[0]] + list(out_trailing))
    if kind == "perm":
        return mx.transpose(x, [0] + [1 + i for i in arg])
    if kind == "reshape":
        return mx.reshape(x, [x.shape[0]] + list(arg))
    if kind == "slice":
        return x[tuple([slice(None)] + [slice(a, b, c) for a, b, c in arg])]
    if kind == "dotperm":
        perm, trailing = arg
        if perm != sorted(perm):
            x = mx.transpose(x, [0] + [1 + i for i in perm])
        return mx.reshape(x, [x.shape[0]] + list(trailing))
    raise RuntimeError(f"moe: unknown view {kind}")


def _emit_elem(ctx, node):
    from metaljax.interpreter import REGISTRY

    ins = [ctx.vals[id(s)] for s in node.srcs]
    if node.op.name == "stablehlo.concatenate":
        d = _ir.int_attr(node.op, "dimension")
        axis = 1 + _tpos(node.shape, node.ea, node.ta, d)
        lead = max(x.shape[0] for x in ins)
        ins = [x if x.shape[0] == lead
               else mx.broadcast_to(x, [lead] + list(x.shape[1:]))
               for x in ins]
        return mx.concatenate(ins, axis=axis)
    out = REGISTRY[node.op.name](ctx.interp, node.op, ins, {})
    return out[0] if isinstance(out, list) else out


def _split_experts(a, E):
    """A qmm pack's leading `E * N` rows re-viewed as `[E, N, ...]`."""
    if len(a.shape) >= 3:
        return a
    return mx.reshape(a, (E, a.shape[0] // E, a.shape[1]))


def _emit_dot(ctx, node):
    m = ctx.m
    P = m.P
    data = node.data
    if isinstance(data, _Ext) and data.ea is None and data.op is None:
        # Token-indexed and nothing else: hand gather_* the ungathered rows
        # and let it do the indexing, instead of materializing K copies.
        dense = ctx.env[data.value]
        ta = data.ta
        rest = [i for i in range(len(dense.shape)) if i != ta]
        x = mx.reshape(mx.transpose(dense, [ta] + rest),
                       (dense.shape[ta], node.M, node.K))
        lhs_idx = ctx.tidx
    else:
        x = _pair_lead(ctx.vals[id(data)], data)
        if x.shape[0] != P:
            x = mx.broadcast_to(x, [P] + list(x.shape[1:]))
        x = mx.reshape(x, (P, node.M, node.K))
        lhs_idx = mx.arange(P, dtype=mx.uint32)
    if len(x.shape) != 3:
        # gather_* silently treats a 2-D operand as [M, K] with NO batch
        # dims and broadcasts the indices against it -- wrong, and quiet.
        raise RuntimeError("moe: gather operand is not [batch, M, K]")
    rhs_idx = ctx.eidx

    pack = node.pack
    if pack is not None:
        from metaljax import qmm
        vals = qmm.pack_arrays(ctx.interp, pack)
        w = _split_experts(vals[0], m.E)
        scales = _split_experts(vals[1], m.E)
        biases = (_split_experts(vals[2], m.E) if pack.mode == "affine"
                  else None)
        if pack.has_perm:
            x = mx.take(x, vals[-1], axis=-1)
        if x.dtype not in (mx.float32, mx.float16, mx.bfloat16):
            x = x.astype(node.out_dtype)
        y = mx.gather_qmm(x, w, scales, biases, lhs_idx, rhs_idx,
                          transpose=True, group_size=pack.gs, bits=pack.bits,
                          mode=pack.mode)
        STATS["gather_qmm"] += 1
    else:
        wv = ctx.env[node.weight]
        if node.n_first:
            # gather_mm computes `a @ b`, so an [E, N, K] weight has to be
            # transposed. As a LAZY view: materializing it would copy the
            # whole expert set, which is the traffic being avoided. (Measured
            # on the gemma4-26B shape: the lazy view is the fastest of the
            # four orientations, 233 GB/s vs 192 for weight-on-the-left.)
            w3 = mx.swapaxes(mx.reshape(wv, (m.E, node.N, node.K)), -1, -2)
        else:
            w3 = mx.reshape(wv, (m.E, node.K, node.N))
        y = mx.gather_mm(x, w3, lhs_idx, rhs_idx)
        STATS["gather_mm"] += 1
    y = mx.reshape(y, (P, node.M, node.N))
    if y.dtype != node.out_dtype:
        y = y.astype(node.out_dtype)
    return mx.reshape(y, [P] + list(node.mshape) + list(node.nshape))


def emit(interp, m, env):
    """The gathered expert dispatch, in place of the dense weighted sum."""
    r = m.router
    idx, wgt = env[r.indices], env[r.weights]
    ctx = _Ctx()
    ctx.interp, ctx.env, ctx.m = interp, env, m
    ctx.vals = {}
    ctx.eidx = mx.reshape(idx, (m.P,)).astype(mx.uint32)
    ctx.tidx = mx.repeat(mx.arange(m.T, dtype=mx.uint32), m.K)
    for node in m.order:
        if isinstance(node, _Ext):
            v = _emit_ext(ctx, node)
        elif isinstance(node, _Dot):
            v = _emit_dot(ctx, node)
        elif isinstance(node, _View):
            v = _emit_view(ctx, node)
        else:
            v = _emit_elem(ctx, node)
        ctx.vals[id(node)] = v
    y = _pair_lead(ctx.vals[id(m.out)], m.out)
    trailing = list(y.shape[1:])
    if y.shape[0] != m.P:
        y = mx.broadcast_to(y, [m.P] + trailing)
    w = mx.reshape(wgt, (m.P,)).astype(y.dtype)
    y = y * mx.reshape(w, [m.P] + [1] * len(trailing))
    if y.dtype != m.sum_dtype:
        # The dense graph converts the products before summing them.
        y = y.astype(m.sum_dtype)
    y = mx.sum(mx.reshape(y, [m.T, m.K] + trailing), axis=1)
    if m.out_axis:
        y = mx.moveaxis(y, 0, m.out_axis)
    if list(y.shape) != list(m.out_shape):
        raise RuntimeError(f"moe: produced {list(y.shape)}, expected "
                           f"{m.out_shape}")
    if y.dtype != m.out_dtype:
        y = y.astype(m.out_dtype)
    return [y]
