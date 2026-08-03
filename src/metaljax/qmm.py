"""Recognizer that maps dequantize-then-matmul chains onto mx.quantized_matmul.

Weight-only quantized models (keras `quantize("int4")`, `jnp.int4` weights,
anything that stores integer codes plus a scale/zero-point map) lower to
StableHLO that *materializes the whole dequantized weight* and then runs a
float dot:

    w_f = (convert(codes) - convert(zeros)) * scales     # full weight size
    y   = dot_general(x, w_f)

XLA:CPU fuses that reconstruction into the dot's operand loop and pays
almost nothing; interpreting it literally costs ~20 full-weight-sized MLX
kernels per matmul per token, which is why LLM decode on packed-int4 weights
was 5x slower here than on the CPU backend.

This module recognizes the pattern structurally at compile time, repacks the
codes ONCE into MLX's affine layout at the first execute, and replaces the
whole chain with a single `mx.quantized_matmul`.

Two forms are recognized (both emitted by keras 3.15):

  sub-channel / asymmetric   dot(x, [reshape/transpose](mul(sub(cvt(codes),
                                                    cvt(zeros)), scales)))
  per-channel / symmetric    divide(dot(x, cvt(codes)), broadcast(scale))

The chain between the integer codes and the dot is NOT pattern-matched op by
op (keras' nibble unpack is a dozen ops of `and`/`shift`/`xor`/`concatenate`
whose exact shape is not contractual). Instead the recognizer verifies the
top-level affine form, records the three operand subtrees, and defers every
VALUE-level question to the first execute, where the subtrees are evaluated
once on concrete buffers with the interpreter's own op handlers:

  * codes must be integral and fit in 4 or 8 bits;
  * the scale and zero maps must be constant inside contiguous groups along
    the contraction axis (this subsumes checking that keras' `g_idx` gather
    is the identity ramp -- any group-preserving permutation is fine because
    the group scales are read off the MATERIALIZED map, not the source
    argument);
  * zero points must be integers.

Any check failing permanently disables that one dot (it falls back to
executing the chain literally, which is what happened before this module
existed).

Packing is exact: `w = scale * (q - zp)` maps onto MLX's affine dequant
`w = scales * q_hat + biases` with an unsigned `q_hat = q + offset` and
`biases = -scale * (offset + zp)`; MLX's kernel evaluates that as a single
FMA, so whenever the bias is exactly representable the reconstructed weight
is bit-identical to a float32 dequantization. `offset` is `2^(bits-1)` (the
signed-code convention) whenever the codes fit it, so a zero-point-free
quantization keeps a power-of-two bias. Scales/biases are kept in the source
dtype when that downcast is lossless and widened to f32 otherwise -- see
METALJAX_QMM_SCALES.

Packing needs CONCRETE buffers and must never run inside an `mx.compile`
trace, so it happens in an eager prologue (`prologue()`, called by
engine.execute) and the packed arrays are threaded into traced code as
explicit inputs -- never as captured constants, which mx.compile bakes by
value and would silently freeze at their first value. Decode loops need one
more thing: jax lowers a while_loop's closed-over weights as loop-carried
state rather than region captures, so the operand subtrees are followed out
through invariant carries (`_hoist`) to the values the loop was handed.

METALJAX_QMM=0 disables the whole recognizer.
"""

from __future__ import annotations

import os
import weakref
from collections import deque

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes

ENABLED = os.environ.get("METALJAX_QMM", "1") != "0"
_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"

# mx.quantize supports these only, and requires K % group_size == 0.
# Largest first: gs=32 measured 1.8x slower than 64/128 at decode.
_GROUP_SIZES = (128, 64, 32)
# Packs retained per recognized dot (keyed by source-buffer identity).
_MAX_PACKS = int(os.environ.get("METALJAX_QMM_CACHE", "2"))
# Give up on a dot whose operands keep changing: repacking every execute
# (weights being trained, or a "scale" that is really per-call data) costs
# far more than the literal chain it replaces.
_MAX_REPACKS = int(os.environ.get("METALJAX_QMM_REPACKS", "8"))
# Width of the scale/bias tables. "auto" keeps the source (bf16/f16) width
# when the folded bias is exactly representable in it -- always true for a
# zero-point-free quantization -- and widens to f32 otherwise, which keeps
# the reconstruction exact at the cost of 3-12% of the matmul (measured, M=1,
# 4-bit: scale traffic is +12.5% of the weight at group 128 instead of
# +6.25%). "source" always narrows (faster, bias rounded to <=0.5 ULP);
# "f32" never narrows.
_SCALE_WIDTH = os.environ.get("METALJAX_QMM_SCALES", "auto")

# Diagnostics (tests assert on these; nothing else reads them).
STATS = {"recognized": 0, "packs": 0, "fallbacks": 0}


def stats() -> dict:
    return dict(STATS)


def reset_stats():
    for k in STATS:
        STATS[k] = 0


_INT_ELS = {"i4", "ui4", "i8", "i16", "i32", "i64",
            "ui8", "ui16", "ui32", "ui64"}
_FLOAT_ELS = {"f16", "f32", "bf16"}
_FLOAT_MX = (mx.float32, mx.float16, mx.bfloat16)
_SHAPE_OPS = ("stablehlo.reshape", "stablehlo.transpose")


class _Reject(Exception):
    """A candidate does not match (or cannot be trusted): run it literally."""


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


def pack_codes(codes: mx.array, bits: int) -> mx.array:
    """Unsigned codes `[..., K]` -> uint32 words `[..., K*bits//32]`.

    MLX packs each row of the last axis as one contiguous little-endian bit
    stream, LSB first: element i occupies bits [i*bits, (i+1)*bits). Only
    bits that divide 32 (2/4/8/16) are handled here -- the ones we emit --
    so no value ever straddles a word boundary.
    """
    if 32 % bits:
        raise ValueError(f"pack_codes: bits={bits} does not divide 32")
    per = 32 // bits
    k = codes.shape[-1]
    if k % per:
        raise ValueError(f"pack_codes: K={k} not a multiple of {per}")
    lead = list(codes.shape[:-1])
    c = mx.reshape(codes.astype(mx.uint32), lead + [k // per, per])
    mask = mx.array((1 << bits) - 1, mx.uint32)
    out = None
    for i in range(per):
        # Defensive mask: an out-of-range code would otherwise spill into the
        # NEXT element's bits and corrupt an unrelated weight. Callers are
        # expected to have range-checked (see _build_pack).
        v = mx.bitwise_and(c[..., i], mask)
        if i:
            v = mx.left_shift(v, mx.array(i * bits, mx.uint32))
        out = v if out is None else mx.bitwise_or(out, v)
    return out


def pack_exact(codes: mx.array, scales: mx.array, zeros, bits: int,
               scale_dtype=None, offset=None):
    """Repack already-quantized weights into MLX's affine format.

    `codes` are signed codes `[N, K]`; `scales` and `zeros` are the per-group
    values `[N, K/group_size]` of the source quantization
    `w = scale * (code - zero)`. Returns `(packed_u32, scales, biases)` for
    `mx.quantized_matmul(..., bits=bits)`.

    The algebra: MLX dequantizes `scales*q_hat + biases` with an UNSIGNED
    `q_hat`, so shifting the codes by any integer `offset` that makes them
    non-negative works, as long as the shift is undone in the bias:

        q_hat  = code + offset          (must fit in `bits`)
        biases = -scale * (offset + zero)

    `offset` defaults to `2^(bits-1)`, which is exactly the signed-code
    convention. Whenever `scale * (offset + zero)` is representable the
    reconstruction is bit-identical to a float32 dequantization, because
    MLX's kernel evaluates the dequant as a single FMA.
    """
    off = (1 << (bits - 1)) if offset is None else int(offset)
    packed = pack_codes(mx.contiguous(codes).astype(mx.int32) + off, bits)
    s32 = mx.contiguous(scales).astype(mx.float32)
    z32 = (mx.contiguous(zeros).astype(mx.float32)
           if zeros is not None else mx.array(0.0, mx.float32))
    b32 = -(s32 * (z32 + off))
    if (scale_dtype is not None and scale_dtype != mx.float32
            and _SCALE_WIDTH != "f32"):
        # Keep the source (bf16/f16) width when nothing is lost by it: it
        # halves scale traffic and keeps the output in the compute dtype.
        # zp == 0 always qualifies (the bias is a power-of-two multiple of
        # the scale); a general zero point usually does not.
        if _SCALE_WIDTH == "source" or (_lossless(s32, scale_dtype)
                                        and _lossless(b32, scale_dtype)):
            return packed, s32.astype(scale_dtype), b32.astype(scale_dtype)
    return packed, s32, b32


def _lossless(x: mx.array, dtype) -> bool:
    return bool(mx.all(x.astype(dtype).astype(mx.float32) == x).item())


# --------------------------------------------------------------------------
# structural matching
# --------------------------------------------------------------------------


def _owner(v):
    return v.owner.operation if isinstance(v, ir.OpResult) else None


def _okey(op):
    """Identity key for an operation: its first result.

    ir.Value hashes by the underlying MLIR value; the python wrappers around
    operations are transient, and this project has twice been bitten by
    keying sets on id() of such wrappers.
    """
    res = op.results
    return res[0] if len(res) else None


def _el(v) -> str:
    return str(_ir.tensor_type(v).element_type)


def _is_int(v) -> bool:
    try:
        return _el(v) in _INT_ELS
    except Exception:
        return False


def _shape(v) -> list[int]:
    return list(_ir.tensor_type(v).shape)


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _strip(v, names):
    """Walk down through `names`-kind ops. Returns (base, ops-outermost-last)."""
    chain = []
    while True:
        o = _owner(v)
        if o is None or o.name not in names:
            return v, list(reversed(chain))
        chain.append(o)
        v = o.operands[0]


def _strip_shape(v):
    return _strip(v, _SHAPE_OPS + ("stablehlo.convert",))


def _strip_converts(v):
    return _strip(v, ("stablehlo.convert",))


_SIGNED_WIDTH = {"i4": 4, "i8": 8, "i16": 16, "i32": 32, "i64": 64}


def _parse_codes(v):
    """`v` as an integer-code operand: (codes, zero_point, subtract_range).

    `subtract_range` is the (lo, hi) the graph's `codes - zero` is computed
    in when that subtraction happens in INTEGER arithmetic -- the rewrite
    evaluates it exactly, so a graph that would wrap must not be rewritten.
    It is None when the subtraction is done in floating point (what keras
    emits: both operands are converted first).
    """
    base, _ = _strip_converts(v)
    o = _owner(base)
    if o is not None and o.name == "stablehlo.subtract":
        c, _ = _strip_converts(o.operands[0])
        if not _is_int(c):
            return None
        rng = None
        if _is_int(base):
            w = _SIGNED_WIDTH.get(_el(base))
            if w is None:
                return None  # unsigned wrap semantics: not worth modelling
            rng = (-(1 << (w - 1)), (1 << (w - 1)) - 1)
        return c, o.operands[1], rng
    if _is_int(base):
        return base, None, None
    return None


def _dot_dims(op):
    from metaljax.ops.linalg import _dot_dims as impl
    return impl(op)


def _bcast_chain(v):
    """Peel a chain of broadcast_in_dim (jax emits one per rank step).

    Returns (base, dims) where `dims[i]` is the dimension of `v` that base's
    dimension i lands in, or (v, None) when the head is not a broadcast.
    """
    dims = list(range(len(_shape(v))))
    base = None
    while True:
        o = _owner(v)
        if o is None:
            break
        if o.name == "stablehlo.broadcast_in_dim":
            d = _ir.i64_list(o, "broadcast_dimensions")
            dims = [dims[j] for j in d]
            v = o.operands[0]
            base = v
            continue
        if o.name == "stablehlo.convert":
            v = o.operands[0]
            continue
        break
    return (v, dims) if base is not None else (v, None)


class Match:
    """One recognized quantized matmul."""

    __slots__ = ("root", "key", "lhs", "codes", "zero", "scale", "recip",
                 "post", "bcast_dims", "ops", "arg_indices", "required",
                 "lperm", "rfree", "rc", "rshape", "mshape", "nshape",
                 "M", "K", "N", "out_dtype", "disabled", "slot", "gs", "bits",
                 "packs", "repacks", "name", "sub_range")

    def __init__(self):
        self.recip = False
        self.zero = None
        self.sub_range = None
        self.post = []
        self.bcast_dims = None
        self.ops = {}
        self.required = []
        self.arg_indices = ()
        self.disabled = False
        self.slot = -1
        self.gs = 0
        self.bits = 0
        self.packs = []
        self.repacks = 0
        self.name = "?"


class _Pack:
    __slots__ = ("refs", "w", "scales", "biases", "gs", "bits")

    def __init__(self, leaves, w, scales, biases, gs, bits):
        # Weak references: identity is what we care about, and pinning a
        # multi-GB weight buffer alive because we packed a copy of it would
        # be its own bug. A dead referent simply misses and repacks.
        self.refs = [weakref.ref(a) for a in leaves]
        self.w, self.scales, self.biases = w, scales, biases
        self.gs, self.bits = gs, bits

    def matches(self, leaves) -> bool:
        if len(leaves) != len(self.refs):
            return False
        return all(r() is a for r, a in zip(self.refs, leaves))


class State:
    """Per-program recognizer state."""

    __slots__ = ("matches", "skip", "roots", "values", "active")

    def __init__(self):
        self.matches = []
        self.skip = frozenset()
        self.roots = {}
        self.values = []
        self.active = False

    def rebuild(self):
        live = [m for m in self.matches if not m.disabled]
        self.roots = {m.key: m for m in live}
        skip = set()
        for m in live:
            skip.update(m.ops)
        self.skip = frozenset(skip)
        self.active = bool(self.roots)


def _walk_blocks(block):
    yield block
    for op in block.operations:
        for region in op.operation.regions:
            for b in region.blocks:
                yield from _walk_blocks(b)


_OPAQUE = ("stablehlo.while", "stablehlo.if", "stablehlo.case",
           "stablehlo.custom_call", "stablehlo.optimization_barrier")


def _hoist(v):
    """Follow a value out of the loops that merely carry it around.

    jax does NOT lower a while_loop's closed-over constants as region
    captures: they become loop-carried state that the body returns
    unchanged (`%iterArg_0 = %arg0` ... `stablehlo.return %iterArg_0`).
    A decode loop's weights therefore arrive as body block arguments, and
    the value that is actually constant for the whole loop -- the one the
    packing prologue can evaluate -- is the while's initial operand.
    """
    for _ in range(8):  # depth guard; real programs nest 1-2 loops
        if not isinstance(v, ir.BlockArgument):
            return v
        blk = v.owner
        owner = blk.owner
        if owner is None:
            return v
        op = owner.operation if hasattr(owner, "operation") else owner
        if op.name != "stablehlo.while":
            return v
        i = v.arg_number
        body = op.regions[1].blocks[0]
        term = list(body.operations)[-1].operation
        if term.operands[i] != list(body.arguments)[i]:
            return v  # the carry changes: not loop-invariant
        v = op.operands[i]
    return v


def _closure(values, main_args, ops, arg_indices, register=()):
    """Backward closure of `values`, stopping at block arguments/constants.

    Every block argument reached must be one of @main's -- otherwise the
    subtree depends on a loop carry and cannot be packed once. `register`
    ops join the absorbed set without having their own operands walked (the
    caller passes the operands it wants absorbed in `values`; the per-channel
    form absorbs a dot's weight operand but not its activations).
    """
    for o in register:
        ops[_okey(o)] = o
    stack = list(values)
    seen = set()
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        if isinstance(v, ir.BlockArgument):
            idx = main_args.get(v)
            if idx is not None:
                arg_indices.add(idx)
                continue
            outer = _hoist(v)
            if outer is v:
                raise _Reject("operand depends on an inner block argument")
            stack.append(outer)
            continue
        o = _owner(v)
        if o is None:
            raise _Reject("operand is neither a block argument nor a result")
        if o.name in _OPAQUE:
            raise _Reject(f"operand subtree contains {o.name}")
        ops.setdefault(_okey(o), o)
        stack.extend(o.operands)


def _finish(ctx, m, root, absorb, dot, required=(), register=()):
    lb, rb, lc, rc = _dot_dims(dot)
    if lb or rb:
        raise _Reject("batching dimensions")
    lhs, rhs = dot.operands
    if _el(lhs) not in _FLOAT_ELS:
        raise _Reject(f"lhs element type {_el(lhs)}")
    out_el = _el(root.results[0])
    if out_el not in _FLOAT_ELS:
        raise _Reject(f"result element type {out_el}")
    lshape, rshape = _shape(lhs), _shape(rhs)
    lfree = [d for d in range(len(lshape)) if d not in lc]
    rfree = [d for d in range(len(rshape)) if d not in rc]
    M, K, N = (_prod(lshape[d] for d in lfree), _prod(lshape[d] for d in lc),
               _prod(rshape[d] for d in rfree))
    if M == 0 or K == 0 or N == 0:
        raise _Reject("empty matmul")
    if K % _GROUP_SIZES[-1]:
        raise _Reject(f"K={K} is not a multiple of {_GROUP_SIZES[-1]}")
    if K != _prod(rshape[d] for d in rc):
        raise _Reject("contracting dimensions disagree")

    m.root = root
    m.key = _okey(root)
    m.lhs = lhs
    m.lperm = lfree + lc
    m.rfree, m.rc, m.rshape = rfree, rc, rshape
    m.mshape = [lshape[d] for d in lfree]
    m.nshape = [rshape[d] for d in rfree]
    m.M, m.K, m.N = M, K, N
    m.out_dtype = dtypes.mx_result_dtype(root.results[0])
    m.name = f"{m.M}x{m.K}x{m.N}"

    main_args, donated = ctx
    arg_indices = set()
    _closure(absorb, main_args, m.ops, arg_indices, register)
    if arg_indices & donated:
        raise _Reject("quantized operand is donated (buffer may be reused)")
    m.arg_indices = tuple(sorted(arg_indices))
    # Ops that MUST end up skipped for the rewrite to be a win: everything
    # between the affine reconstruction and the dot. If some other consumer
    # forces the dequantized weight to be materialized anyway, running a
    # quantized matmul on top of it would only add work.
    m.required = [_okey(o) for o in required]
    m.required += [_okey(o) for o in m.post]
    return m


def _try_affine(ctx, dot):
    """dot(x, [shape ops](mul(sub(cvt(codes), cvt(zeros)), scales)))."""
    rhs = dot.operands[1]
    base, post = _strip_shape(rhs)
    mul = _owner(base)
    if mul is None or mul.name != "stablehlo.multiply":
        raise _Reject("rhs is not a multiply")
    parsed = scale = None
    for i in (0, 1):
        parsed = _parse_codes(mul.operands[i])
        if parsed is not None:
            scale = mul.operands[1 - i]
            break
    if parsed is None:
        raise _Reject("neither multiply operand is an integer code tensor")
    if _el(scale) not in _FLOAT_ELS:
        raise _Reject(f"scale element type {_el(scale)}")
    m = Match()
    m.codes, m.zero, m.sub_range = parsed
    m.scale = scale
    m.post = [o for o in post if o.name in _SHAPE_OPS]
    _finish(ctx, m, dot, [rhs], dot, required=(mul,))
    return m


def _try_perchannel(ctx, div):
    """divide(dot(x, [shape ops](cvt(codes))), broadcast(scale))."""
    num, den = div.operands
    dot = _owner(num)
    if dot is None or dot.name != "stablehlo.dot_general":
        raise _Reject("divide numerator is not a dot")
    codes, post = _strip_shape(dot.operands[1])
    if not _is_int(codes):
        raise _Reject("dot rhs is not an integer code tensor")
    scale, dims = _bcast_chain(den)
    if dims is None:
        raise _Reject("divisor is not a broadcast")
    if _el(scale) not in _FLOAT_ELS:
        raise _Reject("divisor is not float")
    lb, rb, lc, _rc = _dot_dims(dot)
    if lb or rb:
        raise _Reject("batching dimensions")
    nm = len(_shape(dot.operands[0])) - len(lc)  # leading output (M) dims
    if any(d < nm for d in dims):
        # A divisor that varies along the M axis is not a weight scale.
        raise _Reject("divisor depends on the batch axis")
    m = Match()
    m.recip = True
    m.codes = codes
    m.scale = scale
    m.bcast_dims = [d - nm for d in dims]
    m.post = [o for o in post if o.name in _SHAPE_OPS]
    _finish(ctx, m, div, [dot.operands[1], den], dot,
            required=(dot,), register=(dot,))
    return m


def _users(op):
    for r in op.results:
        for use in r.uses:
            yield use.owner.operation


def _prune(cands):
    """Decide which absorbed ops may actually be skipped.

    An op can only be skipped if every consumer of every result is itself
    skipped (or is a rewritten root) -- otherwise the value is still needed.
    Constants and their broadcasts are routinely shared between layers by
    CSE; those simply stay, at a couple of op-units each. A candidate whose
    reconstruction ops cannot be skipped is dropped entirely.
    """
    live = list(cands)
    while True:
        ops = {}
        for m in live:
            ops.update(m.ops)
        roots = {m.key for m in live}
        work = deque(ops)
        while work:
            key = work.popleft()
            op = ops.get(key)
            if op is None:
                continue
            keep = True
            for u in _users(op):
                ukey = _okey(u)
                if ukey is None or (ukey not in ops and ukey not in roots):
                    keep = False
                    break
            if keep:
                continue
            del ops[key]
            for opd in op.operands:
                if isinstance(opd, ir.OpResult):
                    k = _okey(_owner(opd))
                    if k in ops:
                        work.append(k)
        kept = []
        for m in live:
            missing = [k for k in m.required if k not in ops]
            if missing:
                if _DEBUG:
                    print(f"[metaljax] qmm: dropped {m.name} "
                          f"(reconstruction is used elsewhere)", flush=True)
                continue
            kept.append(m)
        if len(kept) == len(live):
            for m in live:
                m.ops = {k: v for k, v in m.ops.items() if k in ops}
            return live
        live = kept


def analyze(interp) -> State:
    """Structural analysis of a program, once. Never touches values."""
    st = interp._qmm
    if st is not None:
        return st
    st = State()
    interp._qmm = st
    if not ENABLED:
        return st
    try:
        with interp.context:
            ctx = ({a: i for i, a in
                    enumerate(interp._main_block().arguments)},
                   set(interp.donated_args))
            cands = []
            for block in _walk_blocks(interp._main_block()):
                for op in block.operations:
                    o = op.operation
                    try:
                        if o.name == "stablehlo.dot_general":
                            cands.append(_try_affine(ctx, o))
                        elif o.name == "stablehlo.divide":
                            cands.append(_try_perchannel(ctx, o))
                    except _Reject:
                        pass
            # A candidate whose result feeds another candidate's operand
            # subtree is evaluated during that one's packing prologue; it
            # cannot also be rewritten at run time.
            absorbed = set()
            for m in cands:
                absorbed.update(m.ops)
            cands = [m for m in cands if m.key not in absorbed]
            st.matches = _prune(cands) if cands else []
            STATS["recognized"] += len(st.matches)
            if _DEBUG and st.matches:
                print(f"[metaljax] qmm: {len(st.matches)} quantized "
                      f"matmul(s) recognized", flush=True)
    except Exception as e:  # analysis must never break a program
        if _DEBUG:
            print(f"[metaljax] qmm: analysis failed ({e})", flush=True)
        st.matches = []
    st.rebuild()
    return st


# --------------------------------------------------------------------------
# first-execute verification + packing
# --------------------------------------------------------------------------


def _eval(interp, value, env):
    """Evaluate an operand subtree eagerly on concrete arguments."""
    from metaljax.interpreter import REGISTRY

    cached = env.get(value)
    if cached is not None:
        return cached
    if isinstance(value, ir.BlockArgument):
        # A loop-invariant carry: evaluate what the loop was handed.
        outer = _hoist(value)
        if outer is value:
            raise _Reject("unbound value in operand subtree")
        env[value] = _eval(interp, outer, env)
        return env[value]
    o = _owner(value)
    if o is None:
        raise _Reject("unbound value in operand subtree")
    ins = [_eval(interp, x, env) for x in o.operands]
    handler = REGISTRY.get(o.name)
    if handler is None:
        raise _Reject(f"no handler for {o.name}")
    out = handler(interp, o, ins, env)
    if isinstance(out, mx.array):
        out = [out]
    for r, v in zip(o.results, out):
        env[r] = v
    return env[value]


def _replay(x, post):
    for o in post:
        if o.name == "stablehlo.reshape":
            x = mx.reshape(x, list(_ir.tensor_type(o.results[0]).shape))
        else:
            x = mx.transpose(x, _ir.i64_list(o, "permutation"))
    return x


def _to_nk(x, m):
    """A rhs-shaped tensor as the [N, K] matrix `mx.quantized_matmul` wants.

    Materialized on the spot: these are full-weight-sized reconstructions
    (a 262k-vocab head is ~800 MB per map), and leaving them lazy would keep
    every intermediate of the unpack chain alive until the pack is finished.
    """
    x = _replay(x, m.post)
    if list(x.shape) != m.rshape:
        raise _Reject(f"expected rhs shape {m.rshape}, got {list(x.shape)}")
    out = mx.contiguous(mx.reshape(mx.transpose(x, m.rfree + m.rc),
                                   (m.N, m.K)))
    mx.eval(out)
    return out


def _group_const(x, gs) -> bool:
    n, k = x.shape
    v = mx.reshape(x, (n, k // gs, gs))
    return bool(mx.all(v == v[:, :, :1]).item())


def _build_pack(interp, m, args, leaves):
    # One environment per operand subtree: they share only cheap prefixes
    # (the group-index cast), while each one's intermediates are full weight
    # size -- keeping all three alive at once would triple the peak.
    def evaluate(value):
        env = dict(zip(interp._main_block().arguments, args))
        return _to_nk(_eval(interp, value, env), m)

    codes = evaluate(m.codes)
    lo = int(mx.min(codes).item())
    hi = int(mx.max(codes).item())
    # The codes only have to FIT: any integer shift that makes them
    # non-negative is undone in the bias, so the code range decides the
    # width (a symmetric [-8, 7] weight lands on 4 bits either way).
    if hi - lo < 16:
        bits = 4
    elif hi - lo < 256:
        bits = 8
    else:
        raise _Reject(f"codes span [{lo}, {hi}]: more than 8 bits")
    # Prefer the signed-code convention (offset = 2^(bits-1)): its bias is a
    # power-of-two multiple of the scale, which is what keeps a zero-point-
    # free quantization exactly representable in bf16.
    offset = 1 << (bits - 1)
    if lo + offset < 0 or hi + offset >= (1 << bits):
        offset = -lo

    scale_dtype = None
    if m.recip:
        s = _eval(interp, m.scale,
                  dict(zip(interp._main_block().arguments, args)))
        scale_dtype = s.dtype  # a reciprocal is rarely exact in bf16, so
        # "auto" will widen to f32 here; METALJAX_QMM_SCALES=source forces
        # the narrow form for the traffic saving.
        interim = [1] * len(m.nshape)
        dims = m.bcast_dims
        if dims != sorted(dims):
            s = mx.transpose(s, sorted(range(len(dims)), key=lambda i: dims[i]))
            dims = sorted(dims)
        for i, d in enumerate(dims):
            interim[d] = s.shape[i]
        s = mx.reshape(mx.broadcast_to(mx.reshape(s, interim), m.nshape),
                       (m.N, 1))
        # The graph divides the OUTPUT by a per-output-channel scale; fold
        # the reciprocal into the weight scale. (A rounding change: 1/s is
        # computed once in f32 instead of dividing every output element.)
        scales = mx.divide(mx.array(1.0, mx.float32), s.astype(mx.float32))
        gs = next((g for g in _GROUP_SIZES if m.K % g == 0), None)
        if gs is None:
            raise _Reject(f"K={m.K} has no legal group size")
        scales = mx.broadcast_to(scales, (m.N, m.K // gs))
        zeros = None
    else:
        scale_map = evaluate(m.scale)
        zero_map = evaluate(m.zero) if m.zero is not None else None
        gs = None
        for g in _GROUP_SIZES:
            if m.K % g:
                continue
            if _group_const(scale_map, g) and (
                    zero_map is None or _group_const(zero_map, g)):
                gs = g
                break
        if gs is None:
            raise _Reject("scales/zeros are not constant within any group")
        scale_dtype = scale_map.dtype
        scales = mx.contiguous(
            mx.reshape(scale_map, (m.N, m.K // gs, gs))[:, :, 0])
        zeros = None
        if zero_map is not None:
            # In f32 throughout: the zero map may be an integer tensor, and
            # MLX refuses to compare one against out-of-range literals.
            zeros = mx.contiguous(
                mx.reshape(zero_map, (m.N, m.K // gs, gs))[:, :, 0]
            ).astype(mx.float32)
            if not bool(mx.all(zeros == mx.round(zeros)).item()):
                raise _Reject("zero points are not integers")
            if bool(mx.any(mx.abs(zeros) > 32768).item()):
                raise _Reject("zero points out of range")
            if m.sub_range is not None:
                # The graph subtracts the zero point in INTEGER arithmetic,
                # which wraps; the rewrite computes it exactly. Only fuse
                # when nothing can wrap.
                zlo = int(mx.min(zeros).item())
                zhi = int(mx.max(zeros).item())
                if (lo - zhi) < m.sub_range[0] or (hi - zlo) > m.sub_range[1]:
                    raise _Reject("integer zero-point subtraction can wrap")
        # Free the full-size maps before packing: only the per-group values
        # are needed from here on.
        mx.eval(*[a for a in (scales, zeros) if a is not None])
        scale_map = zero_map = None

    w, scales, biases = pack_exact(codes, scales, zeros, bits,
                                   scale_dtype=scale_dtype, offset=offset)
    # Materialize: a lazy packed weight would pin the whole reconstruction
    # graph (and its full-size intermediates) for the life of the cache.
    mx.eval(w, scales, biases)
    STATS["packs"] += 1
    if _DEBUG:
        print(f"[metaljax] qmm: packed {m.name} bits={bits} group={gs} "
              f"offset={offset} scales={scales.dtype}", flush=True)
    return _Pack(leaves, w, scales, biases, gs, bits)


def _resolve(interp, m, args):
    """(pack, freshly_built) for these argument buffers."""
    leaves = [args[i] for i in m.arg_indices]
    for i, pk in enumerate(m.packs):
        if pk.matches(leaves):
            if i:
                m.packs.insert(0, m.packs.pop(i))
            return pk, False
    m.repacks += 1
    if m.repacks > _MAX_REPACKS:
        raise _Reject(f"operands changed {m.repacks} times; repacking costs "
                      f"more than the chain it replaces")
    pk = _build_pack(interp, m, args, leaves)
    m.packs.insert(0, pk)
    del m.packs[_MAX_PACKS:]
    return pk, True


def prologue(interp, args) -> bool:
    """Pack every recognized weight against these arguments (eagerly).

    Returns True when the set of rewritten dots changed, i.e. any cached
    trace built around the previous structure has to be dropped.
    """
    st = interp._qmm
    if st is None:
        st = analyze(interp)
    if not st.matches:
        return False
    changed = False
    values = []
    packed_any = False
    with interp.context:
        for m in st.matches:
            if m.disabled:
                continue
            try:
                pk, fresh = _resolve(interp, m, args)
                packed_any = packed_any or fresh
            except Exception as e:
                m.disabled = True
                changed = True
                STATS["fallbacks"] += 1
                if _DEBUG:
                    print(f"[metaljax] qmm: {m.name} falls back to the "
                          f"literal chain ({e})", flush=True)
                continue
            if (m.gs, m.bits) != (pk.gs, pk.bits):
                # Baked into any trace built earlier.
                changed = changed or m.gs != 0
                m.gs, m.bits = pk.gs, pk.bits
            m.slot = len(values)
            values.extend((pk.w, pk.scales, pk.biases))
        if changed:
            st.rebuild()
    st.values = values
    if packed_any:
        # The reconstruction ran once at full weight size; its intermediates
        # are dead now and Metal counts live buffers, not bytes.
        mx.clear_cache()
    return changed


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------


def values(interp):
    """Packed arrays for the current context (traced ones inside a trace)."""
    st = getattr(interp, "_qmm", None)
    return st.values if st is not None else ()


def push(interp, vals):
    """Rebind the packed arrays to a traced function's own inputs."""
    st = getattr(interp, "_qmm", None)
    if st is None or not st.values:
        return None
    prev = st.values
    st.values = list(vals)
    return (prev,)


def pop(interp, token):
    if token is not None:
        interp._qmm.values = token[0]


def emit(interp, m, env):
    """One `mx.quantized_matmul` in place of the whole dequant-and-dot."""
    st = interp._qmm
    w = st.values[m.slot]
    scales = st.values[m.slot + 1]
    biases = st.values[m.slot + 2]
    x = env[m.lhs]
    if m.lperm != list(range(len(x.shape))):
        x = mx.transpose(x, m.lperm)
    x = mx.reshape(x, (m.M, m.K))
    if x.dtype not in _FLOAT_MX:
        x = x.astype(m.out_dtype)
    y = mx.quantized_matmul(x, w, scales, biases, transpose=True,
                            group_size=m.gs, bits=m.bits)
    y = mx.reshape(y, m.mshape + m.nshape)
    if y.dtype != m.out_dtype:
        y = y.astype(m.out_dtype)
    return [y]
