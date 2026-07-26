"""Persistent-kernel execution for elementwise scan loops.

Recognizes the counted while-loops JAX emits for `lax.scan` whose per-step
computation is a pure elementwise DAG (no cross-feature mixing): stacked
inputs read via dynamic_slice at an affine function of the induction
variable, an elementwise cell (possibly behind func.calls), and
dynamic_update_slice writes of per-step outputs. Such loops — texmo's
mingru/rglru/lrnn family, forward AND their AD-generated backward — are
compiled into a single generated Metal kernel: one GPU thread per
batch/feature lane runs the whole time loop with state in registers.

Everything here is generic IR pattern-matching; nothing is layer-specific.
Any mismatch falls back to the interpreter's normal loop execution.
"""

from __future__ import annotations

import os

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes

ENABLED = os.environ.get("METALJAX_MSL", "1") != "0"
_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"
# Max width of a register-tail vector (per-lane block state) in vector mode:
# bodies containing small dot_generals hold the whole block in registers and
# unroll the matvec in-lane.
_REG_LIMIT = int(os.environ.get("METALJAX_MSL_REG", "16"))
_KERNEL_SEQ = 0


class _Unsupported(Exception):
    pass


# ------------------------------------------------------------ symbolic IR

class Sym:
    """A value in the loop body, symbolically."""
    __slots__ = ("shape", "dtype")


class SymCounter(Sym):
    """Integer scalar, affine in the induction variable: a*iv + b.
    `base` identifies which counter carry it derives from."""
    __slots__ = ("a", "b", "base")

    def __init__(self, a, b, base):
        self.a, self.b, self.base = a, b, base
        self.shape, self.dtype = (), "int"


class SymConst(Sym):
    __slots__ = ("value",)

    def __init__(self, value, dtype, shape=()):
        self.value, self.dtype, self.shape = value, dtype, shape


class SymLeaf(Sym):
    """A tensor materialized per-lane from a device buffer.
    kind: 'read' (slice of a stacked source at index `idx`),
          'whole' (loop-invariant tensor), 'state' (loop-carried).
    inner_shape: for reads, the buffer's per-step block shape (immutable
    under reshape/broadcast bookkeeping on .shape)."""
    __slots__ = ("kind", "source", "idx", "state_pos", "inner_shape",
                 "strides", "offset")

    def __init__(self, kind, shape, dtype, source=None, idx=None,
                 state_pos=None, inner_shape=None, strides=None, offset=0):
        self.kind, self.shape, self.dtype = kind, shape, dtype
        self.source, self.idx, self.state_pos = source, idx, state_pos
        self.inner_shape = inner_shape
        self.strides = strides if strides is not None else _rowmajor(shape)
        self.offset = offset


class SymElem(Sym):
    __slots__ = ("op", "args", "extra")

    def __init__(self, op, args, dtype, shape, extra=None):
        self.op, self.args, self.extra = op, args, extra
        self.dtype, self.shape = dtype, shape


class SymDot(Sym):
    """Small in-lane matvec: data (lane..., C) x invariant weights -> reg D.

    roles: for each dim of .shape, either ("data", i) — follows dim i of the
    data operand's lane part — or ("reg",) for the output-feature dim.
    widx: for each dim of the weight tensor, ("data", i) | ("c",) | ("d",).
    """
    __slots__ = ("data", "weight", "roles", "widx", "csize", "dsize")

    def __init__(self, data, weight, roles, widx, csize, dsize, dtype, shape):
        self.data, self.weight = data, weight
        self.roles, self.widx = roles, widx
        self.csize, self.dsize = csize, dsize
        self.dtype, self.shape = dtype, shape


class SymPerm(Sym):
    """Deferred transpose of an elementwise subtree. Never consumed by
    elementwise ops; dots absorb the permutation into their dims."""
    __slots__ = ("inner", "perm")

    def __init__(self, inner, perm):
        self.inner, self.perm = inner, tuple(perm)
        self.shape = tuple(inner.shape[p] for p in perm)
        self.dtype = inner.dtype


class SymPad(Sym):
    """Zero-padding along the register (last) dim only."""
    __slots__ = ("inner", "lo", "n")

    def __init__(self, inner, lo, n, shape):
        self.inner, self.lo, self.n = inner, lo, n
        self.shape, self.dtype = shape, inner.dtype


class SymStack(Sym):
    """A small tensor assembled by static-index dynamic_update_slices
    (the unrolled form of an inner scan's output stack). Never emitted:
    static dynamic_slices read parts back out and the node dissolves."""
    __slots__ = ("base", "parts", "axis")

    def __init__(self, base, parts, shape, dtype, axis=0):
        self.base, self.parts = base, parts
        self.axis = axis
        self.shape, self.dtype = tuple(shape), dtype


class SymIota(Sym):
    """Coordinate ramp along one axis (loop-invariant): value = coord+start.
    axis indexes .shape; the emitters map it to the register index (last
    axis) or a lane coordinate."""
    __slots__ = ("axis", "start")

    def __init__(self, axis, start, shape, dtype):
        self.axis, self.start = axis, start
        self.shape, self.dtype = tuple(shape), dtype


class SymRedReg(Sym):
    """In-lane reduction (add) over the register (last) dim of a
    register-resident expression; value has register width 1."""
    __slots__ = ("inner",)

    def __init__(self, inner, shape, dtype):
        self.inner = inner
        self.shape, self.dtype = tuple(shape), dtype


class SymAccRed(Sym):
    """Cross-lane reduce-sum (e.g. bias-gradient accumulation): hoisted out
    of the kernel as a post-kernel sum over the stacked operand."""
    __slots__ = ("inner", "dims", "perm")

    def __init__(self, inner, dims, shape, dtype, perm=None):
        self.inner, self.dims = inner, tuple(dims)
        self.shape, self.dtype = shape, dtype
        self.perm = perm or tuple(range(len(shape)))


class SymAccDot(Sym):
    """Cross-lane dot (e.g. per-step weight-gradient contributions): cannot
    run per-lane; hoisted out of the kernel as one einsum over stacked
    operands (loop fission). Only legal directly inside a state-accumulator
    update: state' = state + acc_dot."""
    __slots__ = ("lhs", "rhs", "dims", "perm")

    def __init__(self, lhs, rhs, dims, shape, dtype, perm=None):
        self.lhs, self.rhs, self.dims = lhs, rhs, dims
        self.shape, self.dtype = shape, dtype
        self.perm = perm or tuple(range(len(shape)))


_MSL_DTYPE = {"f32": "float", "f16": "half", "i32": "int", "i1": "bool",
              "i64": "long", "i16": "short", "i8": "char",
              "ui64": "ulong", "ui32": "uint", "ui16": "ushort",
              "ui8": "uchar"}


def _dt(t: ir.Type) -> str:
    s = str(t)
    if s == "f64" and dtypes.F64_DOWNCAST:
        return "f32"  # engine stores f64 as f32 device-wide in downcast mode
    if s in _MSL_DTYPE:
        return s
    raise _Unsupported(f"dtype {s}")


def _ttype(v: ir.Value) -> ir.RankedTensorType:
    return ir.RankedTensorType(v.type)


_UNARY = {
    "stablehlo.negate": "(-{0})",
    "stablehlo.abs": "metal::abs({0})",
    "stablehlo.exponential": "metal::precise::exp({0})",
    "stablehlo.log": "metal::precise::log({0})",
    "stablehlo.log_plus_one": "metal::precise::log(1.0f + {0})",
    "stablehlo.exponential_minus_one": "(metal::precise::exp({0}) - 1.0f)",
    "stablehlo.tanh": "metal::precise::tanh({0})",
    "stablehlo.logistic": "(1.0f / (1.0f + metal::precise::exp(-({0}))))",
    "stablehlo.sqrt": "metal::precise::sqrt({0})",
    "stablehlo.rsqrt": "metal::precise::rsqrt({0})",
    "stablehlo.floor": "metal::floor({0})",
    "stablehlo.ceil": "metal::ceil({0})",
    "stablehlo.sign": "metal::sign({0})",
    "stablehlo.cosine": "metal::precise::cos({0})",
    "stablehlo.sine": "metal::precise::sin({0})",
    "stablehlo.not": "(!({0}))",
}

_BINARY = {
    "stablehlo.add": "({0} + {1})",
    "stablehlo.subtract": "({0} - {1})",
    "stablehlo.multiply": "({0} * {1})",
    "stablehlo.divide": "({0} / {1})",
    "stablehlo.maximum": "metal::max({0}, {1})",
    "stablehlo.minimum": "metal::min({0}, {1})",
    "stablehlo.power": "metal::precise::pow({0}, {1})",
    "stablehlo.remainder": "metal::fmod({0}, {1})",
    "stablehlo.and": "({0} && {1})",
    "stablehlo.or": "({0} || {1})",
    "stablehlo.xor": "({0} != {1})",
}

_COMPARE = {"EQ": "==", "NE": "!=", "LT": "<", "LE": "<=", "GT": ">", "GE": ">="}


# ------------------------------------------------------------ analysis

class _Analyzer:
    """Symbolically evaluates a loop body block into Sym expressions."""

    def __init__(self, interp, body_block, counter_pos):
        self.interp = interp
        self.body = body_block
        self.counter_pos = counter_pos
        self.args = list(body_block.arguments)
        self.reads = []          # SymLeaf('read'/'whole') in discovery order
        self.hoist_ok = set()    # ir results representable via hoisting
        self.hoisted_read_srcs = set()  # carries read at fixed index by hoists
        self.invariant_vals = set()
        self.updates = {}        # carry pos -> (idx SymCounter, value Sym)
        self.free = []           # captured ir.Values used as whole tensors

    def analyze(self):
        env = {}
        self.counter_seeded = set()
        for i, a in enumerate(self.args):
            t = _ttype(a)
            shape = tuple(t.shape)
            el = str(t.element_type)
            if i == self.counter_pos or (el in ("i32", "i64") and shape == ()):
                env[a] = SymCounter(1, 0, i)
                self.counter_seeded.add(i)
            else:
                # passthrough source / state / stacked-output: decided lazily
                env[a] = SymLeaf("arg", shape, _dt(t.element_type), source=("carry", i))
        rets = self.eval_block(self.body, env)
        return rets

    def lookup(self, env, v):
        s = env.get(v)
        if s is None:
            # Captured from an enclosing scope. Splat constants fold to
            # SymConst (jax hoists e.g. the loop increment out of the body);
            # anything else is a loop-invariant tensor/scalar input.
            t = _ttype(v)
            s = None
            if isinstance(v, ir.OpResult):
                owner = v.owner
                o = getattr(owner, "operation", owner)
                if o.name == "stablehlo.constant":
                    attr = ir.DenseElementsAttr(o.attributes["value"])
                    if attr.is_splat or not tuple(t.shape):
                        val = _ir.dense_to_np(o.attributes["value"], t)
                        flat = val.reshape(-1)
                        s = SymConst(flat[0] if flat.size else 0,
                                     _dt(t.element_type), tuple(t.shape))
            if s is None:
                s = SymLeaf("whole", tuple(t.shape), _dt(t.element_type),
                            source=("free", v))
            env[v] = s
        return s

    def _op_invariant(self, o, block_args):
        """True when every transitive input of `o` within the body comes
        from captures or other invariant ops (never a block argument), so
        its value is identical on every iteration."""
        if o.regions or o.name in ("func.call", "stablehlo.composite"):
            return False
        for v in o.operands:
            if v in block_args:
                return False
            if isinstance(v, ir.OpResult) and v.owner.operation.parent is not None:
                if v in self.invariant_vals:
                    continue
                # defined inside the body by a non-invariant op?
                owner_blk = v.owner.operation.parent
                if owner_blk == self._cur_block_op:
                    return False
            # captures (defined outside) are invariant
        return True

    def eval_block(self, block, env):
        result = None
        block_args = set(block.arguments)
        self._cur_block_op = getattr(block.owner, "operation", None)
        for op in block.operations:
            o = op.operation
            name = o.name
            if name in ("func.return", "stablehlo.return"):
                result = [self.lookup(env, v) for v in o.operands]
                break
            try:
                outs = self.eval_op(o, env)
            except _Unsupported:
                outs = self._try_hoist(o, env, block_args)
                if outs is None:
                    raise
            for r, s in zip(o.results, outs):
                env[r] = s
        return result

    def _try_hoist(self, o, env, block_args):
        """Represent a loop-invariant op we can't symify as a 'whole'
        input: the plan evaluates its IR subgraph once per call through
        the regular interpreter and feeds the result as a buffer."""
        if o.regions or os.environ.get("MJDBG_NOHOIST"):
            return None

        def sym_invariant(sym):
            if isinstance(sym, (SymConst, SymIota)):
                return True
            if isinstance(sym, SymLeaf):
                if sym.kind == "whole":
                    return True
                if sym.kind == "read" and getattr(sym.idx, "a", 1) == 0:
                    # fixed-index slice: constant across iterations as long
                    # as the source carry is never mutated (checked at Plan
                    # classification via hoisted_read_srcs)
                    if sym.source[0] == "carry":
                        self.hoisted_read_srcs.add(sym.source[1])
                    return True
                return False
            if isinstance(sym, SymElem):
                return all(sym_invariant(a) for a in sym.args)
            if isinstance(sym, (SymPad, SymPerm, SymRedReg)):
                return sym_invariant(sym.inner)
            if isinstance(sym, SymDot):
                return sym_invariant(sym.data)  # weights are whole leaves
            return False

        def val_invariant(v):
            if v in block_args:
                return False
            sym = env.get(v)
            if sym is not None:
                return v in self.hoist_ok or sym_invariant(sym)
            return True  # capture from an enclosing scope

        if not all(val_invariant(v) for v in o.operands):
            if _DEBUG:
                def why(sym, d=0):
                    if d > 6 or sym_invariant(sym):
                        return None
                    if isinstance(sym, SymElem):
                        for a in sym.args:
                            w = why(a, d + 1)
                            if w:
                                return w
                        return f"elem {sym.op}"
                    if isinstance(sym, SymLeaf):
                        return (f"leaf kind={sym.kind} idx={type(sym.idx).__name__}"
                                f" a={getattr(sym.idx, 'a', '?')} src={sym.source[0]}")
                    return type(sym).__name__
                det = [why(env.get(v)) for v in o.operands
                       if env.get(v) is not None and not val_invariant(v)]
                print(f"[metaljax] hoist refused {o.name}: {det}", flush=True)
            return None
        from metaljax.interpreter import REGISTRY
        if o.name not in REGISTRY:
            return None
        outs = []
        for r in o.results:
            t = ir.RankedTensorType(r.type)
            leaf = SymLeaf("whole", tuple(t.shape), _dt(t.element_type),
                           source=("hoist", r))
            self.hoist_ok.add(r)
            outs.append(leaf)
        return outs

    def eval_op(self, o, env):
        name = o.name
        ins = [self.lookup(env, v) for v in o.operands]

        if name == "stablehlo.constant":
            t = ir.RankedTensorType(o.results[0].type)
            attr = ir.DenseElementsAttr(o.attributes["value"])
            if not attr.is_splat and tuple(t.shape):
                raise _Unsupported("non-splat constant")
            val = _ir.dense_to_np(o.attributes["value"], t)
            flat = val.reshape(-1)
            v0 = flat[0] if flat.size else 0
            return [SymConst(v0, _dt(t.element_type), tuple(t.shape))]

        if name == "func.call":
            callee = ir.FlatSymbolRefAttr(o.attributes["callee"]).value
            fn = self.interp.funcs[callee]
            blk = fn.regions[0].blocks[0]
            sub = {a: s for a, s in zip(blk.arguments, ins)}
            return self.eval_block(blk, sub)

        if name in ("stablehlo.add", "stablehlo.subtract", "stablehlo.multiply"):
            a, b = ins
            if isinstance(a, (SymCounter, SymConst)) and isinstance(b, (SymCounter, SymConst)) \
               and a.shape == () and b.shape == () and {a.dtype, b.dtype} <= {"int", "i32", "i64"}:
                return [self._counter_arith(name, a, b)]
            if _DEBUG and any(isinstance(x, SymCounter) for x in ins):
                print(f"[metaljax] msl_scan: counter arith missed: "
                      f"{[(type(x).__name__, x.dtype, x.shape) for x in ins]}",
                      flush=True)

        if name == "stablehlo.reshape":
            (x,) = ins
            out_shape = tuple(ir.RankedTensorType(o.results[0].type).shape)
            return [self._reshaped(x, out_shape)]

        if name == "stablehlo.broadcast_in_dim":
            (x,) = ins
            out_shape = tuple(ir.RankedTensorType(o.results[0].type).shape)
            dims = _ir.i64_list(o, "broadcast_dimensions")
            return [self._broadcasted(x, out_shape, dims)]

        if name == "stablehlo.while":
            if os.environ.get("MJDBG_NONESTED"):
                raise _Unsupported("op stablehlo.while (disabled)")
            # Symbolically unroll small statically-counted nested loops
            # (lrnn's rank recursion nests a tiny scan inside the sequence
            # scan). The inner counter becomes a constant per iteration.
            from metaljax.ops.control import _analyze_counted, _static_start
            counted = _analyze_counted(self.interp, o)
            if counted is None or not isinstance(counted[1], int):
                raise _Unsupported("op stablehlo.while (dynamic nested)")
            k, bound = counted
            start = _static_start(o, k)
            if start is None:
                st_sym = ins[k]
                if isinstance(st_sym, SymConst):
                    start = int(st_sym.value)
                elif isinstance(st_sym, SymCounter) and st_sym.a == 0:
                    start = int(st_sym.b)
            if start is None:
                raise _Unsupported("op stablehlo.while (dynamic nested start)")
            trip = bound - start
            if trip <= 0 or trip > 64:
                raise _Unsupported(f"op stablehlo.while (nested trip {trip})")
            body = o.regions[1].blocks[0]
            bargs = list(body.arguments)
            vals = list(ins)
            for it in range(trip):
                env2 = dict(env)
                for a, v in zip(bargs, vals):
                    env2[a] = v
                env2[bargs[k]] = SymConst(start + it, "i32")
                vals = self.eval_block(body, env2)
            return vals

        if name == "stablehlo.iota":
            t = ir.RankedTensorType(o.results[0].type)
            out_shape = tuple(t.shape)
            d = int(ir.IntegerAttr(o.attributes["iota_dimension"]).value)
            return [SymIota(d, 0, out_shape, _dt(t.element_type))]

        if name == "stablehlo.concatenate":
            dim = _ir.int_attr(o, "dimension")
            out_shape = tuple(ir.RankedTensorType(o.results[0].type).shape)
            if dim != len(out_shape) - 1:
                raise _Unsupported(
                    f"concat on non-feature dim {dim} of {out_shape}: "
                    + "; ".join(f"{type(x).__name__}{getattr(x,'shape',())}"
                                for x in ins))
            total = out_shape[-1]
            acc = None
            off = 0
            for x in ins:
                part = SymPad(x, off, total, out_shape)
                off += x.shape[-1]
                acc = part if acc is None else SymElem(
                    "stablehlo.add", [acc, part], x.dtype, out_shape)
            return [acc]

        if name == "stablehlo.slice":
            (x,) = ins
            starts = _ir.i64_list(o, "start_indices")
            limits = _ir.i64_list(o, "limit_indices")
            steps = _ir.i64_list(o, "strides")
            if any(st != 1 for st in steps):
                raise _Unsupported("strided slice")
            sizes = [l - a for a, l in zip(starts, limits)]
            return [self._sliced(x, starts, tuple(sizes))]

        if name == "stablehlo.pad":
            x, pv = ins
            if not (isinstance(pv, SymConst) and float(pv.value) == 0.0):
                raise _Unsupported("non-zero pad")
            low = _ir.i64_list(o, "edge_padding_low")
            high = _ir.i64_list(o, "edge_padding_high")
            interior = _ir.i64_list(o, "interior_padding")
            if any(i != 0 for i in interior):
                raise _Unsupported("interior pad")
            out_shape = tuple(ir.RankedTensorType(o.results[0].type).shape)
            if any(l != 0 or h != 0 for l, h in
                   zip(low[:-1], high[:-1])):
                raise _Unsupported("pad on lane dims")
            if not low or (low[-1] == 0 and high[-1] == 0):
                return [x]
            return [SymPad(x, low[-1], x.shape[-1], out_shape)]

        if name == "stablehlo.reduce":
            if len(ins) != 2:
                raise _Unsupported("variadic reduce")
            x, init = ins
            if not (isinstance(init, SymConst) and float(init.value) == 0.0):
                raise _Unsupported("reduce with non-zero init")
            body_ops = [b.operation
                        for b in o.regions[0].blocks[0].operations]
            if len(body_ops) != 2 or body_ops[0].name != "stablehlo.add":
                raise _Unsupported("non-add reduce")
            dims = _ir.i64_list(o, "dimensions")
            out_shape = tuple(ir.RankedTensorType(o.results[0].type).shape)
            if all(x.shape[d] == 1 for d in dims):
                # reducing unit dims is a squeeze, not a contraction
                if isinstance(x, (SymAccDot, SymAccRed, SymDot)):
                    return [self._reshaped(x, out_shape)]
                out = x
                for d in sorted(dims, reverse=True):
                    out = self._dropdim(out, d)
                return [out]
            dot = self._reduce_as_dot(x, dims, out_shape)
            if dot is not None:
                return [dot]
            if (list(dims) == [len(x.shape) - 1]
                    and not os.environ.get("MJDBG_NOREDREG")):
                # in-lane: reduce over the register dim (texmo's lrnn
                # readout contracts register-resident products)
                return [SymRedReg(x, out_shape, x.dtype)]
            return [SymAccRed(x, dims, out_shape, x.dtype)]

        if name == "stablehlo.dynamic_slice":
            return [self._dynamic_slice(o, ins)]

        if name == "stablehlo.dynamic_update_slice":
            return [self._dynamic_update(o, ins)]

        if name == "stablehlo.dot_general":
            return [self._dot_general(o, ins)]

        if name == "stablehlo.transpose":
            (x,) = ins
            perm = _ir.i64_list(o, "permutation")
            return [_transpose_sym(x, perm)]

        if name == "stablehlo.convert":
            (x,) = ins
            dt = _dt(ir.RankedTensorType(o.results[0].type).element_type)
            return [SymElem("convert", [x], dt, x.shape, extra=dt)]

        if name == "stablehlo.compare":
            direction = None
            s = str(o.attributes["comparison_direction"])
            for d in _COMPARE:
                if d in s:
                    direction = d
                    break
            if direction is None:
                raise _Unsupported("compare direction")
            return [SymElem("compare", ins, "i1",
                            _bshape(ins[0].shape, ins[1].shape), extra=direction)]

        if name == "stablehlo.select":
            p, a, b = ins
            return [SymElem("select", ins, a.dtype,
                            _bshape(_bshape(p.shape, a.shape), b.shape))]

        if name == "stablehlo.clamp":
            lo, x, hi = ins
            return [SymElem("clamp", ins, x.dtype, x.shape)]

        if name in _UNARY:
            (x,) = ins
            return [SymElem(name, ins, x.dtype, x.shape)]

        if name in _BINARY:
            a, b = ins
            return [SymElem(name, ins, a.dtype, _bshape(a.shape, b.shape))]

        raise _Unsupported(f"op {name}")

    def _varies(self, s, axis):
        """Does `s` (rank-aligned to its own shape) vary along `axis`?
        Conservative toward True for unknown node kinds."""
        shape = getattr(s, "shape", ()) or ()
        if axis >= len(shape) or shape[axis] == 1:
            return False
        if isinstance(s, (SymConst, SymCounter)):
            return False
        if isinstance(s, SymIota):
            return axis == s.axis
        if isinstance(s, SymLeaf):
            return s.strides[axis] != 0
        if isinstance(s, SymElem):
            k = len(shape)
            for a in s.args:
                ashape = getattr(a, "shape", ()) or ()
                ax = axis - (k - len(ashape))
                if ax >= 0 and self._varies(a, ax):
                    return True
            return False
        return True

    def _dropdim(self, s, axis):
        """Remove a size-1 axis from a sym (leaves drop the stride entry)."""
        shape = tuple(s.shape[:axis]) + tuple(s.shape[axis + 1:])
        if isinstance(s, SymConst):
            return SymConst(s.value, s.dtype, shape)
        if isinstance(s, SymLeaf):
            out = _clone_leafish(s)
            out.shape = shape
            out.strides = (tuple(s.strides[:axis])
                           + tuple(s.strides[axis + 1:]))
            return out
        if isinstance(s, SymElem):
            k = len(s.shape)
            args = []
            for a in s.args:
                ashape = getattr(a, "shape", ()) or ()
                ax = axis - (k - len(ashape))
                if ax < 0:
                    args.append(a)
                elif ashape[ax] == 1:
                    args.append(self._dropdim(a, ax))
                else:
                    raise _Unsupported("dropdim of varying arg")
            return SymElem(s.op, args, s.dtype, shape, extra=s.extra)
        if isinstance(s, SymRedReg):
            return SymRedReg(s.inner, shape, s.dtype)
        if isinstance(s, SymIota):
            if axis == s.axis:
                raise _Unsupported("dropdim of iota axis")
            na = s.axis - (1 if axis < s.axis else 0)
            return SymIota(na, s.start, shape, s.dtype)
        if isinstance(s, SymStack):
            if axis == s.axis:
                if s.shape[axis] == 1 and 0 in s.parts:
                    # a one-slot stack IS its single part
                    return self._dropdim(s.parts[0], axis)
                raise _Unsupported("dropdim of stack axis")
            na = s.axis - (1 if axis < s.axis else 0)
            return SymStack(self._dropdim(s.base, axis),
                            {k2: self._dropdim(v2, axis)
                             for k2, v2 in s.parts.items()},
                            shape, s.dtype, axis=na)
        raise _Unsupported(f"dropdim of {type(s).__name__}")

    def _squeeze_axis(self, s, axis):
        """Slice a non-varying axis to width 1 and drop it."""
        starts = [0] * len(s.shape)
        sizes = list(s.shape)
        sizes[axis] = 1
        return self._dropdim(self._sliced(s, starts, tuple(sizes)), axis)

    def _reduce_as_dot(self, x, dims, out_shape):
        """Recognize reduce(add, multiply(a, b), dims) as a contraction.

        texmo's lrnn lowers its block matvec as broadcast-multiply +
        reduce instead of dot_general. Two shapes come up:
        - one side is a loop-invariant weight leaf and the reduce is over
          the trailing register dims -> an in-lane SymDot (fwd and the
          dh backward);
        - both sides vary and the reduce covers lane dims -> a cross-lane
          SymAccDot (the dW accumulator), evaluated post-kernel.
        Returns a Sym or None.
        """
        if not (isinstance(x, SymElem) and x.op == "stablehlo.multiply"
                and len(x.args) == 2):
            return None
        rank = len(x.shape)
        a, b = x.args

        def var_axes(s):
            k = rank - len(getattr(s, "shape", ()) or ())
            return {ax for ax in range(rank)
                    if ax >= k and self._varies(s, ax - k)}

        va, vb = var_axes(a), var_axes(b)
        if _DEBUG:
            print(f"[metaljax] reduce_as_dot: dims={dims} "
                  f"a={_dump(a)[:120]} va={sorted(va)} "
                  f"b={_dump(b)[:120]} vb={sorted(vb)}", flush=True)

        def invariant_leaf(s):
            return isinstance(s, SymLeaf) and s.kind in ("whole", "arg")

        # Case 1: in-lane dot. Exactly one reduced dim; it and the 'd'
        # output dim must be the two trailing dims.
        if len(dims) == 1:
            red = dims[0]
            for w, d_expr in ((a, b), (b, a)):
                if not (invariant_leaf(w) and red in var_axes(w)
                        and red in var_axes(d_expr)):
                    continue
                vw, vd = var_axes(w), var_axes(d_expr)
                d_axes = [ax for ax in range(rank)
                          if ax != red and ax in vw and ax not in vd]
                if len(d_axes) != 1:
                    continue
                d_ax = d_axes[0]
                if {d_ax, red} != {rank - 2, rank - 1}:
                    continue
                if x.dtype != "f32":
                    continue
                csize, dsize = x.shape[red], x.shape[d_ax]
                if csize > 4096 or dsize > 4096:
                    continue
                # compact weight: drop broadcast (stride-0 or size-1) dims
                k = rank - len(w.shape)
                wleaf = _clone_leafish(w)
                keep = [i for i in range(len(w.shape))
                        if (i + k) in vw]
                wleaf.shape = tuple(w.shape[i] for i in keep)
                wleaf.strides = tuple(w.strides[i] for i in keep)
                widx = []
                lane_map = {}
                lane_axes = [ax for ax in range(rank)
                             if ax not in (red, d_ax)]
                for li, ax in enumerate(lane_axes):
                    lane_map[ax] = li
                for i in keep:
                    ax = i + k
                    if ax == red:
                        widx.append(("c",))
                    elif ax == d_ax:
                        widx.append(("d",))
                    else:
                        widx.append(("data", lane_map[ax]))
                data = self._squeeze_axis(d_expr if rank == len(
                    getattr(d_expr, "shape", ())) else
                    self._broadcasted(d_expr, x.shape,
                                      list(range(rank - len(d_expr.shape),
                                                 rank))), d_ax)
                # data now has the reduced dim last
                roles = []
                for ax in range(rank):
                    if ax == red:
                        continue
                    roles.append(("reg",) if ax == d_ax
                                 else ("data", lane_map[ax]))
                return SymDot(data, wleaf, roles, widx, csize, dsize,
                              "f32", tuple(out_shape))

        # Case 2: cross-lane contraction (weight gradients): every reduced
        # dim varies on both sides and none is the register (last) dim;
        # compact each side by squeezing its broadcast axes.
        if (all(ax in va and ax in vb for ax in dims)
                and all(ax != rank - 1 for ax in dims)):
            def compact(s, own_axes):
                k = rank - len(getattr(s, "shape", ()) or ())
                if k:
                    s = self._broadcasted(s, x.shape, list(range(k, rank)))
                mapping = {}
                out = s
                removed = 0
                for ax in range(rank):
                    if ax in own_axes:
                        mapping[ax] = ax - removed
                    else:
                        out = self._squeeze_axis(out, ax - removed)
                        removed += 1
                return out, mapping
            try:
                ca, ma = compact(a, va)
                cb, mb = compact(b, vb)
            except _Unsupported:
                return None
            kept = [ax for ax in range(rank) if ax not in dims]
            if any(ax not in va and ax not in vb and x.shape[ax] != 1
                   for ax in kept):
                return None
            batch = [ax for ax in kept if ax in va and ax in vb]
            a_only = [ax for ax in kept if ax in va and ax not in vb]
            b_only = [ax for ax in kept if ax in vb and ax not in va]
            ones = [ax for ax in kept if ax not in va and ax not in vb]
            if ones:
                return None  # size-1 leftovers: rare, skip
            lb = [ma[ax] for ax in batch]
            rb = [mb[ax] for ax in batch]
            lc = [ma[ax] for ax in dims]
            rc = [mb[ax] for ax in dims]
            einsum_order = batch + a_only + b_only
            perm = tuple(einsum_order.index(ax) for ax in kept)
            return SymAccDot(ca, cb, (lb, rb, lc, rc), tuple(out_shape),
                             "f32", perm=perm)
        return None

    def _dot_general(self, o, ins):
        from metaljax.ops.linalg import _dot_dims
        lb, rb, lc, rc = _dot_dims(o)
        lhs, rhs = ins
        if isinstance(lhs, SymPerm):
            lb = [lhs.perm[d] for d in lb]
            lc = [lhs.perm[d] for d in lc]
            lhs = lhs.inner
        if isinstance(rhs, SymPerm):
            rb = [rhs.perm[d] for d in rb]
            rc = [rhs.perm[d] for d in rc]
            rhs = rhs.inner

        def invariant(s):
            # Loop-invariant weights: a free capture, or an untouched carry
            # (verified to be passthrough at classification time via the
            # mutated-carry check).
            return isinstance(s, SymLeaf) and s.kind in ("whole", "arg")

        shape = tuple(ir.RankedTensorType(o.results[0].type).shape)

        def attempt(w, data, wb, db, wc, dc, w_is_lhs):
            if not invariant(w):
                return None
            if len(wc) != 1 or len(dc) != 1:
                return None
            if dc[0] != len(data.shape) - 1:
                return None
            csize = data.shape[dc[0]]
            wfree = [i for i in range(len(w.shape))
                     if i not in wb and i != wc[0]]
            if len(wfree) != 1:
                return None
            dsize = w.shape[wfree[0]]
            if csize > 4096 or dsize > 4096:
                return None
            if w.dtype != "f32" or data.dtype != "f32":
                return None
            widx = [None] * len(w.shape)
            for wdim, ddim in zip(wb, db):
                widx[wdim] = ("data", ddim)
            widx[wc[0]] = ("c",)
            widx[wfree[0]] = ("d",)
            dfree = [i for i in range(len(data.shape) - 1) if i not in db]
            roles = [("data", d) for d in db]
            if w_is_lhs:
                roles += [("reg",)] + [("data", d) for d in dfree]
            else:
                roles += [("data", d) for d in dfree] + [("reg",)]
            return SymDot(data, w, roles, widx, csize, dsize, "f32", shape)

        out = attempt(rhs, lhs, rb, lb, rc, lc, False)
        if out is None:
            out = attempt(lhs, rhs, lb, rb, lc, rc, True)
        if out is None:
            if _DEBUG:
                print(f"[metaljax] msl_scan: accdot lhs={_dump(lhs)} "
                      f"rhs={_dump(rhs)} dims={(lb, rb, lc, rc)}", flush=True)
            # Cross-lane contraction (weight-gradient accumulation):
            # representable only as a hoisted post-kernel einsum.
            out = SymAccDot(lhs, rhs, (lb, rb, lc, rc), shape, "f32")
        return out

    def _counter_arith(self, name, a, b):
        def parts(s):
            if isinstance(s, SymCounter):
                return s.a, s.b, s.base
            return 0, int(s.value), None
        aa, ab, abase = parts(a)
        ba, bb, bbase = parts(b)
        if abase is not None and bbase is not None:
            raise _Unsupported("counter + counter")
        base = abase if abase is not None else bbase
        if name == "stablehlo.add":
            r = (aa + ba, ab + bb)
        elif name == "stablehlo.subtract":
            r = (aa - ba, ab - bb)
        else:  # multiply
            if abase is not None and bbase is not None:
                raise _Unsupported("counter * counter")
            if abase is not None:
                r = (aa * bb, ab * bb)
            else:
                r = (ba * ab, bb * ab)
        if base is None:
            return SymConst(r[1], "int")
        return SymCounter(r[0], r[1], base)

    def _sliced(self, x, starts, sizes):
        if isinstance(x, SymConst):
            return SymConst(x.value, x.dtype, sizes)
        if isinstance(x, SymLeaf):
            out = _clone_leafish(x)
            out.offset = x.offset + sum(a * st for a, st in
                                        zip(starts, x.strides))
            out.shape = sizes
            return out
        if isinstance(x, SymElem):
            args = []
            for a in x.args:
                if isinstance(a, (SymConst, SymCounter)) or not a.shape \
                        or all(d == 1 for d in a.shape):
                    args.append(a)
                    continue
                k = len(x.shape) - len(a.shape)
                asub = [min(st, max(0, d - 1)) if d == 1 else st
                        for st, d in zip(starts[k:], a.shape)]
                asz = [1 if d == 1 else z
                       for z, d in zip(sizes[k:], a.shape)]
                args.append(self._sliced(a, asub, tuple(asz)))
            return SymElem(x.op, args, x.dtype, sizes, extra=x.extra)
        if isinstance(x, SymIota):
            return SymIota(x.axis, x.start + starts[x.axis], sizes, x.dtype)
        if isinstance(x, SymDot):
            reg_axis = next(
                (i for i, r in enumerate(x.roles) if r[0] == "reg"), None)
            if reg_axis is not None and all(
                    a == 0 and z == d
                    for i, (a, z, d) in enumerate(zip(starts, sizes, x.shape))
                    if i != reg_axis):
                lo = starts[reg_axis]
                nsize = sizes[reg_axis]
                wd = next(i for i, r in enumerate(x.widx) if r[0] == "d")
                w = _clone_leafish(x.weight)
                w.offset = x.weight.offset + lo * x.weight.strides[wd]
                wshape = list(w.shape)
                wshape[wd] = nsize
                w.shape = tuple(wshape)
                nshape = list(x.shape)
                nshape[reg_axis] = nsize
                return SymDot(x.data, w, x.roles, x.widx, x.csize, nsize,
                              x.dtype, tuple(nshape))
        raise _Unsupported(f"slice of {type(x).__name__}")

    def _reshaped(self, x, out_shape):
        if x.shape == out_shape:
            return x
        # only allow dropping/adding leading 1s
        if tuple(d for d in x.shape if d != 1) != tuple(d for d in out_shape if d != 1):
            raise _Unsupported(f"reshape {x.shape} -> {out_shape}")
        if isinstance(x, SymConst):
            return SymConst(x.value, x.dtype, out_shape)
        if isinstance(x, SymElem):
            args = [a if isinstance(a, (SymConst, SymCounter))
                    else self._reshaped(a, out_shape) for a in x.args]
            return SymElem(x.op, args, x.dtype, out_shape, extra=x.extra)
        if isinstance(x, SymAccRed):
            return SymAccRed(x.inner, x.dims, out_shape, x.dtype, perm=x.perm)
        if isinstance(x, SymAccDot):
            return SymAccDot(x.lhs, x.rhs, x.dims, out_shape, x.dtype,
                             perm=x.perm)
        if isinstance(x, SymRedReg):
            return SymRedReg(x.inner, out_shape, x.dtype)
        if isinstance(x, SymDot):
            # unit-dim insertions/removals: rebuild roles positionally
            roles = []
            i = 0
            xs = list(x.shape)
            for d in out_shape:
                if i < len(xs) and xs[i] == d:
                    roles.append(x.roles[i])
                    i += 1
                elif d == 1:
                    roles.append(("one",))
                elif (i < len(xs) and xs[i] == 1
                      and x.roles[i] == ("one",)):
                    i += 1  # dropped unit dim; retry this out dim
                    if i < len(xs) and xs[i] == d:
                        roles.append(x.roles[i])
                        i += 1
                    else:
                        raise _Unsupported(
                            f"reshape {x.shape} -> {out_shape} of dot")
                else:
                    raise _Unsupported(
                        f"reshape {x.shape} -> {out_shape} of dot")
            while i < len(xs):
                if xs[i] != 1 or x.roles[i] != ("one",):
                    raise _Unsupported(
                        f"reshape {x.shape} -> {out_shape} of dot")
                i += 1
            return SymDot(x.data, x.weight, roles, x.widx, x.csize,
                          x.dsize, x.dtype, tuple(out_shape))
        if not isinstance(x, SymLeaf):
            raise _Unsupported(f"reshape of {type(x).__name__}")
        s = _clone_leafish(x)
        s.strides = _remap_strides(x.shape, x.strides, out_shape)
        s.shape = out_shape
        return s

    def _broadcasted(self, x, out_shape, dims):
        # Represent as a reshape to a right-aligned shape with 1s; the lane
        # mapping resolves broadcasting via zero strides.
        interim = [1] * len(out_shape)
        for i, d in enumerate(dims):
            interim[d] = x.shape[i]
        if list(dims) != sorted(dims):
            raise _Unsupported("unsorted broadcast dims")
        if isinstance(x, SymConst):
            return SymConst(x.value, x.dtype, tuple(out_shape))
        if isinstance(x, SymElem):
            # broadcast commutes with elementwise: push down to leaves.
            # Args may have lower rank (right-aligned broadcasting): adjust
            # the dims mapping accordingly; rank-0/all-1 args pass through.
            args = []
            for a in x.args:
                if isinstance(a, (SymConst, SymCounter)) or not a.shape \
                        or all(d == 1 for d in a.shape):
                    args.append(a)
                    continue
                adims = list(dims)[len(x.shape) - len(a.shape):]
                args.append(self._broadcasted(a, out_shape, adims))
            return SymElem(x.op, args, x.dtype, tuple(out_shape), extra=x.extra)
        if isinstance(x, SymDot):
            # only pure dim-insertion (sizes preserved) is representable
            for i, d in enumerate(dims):
                if x.shape[i] != out_shape[d]:
                    raise _Unsupported("size-changing broadcast of dot")
            roles = [("one",)] * len(out_shape)
            for i, d in enumerate(dims):
                roles[d] = x.roles[i]
            return SymDot(x.data, x.weight, roles, x.widx,
                          x.csize, x.dsize, x.dtype, tuple(out_shape))
        if isinstance(x, SymRedReg):
            # width-1 value: broadcasting is free at emission
            return SymRedReg(x.inner, out_shape, x.dtype)
        if isinstance(x, SymIota):
            return SymIota(dims[x.axis], x.start, out_shape, x.dtype)
        if isinstance(x, SymStack):
            if all(x.shape[i] == out_shape[d] for i, d in enumerate(dims)):
                base = self._broadcasted(x.base, out_shape, dims)
                return SymStack(base, x.parts, out_shape, x.dtype,
                                axis=dims[x.axis])
            raise _Unsupported("size-changing broadcast of a stack")
        if isinstance(x, SymPad):
            # The pad window lives on the last dim; a broadcast that keeps
            # the last dim mapped last commutes with it.
            if dims and dims[-1] == len(out_shape) - 1 \
                    and x.shape[-1] == out_shape[-1]:
                inner = self._broadcasted(
                    x.inner, tuple(out_shape[:-1]) + (x.inner.shape[-1],),
                    dims)
                return SymPad(inner, x.lo, x.n, tuple(out_shape))
            raise _Unsupported("broadcast reshaping a pad window")
        if not isinstance(x, SymLeaf):
            raise _Unsupported(f"broadcast of {type(x).__name__} "
                               f"{x.shape}->{tuple(out_shape)} dims={list(dims)}")
        s = _clone_leafish(x)
        new_strides = [0] * len(out_shape)
        for i, d in enumerate(dims):
            if x.shape[i] != 1:
                if x.shape[i] != out_shape[d]:
                    raise _Unsupported("broadcast size mismatch")
                new_strides[d] = x.strides[i]
        s.shape = tuple(out_shape)
        s.strides = tuple(new_strides)
        return s

    @staticmethod
    def _static_index(sym):
        if isinstance(sym, SymConst):
            return int(sym.value)
        if isinstance(sym, SymCounter) and sym.a == 0:
            return int(sym.b)
        return None

    def _dynamic_slice(self, o, ins):
        x = ins[0]
        starts = ins[1:]
        sizes = _ir.i64_list(o, "slice_sizes")
        static = [self._static_index(st) for st in starts]
        if all(st is not None for st in static):
            # all-static dynamic_slice is a plain slice (nested-loop
            # unrolling turns inner-counter indices into constants)
            if isinstance(x, SymStack):
                ax = x.axis
                full_rest = all(
                    (d == ax and sizes[d] == 1)
                    or (d != ax and static[d] == 0 and sizes[d] == x.shape[d])
                    for d in range(len(x.shape)))
                if full_rest and static[ax] in x.parts:
                    part = x.parts[static[ax]]
                    want = tuple(sizes)
                    if tuple(part.shape) != want:
                        part = self._reshaped(part, want)
                    return part
                x = x.base
            return self._sliced(x, static, tuple(sizes))
        if not isinstance(x, SymLeaf) or x.kind not in ("arg", "whole"):
            raise _Unsupported(f"dynamic_slice of computed value: {type(x).__name__} {_dump(x)[:90]}")
        shape = x.shape
        if len(shape) < 1 or sizes[0] != 1:
            raise _Unsupported("dynamic_slice shape")
        for d in range(1, len(shape)):
            if sizes[d] != shape[d]:
                raise _Unsupported("partial dynamic_slice")
            s = starts[d]
            if not (isinstance(s, SymConst) and int(s.value) == 0) and not (
                isinstance(s, SymCounter) and s.a == 0 and s.b == 0
            ):
                raise _Unsupported("nonzero inner start")
        idx = starts[0]
        if isinstance(idx, SymConst):
            idx = SymCounter(0, int(idx.value), None)
        if not isinstance(idx, SymCounter):
            raise _Unsupported("non-affine slice index")
        leaf = SymLeaf("read", (1,) + tuple(shape[1:]), x.dtype,
                       source=x.source, idx=idx,
                       inner_shape=tuple(shape[1:]))
        self.reads.append(leaf)
        return leaf

    def _dynamic_update(self, o, ins):
        x, upd = ins[0], ins[1]
        starts = ins[2:]
        static = [self._static_index(st) for st in starts]
        if (all(st is not None for st in static)
                and all(st == 0 for st in static[1:])
                and isinstance(x, (SymConst, SymStack))
                and upd.shape and upd.shape[0] == 1
                and tuple(upd.shape[1:]) == tuple(x.shape[1:])):
            if isinstance(x, SymStack):
                parts = dict(x.parts)
                base = x.base
            else:
                parts = {}
                base = x
            parts[static[0]] = upd
            return SymStack(base, parts, x.shape, x.dtype)
        if not (isinstance(x, SymLeaf) and x.kind == "arg"
                and x.source[0] == "carry"):
            raise _Unsupported(f"dynamic_update_slice target: {type(x).__name__} {_dump(x)[:90]}")
        shape = x.shape
        if upd.shape[0] != 1 or tuple(upd.shape[1:]) != tuple(shape[1:]):
            raise _Unsupported("dus update shape")
        for s in starts[1:]:
            if not (isinstance(s, SymConst) and int(s.value) == 0) and not (
                isinstance(s, SymCounter) and s.a == 0 and s.b == 0
            ):
                raise _Unsupported("dus inner start")
        idx = starts[0]
        if isinstance(idx, SymConst):
            idx = SymCounter(0, int(idx.value), None)
        if not isinstance(idx, SymCounter):
            raise _Unsupported("dus non-affine index")
        out = SymLeaf("updated", shape, x.dtype, source=x.source, idx=idx)
        out.state_pos = None
        # remember the update expression on the sym itself
        out_update = (idx, upd)
        self.updates[id(out)] = out_update
        return out


def _off_strided(shape, strides, lane, base=0):
    """Lane-offset expression from explicit element strides (scalar mode)."""
    dims = list(shape)
    sts = list(strides)
    while len(dims) < len(lane):
        dims.insert(0, 1)
        sts.insert(0, 0)
    if len(dims) > len(lane):
        extra = dims[: len(dims) - len(lane)]
        if any(d != 1 for d in extra):
            raise _Unsupported(f"shape {shape} vs lane {lane}")
        sts = sts[len(dims) - len(lane):]
        dims = dims[len(dims) - len(lane):]
    terms = [f"{base}u"] if base else []
    for i, (d, st) in enumerate(zip(dims, sts)):
        if d == 1 or st == 0:
            continue
        if d != lane[i]:
            raise _Unsupported(f"shape {shape} vs lane {lane}")
        terms.append(f"c{i} * {st}u")
    return " + ".join(terms) if terms else "0u"


def _remap_strides(old_shape, old_strides, new_shape):
    """Strides after a reshape that only adds/drops size-1 dims."""
    core = [(d, st) for d, st in zip(old_shape, old_strides) if d != 1]
    out = []
    it = iter(core)
    for d in new_shape:
        if d == 1:
            out.append(0)
        else:
            cd, cst = next(it)
            if cd != d:
                raise _Unsupported("reshape strides")
            out.append(cst)
    return tuple(out)


def _rowmajor(shape):
    strides = []
    acc = 1
    for d in reversed(shape):
        strides.append(acc)
        acc *= d
    strides.reverse()
    return tuple(strides)


def _transpose_sym(x, perm):
    """Transpose commutes with elementwise ops: push it down to the leaves."""
    if isinstance(x, SymAccRed):
        return SymAccRed(x.inner, x.dims,
                         tuple(x.shape[p] for p in perm), x.dtype,
                         perm=tuple(x.perm[p] for p in perm))
    if isinstance(x, SymAccDot):
        return SymAccDot(x.lhs, x.rhs, x.dims,
                         tuple(x.shape[p] for p in perm), x.dtype,
                         perm=tuple(x.perm[p] for p in perm))
    if isinstance(x, SymDot):
        return SymDot(x.data, x.weight,
                      [x.roles[p] for p in perm], x.widx,
                      x.csize, x.dsize, x.dtype,
                      tuple(x.shape[p] for p in perm))
    if isinstance(x, SymLeaf):
        out = _clone_leafish(x)
        out.shape = tuple(x.shape[p] for p in perm)
        out.strides = tuple(x.strides[p] for p in perm)
        return out
    if isinstance(x, SymConst):
        return SymConst(x.value, x.dtype, tuple(x.shape[p] for p in perm))
    if isinstance(x, SymCounter):
        return x
    if isinstance(x, SymPerm):
        return SymPerm(x.inner, tuple(x.perm[p] for p in perm))
    if isinstance(x, SymRedReg):
        return SymRedReg(x.inner, tuple(x.shape[p] for p in perm), x.dtype)
    if isinstance(x, SymElem):
        n = len(x.shape)
        args = []
        for a in x.args:
            ashape = tuple(getattr(a, "shape", ()) or ())
            if not ashape or all(d == 1 for d in ashape):
                args.append(a)
                continue
            k = n - len(ashape)
            if k:
                # expand right-aligned args to full rank so the perm applies
                if isinstance(a, SymLeaf):
                    a = _clone_leafish(a)
                    a.shape = (1,) * k + ashape
                    a.strides = (0,) * k + tuple(a.strides)
                elif isinstance(a, SymConst):
                    a = SymConst(a.value, a.dtype, (1,) * k + ashape)
                elif isinstance(a, SymIota):
                    a = SymIota(a.axis + k, a.start, (1,) * k + ashape,
                                a.dtype)
                else:
                    return SymPerm(x, perm)
            args.append(_transpose_sym(a, perm))
        return SymElem(x.op, args, x.dtype,
                       tuple(x.shape[p] for p in perm), extra=x.extra)
    return SymPerm(x, perm)


def _dump(s, d=0):
    if isinstance(s, SymAccRed):
        return f"ACCRED[dims={list(s.dims)}]({_dump(s.inner)})"
    if isinstance(s, SymPerm):
        return f"PERM{list(s.perm)}({_dump(s.inner)})"
    if isinstance(s, SymElem):
        return (f"{s.op.split('.')[-1]}{list(s.shape)}("
                + ",".join(_dump(a) for a in s.args) + ")")
    if isinstance(s, SymAccDot):
        return f"ACC{list(s.shape)}perm{list(s.perm)}"
    if isinstance(s, SymLeaf):
        return f"{s.kind}{list(s.shape)}st{list(s.strides)}@{s.source}"
    if isinstance(s, SymConst):
        return f"c{s.value}"
    if isinstance(s, SymDot):
        return f"DOT{list(s.shape)}"
    return type(s).__name__


def _match_accum(s, pos, _unused):
    """state' = state + sum of cross-lane accumulators -> [acc syms].

    Unrolled nested loops produce sums: carry + (0 + ACC1 + ACC2 + ...);
    flatten the add tree, require exactly one occurrence of the carry,
    tolerate zero constants, and collect every AccDot/AccRed term."""
    terms = []
    def flatten(x):
        if (isinstance(x, SymElem) and x.op == "stablehlo.add"):
            for a in x.args:
                flatten(a)
        else:
            terms.append(x)
    flatten(s)
    carry = [t for t in terms
             if isinstance(t, SymLeaf) and t.kind == "arg"
             and t.source == ("carry", pos)
             and t.strides == _rowmajor(t.shape)]
    accs = [t for t in terms if isinstance(t, (SymAccDot, SymAccRed))]
    rest = [t for t in terms
            if t not in carry and t not in accs
            and not (isinstance(t, SymConst) and float(t.value) == 0.0)]
    if len(carry) == 1 and accs and not rest:
        return accs
    return None


def _contains_accdot(root):
    stack = [root]
    seen = set()
    while stack:
        s = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        if isinstance(s, (SymAccDot, SymAccRed)):
            return True
        if isinstance(s, SymElem):
            stack.extend(s.args)
        elif isinstance(s, SymDot):
            stack.append(s.data)
        elif isinstance(s, (SymPad, SymPerm, SymRedReg)):
            stack.append(s.inner)
    return False


def _collect_dots(roots):
    seen = set()
    stack = list(roots)
    out = []
    while stack:
        s = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        if isinstance(s, SymDot):
            out.append(s)
            stack.append(s.data)
        elif isinstance(s, SymElem):
            stack.extend(s.args)
        elif isinstance(s, SymPad):
            stack.append(s.inner)
    return out


def _has_dot(roots):
    seen = set()
    stack = list(roots)
    while stack:
        s = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        if isinstance(s, SymDot):
            return True
        if isinstance(s, SymElem):
            stack.extend(s.args)
        elif isinstance(s, SymPad):
            stack.append(s.inner)
    return False


def _clone_leafish(x):
    if isinstance(x, SymLeaf):
        s = SymLeaf(x.kind, x.shape, x.dtype, source=x.source, idx=x.idx,
                    state_pos=x.state_pos, inner_shape=x.inner_shape,
                    strides=x.strides, offset=x.offset)
        return s
    if isinstance(x, SymElem):
        s = SymElem(x.op, x.args, x.dtype, x.shape, extra=x.extra)
        return s
    raise _Unsupported(f"reshape of {type(x).__name__}")


def _bshape(a, b):
    ra, rb = list(a), list(b)
    while len(ra) < len(rb):
        ra.insert(0, 1)
    while len(rb) < len(ra):
        rb.insert(0, 1)
    out = []
    for x, y in zip(ra, rb):
        if x != y and 1 not in (x, y):
            raise _Unsupported(f"broadcast {a} vs {b}")
        out.append(max(x, y))
    return tuple(out)


# ------------------------------------------------------------ planning

def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _lane_offset(shape, lane_shape):
    """MSL expression for a tensor's flat offset given lane coords c0..cn.
    Broadcasting handled with zero strides."""
    rs = list(shape)
    while len(rs) < len(lane_shape):
        rs.insert(0, 1)
    if len(rs) > len(lane_shape):
        extra = rs[: len(rs) - len(lane_shape)]
        if any(d != 1 for d in extra):
            raise _Unsupported(f"shape {shape} vs lane {lane_shape}")
        rs = rs[len(rs) - len(lane_shape):]
    strides = []
    acc = 1
    for d in reversed(rs):
        strides.append(acc if d != 1 else 0)
        acc *= d
    strides.reverse()
    for d, ld in zip(rs, lane_shape):
        if d != 1 and d != ld:
            raise _Unsupported(f"shape {shape} vs lane {lane_shape}")
    terms = [f"c{i} * {s}u" for i, s in enumerate(strides) if s != 0]
    return " + ".join(terms) if terms else "0u"


_MX_DTYPE = {"f32": mx.float32, "f16": mx.float16, "i32": mx.int32,
             "i1": mx.bool_, "i64": mx.int64, "i16": mx.int16,
             "i8": mx.int8, "ui64": mx.uint64, "ui32": mx.uint32,
             "ui16": mx.uint16, "ui8": mx.uint8}


def _canonicalizer():
    """Structural hash-consing over the Sym DAG (CSE).

    jax's AD-generated loops recompute gate subexpressions once per residual
    (CSE is normally XLA's job), so identical subtrees arrive as distinct
    ops. Interning by structure collapses them so each dot/activation is
    emitted once; emitter memoization by id() then works structurally.
    """
    table: dict = {}
    memo: dict = {}

    def canon(s):
        if s is None:
            return None
        got = memo.get(id(s))
        if got is not None:
            return got
        try:
            if isinstance(s, SymElem):
                s.args = tuple(canon(a) for a in s.args)
                key = ("e", s.op, s.extra, s.dtype, tuple(s.shape),
                       tuple(id(a) for a in s.args))
            elif isinstance(s, SymPad):
                s.inner = canon(s.inner)
                key = ("pad", id(s.inner), s.lo, s.n, tuple(s.shape))
            elif isinstance(s, SymPerm):
                s.inner = canon(s.inner)
                key = ("perm", id(s.inner), s.perm)
            elif isinstance(s, SymDot):
                s.data = canon(s.data)
                s.weight = canon(s.weight)
                key = ("dot", id(s.data), id(s.weight), tuple(s.roles),
                       tuple(s.widx), tuple(s.shape))
            elif isinstance(s, SymAccDot):
                s.lhs = canon(s.lhs)
                s.rhs = canon(s.rhs)
                key = ("accdot", id(s.lhs), id(s.rhs), s.dims, s.perm,
                       tuple(s.shape))
            elif isinstance(s, SymAccRed):
                s.inner = canon(s.inner)
                key = ("accred", id(s.inner), s.dims, s.perm, tuple(s.shape))
            elif isinstance(s, SymRedReg):
                s.inner = canon(s.inner)
                key = ("redreg", id(s.inner), tuple(s.shape))
            elif isinstance(s, SymIota):
                key = ("iota", s.axis, s.start, tuple(s.shape), s.dtype)
            elif isinstance(s, SymStack):
                s.base = canon(s.base)
                s.parts = {k2: canon(v2) for k2, v2 in s.parts.items()}
                key = ("stack", id(s.base), s.axis,
                       tuple(sorted((k2, id(v2))
                                    for k2, v2 in s.parts.items())),
                       tuple(s.shape))
            elif isinstance(s, SymConst):
                key = ("c", s.dtype, tuple(s.shape), s.value)
            elif isinstance(s, SymCounter):
                key = ("t", s.a, s.b, s.base)
            elif isinstance(s, SymLeaf):
                # s.source may hold an ir.Value; those hash/compare by the
                # underlying MLIR value, not the Python wrapper.
                src = s.source
                idx = s.idx
                if idx is not None:
                    idx = canon(idx)
                    s.idx = idx
                key = ("l", s.kind, src, id(idx) if idx is not None else None,
                       tuple(s.shape), tuple(s.strides), s.offset,
                       s.state_pos)
            else:
                key = ("?", id(s))
            out = table.setdefault(key, s)
        except TypeError:  # unhashable field: keep the node as-is
            out = s
        memo[id(s)] = out
        return out

    return canon


class Plan:
    """A compiled persistent-kernel execution plan for one loop body."""

    def __init__(self, interp, body_block, counter_pos, trip, start):
        self.trip = trip
        args = list(body_block.arguments)
        an = _Analyzer(interp, body_block, counter_pos)
        rets = an.analyze()
        if rets is None or len(rets) != len(args):
            raise _Unsupported("terminator mismatch")

        # Re-key stacked-update info by carry position, then intern the DAG.
        upd = {}
        for i, s in enumerate(rets):
            if isinstance(s, SymLeaf) and s.kind == "updated":
                info = an.updates.get(id(s))
                if info is not None:
                    upd[i] = info
        canon = _canonicalizer()
        rets = [canon(s) for s in rets]
        upd = {i: (canon(ix), canon(v)) for i, (ix, v) in upd.items()}

        self.passthrough = []      # positions
        self.counters = []         # (pos, per-iter delta)
        self.states = []           # (pos, expr Sym)
        self.stacked = []          # (pos, idx SymCounter, value Sym)
        self.accums = []           # (pos, SymAccDot)
        self.hidden = []           # (sym, name) per-step stacks for accums
        state_pos_to_id = {}

        for i, s in enumerate(rets):
            if isinstance(s, SymLeaf) and s.kind == "arg" and s.source == ("carry", i):
                self.passthrough.append(i)
            elif isinstance(s, SymCounter):
                if s.base != i or s.a != 1:
                    raise _Unsupported("non-increment counter")
                self.counters.append((i, s.b))
            elif isinstance(s, SymLeaf) and s.kind == "updated":
                if s.source != ("carry", i):
                    raise _Unsupported("update aliasing")
                idx, val = upd[i]
                if _contains_accdot(val):
                    raise _Unsupported(
                        f"acc in stacked write: {_dump(val)[:300]}")
                self.stacked.append((i, idx, val))
            else:
                if i in an.counter_seeded:
                    raise _Unsupported(
                        f"i32 scalar carry {i} with non-affine update: "
                        f"{type(s).__name__} "
                        f"{getattr(s, 'op', getattr(s, 'kind', ''))}")
                accs = _match_accum(s, i, None)
                if accs is not None:
                    self.accums.append((i, accs))
                    continue
                if _contains_accdot(s):
                    raise _Unsupported(
                        f"acc-dot outside accumulator update: {_dump(s)}")
                state_pos_to_id[i] = len(self.states)
                self.states.append((i, s))

        arg_types = [(_ttype(a)) for a in args]
        self.arg_shapes = [tuple(t.shape) for t in arg_types]
        self.arg_dtypes = [_dt(t.element_type) for t in arg_types]

        # Squeeze interior unit dims out of classified values: row-major
        # layout is unchanged by them, and the lane-space unifier expects
        # values shaped (lane..., reg) without stray 1s (broadcast-multiply
        # lowerings leave (B,K,1,C)-style shapes behind).
        _squeeze_cache = {}

        def _squeeze_units(sym):
            shape = getattr(sym, "shape", ()) or ()
            if len(shape) <= 1 or 1 not in shape[:-1]:
                return sym
            hit = _squeeze_cache.get(id(sym))
            if hit is not None:
                return hit
            orig = sym
            try:
                for ax in range(len(shape) - 2, -1, -1):
                    if sym.shape[ax] == 1:
                        sym = an._dropdim(sym, ax)
            except _Unsupported:
                pass
            _squeeze_cache[id(orig)] = sym
            return sym

        self.states = [(i, _squeeze_units(v)) for i, v in self.states]
        self.stacked = [(i, idx, _squeeze_units(v))
                        for i, idx, v in self.stacked]

        # Resolve accumulator operands: reads of existing stacks are used
        # directly (with flips for reversed indexing); loop-computed values
        # get hidden per-step stacked outputs.
        self.acc_plans = []       # (pos, spec) resolved in run()

        def resolve(sym):
            if (isinstance(sym, SymLeaf) and sym.kind == "read"
                    and sym.strides == _rowmajor(sym.shape)
                    and sym.idx.a in (1, -1)
                    and sym.offset == 0
                    and (sym.inner_shape is None
                         or _numel(sym.shape)
                         == _numel((1,) + tuple(sym.inner_shape)))):
                # only a FULL per-step block can be consumed directly as a
                # buffer slice; offset sub-windows need a hidden stack
                return ("buffer", sym)
            if isinstance(sym, SymAccRed):
                return ("red", resolve(sym.inner), sym.dims, sym.perm,
                        tuple(sym.shape))
            if isinstance(sym, SymAccDot):
                return ("dot", resolve(sym.lhs), resolve(sym.rhs), sym.dims,
                        sym.perm, len(sym.lhs.shape), len(sym.rhs.shape))
            if _contains_accdot(sym):
                raise _Unsupported(
                    f"acc under elementwise: {_dump(sym)[:200]}")
            if sym.dtype != "f32":
                raise _Unsupported("non-f32 accumulator operand")
            for hi, (hsym, _) in enumerate(self.hidden):
                if hsym is sym:
                    return ("hidden", sym, hi)
            sym = _squeeze_units(sym)
            for hi, (hsym, _) in enumerate(self.hidden):
                if hsym is sym:
                    return ("hidden", sym, hi)
            name = f"hid{len(self.hidden)}"
            self.hidden.append((sym, name))
            return ("hidden", sym, len(self.hidden) - 1)

        for pos, accs in self.accums:
            self.acc_plans.append((pos, [resolve(a) for a in accs]))

        # Vector mode: bodies with small in-lane matvecs hold the trailing
        # (feature/block) dim of every tensor in registers.
        exprs0 = ([s for _, s in self.states]
                  + [v for _, _, v in self.stacked]
                  + [h for h, _ in self.hidden])
        dots = _collect_dots(exprs0)
        _dcap = _REG_LIMIT if os.environ.get("MJDBG_NODSIZE") else 4 * _REG_LIMIT
        if dots and all(d.csize <= _dcap and d.dsize <= _dcap
                        for d in dots):
            # dsize (output features of the in-lane matvec) costs registers
            # but no extra serial work per element beyond the unrolled loop
            self.mode = "vector"
        elif dots or self.accums:
            self.mode = "coop"
        else:
            self.mode = "scalar"
        self.vector = self.mode == "vector"
        if self.mode == "coop":
            widths = {sh[-1] for _, sh in
                      ((p_, self.arg_shapes[p_]) for p_, _ in self.states)
                      if sh}
            if len(widths) != 1:
                raise _Unsupported(f"coop: mixed state widths {widths}")
            self.F = widths.pop()
            if not (16 < self.F <= 1024) and dots:
                if self.F > 1024:
                    raise _Unsupported(f"coop: width {self.F} > 1024")
            shared_bytes = 0
            seen_data = set()
            for d in dots:
                if d.csize % self.F or d.dsize % self.F:
                    raise _Unsupported(
                        f"coop: dot {d.csize}x{d.dsize} not a multiple "
                        f"of F={self.F}")
                if d.csize > 4096 or d.dsize > 4096:
                    raise _Unsupported("coop: dot too large")
                if id(d.data) not in seen_data:
                    seen_data.add(id(d.data))
                    shared_bytes += 4 * d.csize
                if any(r[0] == "data" for r in d.widx):
                    raise _Unsupported("coop: batched dot")
            if shared_bytes > 32768:
                raise _Unsupported(
                    f"coop: threadgroup memory {shared_bytes}B > 32KB")

        # Lane space
        lane = ()
        if self.mode == "coop":
            for i, st_ in self.states:
                lane = _bshape(lane, self.arg_shapes[i][:-1] if
                               self.arg_shapes[i] else ())
            for i, idx, val in self.stacked:
                w = self.arg_shapes[i][-1]
                if w % self.F:
                    raise _Unsupported("coop: width not multiple of F")
                lane = _bshape(lane, tuple(val.shape[1:-1]))
            for h, _ in self.hidden:
                lane = _bshape(lane, h.shape[:-1] if h.shape else ())
            lane = lane + (self.F,)
        elif self.vector:
            for i, s in self.states:
                lane = _bshape(lane, s.shape[:-1] if s.shape else ())
            for i, idx, val in self.stacked:
                lane = _bshape(lane, tuple(val.shape[1:-1]))
            for h, _ in self.hidden:
                lane = _bshape(lane, h.shape[:-1] if h.shape else ())
        else:
            for i, s in self.states:
                lane = _bshape(lane, s.shape)
            for i, idx, val in self.stacked:
                lane = _bshape(lane, tuple(val.shape[1:]))
        self.lane_shape = lane
        self.N = _numel(lane)
        if self.N == 0:
            raise _Unsupported("empty lane space")

        # Device inputs: unique sources for reads/wholes + state inits
        self.sources = []          # list of ('carry', i) | ('free', ir.Value)
        self._source_ids = {}

        def source_id(src):
            if src[0] == "carry":
                key = src
            elif src[0] == "hoist":
                key = ("hoist", src[1])
            else:
                key = ("free", id(src[1]), src[1])
            if key not in self._source_ids:
                self._source_ids[key] = len(self.sources)
                self.sources.append(src)
            return self._source_ids[key]

        # Walk all expressions to collect leaves
        self._reads = {}           # (srcid, a, b, start-folded) -> name
        self._wholes = {}          # srcid -> name
        self._state_args = state_pos_to_id
        self.start = start

        self._weights = {}         # srcid -> weight leaf (indexed in-loop)
        self._weight_dots = {}     # srcid -> [SymDot] using that weight
        exprs = exprs0
        self._collect(exprs, source_id)
        if set(self._weights) & set(self._wholes):
            raise _Unsupported("weights also used elementwise")
        # Reads must come from passthrough carries or free captures — never
        # from tensors this loop itself mutates.
        mutated = ({pos for pos, _, _ in self.stacked}
                   | {pos for pos, _ in self.states}
                   | {pos for pos, _ in self.accums})
        for src in self.sources:
            if src[0] == "carry" and src[1] in mutated:
                raise _Unsupported("read of a mutated carry")
        if an.hoisted_read_srcs & mutated:
            raise _Unsupported("hoisted read of a mutated carry")

        # Canonicalize weight layout to (data..., c, d): the output-feature
        # dim at unit stride makes threadgroup dot reads coalesced (the AD
        # backward pass contracts against W^T, which would otherwise read
        # stride-F apart across adjacent threads — ~100x slower in coop mode).
        # Non-canonical weights are materialized contiguous once per call.
        self._weight_norms = {}
        if os.environ.get("METALJAX_MSL_WNORM", "1") != "0":
            for sid, wleaf in list(self._weights.items()):
                dots = self._weight_dots.get(sid, [])
                if not dots:
                    continue
                widx0 = tuple(dots[0].widx)
                if any(tuple(d.widx) != widx0 for d in dots[1:]):
                    continue
                # All dots must read the SAME window of this source: the
                # normalization materializes one transformed buffer per
                # source, which cannot serve two different slices (gate
                # splits of a fused weight index distinct windows).
                w0 = dots[0].weight
                if any(d.weight.shape != w0.shape
                       or tuple(d.weight.strides) != tuple(w0.strides)
                       or d.weight.offset != w0.offset for d in dots[1:]):
                    continue
                nd = len(wleaf.shape)
                perm = tuple(
                    [i for i in range(nd) if widx0[i][0] == "data"]
                    + [i for i in range(nd) if widx0[i][0] == "c"]
                    + [i for i in range(nd) if widx0[i][0] == "d"])
                if (perm == tuple(range(nd))
                        and tuple(wleaf.strides) == _rowmajor(wleaf.shape)
                        and not wleaf.offset):
                    continue
                self._weight_norms[sid] = (tuple(wleaf.shape),
                                           tuple(wleaf.strides),
                                           wleaf.offset, perm)
                new_shape = tuple(wleaf.shape[p] for p in perm)
                wleaf.shape = new_shape
                wleaf.strides = _rowmajor(new_shape)
                wleaf.offset = 0
                new_widx = tuple(widx0[p] for p in perm)
                for d in dots:
                    d.widx = new_widx
        if self.mode == "coop":
            src = self._emit_coop()
        elif self.mode == "vector":
            src = self._emit_vector()
        else:
            src = self._emit(counter_pos)
        self.source = src
        # NB kernel names must be process-unique: MLX caches compiled
        # kernels and id()-based names collide once Python reuses addresses
        # across executables (order-1 wrong results in config sweeps).
        global _KERNEL_SEQ
        _KERNEL_SEQ += 1
        name = f"mj_scan_{_KERNEL_SEQ}"
        self.kernel = mx.fast.metal_kernel(
            name=name,
            input_names=[f"inp{i}" for i in range(len(self.sources))]
            + [f"init{j}" for j in range(len(self.states))],
            output_names=[f"out{q}" for q in range(len(self.stacked))]
            + [nm for _, nm in self.hidden]
            + [f"fin{j}" for j in range(len(self.states))],
            source=src,
        )

    # ---- collection

    def _collect(self, roots, source_id):
        seen = set()

        def walk(s):
            if id(s) in seen:
                return
            seen.add(id(s))
            if isinstance(s, SymElem):
                for a in s.args:
                    walk(a)
            elif isinstance(s, SymPad):
                walk(s.inner)
            elif isinstance(s, SymRedReg):
                walk(s.inner)
            elif isinstance(s, SymDot):
                walk(s.data)
                sid = source_id(s.weight.source)
                self._weights[sid] = s.weight
                self._weight_dots.setdefault(sid, []).append(s)
            elif isinstance(s, SymLeaf):
                if s.kind == "read":
                    sid = source_id(s.source)
                    key = (sid, s.idx.a, s.idx.b, s.shape, s.strides, s.offset)
                    self._reads.setdefault(key, (s, f"r{len(self._reads)}"))
                elif s.kind == "arg":
                    pos = s.source[1]
                    if pos in self._state_args:
                        pass  # register
                    else:
                        sid = source_id(s.source)
                        self._wholes.setdefault(sid, (s, f"w{sid}"))
                elif s.kind == "whole":
                    sid = source_id(s.source)
                    self._wholes.setdefault(sid, (s, f"w{sid}"))
                elif s.kind == "updated":
                    raise _Unsupported("nested update use")

        for r in roots:
            walk(r)

    # ---- emission

    def _emit(self, counter_pos):
        L = []
        lane = self.lane_shape
        L.append("uint lane = thread_position_in_grid.x;")
        L.append(f"if (lane >= {self.N}u) return;")
        # lane coords
        rem = "lane"
        tail = _numel(lane)
        for i, d in enumerate(lane):
            tail //= d
            L.append(f"uint c{i} = ({rem}) / {tail}u;")
            rem = f"lane % {tail}u" if False else f"(lane % {tail}u)"
        # (recompute coords properly: c_i = (lane / prod(after)) % d_i)
        L = L[:2]
        tail = _numel(lane)
        for i, d in enumerate(lane):
            tail //= d
            L.append(f"uint c{i} = (lane / {tail}u) % {d}u;")

        mslt = lambda dt: _MSL_DTYPE[dt]

        # whole (invariant) tensors: load once. 0-dim inputs arrive by value.
        for sid, (leaf, name) in sorted(self._wholes.items()):
            if len(self._buffer_shape(leaf)) == 0:
                L.append(f"{mslt(leaf.dtype)} {name} = inp{sid};")
            else:
                off = _off_strided(leaf.shape, leaf.strides, lane, leaf.offset)
                L.append(f"{mslt(leaf.dtype)} {name} = inp{sid}[{off}];")
        # state registers
        for j, (pos, expr) in enumerate(self.states):
            if len(self.arg_shapes[pos]) == 0:
                L.append(f"{mslt(self.arg_dtypes[pos])} st{j} = init{j};")
            else:
                off = _lane_offset(self.arg_shapes[pos], lane)
                L.append(f"{mslt(self.arg_dtypes[pos])} st{j} = init{j}[{off}];")

        L.append(f"for (uint t_ = 0; t_ < {self.trip}u; t_++) {{")
        # WORKAROUND: the Metal shader compiler miscompiles multi-
        # iteration loops here, illegally caching t-derived reads
        # across iterations (verified vs a numpy emulation of the
        # emitted source; volatile forces re-evaluation, bit-exact).
        L.append("  volatile uint t = t_;")

        # per-iteration reads
        for (sid, a, b, *_), (leaf, name) in self._reads.items():
            inner = _numel(leaf.inner_shape)
            off = _off_strided(leaf.shape, leaf.strides, lane, leaf.offset)
            idx = f"((int)t + {self.start}) * {a} + {b}" if a != 1 or b != 0 or self.start \
                else "(int)t"
            L.append(f"  {mslt(leaf.dtype)} {name} = "
                     f"inp{sid}[(uint)({idx}) * {inner}u + ({off})];")

        # expression emission with memo
        memo = {}
        tmp = [0]
        body = []

        def emit(s):
            if id(s) in memo:
                return memo[id(s)]
            if isinstance(s, SymConst):
                v = _literal(s)
            elif isinstance(s, SymCounter):
                base_expr = f"((int)t + {self.start})"
                v = f"({s.a} * {base_expr} + {s.b})"
            elif isinstance(s, SymLeaf):
                if s.kind == "read":
                    sid = self._source_key(s.source)
                    v = self._reads[
                        (sid, s.idx.a, s.idx.b, s.shape, s.strides, s.offset)][1]
                elif s.kind == "arg":
                    pos = s.source[1]
                    if pos in self._state_args:
                        v = f"st{self._state_args[pos]}"
                    else:
                        v = self._wholes[self._source_key(s.source)][1]
                elif s.kind == "whole":
                    v = self._wholes[self._source_key(s.source)][1]
                else:
                    raise _Unsupported("leaf kind")
            elif isinstance(s, SymElem):
                args = [emit(a) for a in s.args]
                if s.op == "convert":
                    v = f"(({_MSL_DTYPE[s.extra]})({args[0]}))"
                elif s.op == "compare":
                    v = f"({args[0]} {_COMPARE[s.extra]} {args[1]})"
                elif s.op == "select":
                    v = f"({args[0]} ? {args[1]} : {args[2]})"
                elif s.op == "clamp":
                    v = f"metal::min(metal::max({args[1]}, {args[0]}), {args[2]})"
                elif s.op in _UNARY:
                    v = _UNARY[s.op].format(*args)
                elif s.op in _BINARY:
                    v = _BINARY[s.op].format(*args)
                else:
                    raise _Unsupported(f"emit {s.op}")
                name = f"v{tmp[0]}"
                tmp[0] += 1
                body.append(f"  {_MSL_DTYPE[s.dtype]} {name} = {v};")
                v = name
            else:
                raise _Unsupported("emit type")
            memo[id(s)] = v
            return v

        # stacked writes + state updates (compute all, then assign states)
        writes = []
        for q, (pos, idx, val) in enumerate(self.stacked):
            v = emit(val)
            inner = _numel(self.arg_shapes[pos][1:])
            off = _lane_offset(tuple(self.arg_shapes[pos][1:]), lane)
            ii = f"((int)t + {self.start}) * {idx.a} + {idx.b}"
            writes.append(f"  out{q}[(uint)({ii}) * {inner}u + ({off})] = {v};")
        news = []
        for j, (pos, expr) in enumerate(self.states):
            v = emit(expr)
            news.append((j, v))
        L.extend(body)
        L.extend(writes)
        for j, v in news:
            L.append(f"  st{j} = {v};")
        L.append("}")
        for j, (pos, expr) in enumerate(self.states):
            off = _lane_offset(self.arg_shapes[pos], lane)
            L.append(f"fin{j}[{off}] = st{j};")
        return "\n".join(L)

    # ---- vector-mode emission (register-tail lanes)

    def _R(self, s):
        if isinstance(s, (SymConst, SymCounter)):
            return 1
        if isinstance(s, SymDot):
            return s.dsize
        if isinstance(s, SymPad):
            return s.shape[-1]
        return s.shape[-1] if s.shape else 1

    def _vec_off(self, shape, strides=None, base=0):
        """Lane-part offset for a tensor whose last dim is the register tail
        (indexed by `r` when > 1), from explicit element strides."""
        lane = self.lane_shape
        dims = list(shape)
        sts = list(strides) if strides is not None else list(_rowmajor(shape))
        reg = dims[-1] if dims else 1
        reg_stride = sts[-1] if dims else 0
        lane_dims = dims[:-1] if dims else []
        lane_sts = sts[:-1] if dims else []
        # unit dims contribute no offset and may sit anywhere (broadcast
        # lowerings leave interior 1s); drop them before lane alignment
        keep = [i for i, d in enumerate(lane_dims) if d != 1]
        lane_dims = [lane_dims[i] for i in keep]
        lane_sts = [lane_sts[i] for i in keep]
        pad = len(lane) - len(lane_dims)
        if pad < 0:
            n = len(lane_dims) - len(lane)
            if any(d != 1 for d in lane_dims[:n]):
                raise _Unsupported(f"vec shape {shape} vs lane {lane}")
            lane_dims, lane_sts = lane_dims[n:], lane_sts[n:]
            pad = 0
        terms = [f"{base}u"] if base else []
        for i, (d, st) in enumerate(zip(lane_dims, lane_sts)):
            if d == 1 or st == 0:
                continue
            if d != lane[i + pad]:
                raise _Unsupported(f"vec shape {shape} vs lane {lane}")
            terms.append(f"c{i + pad} * {st}u")
        expr = " + ".join(terms) if terms else "0u"
        if reg > 1 and reg_stride != 0:
            return f"{expr} + r * {reg_stride}u"
        return expr

    def _emit_vector(self):
        lane = self.lane_shape
        out = []
        out.append("uint lane = thread_position_in_grid.x;")
        out.append(f"if (lane >= {self.N}u) return;")
        tail = _numel(lane)
        for i, d in enumerate(lane):
            tail //= d
            out.append(f"uint c{i} = (lane / {tail}u) % {d}u;")

        T = lambda dt: _MSL_DTYPE[dt]

        def declare(name, dtype, R):
            return (f"{T(dtype)} {name}[{R}];" if R > 1 else f"{T(dtype)} {name};")

        def load(dst, dtype, R, buf, off, indent=""):
            ls = []
            if R > 1:
                ls.append(f"{indent}for (int r = 0; r < {R}; r++) "
                          f"{dst}[r] = {buf}[{off}];")
            else:
                ls.append(f"{indent}{dst} = {buf}[{off}];")
            return ls

        # invariant tensors (not weights): preload
        for sid, (leaf, name) in sorted(self._wholes.items()):
            R = self._R(leaf)
            bshape = self._buffer_shape(leaf)
            out.append(declare(name, leaf.dtype, R))
            if len(bshape) == 0:
                out.append(f"{name} = inp{sid};" if R == 1 else "")
            else:
                out.extend(load(name, leaf.dtype, R, f"inp{sid}",
                                self._vec_off(leaf.shape, leaf.strides,
                                              leaf.offset)))
        # states
        for j, (pos, expr) in enumerate(self.states):
            shape = self.arg_shapes[pos]
            R = shape[-1] if shape else 1
            out.append(declare(f"st{j}", self.arg_dtypes[pos], R))
            if len(shape) == 0:
                out.append(f"st{j} = init{j};")
            else:
                out.extend(load(f"st{j}", self.arg_dtypes[pos], R,
                                f"init{j}", self._vec_off(shape)))

        out.append(f"for (uint t_ = 0; t_ < {self.trip}u; t_++) {{")
        out.append("  volatile uint t = t_;")

        # reads
        for (sid, a, b, *_), (leaf, name) in self._reads.items():
            R = self._R(leaf)
            inner = _numel(leaf.inner_shape)
            idx = (f"((int)t + {self.start}) * {a} + {b}"
                   if a != 1 or b != 0 or self.start else "(int)t")
            off = self._vec_off(leaf.shape, leaf.strides, leaf.offset)
            out.append("  " + declare(name, leaf.dtype, R))
            out.extend(load(name, leaf.dtype, R, f"inp{sid}",
                            f"(uint)({idx}) * {inner}u + ({off})", "  "))

        memo = {}
        tmp = [0]
        body = []

        def scalarize(s, val):
            # access expression for one register component
            name, R = val
            return f"{name}[r]" if R > 1 else name

        def emit(s):
            if id(s) in memo:
                return memo[id(s)]
            if isinstance(s, SymConst):
                v = (_literal(s), 1)
            elif isinstance(s, SymCounter):
                v = (f"({s.a} * ((int)t + {self.start}) + {s.b})", 1)
            elif isinstance(s, SymLeaf):
                if s.kind == "read":
                    sid = self._source_key(s.source)
                    leaf, nm = self._reads[
                        (sid, s.idx.a, s.idx.b, s.shape, s.strides, s.offset)]
                    v = (nm, self._R(leaf))
                elif s.kind == "arg":
                    pos = s.source[1]
                    if pos in self._state_args:
                        shape = self.arg_shapes[pos]
                        v = (f"st{self._state_args[pos]}",
                             shape[-1] if shape else 1)
                    else:
                        leaf, nm = self._wholes[self._source_key(s.source)]
                        v = (nm, self._R(leaf))
                elif s.kind == "whole":
                    leaf, nm = self._wholes[self._source_key(s.source)]
                    v = (nm, self._R(leaf))
                else:
                    raise _Unsupported("leaf kind in vector mode")
            elif isinstance(s, SymPad):
                iv = emit(s.inner)
                inm, iR = iv
                R = s.shape[-1]
                name = f"v{tmp[0]}"
                tmp[0] += 1
                body.append(f"  {T(s.dtype)} {name}[{R}];")
                src = f"{inm}[r - {s.lo}]" if iR > 1 else inm
                body.append(
                    f"  for (int r = 0; r < {R}; r++) {name}[r] = "
                    f"(r >= {s.lo} && r < {s.lo + s.n}) ? "
                    f"({src}) : ({T(s.dtype)})0;")
                v = (name, R)
            elif isinstance(s, SymRedReg):
                inm, iR = emit(s.inner)
                true_w = s.inner.shape[-1] if s.inner.shape else 1
                if iR != true_w:
                    # the reduced dim is not (fully) register-resident:
                    # summing registers would silently skip lane elements
                    raise _Unsupported("register reduce over non-register dim")
                name = f"v{tmp[0]}"
                tmp[0] += 1
                body.append(f"  {T(s.dtype)} {name};")
                if iR == 1:
                    body.append(f"  {name} = {inm};")
                else:
                    body.append(
                        f"  {{ {T(s.dtype)} _s = ({T(s.dtype)})0; "
                        f"for (int r = 0; r < {iR}; r++) _s += {inm}[r]; "
                        f"{name} = _s; }}")
                v = (name, 1)
            elif isinstance(s, SymIota):
                name = f"v{tmp[0]}"
                tmp[0] += 1
                if s.axis == len(s.shape) - 1 and s.shape[-1] > 1:
                    R = s.shape[-1]
                    body.append(f"  {T(s.dtype)} {name}[{R}];")
                    body.append(f"  for (int r = 0; r < {R}; r++) "
                                f"{name}[r] = ({T(s.dtype)})(r + {s.start});")
                    v = (name, R)
                else:
                    # lane-coordinate ramp (or size-1 axis: constant)
                    if s.shape[s.axis] == 1:
                        expr = f"({T(s.dtype)}){s.start}"
                    else:
                        pad = len(self.lane_shape) - (len(s.shape) - 1)
                        expr = f"({T(s.dtype)})(c{s.axis + pad} + {s.start})"
                    body.append(f"  {T(s.dtype)} {name} = {expr};")
                    v = (name, 1)
            elif isinstance(s, SymDot):
                v = emit_dot(s)
            elif isinstance(s, SymElem):
                args = [emit(a) for a in s.args]
                R = max(r for _, r in args) if args else 1
                R = max(R, self._R(s))
                for _, r in args:
                    if r not in (1, R):
                        raise _Unsupported("register width mismatch")
                name = f"v{tmp[0]}"
                tmp[0] += 1
                body.append("  " + declare(name, s.dtype, R))
                acc = [scalarize(s, a) for a in args]
                if s.op == "convert":
                    e = f"(({_MSL_DTYPE[s.extra]})({acc[0]}))"
                elif s.op == "compare":
                    e = f"({acc[0]} {_COMPARE[s.extra]} {acc[1]})"
                elif s.op == "select":
                    e = f"({acc[0]} ? {acc[1]} : {acc[2]})"
                elif s.op == "clamp":
                    e = f"metal::min(metal::max({acc[1]}, {acc[0]}), {acc[2]})"
                elif s.op in _UNARY:
                    e = _UNARY[s.op].format(*acc)
                elif s.op in _BINARY:
                    e = _BINARY[s.op].format(*acc)
                else:
                    raise _Unsupported(f"vec emit {s.op}")
                if R > 1:
                    body.append(f"  for (int r = 0; r < {R}; r++) "
                                f"{name}[r] = {e};")
                else:
                    body.append(f"  {name} = {e};")
                v = (name, R)
            else:
                raise _Unsupported(f"vec emit type {type(s).__name__} {_dump(s)[:80]}")
            memo[id(s)] = v
            return v

        def emit_dot(s):
            # validate canonical orientation: lane dims ascending, reg last
            roles = [r for r in s.roles if r != ("one",)]
            if not roles or roles[-1] != ("reg",):
                raise _Unsupported("dot output reg dim not last")
            dorder = [r[1] for r in roles[:-1]]
            if dorder != sorted(dorder):
                raise _Unsupported("dot output lane dims permuted")
            dname, dR = emit(s.data)
            if dR != s.csize:
                raise _Unsupported("dot data register width mismatch")
            wsid = self._source_key(s.weight.source)
            wstrides = list(s.weight.strides)
            pad = len(self.lane_shape) - (len(s.data.shape) - 1)
            terms = []
            for wdim, role in enumerate(s.widx):
                st = wstrides[wdim]
                if role[0] == "data":
                    if s.data.shape[role[1]] == 1:
                        continue
                    terms.append(f"c{role[1] + pad} * {st}u")
                elif role[0] == "c":
                    terms.append(f"(uint)cc * {st}u")
                else:
                    terms.append(f"(uint)d * {st}u")
            if s.weight.offset:
                terms.insert(0, f"{s.weight.offset}u")
            name = f"v{tmp[0]}"
            tmp[0] += 1
            dacc = f"{dname}[cc]" if dR > 1 else dname
            if s.dsize > 1:
                woff = " + ".join(terms) if terms else "0u"
                body.append(f"  float {name}[{s.dsize}];")
                body.append(
                    f"  for (int d = 0; d < {s.dsize}; d++) {{ float _a = 0.0f; "
                    f"for (int cc = 0; cc < {s.csize}; cc++) "
                    f"_a += {dacc} * inp{wsid}[{woff}]; {name}[d] = _a; }}")
            else:
                t0 = [t for t in terms if "(uint)d" not in t]
                woff = " + ".join(t0) if t0 else "0u"
                body.append(f"  float {name};")
                body.append(
                    f"  {{ float _a = 0.0f; "
                    f"for (int cc = 0; cc < {s.csize}; cc++) "
                    f"_a += {dacc} * inp{wsid}[{woff}]; {name} = _a; }}")
            return (name, s.dsize)

        def emit_write(val, per_shape, dest_expr):
            """Emit writes of `val` into a buffer whose per-step layout is
            row-major `per_shape`; absorbs top-level transposes into the
            write strides and materializes SymStacks part by part."""
            if isinstance(val, SymStack):
                extra = len(val.shape) - len(per_shape)
                if (extra > 0 and val.axis >= extra
                        and all(d == 1 for d in val.shape[:extra])):
                    val = SymStack(val.base, val.parts, val.shape[extra:],
                                   val.dtype, axis=val.axis - extra)
                if len(val.shape) != len(per_shape):
                    raise _Unsupported(
                        f"stack rank vs write target mismatch: "
                        f"stack{val.shape} axis={val.axis} vs per{per_shape}")
                st_t = _rowmajor(per_shape)
                ax_stride = st_t[val.axis]
                if set(val.parts) != set(range(val.shape[val.axis])):
                    raise _Unsupported("partial stack write")
                st_sub = [d for i2, d in enumerate(st_t) if i2 != val.axis]
                for idx0, part in sorted(val.parts.items()):
                    nm2, R2 = emit(part)
                    st_al = list(st_sub)
                    while len(st_al) < len(part.shape):
                        st_al.insert(0, 0)
                    st_al = st_al[len(st_al) - len(part.shape):]
                    off2 = self._vec_off(tuple(part.shape), tuple(st_al))
                    base_off = idx0 * ax_stride
                    if R2 > 1:
                        writes.append(
                            f"  for (int r = 0; r < {R2}; r++) "
                            f"{dest_expr}({off2}) + {base_off}u] = {nm2}[r];")
                    else:
                        writes.append(
                            f"  {dest_expr}({off2}) + {base_off}u] = {nm2};")
                return
            wr_strides2 = None
            if isinstance(val, SymPerm):
                st_t = _rowmajor(per_shape)
                wr_strides2 = tuple(st_t[val.perm.index(i)]
                                    for i in range(len(val.perm)))
                val = val.inner
            nm2, R2 = emit(val)
            if wr_strides2 is not None:
                off2 = self._vec_off(tuple(val.shape), wr_strides2)
                tgt_R2 = val.shape[-1] if val.shape else 1
            elif (R2 == 1 and per_shape
                    and _numel(per_shape) == _numel(self.lane_shape)):
                # width-1 value: every target dim is a lane dim; a fake
                # unit register dim maps them all to lane coordinates
                off2 = self._vec_off(tuple(per_shape) + (1,))
                tgt_R2 = 1
            else:
                off2 = self._vec_off(tuple(val.shape)
                                     if val.shape else per_shape)
                tgt_R2 = val.shape[-1] if val.shape else 1
            src2 = f"{nm2}[r]" if R2 > 1 else nm2
            if tgt_R2 > 1:
                writes.append(f"  for (int r = 0; r < {tgt_R2}; r++) "
                              f"{dest_expr}({off2})] = {src2};")
            else:
                writes.append(f"  {dest_expr}({off2})] = {src2};")

        writes = []
        for q, (pos, idx, val) in enumerate(self.stacked):
            inner = _numel(self.arg_shapes[pos][1:])
            per = tuple(self.arg_shapes[pos][1:])
            ii = f"((int)t + {self.start}) * {idx.a} + {idx.b}"
            emit_write(val, per, f"out{q}[(uint)({ii}) * {inner}u + ")
        for q, (sym, hname) in enumerate(self.hidden):
            numel = _numel(sym.shape)
            emit_write(sym, tuple(sym.shape),
                       f"{hname}[t * {numel}u + ")
        news = []
        for j, (pos, expr) in enumerate(self.states):
            nm, R = emit(expr)
            shape = self.arg_shapes[pos]
            sR = shape[-1] if shape else 1
            news.append((j, nm, R, sR))
        out.extend(body)
        out.extend(writes)
        for j, nm, R, sR in news:
            src = f"{nm}[r]" if R > 1 else nm
            if sR > 1:
                out.append(f"  for (int r = 0; r < {sR}; r++) st{j}[r] = {src};")
            else:
                out.append(f"  st{j} = {src};")
        out.append("}")
        for j, (pos, expr) in enumerate(self.states):
            shape = self.arg_shapes[pos]
            sR = shape[-1] if shape else 1
            if len(shape) == 0:
                out.append(f"fin{j}[0u] = st{j};")
            elif sR > 1:
                out.append(f"for (int r = 0; r < {sR}; r++) "
                           f"fin{j}[{self._vec_off(shape)}] = st{j}[r];")
            else:
                out.append(f"fin{j}[{self._vec_off(shape)}] = st{j};")
        return "\n".join(out)

    # ---- cooperative emission (threadgroup per batch element)

    def _coop_R(self, s):
        if isinstance(s, (SymConst, SymCounter)):
            return 1
        if isinstance(s, SymDot):
            return s.dsize // self.F  # output chunks per thread
        if isinstance(s, SymPad):
            w = s.shape[-1]
        else:
            w = s.shape[-1] if s.shape else 1
        if w == 1:
            return 1
        if w % self.F:
            raise _Unsupported(f"coop width {w} not multiple of F={self.F}")
        return w // self.F

    def _coop_off(self, shape, strides=None, base=0):
        """Offset for a tensor whose feature (last) dim is chunked: thread f
        owns feature g*F + f for component g (loop var `r`)."""
        lane = self.lane_shape          # (..., F); feature coord = last
        fcoord = f"c{len(lane) - 1}"
        dims = list(shape)
        sts = list(strides) if strides is not None else list(_rowmajor(shape))
        w = dims[-1] if dims else 1
        fst = sts[-1] if dims else 0
        lane_dims = dims[:-1] if dims else []
        lane_sts = sts[:-1] if dims else []
        pad = (len(lane) - 1) - len(lane_dims)
        if pad < 0:
            n = -pad
            if any(d != 1 for d in lane_dims[:n]):
                raise _Unsupported(f"coop shape {shape} vs lane {lane}")
            lane_dims, lane_sts = lane_dims[n:], lane_sts[n:]
            pad = 0
        terms = [f"{base}u"] if base else []
        for i, (d, st) in enumerate(zip(lane_dims, lane_sts)):
            if d == 1 or st == 0:
                continue
            if d != lane[i + pad]:
                raise _Unsupported(f"coop shape {shape} vs lane {lane}")
            terms.append(f"c{i + pad} * {st}u")
        expr = " + ".join(terms) if terms else "0u"
        if w > 1 and fst != 0:
            if w == self.F:
                return f"{expr} + {fcoord} * {fst}u"
            return f"{expr} + (r * {self.F}u + {fcoord}) * {fst}u"
        return expr

    def _emit_coop(self):
        lane = self.lane_shape
        F = self.F
        fcoord = f"c{len(lane) - 1}"
        out = []
        out.append("uint lane = thread_position_in_grid.x;")
        out.append(f"if (lane >= {self.N}u) return;")
        tail = _numel(lane)
        for i, d in enumerate(lane):
            tail //= d
            out.append(f"uint c{i} = (lane / {tail}u) % {d}u;")

        T = lambda dt: _MSL_DTYPE[dt]

        def declare(name, dtype, R):
            return (f"{T(dtype)} {name}[{R}];" if R > 1 else f"{T(dtype)} {name};")

        def load(dst, dtype, R, buf, off, indent=""):
            if R > 1:
                return [f"{indent}for (int r = 0; r < {R}; r++) "
                        f"{dst}[r] = {buf}[{off}];"]
            return [f"{indent}{dst} = {buf}[{off}];"]

        # threadgroup mirrors for dot data (one per distinct data sym)
        self._shared = {}
        shared_sizes = {}
        for d in _collect_dots(
            [s for _, s in self.states] + [v for _, _, v in self.stacked]
            + [h for h, _ in self.hidden]
        ):
            if id(d.data) not in self._shared:
                nm = f"sh{len(self._shared)}"
                self._shared[id(d.data)] = nm
                shared_sizes[nm] = d.csize
        for nm, sz in shared_sizes.items():
            out.append(f"threadgroup float {nm}[{sz}];")

        for sid, (leaf, name) in sorted(self._wholes.items()):
            R = self._coop_R(leaf)
            bshape = self._buffer_shape(leaf)
            out.append(declare(name, leaf.dtype, R))
            if len(bshape) == 0:
                out.append(f"{name} = inp{sid};")
            else:
                out.extend(load(name, leaf.dtype, R, f"inp{sid}",
                                self._coop_off(leaf.shape, leaf.strides,
                                               leaf.offset)))
        for j, (pos, expr) in enumerate(self.states):
            shape = self.arg_shapes[pos]
            R = self._coop_R_shape(shape)
            out.append(declare(f"st{j}", self.arg_dtypes[pos], R))
            if len(shape) == 0:
                out.append(f"st{j} = init{j};")
            else:
                out.extend(load(f"st{j}", self.arg_dtypes[pos], R,
                                f"init{j}", self._coop_off(shape)))

        out.append(f"for (uint t_ = 0; t_ < {self.trip}u; t_++) {{")
        out.append("  volatile uint t = t_;")

        for (sid, a, b, *_), (leaf, name) in self._reads.items():
            R = self._coop_R(leaf)
            inner = _numel(leaf.inner_shape)
            idx = (f"((int)t + {self.start}) * {a} + {b}"
                   if a != 1 or b != 0 or self.start else "(int)t")
            off = self._coop_off(leaf.shape, leaf.strides, leaf.offset)
            out.append("  " + declare(name, leaf.dtype, R))
            out.extend(load(name, leaf.dtype, R, f"inp{sid}",
                            f"(uint)({idx}) * {inner}u + ({off})", "  "))

        memo = {}
        tmp = [0]
        body = []

        def scalarize(s, val):
            name, R = val
            return f"{name}[r]" if R > 1 else name

        def emit(s):
            if id(s) in memo:
                return memo[id(s)]
            if isinstance(s, SymConst):
                v = (_literal(s), 1)
            elif isinstance(s, SymCounter):
                v = (f"({s.a} * ((int)t + {self.start}) + {s.b})", 1)
            elif isinstance(s, SymLeaf):
                if s.kind == "read":
                    sid = self._source_key(s.source)
                    leaf, nm = self._reads[
                        (sid, s.idx.a, s.idx.b, s.shape, s.strides, s.offset)]
                    v = (nm, self._coop_R(leaf))
                elif s.kind == "arg":
                    pos = s.source[1]
                    if pos in self._state_args:
                        v = (f"st{self._state_args[pos]}",
                             self._coop_R_shape(self.arg_shapes[pos]))
                    else:
                        leaf, nm = self._wholes[self._source_key(s.source)]
                        v = (nm, self._coop_R(leaf))
                elif s.kind == "whole":
                    leaf, nm = self._wholes[self._source_key(s.source)]
                    v = (nm, self._coop_R(leaf))
                else:
                    raise _Unsupported("coop leaf kind")
            elif isinstance(s, SymPad):
                iv = emit(s.inner)
                inm, iR = iv
                # pad on the feature axis: component/feature-window shift
                R = self._coop_R(s)
                if s.lo % F or s.n % F:
                    raise _Unsupported("coop pad not F-aligned")
                glo, gn = s.lo // F, s.n // F
                name = f"v{tmp[0]}"; tmp[0] += 1
                body.append("  " + declare(name, s.dtype, R))
                src = f"{inm}[r - {glo}]" if iR > 1 else inm
                if R > 1:
                    body.append(
                        f"  for (int r = 0; r < {R}; r++) {name}[r] = "
                        f"(r >= {glo} && r < {glo + gn}) ? ({src}) "
                        f": ({T(s.dtype)})0;")
                else:
                    body.append(f"  {name} = {src};")
                v = (name, R)
            elif isinstance(s, SymDot):
                # Structural CSE: the same dot may appear once per residual
                # stack differing only in unit dims; the per-thread scalar
                # value is identical.
                wsid = self._source_key(s.weight.source)
                dkey = ("dotval", id(s.data), wsid,
                        tuple(tuple(r) for r in s.widx),
                        s.weight.offset, tuple(s.weight.shape))
                if dkey in memo:
                    return memo[dkey]
                shname = self._shared[id(s.data)]
                dv, dR = emit(s.data)
                if dR * F != s.csize:
                    raise _Unsupported("coop dot data width mismatch")
                key = ("shared_written", id(s.data))
                if key not in memo:
                    body.append("  threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
                    if dR > 1:
                        body.append(
                            f"  for (int r = 0; r < {dR}; r++) "
                            f"{shname}[r * {F}u + {fcoord}] = {dv}[r];")
                    else:
                        body.append(f"  {shname}[{fcoord}] = {dv};")
                    body.append("  threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
                    memo[key] = True
                wst = list(s.weight.strides)
                k_out = s.dsize // F
                terms = []
                for wdim, role in enumerate(s.widx):
                    st = wst[wdim]
                    if role[0] == "c":
                        terms.append(f"(uint)cc * {st}u")
                    elif role[0] == "d":
                        if k_out > 1:
                            terms.append(
                                f"((uint)g * {F}u + {fcoord}) * {st}u")
                        else:
                            terms.append(f"{fcoord} * {st}u")
                if s.weight.offset:
                    terms.insert(0, f"{s.weight.offset}u")
                woff = " + ".join(terms) if terms else "0u"
                name = f"v{tmp[0]}"; tmp[0] += 1
                if k_out > 1:
                    body.append(f"  float {name}[{k_out}];")
                    body.append(
                        f"  for (int g = 0; g < {k_out}; g++) "
                        f"{{ float _a = 0.0f; for (int cc = 0; cc < {s.csize}; cc++) "
                        f"_a += {shname}[cc] * inp{wsid}[{woff}]; {name}[g] = _a; }}")
                else:
                    body.append(f"  float {name};")
                    body.append(
                        f"  {{ float _a = 0.0f; for (int cc = 0; cc < {s.csize}; cc++) "
                        f"_a += {shname}[cc] * inp{wsid}[{woff}]; {name} = _a; }}")
                v = (name, k_out)
                memo[dkey] = v
            elif isinstance(s, SymElem):
                args = [emit(a) for a in s.args]
                R = max([r for _, r in args] + [self._coop_R(s)])
                for _, r in args:
                    if r not in (1, R):
                        raise _Unsupported("coop width mismatch")
                name = f"v{tmp[0]}"; tmp[0] += 1
                body.append("  " + declare(name, s.dtype, R))
                acc = [scalarize(s, a) for a in args]
                if s.op == "convert":
                    e = f"(({_MSL_DTYPE[s.extra]})({acc[0]}))"
                elif s.op == "compare":
                    e = f"({acc[0]} {_COMPARE[s.extra]} {acc[1]})"
                elif s.op == "select":
                    e = f"({acc[0]} ? {acc[1]} : {acc[2]})"
                elif s.op == "clamp":
                    e = f"metal::min(metal::max({acc[1]}, {acc[0]}), {acc[2]})"
                elif s.op in _UNARY:
                    e = _UNARY[s.op].format(*acc)
                elif s.op in _BINARY:
                    e = _BINARY[s.op].format(*acc)
                else:
                    raise _Unsupported(f"coop emit {s.op}")
                if R > 1:
                    body.append(f"  for (int r = 0; r < {R}; r++) "
                                f"{name}[r] = {e};")
                else:
                    body.append(f"  {name} = {e};")
                v = (name, R)
            else:
                raise _Unsupported(f"coop emit type {_dump(s)[:200]}")
            memo[id(s)] = v
            return v

        writes = []
        for q, (pos, idx, val) in enumerate(self.stacked):
            nm, R = emit(val)
            inner = _numel(self.arg_shapes[pos][1:])
            off = self._coop_off(tuple(self.arg_shapes[pos][1:]))
            ii = f"((int)t + {self.start}) * {idx.a} + {idx.b}"
            tgt_R = self._coop_R_shape(tuple(self.arg_shapes[pos][1:]))
            src = f"{nm}[r]" if R > 1 else nm
            if tgt_R > 1:
                writes.append(f"  for (int r = 0; r < {tgt_R}; r++) "
                              f"out{q}[(uint)({ii}) * {inner}u + ({off})] = {src};")
            else:
                writes.append(f"  out{q}[(uint)({ii}) * {inner}u + ({off})] = {src};")
        for q, (sym, hname) in enumerate(self.hidden):
            if _DEBUG:
                print(f"[metaljax] hidden {hname}: {_dump(sym)[:220]}",
                      flush=True)
            nm, R = emit(sym)
            numel = _numel(sym.shape)
            off = self._coop_off(sym.shape)
            tgt_R = self._coop_R_shape(tuple(sym.shape))
            src = f"{nm}[r]" if R > 1 else nm
            if tgt_R > 1:
                writes.append(f"  for (int r = 0; r < {tgt_R}; r++) "
                              f"{hname}[t * {numel}u + ({off})] = {src};")
            else:
                writes.append(f"  {hname}[t * {numel}u + ({off})] = {src};")
        news = []
        for j, (pos, expr) in enumerate(self.states):
            nm, R = emit(expr)
            sR = self._coop_R_shape(self.arg_shapes[pos])
            news.append((j, nm, R, sR))
        out.extend(body)
        out.extend(writes)
        for j, nm, R, sR in news:
            src = f"{nm}[r]" if R > 1 else nm
            if sR > 1:
                out.append(f"  for (int r = 0; r < {sR}; r++) st{j}[r] = {src};")
            else:
                out.append(f"  st{j} = {src};")
        out.append("}")
        for j, (pos, expr) in enumerate(self.states):
            shape = self.arg_shapes[pos]
            sR = self._coop_R_shape(shape)
            if len(shape) == 0:
                out.append(f"fin{j}[0u] = st{j};")
            elif sR > 1:
                out.append(f"for (int r = 0; r < {sR}; r++) "
                           f"fin{j}[{self._coop_off(shape)}] = st{j}[r];")
            else:
                out.append(f"fin{j}[{self._coop_off(shape)}] = st{j};")
        return "\n".join(out)

    def _coop_R_shape(self, shape):
        w = shape[-1] if shape else 1
        if w == 1:
            return 1
        if w % self.F:
            raise _Unsupported(f"coop width {w} not multiple of F={self.F}")
        return w // self.F

    def _source_key(self, src):
        key = src if src[0] == "carry" else ("free", id(src[1]), src[1])
        return self._source_ids[key]

    def _buffer_shape(self, leaf):
        if leaf.kind == "arg":
            return self.arg_shapes[leaf.source[1]]
        if leaf.source[0] == "carry":
            return self.arg_shapes[leaf.source[1]]
        return tuple(_ttype(leaf.source[1]).shape)

    # ---- runtime

    def run(self, interp, ins, env):
        from metaljax.interpreter import REGISTRY

        hoist_cache = {}

        def hoisted(v):
            got = hoist_cache.get(v)
            if got is not None:
                return got
            if v in env:
                out = env[v]
            else:
                o = v.owner.operation
                ins_ = [hoisted(x) for x in o.operands]
                res = REGISTRY[o.name](interp, o, ins_, env)
                if isinstance(res, mx.array):
                    res = [res]
                for r, x in zip(list(o.results), res or []):
                    hoist_cache[r] = x
                out = hoist_cache[v]
            hoist_cache[v] = out
            return out

        bufs = []
        for sid, src in enumerate(self.sources):
            if src[0] == "carry":
                buf = ins[src[1]]
            elif src[0] == "hoist":
                buf = hoisted(src[1])
            else:
                buf = env[src[1]]
            norm = self._weight_norms.get(sid)
            if norm is not None:
                shape, strides, offset, perm = norm
                buf = mx.as_strided(buf, shape, strides, offset)
                buf = mx.contiguous(mx.transpose(buf, perm))
            bufs.append(buf)
        for pos, _ in self.states:
            bufs.append(ins[pos])
        out_shapes = ([self.arg_shapes[pos] for pos, _, _ in self.stacked]
                      + [(self.trip,) + tuple(h.shape) for h, _ in self.hidden]
                      + [self.arg_shapes[pos] for pos, _ in self.states])
        out_dtypes = ([_MX_DTYPE[self.arg_dtypes[pos]] for pos, _, _ in self.stacked]
                      + [mx.float32 for _ in self.hidden]
                      + [_MX_DTYPE[self.arg_dtypes[pos]] for pos, _ in self.states])
        self._last_bufs = bufs
        tg = (self.F if self.mode == "coop" else min(self.N, 256), 1, 1)
        outs = self.kernel(
            inputs=bufs,
            grid=(self.N, 1, 1),
            threadgroup=tg,
            output_shapes=out_shapes,
            output_dtypes=out_dtypes,
        )
        if not getattr(self, "_validated", False) and not interp._in_trace:
            # Force the first evaluation here: a Metal build error surfacing
            # later inside an async eval worker would abort the process.
            # (Inside an mx.compile trace evaluation is impossible; the
            # engine's compile-failure fallback covers that path.)
            mx.eval(*outs)
            self._validated = True
        ns, nh = len(self.stacked), len(self.hidden)
        vals = [None] * len(ins)
        for i in self.passthrough:
            vals[i] = ins[i]
        for i, delta in self.counters:
            vals[i] = ins[i] + mx.array(delta * self.trip, dtype=ins[i].dtype)
        for q, (pos, _, _) in enumerate(self.stacked):
            vals[pos] = outs[q]
        for j, (pos, _) in enumerate(self.states):
            vals[pos] = outs[ns + nh + j]
        def stacked(spec):
            """(L, *per_step_shape) array for a resolved accumulator node."""
            kind = spec[0]
            if kind == "hidden":
                return outs[ns + spec[2]]
            if kind == "buffer":
                leaf = spec[1]
                src = (ins[leaf.source[1]] if leaf.source[0] == "carry"
                       else env[leaf.source[1]])
                a = leaf.idx.a
                b2 = a * self.start + leaf.idx.b
                if a == 1:
                    sl = src[b2:b2 + self.trip]
                else:
                    lo = b2 - self.trip + 1
                    sl = src[lo:b2 + 1][::-1]
                # the source may carry interior unit dims the compact leaf
                # dropped; row-major layout makes the reshape free
                return mx.reshape(sl, (self.trip,) + tuple(leaf.shape))
            if kind == "red":
                _, inner, dims, perm, shape = spec
                arr = mx.sum(stacked(inner), axis=tuple(d + 1 for d in dims))
                if tuple(perm) != tuple(range(len(perm))):
                    arr = mx.transpose(arr, (0,) + tuple(p + 1 for p in perm))
                return mx.reshape(arr, (self.trip,) + shape)
            # dot, keeping the time axis: batched matmul with z as batch
            _, lspec, rspec, dims, perm, lrank, rrank = spec
            lb, rb, lc, rc = dims
            A, Bv = stacked(lspec), stacked(rspec)
            zb_l = [0] + [d + 1 for d in lb]
            zb_r = [0] + [d + 1 for d in rb]
            c_l = [d + 1 for d in lc]
            c_r = [d + 1 for d in rc]
            f_l = [i for i in range(lrank + 1) if i not in zb_l and i not in c_l]
            f_r = [i for i in range(rrank + 1) if i not in zb_r and i not in c_r]

            def prod(xs):
                p_ = 1
                for x in xs:
                    p_ *= x
                return p_

            At = mx.transpose(A, zb_l + f_l + c_l)
            Bt = mx.transpose(Bv, zb_r + c_r + f_r)
            bsz = prod([A.shape[i] for i in zb_l])
            m = prod([A.shape[i] for i in f_l])
            kk = prod([A.shape[i] for i in c_l])
            n2 = prod([Bv.shape[i] for i in f_r])
            out = mx.matmul(mx.reshape(At, (bsz, m, kk)),
                            mx.reshape(Bt, (bsz, kk, n2)))
            out_shape = ([A.shape[i] for i in zb_l]
                         + [A.shape[i] for i in f_l]
                         + [Bv.shape[i] for i in f_r])
            arr = mx.reshape(out, out_shape)
            if tuple(perm) != tuple(range(len(perm))):
                arr = mx.transpose(arr, (0,) + tuple(p + 1 for p in perm))
            return arr

        for pos, specs in self.acc_plans:
            total = None
            for spec in specs:
                t_ = mx.sum(stacked(spec), axis=0)
                t_ = mx.reshape(t_, ins[pos].shape)
                total = t_ if total is None else total + t_
            vals[pos] = ins[pos] + total
        return vals


def _literal(s: SymConst):
    if s.dtype == "i1":
        return "true" if s.value else "false"
    if s.dtype == "int" or (s.dtype.lstrip("u").startswith("i")
                            and s.dtype != "i1"):
        v = int(s.value)
        return f"{v}ull" if v > 2**63 - 1 else f"{v}"
    v = float(s.value)
    if v != v:
        return "NAN"
    if v == float("inf"):
        return "INFINITY"
    if v == float("-inf"):
        return "(-INFINITY)"
    return f"{v!r}f"


# ------------------------------------------------------------ entry point

def try_run(interp, op, ins, env, trip, start, counter_pos):
    """Execute a counted while via a generated persistent kernel.
    Returns the final carry values, or None if the loop doesn't qualify."""
    if not ENABLED or trip <= 0:
        return None
    from metaljax.ops.control import _msl_plan_for
    plan = _msl_plan_for(interp, op)
    if plan is None:
        return None
    try:
        res = plan.run(interp, ins, env)
        if os.environ.get("MJDBG_VERIFY_MSL"):
          try:
            from metaljax.ops import control as _c
            fn, free, _ = _c._body_fn(interp, op.regions[1].blocks[0],
                                      compile_body=False)
            caps = [env[v] for v in free]
            ref = list(ins)
            for _ in range(trip):
                ref = list(fn(*ref, *caps))
            import numpy as _np
            mx.eval(*[x for x in res if x is not None])
            mx.eval(*ref)
            worst, wi = 0.0, -1
            for i, (a, b) in enumerate(zip(res, ref)):
                if a is None:
                    continue
                da = _np.array(a, copy=False).astype(_np.float64)
                db_ = _np.array(b, copy=False).astype(_np.float64)
                d = float(_np.abs(da - db_).max()) if da.size else 0.0
                sc = max(float(_np.abs(db_).max()), 1e-6) if db_.size else 1.0
                if d / sc > worst:
                    worst, wi = d / sc, i
            if worst > 1e-4:
                cls = "?"
                if any(p_ == wi for p_, _ in plan.states):
                    cls = "state"
                elif any(p_ == wi for p_, _, _ in plan.stacked):
                    cls = "stacked"
                elif any(p_ == wi for p_, _ in plan.acc_plans):
                    cls = "accum"
                elif wi in plan.passthrough:
                    cls = "passthrough"
                elif any(p_ == wi for p_, _ in plan.counters):
                    cls = "counter"
                print(f"[metaljax] MSL PLAN WRONG: trip={trip} mode={plan.mode} "
                      f"lanes={plan.N} stacked={len(plan.stacked)} "
                      f"hidden={len(plan.hidden)} worst={worst:.3e} "
                      f"out[{wi}]={cls}", flush=True)
                import pathlib
                pathlib.Path("/tmp/mj_wrong_plan.metal").write_text(plan.source)
                a = _np.array(res[wi], copy=False)
                b = _np.array(ref[wi], copy=False)
                print(f"  plan[{wi}][:2]: {a.reshape(-1)[:6]}", flush=True)
                print(f"  raw [{wi}][:2]: {b.reshape(-1)[:6]}", flush=True)
                d = _np.abs(a - b)
                print(f"  wrong elems: {(d > 1e-4 * max(float(_np.abs(b).max()), 1e-6)).sum()}"
                      f"/{d.size} shape={a.shape}", flush=True)
                # 1-trip discriminator: single-iteration miscompile vs
                # cross-iteration state handling
                try:
                    from metaljax.ops.control import _analyze_counted, _static_start
                    k1, _b1 = _analyze_counted(interp, op)
                    st1 = _static_start(op, k1)
                    p1 = Plan(interp, op.regions[1].blocks[0], k1, 1, st1)
                    r1p = p1.run(interp, ins, env)
                    r1r = list(fn(*ins, *caps))
                    mx.eval(*[x for x in r1p if x is not None]); mx.eval(*r1r)
                    w1 = 0.0
                    for i2, (x2, y2) in enumerate(zip(r1p, r1r)):
                        if x2 is None:
                            continue
                        dx = _np.array(x2, copy=False).astype(_np.float64)
                        dy = _np.array(y2, copy=False).astype(_np.float64)
                        dd = float(_np.abs(dx - dy).max()) if dx.size else 0
                        sc2 = max(float(_np.abs(dy).max()), 1e-6)
                        w1 = max(w1, dd / sc2)
                    print(f"  1-trip plan vs 1-iter raw: worst={w1:.3e}", flush=True)
                except Exception as e1:
                    print(f"  1-trip probe failed: {e1}", flush=True)
                try:
                    # two 1-trip plans composed vs the 2-trip plan
                    p1b = Plan(interp, op.regions[1].blocks[0], k1, 1, st1)
                    mid = p1b.run(interp, ins, env)
                    p1c = Plan(interp, op.regions[1].blocks[0], k1, 1,
                               st1 + 1)
                    fin2 = p1c.run(interp, mid, env)
                    mx.eval(*[x for x in fin2 if x is not None])
                    w2 = 0.0
                    for i3, (x3, y3) in enumerate(zip(fin2, ref)):
                        if x3 is None:
                            continue
                        dx = _np.array(x3, copy=False).astype(_np.float64)
                        dy = _np.array(y3, copy=False).astype(_np.float64)
                        dd = float(_np.abs(dx - dy).max()) if dx.size else 0
                        sc3 = max(float(_np.abs(dy).max()), 1e-6)
                        w2 = max(w2, dd / sc3)
                    print(f"  composed 2x1-trip vs raw: worst={w2:.3e}", flush=True)
                    import pathlib
                    pathlib.Path("/tmp/mj_plan1.metal").write_text(p1b.source)
                    # determinism + input-aliasing probes on the 2-trip plan
                    resB = plan.run(interp, ins, env)
                    mx.eval(*[x for x in resB if x is not None])
                    dAB = max(float(_np.abs(_np.array(a2, copy=False).astype(_np.float64)
                                    - _np.array(b2, copy=False).astype(_np.float64)).max())
                              for a2, b2 in zip(res, resB)
                              if a2 is not None and _np.array(a2).size)
                    print(f"  rerun determinism: max abs diff {dAB:.3e}", flush=True)
                    ins_c = [mx.contiguous(x) + 0 for x in ins]
                    env_keys = list(env)
                    resC = plan.run(interp, ins_c, env)
                    mx.eval(*[x for x in resC if x is not None])
                    wC = 0.0
                    for x4, y4 in zip(resC, ref):
                        if x4 is None:
                            continue
                        dx = _np.array(x4, copy=False).astype(_np.float64)
                        dy = _np.array(y4, copy=False).astype(_np.float64)
                        if dx.size:
                            wC = max(wC, float(_np.abs(dx - dy).max())
                                     / max(float(_np.abs(dy).max()), 1e-6))
                    print(f"  fresh-copied inputs vs raw: worst={wC:.3e}", flush=True)
                    import pathlib as _pl
                    if not _pl.Path("/tmp/mj_dump.npz").exists():
                        _pl.Path("/tmp/mj_dump_src.metal").write_text(plan.source)
                        dump = {}
                        for si, bx in enumerate(plan._last_bufs):
                            dump[f"inp{si}"] = _np.array(bx, copy=False)
                        for sj, (pos_, _e) in enumerate(plan.states):
                            dump[f"init{sj}"] = _np.array(ins[pos_], copy=False)
                        for oi, x5 in enumerate(res):
                            if x5 is not None:
                                dump[f"res{oi}"] = _np.array(x5, copy=False)
                        for oi, y5 in enumerate(ref):
                            dump[f"ref{oi}"] = _np.array(y5, copy=False)
                        dump["state_pos"] = _np.array([p_ for p_, _ in plan.states])
                        _np.savez("/tmp/mj_dump.npz", **dump)
                        print("  dumped buffers to /tmp/mj_dump.npz", flush=True)
                except Exception as e2:
                    print(f"  composed probe failed: {e2}", flush=True)
          except Exception as ve:
            print(f"[metaljax] MSL VERIFY ERROR: {ve}", flush=True)
        return res
    except Exception as e:
        if _DEBUG:
            print(f"[metaljax] msl_scan: run failed ({e}); disabling", flush=True)
        body = op.regions[1].blocks[0]
        interp._msl_cache[(body, trip, start)] = None
        return None
