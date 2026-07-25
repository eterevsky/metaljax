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


_MSL_DTYPE = {"f32": "float", "f16": "half", "i32": "int", "i1": "bool"}


def _dt(t: ir.Type) -> str:
    s = str(t)
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
        self.updates = {}        # carry pos -> (idx SymCounter, value Sym)
        self.free = []           # captured ir.Values used as whole tensors

    def analyze(self):
        env = {}
        self.counter_seeded = set()
        for i, a in enumerate(self.args):
            t = _ttype(a)
            shape = tuple(t.shape)
            el = str(t.element_type)
            if i == self.counter_pos or (el == "i32" and shape == ()):
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

    def eval_block(self, block, env):
        result = None
        for op in block.operations:
            o = op.operation
            name = o.name
            if name in ("func.return", "stablehlo.return"):
                result = [self.lookup(env, v) for v in o.operands]
                break
            outs = self.eval_op(o, env)
            for r, s in zip(o.results, outs):
                env[r] = s
        return result

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
               and a.shape == () and b.shape == () and {a.dtype, b.dtype} <= {"int", "i32"}:
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
            if csize > _REG_LIMIT or dsize > _REG_LIMIT:
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
        if not isinstance(x, SymLeaf):
            raise _Unsupported("broadcast of unsupported value")
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

    def _dynamic_slice(self, o, ins):
        x = ins[0]
        starts = ins[1:]
        sizes = _ir.i64_list(o, "slice_sizes")
        if not isinstance(x, SymLeaf) or x.kind not in ("arg", "whole"):
            raise _Unsupported("dynamic_slice of computed value")
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
        if not (isinstance(x, SymLeaf) and x.kind == "arg"
                and x.source[0] == "carry"):
            raise _Unsupported("dynamic_update_slice target")
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
    if isinstance(x, SymElem):
        return SymPerm(x, perm)
    raise _Unsupported("transpose of unsupported value")


def _dump(s, d=0):
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
    """state' = state + cross_lane_dot -> the SymAccDot, else None."""
    if not (isinstance(s, SymElem) and s.op == "stablehlo.add"
            and len(s.args) == 2):
        return None
    a, b = s.args
    for x, y in ((a, b), (b, a)):
        if (isinstance(x, SymLeaf) and x.kind == "arg"
                and x.source == ("carry", pos)
                and x.strides == _rowmajor(x.shape)
                and isinstance(y, SymAccDot)):
            return y
    return None


def _contains_accdot(root):
    stack = [root]
    seen = set()
    while stack:
        s = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        if isinstance(s, SymAccDot):
            return True
        if isinstance(s, SymElem):
            stack.extend(s.args)
        elif isinstance(s, SymDot):
            stack.append(s.data)
    return False


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
             "i1": mx.bool_}


class Plan:
    """A compiled persistent-kernel execution plan for one loop body."""

    def __init__(self, interp, body_block, counter_pos, trip, start):
        self.trip = trip
        args = list(body_block.arguments)
        an = _Analyzer(interp, body_block, counter_pos)
        rets = an.analyze()
        if rets is None or len(rets) != len(args):
            raise _Unsupported("terminator mismatch")

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
                idx, val = an.updates[id(s)]
                self.stacked.append((i, idx, val))
            else:
                if i in an.counter_seeded:
                    raise _Unsupported(
                        f"i32 scalar carry {i} with non-affine update: "
                        f"{type(s).__name__} "
                        f"{getattr(s, 'op', getattr(s, 'kind', ''))}")
                acc = _match_accum(s, i, self.arg_shapes if False else None)
                if acc is not None:
                    self.accums.append((i, acc))
                    continue
                if _contains_accdot(s):
                    raise _Unsupported(
                        f"acc-dot outside accumulator update: {_dump(s)}")
                state_pos_to_id[i] = len(self.states)
                self.states.append((i, s))

        arg_types = [(_ttype(a)) for a in args]
        self.arg_shapes = [tuple(t.shape) for t in arg_types]
        self.arg_dtypes = [_dt(t.element_type) for t in arg_types]

        # Resolve accumulator operands: reads of existing stacks are used
        # directly (with flips for reversed indexing); loop-computed values
        # get hidden per-step stacked outputs.
        self.acc_plans = []       # (pos, spec) resolved in run()
        for pos, acc in self.accums:
            ops = []
            for side in (acc.lhs, acc.rhs):
                if (isinstance(side, SymLeaf) and side.kind == "read"
                        and side.strides == _rowmajor(side.shape)
                        and side.idx.a in (1, -1)):
                    ops.append(("buffer", side))
                else:
                    if side.dtype != "f32":
                        raise _Unsupported("non-f32 accumulator operand")
                    name = f"hid{len(self.hidden)}"
                    self.hidden.append((side, name))
                    ops.append(("hidden", side, len(self.hidden) - 1))
            self.acc_plans.append((pos, acc, ops))

        # Vector mode: bodies with small in-lane matvecs hold the trailing
        # (feature/block) dim of every tensor in registers.
        exprs0 = ([s for _, s in self.states]
                  + [v for _, _, v in self.stacked]
                  + [h for h, _ in self.hidden])
        self.vector = _has_dot(exprs0)
        if self.accums and not self.vector:
            raise _Unsupported("accumulators outside vector mode")

        # Lane space
        lane = ()
        if self.vector:
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
            key = src if src[0] == "carry" else ("free", id(src[1]), src[1])
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

        src = self._emit_vector() if self.vector else self._emit(counter_pos)
        self.source = src
        name = f"mj_scan_{abs(hash((id(body_block), trip, start))) % 10**8}"
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
            elif isinstance(s, SymDot):
                walk(s.data)
                sid = source_id(s.weight.source)
                self._weights[sid] = s.weight
            elif isinstance(s, SymLeaf):
                if s.kind == "read":
                    sid = source_id(s.source)
                    key = (sid, s.idx.a, s.idx.b)
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

        L.append(f"for (uint t = 0; t < {self.trip}u; t++) {{")

        # per-iteration reads
        for (sid, a, b), (leaf, name) in self._reads.items():
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
                    v = self._reads[(sid, s.idx.a, s.idx.b)][1]
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

        out.append(f"for (uint t = 0; t < {self.trip}u; t++) {{")

        # reads
        for (sid, a, b), (leaf, name) in self._reads.items():
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
                    leaf, nm = self._reads[(sid, s.idx.a, s.idx.b)]
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
                raise _Unsupported("vec emit type")
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

        writes = []
        for q, (pos, idx, val) in enumerate(self.stacked):
            nm, R = emit(val)
            inner = _numel(self.arg_shapes[pos][1:])
            off = self._vec_off(tuple(self.arg_shapes[pos][1:]))
            ii = f"((int)t + {self.start}) * {idx.a} + {idx.b}"
            tgt_R = self.arg_shapes[pos][-1] if len(self.arg_shapes[pos]) > 1 else 1
            src = f"{nm}[r]" if R > 1 else nm
            if tgt_R > 1:
                writes.append(f"  for (int r = 0; r < {tgt_R}; r++) "
                              f"out{q}[(uint)({ii}) * {inner}u + ({off})] = {src};")
            else:
                writes.append(f"  out{q}[(uint)({ii}) * {inner}u + ({off})] = {src};")
        for q, (sym, hname) in enumerate(self.hidden):
            nm, R = emit(sym)
            numel = _numel(sym.shape)
            off = self._vec_off(sym.shape)
            tgt_R = sym.shape[-1] if sym.shape else 1
            src = f"{nm}[r]" if R > 1 else nm
            if tgt_R > 1:
                writes.append(f"  for (int r = 0; r < {tgt_R}; r++) "
                              f"{hname}[t * {numel}u + ({off})] = {src};")
            else:
                writes.append(f"  {hname}[t * {numel}u + ({off})] = {src};")
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
        bufs = []
        for src in self.sources:
            if src[0] == "carry":
                bufs.append(ins[src[1]])
            else:
                bufs.append(env[src[1]])
        for pos, _ in self.states:
            bufs.append(ins[pos])
        out_shapes = ([self.arg_shapes[pos] for pos, _, _ in self.stacked]
                      + [(self.trip,) + tuple(h.shape) for h, _ in self.hidden]
                      + [self.arg_shapes[pos] for pos, _ in self.states])
        out_dtypes = ([_MX_DTYPE[self.arg_dtypes[pos]] for pos, _, _ in self.stacked]
                      + [mx.float32 for _ in self.hidden]
                      + [_MX_DTYPE[self.arg_dtypes[pos]] for pos, _ in self.states])
        outs = self.kernel(
            inputs=bufs,
            grid=(self.N, 1, 1),
            threadgroup=(min(self.N, 256), 1, 1),
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
        for pos, acc, ops in self.acc_plans:
            arrs = []
            for op in ops:
                if op[0] == "hidden":
                    arrs.append(outs[ns + op[2]])
                else:
                    leaf = op[1]
                    src = (ins[leaf.source[1]] if leaf.source[0] == "carry"
                           else env[leaf.source[1]])
                    a = leaf.idx.a
                    b2 = a * self.start + leaf.idx.b
                    if a == 1:
                        arr = src[b2:b2 + self.trip]
                    else:  # a == -1: rows b2, b2-1, ..., b2-trip+1
                        lo = b2 - self.trip + 1
                        arr = src[lo:b2 + 1][::-1]
                    arrs.append(arr)
            lb, rb, lc, rc = acc.dims
            lrank, rrank = len(acc.lhs.shape), len(acc.rhs.shape)
            pool = iter("abcdefghijklmnopqrstuvwxy")
            lsub = [None] * lrank
            rsub = [None] * rrank
            batch_letters = []
            for li, ri in zip(lb, rb):
                c = next(pool)
                lsub[li] = rsub[ri] = c
                batch_letters.append(c)
            for li, ri in zip(lc, rc):
                c = next(pool)
                lsub[li] = rsub[ri] = c
            lfree, rfree = [], []
            for i in range(lrank):
                if lsub[i] is None:
                    lsub[i] = next(pool)
                    lfree.append(lsub[i])
            for i in range(rrank):
                if rsub[i] is None:
                    rsub[i] = next(pool)
                    rfree.append(rsub[i])
            sub = (f"z{''.join(lsub)},z{''.join(rsub)}->"
                   f"{''.join(batch_letters + lfree + rfree)}")
            raw = mx.einsum(sub, arrs[0], arrs[1])
            if tuple(acc.perm) != tuple(range(len(acc.perm))):
                raw = mx.transpose(raw, acc.perm)
            vals[pos] = ins[pos] + raw
        return vals


def _literal(s: SymConst):
    if s.dtype == "i1":
        return "true" if s.value else "false"
    if s.dtype == "i32" or s.dtype == "int":
        return f"{int(s.value)}"
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
        return plan.run(interp, ins, env)
    except Exception as e:
        if _DEBUG:
            print(f"[metaljax] msl_scan: run failed ({e}); disabling", flush=True)
        interp._msl_cache[key] = None
        return None
