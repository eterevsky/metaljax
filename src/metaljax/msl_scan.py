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
    __slots__ = ("kind", "source", "idx", "state_pos", "inner_shape")

    def __init__(self, kind, shape, dtype, source=None, idx=None,
                 state_pos=None, inner_shape=None):
        self.kind, self.shape, self.dtype = kind, shape, dtype
        self.source, self.idx, self.state_pos = source, idx, state_pos
        self.inner_shape = inner_shape


class SymElem(Sym):
    __slots__ = ("op", "args", "extra")

    def __init__(self, op, args, dtype, shape, extra=None):
        self.op, self.args, self.extra = op, args, extra
        self.dtype, self.shape = dtype, shape


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

        if name == "stablehlo.dynamic_slice":
            return [self._dynamic_slice(o, ins)]

        if name == "stablehlo.dynamic_update_slice":
            return [self._dynamic_update(o, ins)]

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

    def _reshaped(self, x, out_shape):
        if x.shape == out_shape:
            return x
        # only allow dropping/adding leading 1s
        if tuple(d for d in x.shape if d != 1) != tuple(d for d in out_shape if d != 1):
            raise _Unsupported(f"reshape {x.shape} -> {out_shape}")
        if isinstance(x, SymConst):
            return SymConst(x.value, x.dtype, out_shape)
        s = _clone_leafish(x)
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
        if tuple(d for d in interim if d != 1) != tuple(d for d in x.shape if d != 1):
            raise _Unsupported("size-expanding broadcast of a leaf")
        s = _clone_leafish(x)
        s.shape = tuple(interim)
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


def _clone_leafish(x):
    if isinstance(x, SymLeaf):
        s = SymLeaf(x.kind, x.shape, x.dtype, source=x.source, idx=x.idx,
                    state_pos=x.state_pos, inner_shape=x.inner_shape)
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
                state_pos_to_id[i] = len(self.states)
                self.states.append((i, s))

        arg_types = [(_ttype(a)) for a in args]
        self.arg_shapes = [tuple(t.shape) for t in arg_types]
        self.arg_dtypes = [_dt(t.element_type) for t in arg_types]

        # Lane space
        lane = ()
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

        exprs = [s for _, s in self.states] + [v for _, _, v in self.stacked]
        self._collect(exprs, source_id)
        # Reads must come from passthrough carries or free captures — never
        # from tensors this loop itself mutates.
        mutated = {pos for pos, _, _ in self.stacked} | {pos for pos, _ in self.states}
        for src in self.sources:
            if src[0] == "carry" and src[1] in mutated:
                raise _Unsupported("read of a mutated carry")

        src = self._emit(counter_pos)
        self.source = src
        name = f"mj_scan_{abs(hash((id(body_block), trip, start))) % 10**8}"
        self.kernel = mx.fast.metal_kernel(
            name=name,
            input_names=[f"inp{i}" for i in range(len(self.sources))]
            + [f"init{j}" for j in range(len(self.states))],
            output_names=[f"out{q}" for q in range(len(self.stacked))]
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
                off = _lane_offset(leaf.shape, lane)
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
            off = _lane_offset(leaf.shape, lane)
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
        out_shapes = [self.arg_shapes[pos] for pos, _, _ in self.stacked] + [
            self.arg_shapes[pos] for pos, _ in self.states]
        out_dtypes = [_MX_DTYPE[self.arg_dtypes[pos]] for pos, _, _ in self.stacked] + [
            _MX_DTYPE[self.arg_dtypes[pos]] for pos, _ in self.states]
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
        vals = [None] * len(ins)
        for i in self.passthrough:
            vals[i] = ins[i]
        for i, delta in self.counters:
            vals[i] = ins[i] + mx.array(delta * self.trip, dtype=ins[i].dtype)
        for q, (pos, _, _) in enumerate(self.stacked):
            vals[pos] = outs[q]
        for j, (pos, _) in enumerate(self.states):
            vals[pos] = outs[len(self.stacked) + j]
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
    body_block = op.regions[1].blocks[0]
    key = (body_block, trip, start)
    plan = interp._msl_cache.get(key, "miss")
    if plan == "miss":
        try:
            plan = Plan(interp, body_block, counter_pos, trip, start)
            if _DEBUG:
                print(f"[metaljax] msl_scan: compiled plan trip={trip} "
                      f"lanes={plan.N} states={len(plan.states)} "
                      f"stacked={len(plan.stacked)}", flush=True)
        except _Unsupported as e:
            if _DEBUG:
                print(f"[metaljax] msl_scan: not eligible ({e})", flush=True)
            plan = None
        except Exception as e:
            if _DEBUG:
                print(f"[metaljax] msl_scan: plan failed ({e})", flush=True)
            plan = None
        interp._msl_cache[key] = plan
    if plan is None:
        return None
    try:
        return plan.run(interp, ins, env)
    except Exception as e:
        if _DEBUG:
            print(f"[metaljax] msl_scan: run failed ({e}); disabling", flush=True)
        interp._msl_cache[key] = None
        return None
