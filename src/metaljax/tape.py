"""Stage 2 M2: lowering an analyzed executable into a native tape.

`lower(interp)` walks main's block once and builds the C++ `Program`
(native/tape.cc) that will replay it: SSA values become integer slots,
attributes become integer vectors, constants are decoded here — by the
same battle-tested paths the eager engine uses — and cross to the device
once. From then on an execute is a switch in C++ with no MLIR, no Python
and no GIL.

The contract that makes this safe to grow: lowering DECLINES WHOLESALE.
Any op outside the supported set, any element type the device stores in
something other than its own bits, anything with a region we do not
recognize — the program returns None and runs on the Python engine
exactly as before. So the native op set grows monotonically and a gap is
a performance question, never a correctness one.

Every handler here has a counterpart in native/tape.cc that is a
transliteration of the Python handler in metaljax.ops. Where a Python
handler branches on dtype, the C++ one branches on dtype too; where it
resolves something static (a broadcast's interim shape, a dot's
transposes, a reduce's monoid), the resolution happens HERE, once. The
differential tests compare output bytes, so any drift shows up as a
failure rather than as a slow divergence.
"""

from __future__ import annotations

import os

from jaxlib.mlir import ir

from metaljax import _ir

_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"

_TERMINATORS = ("func.return", "stablehlo.return")

# stablehlo.compare directions, in the order native/tape.cc switches on.
_DIRECTIONS = {"EQ": 0, "NE": 1, "LT": 2, "LE": 3, "GT": 4, "GE": 5}

# Reduce monoids: body op -> kind, mirroring ops.reduction._REDUCERS and
# _BOOL_REDUCERS (which table applies is decided by the input element
# type, static in the IR). Kinds are (reducer, combine-with-init) pairs on
# the C++ side: 0 sum/add, 1 prod/multiply, 2 max/maximum, 3 min/minimum,
# 4 any/logical_or, 5 all/logical_and.
_REDUCE_KINDS = {
    "stablehlo.add": 0,
    "stablehlo.multiply": 1,
    "stablehlo.maximum": 2,
    "stablehlo.minimum": 3,
}
_BOOL_REDUCE_KINDS = {
    "stablehlo.or": 4,
    "stablehlo.and": 5,
    "stablehlo.add": 4,  # or on i1 sometimes lowers as add
}

_FLOAT_ELEMENTS = ("f16", "f32", "bf16")


class _Decline(Exception):
    """This program does not lower. Carries the reason, for METALJAX_DEBUG."""


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class _Lowering:
    """One pass over main's block."""

    def __init__(self, interp, native):
        self.interp = interp
        self.native = native
        self.opcodes = native.opcodes()
        self.dtypes = native.dtype_codes()
        self.slots: dict = {}
        self.entries: list = []      # (opcode, ins, outs, attrs, payload)
        # Two aliasing taints, both consumed by the output check in `run`.
        # `identity`: slots that may hold the very array object an ARGUMENT
        # holds (see `_is_identity`). `const_view`: slots that may share a
        # CONSTANT's storage, which the Program keeps for the life of the
        # executable — a view of one is as good as the constant itself, so
        # this taint spreads through every shape op, not just the no-ops.
        self.identity: set = set()
        self.const_view: set = set()

    # --- types --------------------------------------------------------

    def _tensor(self, v):
        try:
            t = ir.RankedTensorType(v.type)
        except Exception:
            raise _Decline(f"value is not a ranked tensor: {v.type}") from None
        return t

    def _shape(self, v) -> list[int]:
        t = self._tensor(v)
        shape = list(t.shape)
        for d in shape:
            if d < 0:
                # Shape-poly export: the tape bakes shapes in, so a
                # symbolic dim has nothing to bake.
                raise _Decline("dynamic dimension")
        return shape

    def _element(self, v) -> str:
        return str(self._tensor(v).element_type)

    def _dtype_code(self, el: str) -> int:
        """The extension's code for an MLIR element type, or decline.

        The C++ table holds exactly the types whose device storage IS their
        own bits. Everything else falls out here in one check: complex and
        f64 (no Metal kernel), and the emulated i4/f8/f6/f4 grids, whose
        values live in a WIDER dtype. Declining those is also what keeps
        ops.elementwise's _regrid and _maybe_wrap4 wrappers out of the C++
        handlers — no type that reaches them can trigger either.
        """
        code = self.dtypes.get(el)
        if code is None:
            raise _Decline(f"element type {el}")
        return code

    # --- slots --------------------------------------------------------

    def _slot(self, v) -> int:
        s = self.slots.get(v)
        if s is None:
            raise _Decline("value defined outside the block")
        return s

    def _bind(self, v) -> int:
        s = len(self.slots)
        self.slots[v] = s
        return s

    # --- the walk -----------------------------------------------------

    def run(self):
        func = self.interp.main
        if len(func.regions[0].blocks) != 1:
            raise _Decline("multi-block function")
        block = self.interp._main_block()

        args = list(block.arguments)
        for a in args:
            self._element(a)          # gates tokens and unsupported dtypes
            self._shape(a)
            self.identity.add(self._bind(a))
        nargs = len(args)
        arg_slots = set(range(nargs))

        outputs = None
        for op in block.operations:
            o = op.operation
            if o.name in _TERMINATORS:
                outputs = [self._slot(v) for v in o.operands]
                break
            self._op(o)
        if outputs is None:
            raise _Decline("block without a terminator")

        # An output that IS an argument's array object, or a constant the
        # Program holds for the life of the executable, would alias across
        # calls; XLA's contract wants a fresh buffer. engine.execute fixes
        # the shapes of forwarding it can see statically
        # (forwarded_outputs) and by object identity, and the latter cannot
        # survive the language boundary — so decline the rest here rather
        # than hand back an alias. Returning an argument DIRECTLY is fine:
        # that is the static case, and engine.execute copies it whatever
        # engine ran. (Duplicate outputs, which are one array read twice,
        # native/tape.cc copies; no static analysis needed for those.)
        for s in outputs:
            if s in self.const_view:
                raise _Decline("an output reads a constant's storage")
            if s in self.identity and s not in arg_slots:
                raise _Decline("an argument is forwarded through no-ops")

        drops = self._liveness(outputs)
        prog = self.native.Program(num_slots=len(self.slots), num_args=nargs)
        for (opcode, ins, outs, attrs, payload), drop in zip(self.entries,
                                                             drops):
            prog.add(opcode=opcode, operands=ins, results=outs, attrs=attrs,
                     payload=payload, drops=drop)
        prog.set_outputs(slots=outputs)
        return prog

    def _op(self, o):
        name = o.name
        opcode = self.opcodes.get(name)
        if opcode is None:
            raise _Decline(f"op {name}")
        if o.regions and name != "stablehlo.reduce":
            raise _Decline(f"op {name} carries a region")
        if len(o.results) != 1:
            raise _Decline(f"op {name} has {len(o.results)} results")
        for v in o.operands:
            self._dtype_code(self._element(v))
            self._shape(v)
        self._dtype_code(self._element(o.results[0]))
        self._shape(o.results[0])

        handler = _HANDLERS.get(name)
        attrs, payload = handler(self, o) if handler else ([], None)

        ins = [self._slot(v) for v in o.operands]
        out = self._bind(o.results[0])
        if name == "stablehlo.constant":
            self.const_view.add(out)
        else:
            if any(s in self.const_view for s in ins) and name in _VIEW_OPS:
                self.const_view.add(out)
            if any(s in self.identity for s in ins) and self._is_identity(
                    name, o):
                self.identity.add(out)
        self.entries.append((opcode, ins, [out], attrs, payload))

    def _is_identity(self, name, o) -> bool:
        """Whether MLX may return this op's operand array ITSELF.

        MLX short-circuits a reshape to the same shape, a full-range slice,
        a broadcast to the shape it already has and an astype to its own
        dtype, so such an op can hand back the very object it was given.
        The Python engine catches that at the end of execute by comparing
        `id()`; nanobind's fresh wrappers make that invisible here, so a
        value tainted this way is not allowed to be an output (see `run`).
        Only exact no-ops taint — a real reshape produces a real array, and
        tainting those would decline half the world.
        """
        check = _IDENTITY_CHECKS.get(name)
        if check is None:
            return False
        try:
            return check(self, o)
        except Exception:
            return True  # unreadable attrs: assume the worst

    def _liveness(self, outputs):
        """Per-op drop lists: slots whose last use is that op.

        Straight-line, so this is just "highest index that reads it".
        Mirrors Interpreter.eager_plan: a result nothing reads is let go at
        the op that produced it, and outputs are never dropped (the
        terminator reads them).
        """
        last: dict = {}
        for i, (_, ins, _, _, _) in enumerate(self.entries):
            for s in ins:
                last[s] = i
        for i, (_, _, outs, _, _) in enumerate(self.entries):
            for s in outs:
                last.setdefault(s, i)
        for s in outputs:
            last.pop(s, None)
        drops: list[list[int]] = [[] for _ in self.entries]
        for s, i in last.items():
            drops[i].append(s)
        return drops


# --------------------------------------------------------------------------
# per-op attribute lowering
# --------------------------------------------------------------------------
#
# Each returns (attrs, payload) for the layout documented in
# native/tape.cc. Ops with no attributes have no entry.


def _lower_compare(lo, o):
    from metaljax.ops.elementwise import _comparison_direction
    code = _DIRECTIONS.get(_comparison_direction(o))
    if code is None:
        raise _Decline("compare direction")
    if ("compare_type" in o.attributes
            and "TOTALORDER" in str(o.attributes["compare_type"])
            and lo._element(o.operands[0]) in _FLOAT_ELEMENTS):
        # IEEE totalOrder compares integer keys instead (dtypes.
        # total_order_key); not ported yet.
        raise _Decline("TOTALORDER compare")
    return [code], None


def _lower_convert(lo, o):
    return [lo._dtype_code(lo._element(o.results[0]))], None


def _lower_reshape(lo, o):
    shape = lo._shape(o.results[0])
    return [len(shape), *shape], None


def _lower_transpose(lo, o):
    perm = _ir.i64_list(o, "permutation")
    return [len(perm), *perm], None


def _lower_broadcast_in_dim(lo, o):
    dims = _ir.i64_list(o, "broadcast_dimensions")
    out_shape = lo._shape(o.results[0])
    in_shape = lo._shape(o.operands[0])
    if dims != sorted(dims):
        perm = sorted(range(len(dims)), key=lambda i: dims[i])
        src = [in_shape[p] for p in perm]
        dims = sorted(dims)
        do_transpose = 1
    else:
        perm = list(range(len(in_shape)))
        src = in_shape
        do_transpose = 0
    interim = [1] * len(out_shape)
    for i, d in enumerate(dims):
        interim[d] = src[i]
    return ([do_transpose, len(perm), *perm, len(out_shape), *interim,
             *out_shape], None)


def _lower_slice(lo, o):
    starts = _ir.i64_list(o, "start_indices")
    limits = _ir.i64_list(o, "limit_indices")
    strides = _ir.i64_list(o, "strides")
    return [len(starts), *starts, *limits, *strides], None


def _lower_concatenate(lo, o):
    return [_ir.int_attr(o, "dimension")], None


def _lower_iota(lo, o):
    el = lo._element(o.results[0])
    shape = lo._shape(o.results[0])
    dim = _ir.int_attr(o, "iota_dimension")
    # MLX has no bool arange: ramp in i32 and cast (the Python handler's
    # complex arm is unreachable here — complex declines).
    ramp = "i32" if el == "i1" else el
    return ([dim, lo._dtype_code(ramp), lo._dtype_code(el), len(shape),
             *shape], None)


def _lower_constant(lo, o):
    from metaljax.interpreter import REGISTRY
    # The eager handler, verbatim: splat constants become one-element
    # broadcasts, bf16 crosses through the text/hex decoder, and rank-0
    # floats that %.7g cannot round-trip stay buffer-backed. Called once,
    # at lowering; the array then lives in the Program.
    return [], REGISTRY["stablehlo.constant"](lo.interp, o, [], {})


def _lower_reduce(lo, o):
    n = len(o.operands) // 2
    if n != 1:
        # The (values, indices) argmax/argmin pair and variadic bodies keep
        # the Python engine.
        raise _Decline("variadic reduce")
    body = o.regions[0].blocks[0]
    body_ops = [x.operation for x in body.operations]
    if len(body_ops) != 2:
        raise _Decline("reduce body is not a single op")
    table = (_BOOL_REDUCE_KINDS if lo._element(o.operands[0]) == "i1"
             else _REDUCE_KINDS)
    kind = table.get(body_ops[0].name)
    if kind is None:
        raise _Decline(f"reduce body {body_ops[0].name}")
    dims = _ir.i64_list(o, "dimensions")
    return [kind, len(dims), *dims], None


def _lower_dot_general(lo, o):
    from metaljax.ops.linalg import _dot_dims
    out_el = lo._element(o.results[0])
    if out_el not in _FLOAT_ELEMENTS:
        # Integer dots go through ops.linalg's exact-f32 chunking (or the
        # int64 outer product); neither is ported.
        raise _Decline(f"dot_general result {out_el}")
    lb, rb, lc, rc = _dot_dims(o)
    lhs = lo._shape(o.operands[0])
    rhs = lo._shape(o.operands[1])
    lfree = [d for d in range(len(lhs)) if d not in lb and d not in lc]
    rfree = [d for d in range(len(rhs)) if d not in rb and d not in rc]
    lperm = lb + lfree + lc
    rperm = rb + rc + rfree
    batch = [lhs[d] for d in lb]
    m = [lhs[d] for d in lfree]
    k = [lhs[d] for d in lc]
    n = [rhs[d] for d in rfree]
    out_shape = batch + m + n
    return ([len(lperm), *lperm, len(rperm), *rperm,
             _prod(batch), _prod(m), _prod(k), _prod(n),
             lo._dtype_code(out_el), len(out_shape), *out_shape], None)


_HANDLERS = {
    "stablehlo.compare": _lower_compare,
    "stablehlo.convert": _lower_convert,
    "stablehlo.reshape": _lower_reshape,
    "stablehlo.transpose": _lower_transpose,
    "stablehlo.broadcast_in_dim": _lower_broadcast_in_dim,
    "stablehlo.slice": _lower_slice,
    "stablehlo.concatenate": _lower_concatenate,
    "stablehlo.iota": _lower_iota,
    "stablehlo.constant": _lower_constant,
    "stablehlo.reduce": _lower_reduce,
    "stablehlo.dot_general": _lower_dot_general,
}


# --------------------------------------------------------------------------
# identity recognizers (see _Lowering._note_identity)
# --------------------------------------------------------------------------


def _ident_reshape(lo, o):
    return lo._shape(o.results[0]) == lo._shape(o.operands[0])


def _ident_transpose(lo, o):
    perm = _ir.i64_list(o, "permutation")
    return perm == list(range(len(perm)))


def _ident_convert(lo, o):
    return lo._element(o.results[0]) == lo._element(o.operands[0])


def _ident_slice(lo, o):
    src = lo._shape(o.operands[0])
    return (_ir.i64_list(o, "start_indices") == [0] * len(src)
            and _ir.i64_list(o, "limit_indices") == src
            and _ir.i64_list(o, "strides") == [1] * len(src))


def _ident_broadcast(lo, o):
    dims = _ir.i64_list(o, "broadcast_dimensions")
    return (dims == list(range(len(dims)))
            and lo._shape(o.results[0]) == lo._shape(o.operands[0]))


def _ident_concatenate(lo, o):
    return len(o.operands) == 1


_IDENTITY_CHECKS = {
    "stablehlo.reshape": _ident_reshape,
    "stablehlo.transpose": _ident_transpose,
    "stablehlo.convert": _ident_convert,
    "stablehlo.slice": _ident_slice,
    "stablehlo.broadcast_in_dim": _ident_broadcast,
    "stablehlo.concatenate": _ident_concatenate,
}

# Ops whose result may be a VIEW of an operand rather than new storage.
# Used only to keep a constant's buffer out of an output position, where
# the coarse answer costs a decline and the precise one would cost a rule
# per op per MLX version. (Splat constants are the case that matters: they
# are one-element buffers a broadcast_in_dim spreads over a whole shape.)
_VIEW_OPS = frozenset(_IDENTITY_CHECKS)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def lower(interp):
    """The interpreter's main block as a native Program, or None.

    None means "run this on the Python engine" and is never an error: the
    caller caches it per executable and stops asking.
    """
    from metaljax import engine
    native = engine.NATIVE
    if native is None:
        return None
    try:
        with interp.context:
            return _Lowering(interp, native).run()
    except _Decline as e:
        if _DEBUG:
            print(f"[metaljax] native tape declined: {e}", flush=True)
        return None
