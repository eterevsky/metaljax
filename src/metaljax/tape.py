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
from metaljax.interpreter import free_values, value_bytes

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

# Symbol-carrying call ops and the attribute naming their callee. Both run
# `interp.run_func(callee, ins)` in ops/control.py, so inlining the callee's
# block into the tape is a transliteration of the handler, not an
# optimization: the same ops run on the same arrays in the same order.
_CALL_OPS = {"func.call": "callee", "stablehlo.composite": "decomposition"}

# Ops whose handler is `return list(ins)`: the result IS the operand array.
# Lowered by aliasing slots, so the C++ interpreter never sees them and the
# aliasing taints (identity, const_view) ride along by construction — which
# is the whole reason they are here rather than as no-op opcodes.
_ALIAS_OPS = ("stablehlo.optimization_barrier", "sdy.sharding_constraint",
              "sdy.reshard")

# Ops carrying regions the tape lowers into sub-Programs (see _control).
_REGION_OPS = ("stablehlo.while", "stablehlo.if", "stablehlo.case")


def _rank0_passthrough(o):
    """Operand index a rank-0 dynamic slice/update hands straight back.

    With no index operands there is nothing to slice: ops/shape.py returns
    the operand array ITSELF, so the tape aliases the slot rather than
    emitting an op — which is also what keeps the aliasing taints right.
    """
    if o.name == "stablehlo.dynamic_slice" and len(o.operands) == 1:
        return 0
    if o.name == "stablehlo.dynamic_update_slice" and len(o.operands) == 2:
        return 1  # the update replaces all of the operand
    return None


class _Decline(Exception):
    """This program does not lower. Carries the reason, for METALJAX_DEBUG."""


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class _Lowering:
    """One block, lowered into one native Program.

    Main is one of these; so is every region a control-flow op carries (see
    `_control`), whose Program takes the region block's arguments followed
    by its CAPTURES — the values it reads from the enclosing scope, resolved
    to parent slots here and passed in as ordinary inputs.
    """

    def __init__(self, interp, native):
        self.interp = interp
        self.native = native
        self.opcodes = native.opcodes()
        self.dtypes = native.dtype_codes()
        self.slots: dict = {}
        # Slots are handed out by a monotone counter, NOT by len(self.slots):
        # inlining aliases values onto existing slots (a callee's block
        # argument IS the caller's operand) and re-binds a callee's own
        # values once per call site, so the dict's size tracks neither.
        self.nslots = 0
        self.calls: list = []        # callee symbols currently being inlined
        self.entries: list = []      # (opcode, ins, outs, attrs, payload, ...)
        # Two aliasing taints, both consumed by the output check in `run`.
        # `arg_alias`: slot -> the ARGUMENT slots whose very array object it
        # may be (see `_is_identity`); a region's outputs carry theirs back
        # to the parent, mapped through the call's operands, so a carry that
        # a loop passes through untouched is still recognized as its input.
        # `const_view`: slots that may share a CONSTANT's storage, which the
        # Program keeps for the life of the executable — a view of one is as
        # good as the constant itself, so this taint spreads through every
        # shape op, not just the no-ops.
        self.arg_alias: dict = {}
        self.const_view: set = set()
        self._tcache: dict = {}      # ir.Type -> bytes (see value_bytes)

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

    def _tainted(self, slot) -> tuple:
        """(argument slots this slot may BE, does it view a constant)."""
        return (self.arg_alias.get(slot, frozenset()),
                slot in self.const_view)

    def _bind(self, v) -> int:
        s = self.nslots
        self.nslots += 1
        self.slots[v] = s
        return s

    def _alias(self, v, slot: int) -> None:
        """Bind `v` to a slot that already exists, allocating nothing.

        Inlining is all aliasing: a callee's block argument names the
        caller's operand array and the call's results name whatever the
        callee returned. No op is emitted for either, so the identity and
        const_view taints — which live on slot numbers — ride along by
        construction.
        """
        self.slots[v] = slot

    # --- the walk -----------------------------------------------------

    def lower_block(self, block, captures=()):
        """Walk `block` into this frame; return (Program, output slots).

        `captures` are the values the block reads from enclosing scopes,
        already resolved to parent slots by the caller: they become extra
        arguments of the Program, after the block's own.
        """
        returned = None
        for v in list(block.arguments) + list(captures):
            self._element(v)          # gates tokens and unsupported dtypes
            self._shape(v)
            s = self._bind(v)
            self.arg_alias[s] = frozenset({s})
        for op in block.operations:
            o = op.operation
            if o.name in _TERMINATORS:
                returned = list(o.operands)
                break
            self._op(o)
        if returned is None:
            raise _Decline("block without a terminator")
        outputs = [self._slot(v) for v in returned]
        nargs = len(list(block.arguments)) + len(captures)
        prog = self._build(nargs, outputs)
        return prog, outputs

    def _build(self, nargs, outputs):
        drops = self._liveness(outputs)
        prog = self.native.Program(num_slots=self.nslots, num_args=nargs)
        for (opcode, ins, outs, attrs, payload, regions, nbytes), drop in zip(
                self.entries, drops):
            prog.add(opcode=opcode, operands=ins, results=outs, attrs=attrs,
                     payload=payload, drops=drop, regions=regions,
                     bytes=nbytes)
        # Region programs never need output copies: their results are a
        # loop's carries or a branch's values, which stay inside this
        # engine. Only a whole program's outputs cross to a caller, and
        # `run` sets those.
        prog.set_outputs(slots=outputs)
        return prog

    def run(self, compile_main=False):
        func = self.interp.main
        if len(func.regions[0].blocks) != 1:
            raise _Decline("multi-block function")
        block = self.interp._main_block()

        args = list(block.arguments)
        nargs = len(args)
        arg_slots = set(range(nargs))
        argset = set(args)

        prog, outputs = self.lower_block(block)

        # Which outputs may not be handed out as they stand: one that IS an
        # argument's array object, or one that reads a constant the Program
        # holds for the life of the executable. Either would alias across
        # calls, and XLA's contract wants a fresh buffer — jax asserts on it
        # through unsafe_buffer_pointer, and a consumer that DONATES such an
        # output would clobber a constant for every later call.
        #
        # The Python engine catches its half of this at the end of execute
        # by comparing `id()`, which cannot survive the language boundary
        # (nanobind hands out a fresh wrapper per call), and its constants
        # are rebuilt per call so they never alias at all. So the tape says
        # statically which outputs to copy, and native/tape.cc copies them.
        # (Duplicate outputs — one array read twice — it catches by array
        # identity at run time; no static analysis needed for those.)
        #
        # Returning an argument DIRECTLY is left alone: `forwarded_outputs`
        # sees exactly that syntax and engine.execute copies it whatever
        # engine ran. Inlining can produce the same aliasing WITHOUT it —
        # `call @identity(%a)` returns main's argument through an OpResult —
        # so an argument slot in an output position is only left alone when
        # the terminator really does name the block argument.
        returned = list(list(block.operations)[-1].operation.operands)
        copies = []
        for j, (v, s) in enumerate(zip(returned, outputs)):
            if s in self.const_view:
                copies.append(j)
            elif s in self.arg_alias and not (s in arg_slots and v in argset):
                copies.append(j)
        prog.set_outputs(slots=outputs, copies=copies)

        if compile_main:
            # The whole tape traces through mx::compile. Python decided that
            # (engine.MetalExecutable.can_compile, the same cost and byte
            # gates the Python engine uses); all that is resolved here is
            # which outputs need anchoring against MLX's constant-baking bug.
            from metaljax.ops import control
            prog.set_compile(
                True, anchors=control._underived_outputs(block, []),
                max_repeat=1)
        return prog

    def _op(self, o):
        name = o.name
        if name in _CALL_OPS:
            self._inline(o, _CALL_OPS[name])
            return
        if name in _ALIAS_OPS:
            if len(o.results) != len(o.operands):
                raise _Decline(f"op {name} is not an arity-preserving alias")
            for r, v in zip(o.results, o.operands):
                self._alias(r, self._slot(v))
            return
        passthrough = _rank0_passthrough(o)
        if passthrough is not None:
            self._alias(o.results[0], self._slot(o.operands[passthrough]))
            return
        if name in _REGION_OPS:
            self._control(o)
            return
        opcode = self.opcodes.get(name)
        if opcode is None:
            raise _Decline(f"op {name}")
        if o.regions and name != "stablehlo.reduce":
            raise _Decline(f"op {name} carries a region")
        nres = len(o.results)
        # Multi-result ops exist only where a handler asks for them (the
        # argmax/argmin (values, indices) reduce); everything else has to
        # produce exactly one array, since that is all the tape can bind.
        if nres != 1 and name != "stablehlo.reduce":
            raise _Decline(f"op {name} has {nres} results")
        for v in o.operands:
            self._dtype_code(self._element(v))
            self._shape(v)
        for v in o.results:
            self._dtype_code(self._element(v))
            self._shape(v)

        # The handler is what decides how many results the op may have when
        # more than one is possible, and it may name a different opcode than
        # the op does: one stablehlo.reduce covers both the monoid fold and
        # the (values, indices) pair, which are separate C++ handlers.
        handler = _HANDLERS.get(name)
        res = handler(self, o) if handler else ([], None)
        if len(res) == 3:
            attrs, payload, opname = res
            opcode = self.opcodes.get(opname)
            if opcode is None:
                raise _Decline(f"op {opname}")
        else:
            attrs, payload = res

        ins = [self._slot(v) for v in o.operands]
        outs = [self._bind(v) for v in o.results]
        if name == "stablehlo.constant":
            self.const_view.add(outs[0])
        elif nres == 1:
            if any(s in self.const_view for s in ins) and name in _VIEW_OPS:
                self.const_view.add(outs[0])
            if self._is_identity(name, o):
                src = frozenset().union(
                    *[self.arg_alias.get(s, frozenset()) for s in ins])
                if src:
                    self.arg_alias[outs[0]] = src
        self._emit(opcode, ins, outs, attrs, payload, o)

    def _emit(self, opcode, ins, outs, attrs, payload, o, regions=()):
        """Append one tape entry, charged the result bytes the eager flush
        cadence meters (interpreter.eager_plan's `out_bytes`: the plain
        value_bytes sum, splat corrections deliberately absent — see the
        note there on why a cadence over-counts rather than under-counts)."""
        nbytes = sum(value_bytes(r, self._tcache) for r in o.results)
        self.entries.append((opcode, ins, outs, attrs, payload, list(regions),
                             nbytes))

    # --- control flow -------------------------------------------------
    #
    # Each region becomes a Program of its own; the parent entry carries the
    # sub-Programs and, for a while, the loop policy Python's analysis in
    # ops/control.py resolved. Nothing about the POLICY is re-derived in
    # C++: the cost model, the trace budgets, the counted-loop recognizer
    # and the compile decisions all run here, once, and the tape records
    # their answers.

    def _region(self, block):
        """Lower one region block into a sub-Program.

        Returns (Program, capture slots in the parent, per-output taints,
        the child frame). A region's captures are resolved against THIS
        frame, so a value from further out becomes a capture of this frame
        too — recursively, which is what lets a nested loop read a value
        defined in main.
        """
        if len(list(block.operations)) == 0:
            raise _Decline("empty region")
        free = free_values(block)
        cap_slots = [self._slot(v) for v in free]
        child = _Lowering(self.interp, self.native)
        prog, outputs = child.lower_block(block, free)
        return prog, free, cap_slots, outputs, child

    def _region_taints(self, child, outputs, parent_slots):
        """Per-output (arg-alias set, const-view flag) in the PARENT frame.

        A region output that may be one of the region's own arguments may,
        in the parent, be whatever that argument was: a carry init, a
        capture, and through those an argument of main or a constant the
        Program holds forever. Mapping the taints back is what keeps a loop
        that forwards a carry from smuggling an alias into an output.
        """
        out = []
        for s in outputs:
            srcs, cv = child._tainted(s)
            alias: set = set()
            for i in srcs:
                # Child arg slots are 0..nargs-1, in the order lower_block
                # bound them (block arguments, then captures).
                p = parent_slots[i]
                alias |= self.arg_alias.get(p, frozenset())
                cv = cv or p in self.const_view
            out.append((frozenset(alias), cv))
        return out

    def _control(self, o):
        name = o.name
        opcode = self.opcodes.get(name)
        if opcode is None:
            raise _Decline(f"op {name}")
        for v in list(o.operands) + list(o.results):
            self._dtype_code(self._element(v))
            self._shape(v)
        for region in o.regions:
            if len(region.blocks) != 1:
                raise _Decline(f"op {name} has a multi-block region")
        if name == "stablehlo.while":
            self._while(o, opcode)
        else:
            self._branch(o, opcode)

    def _while(self, o, opcode):
        from metaljax.interpreter import COMPILE_ENABLED
        from metaljax.ops import control

        if control._msl_plan_for(self.interp, o) is not None:
            # A generated persistent kernel beats anything this milestone
            # can do with the loop; native msl_scan is M5.
            raise _Decline("while has an msl_scan plan")

        cond_block = o.regions[0].blocks[0]
        body_block = o.regions[1].blocks[0]
        ncarry = len(o.operands)
        if len(list(body_block.arguments)) != ncarry:
            raise _Decline("while body arity mismatch")
        if len(list(cond_block.arguments)) != ncarry:
            raise _Decline("while cond arity mismatch")

        cond_prog, cond_free, cond_caps, cond_outs, _ = self._region(
            cond_block)
        if len(cond_outs) != 1:
            raise _Decline("while cond does not return one value")
        body_prog, body_free, body_caps, body_outs, body_lo = self._region(
            body_block)
        if len(body_outs) != ncarry:
            raise _Decline("while body result count mismatch")

        counted = control._analyze_counted(self.interp, o)
        is_counted, k, bound_kind, bound = 0, 0, 0, 0
        if counted is not None:
            k, b = counted
            if isinstance(b, int):
                is_counted, bound_kind, bound = 1, 0, b
            elif isinstance(b, tuple):          # ("carry", j): invariant state
                is_counted, bound_kind, bound = 1, 1, b[1]
            elif b in cond_free:
                # Captured from an enclosing scope; the cond compares
                # against it, so it is one of the cond's captures.
                is_counted, bound_kind, bound = 1, 2, cond_free.index(b)
            # else: the bound is out of reach — the Python engine treats
            # that as a dynamic loop rather than a KeyError, and so do we.

        cost = control._block_cost(self.interp, body_block)
        pure = self.interp.block_is_pure(body_block)
        chunks = control._bytes_chunks(self.interp, body_block)
        by_cost = control._TRACE_BUDGET // max(cost, 1)
        # `_bytes_chunks` never returns less than 1 — its callers ask "how
        # many iterations may one trace hold", and the single-step case is
        # gated separately in _body_fn (`_bytes_ok(..., repeat)`), which
        # says NO when one iteration alone is over budget. Solving that
        # gate for `repeat` is this division, and it must not be floored:
        # rounding it up to 1 compiles a body the Python engine refuses,
        # and a compiled body holds every intermediate of an iteration
        # instead of flushing inside it (measured on the byte-gated
        # random.normal init: 1.19 GB peak eager, 2.38 GB compiled).
        if control._COMPILE_BYTES <= 0:
            by_bytes = by_cost
        else:
            by_bytes = control._COMPILE_BYTES // max(
                control._block_bytes(self.interp, body_block), 1)
        chunkable = int(COMPILE_ENABLED
                        and cost <= control._CHUNK_MAX_COST
                        and pure
                        and body_block not in self.interp._no_chunk)
        kmax = max(1, min(by_cost, control._CHUNK_MAX, chunks))
        period = control._flush_period(cost)
        # How many iterations one compiled body may hold: the _body_fn gates
        # (purity, the op budget, the byte budget), solved for `repeat`.
        body_compile_max = 0
        if (COMPILE_ENABLED and control._BODY_COMPILE and pure
                and body_block not in self.interp._no_body_compile):
            body_compile_max = max(0, min(by_cost, by_bytes))
        if body_compile_max:
            body_prog.set_compile(
                True,
                anchors=control._underived_outputs(body_block, body_free),
                max_repeat=body_compile_max)

        attrs = [ncarry, len(cond_caps), len(body_caps), is_counted, k,
                 bound_kind, bound, cost, period, chunkable, kmax,
                 body_compile_max]
        ins = [self._slot(v) for v in o.operands] + cond_caps + body_caps
        outs = [self._bind(v) for v in o.results]

        # A loop's result j is its carry j: the body's output when the loop
        # ran, the INITIAL value when the trip count was zero — and an
        # initial value is very often main's own argument, so which of the
        # two it is decides whether the result may alias one. A statically
        # counted loop answers that here; anything else is charged both.
        static_trip = None
        if is_counted and bound_kind == 0:
            start = control._static_start(o, k)
            if start is not None:
                static_trip = max(bound - start, 0)
        body_parents = [self._slot(v) for v in o.operands] + body_caps
        taints = self._region_taints(body_lo, body_outs, body_parents)
        for j, (alias, cv) in enumerate(taints):
            init = ins[j]
            if static_trip == 0:
                alias, cv = self._tainted(init)   # the body never runs
            elif static_trip is None:
                alias = alias | self.arg_alias.get(init, frozenset())
                cv = cv or init in self.const_view
            if cv:
                self.const_view.add(outs[j])
            if alias:
                self.arg_alias[outs[j]] = frozenset(alias)
        self._emit(opcode, ins, outs, attrs, None, o,
                   regions=[cond_prog, body_prog])

    def _branch(self, o, opcode):
        """stablehlo.if / stablehlo.case: branch blocks take no arguments
        (everything they read is a capture) and the predicate is read on the
        host, exactly as the Python handlers do."""
        attrs: list = []
        ins = [self._slot(o.operands[0])]
        outs = [self._bind(v) for v in o.results]
        regions = []
        per_branch = []
        for region in o.regions:
            blk = region.blocks[0]
            if len(list(blk.arguments)):
                raise _Decline("branch region takes arguments")
            prog, free, caps, b_outs, child = self._region(blk)
            if len(b_outs) != len(outs):
                raise _Decline("branch result count mismatch")
            attrs.append(len(caps))
            ins.extend(caps)
            regions.append(prog)
            per_branch.append(self._region_taints(child, b_outs, caps))
        if not regions:
            raise _Decline("branch op with no regions")
        for j in range(len(outs)):
            alias: set = set()
            cv = False
            for taints in per_branch:
                a, c = taints[j]
                alias |= a
                cv = cv or c
            if cv:
                self.const_view.add(outs[j])
            if alias:
                self.arg_alias[outs[j]] = frozenset(alias)
        self._emit(opcode, ins, outs, attrs, None, o, regions=regions)

    def _inline(self, o, attr):
        """Splice a single-block callee's ops into this tape.

        Purely Python-side: the C++ interpreter never sees a call. The
        callee's block arguments alias the call's operand slots and the
        call's results alias whatever the callee returned, so the inlined
        ops read and write exactly the arrays `interp.run_func` would have
        handed them.
        """
        name = ir.FlatSymbolRefAttr(o.attributes[attr]).value
        fn = self.interp.funcs.get(name)
        if fn is None:
            raise _Decline(f"call to unknown symbol @{name}")
        if name in self.calls:
            # Recursion has no bounded inlining; the Python engine's own
            # recursive run_func keeps it.
            raise _Decline(f"recursive call to @{name}")
        if len(fn.regions[0].blocks) != 1:
            raise _Decline(f"callee @{name} is not single-block")
        block = fn.regions[0].blocks[0]
        args = list(block.arguments)
        if len(args) != len(o.operands):
            raise _Decline(f"callee @{name} arity mismatch")
        for a, v in zip(args, o.operands):
            self._alias(a, self._slot(v))
        self.calls.append(name)
        rets = None
        for inner in block.operations:
            io = inner.operation
            if io.name in _TERMINATORS:
                rets = [self._slot(v) for v in io.operands]
                break
            self._op(io)
        self.calls.pop()
        if rets is None:
            raise _Decline(f"callee @{name} has no terminator")
        if len(rets) != len(o.results):
            raise _Decline(f"callee @{name} result count mismatch")
        for r, s in zip(o.results, rets):
            self._alias(r, s)

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
        for i, e in enumerate(self.entries):
            for s in e[1]:
                last[s] = i
        for i, e in enumerate(self.entries):
            for s in e[2]:
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
    """The two shapes of stablehlo.reduce ops/reduction.py recognizes.

    The order of the tests is the Python handler's: single-operand monoid
    first, then the (values, indices) pair jax lowers argmax/argmin to.
    Anything else — bitwise bodies, variadic min-with-index chains,
    arbitrary combiners — is _generic_reduce, which walks the body block
    op by op and is not ported.
    """
    n = len(o.operands) // 2
    dims = _ir.i64_list(o, "dimensions")
    body = o.regions[0].blocks[0]
    body_ops = [x.operation for x in body.operations]

    if n == 1 and len(body_ops) == 2:
        if len(o.results) != 1:
            raise _Decline("monoid reduce with several results")
        table = (_BOOL_REDUCE_KINDS if lo._element(o.operands[0]) == "i1"
                 else _REDUCE_KINDS)
        kind = table.get(body_ops[0].name)
        if kind is not None:
            return [kind, len(dims), *dims], None

    if n == 2 and len(dims) == 1 and lo._element(o.operands[0]) != "i1":
        from metaljax.ops.elementwise import _comparison_direction
        first = next((_comparison_direction(x) for x in body_ops
                      if x.name == "stablehlo.compare"), None)
        if first in ("GT", "GE", "LT", "LE"):
            if len(o.results) != 2:
                raise _Decline("argmax-pair reduce with wrong result count")
            return ([1 if first in ("GT", "GE") else 0, dims[0]], None,
                    "stablehlo.reduce.arg_pair")

    raise _Decline(f"reduce body {[x.name for x in body_ops]}")


def _lower_dot_general(lo, o):
    from metaljax import dtypes as _dt
    from metaljax.ops.linalg import _dot_dims, _exact_f32_chunk
    out_el = lo._element(o.results[0])
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

    # Which of ops/linalg._dot_general's three arms runs. Every input is
    # static (the operand dtypes and prod(k)), so the choice is made once
    # here and the C++ handler just executes it: 0 float matmul, 1 exact-f32
    # K-chunks, 2 int64 outer product, 3 the same in bool.
    out_dt = _dt.mx_dtype_for(ir.RankedTensorType(o.results[0].type)
                              .element_type)
    l_dt = _dt.mx_dtype_for(ir.RankedTensorType(o.operands[0].type)
                            .element_type)
    r_dt = _dt.mx_dtype_for(ir.RankedTensorType(o.operands[1].type)
                            .element_type)
    chunk = (_exact_f32_chunk(l_dt, r_dt)
             if _dt.is_int(out_dt) and _prod(k) != 0 else None)
    if chunk is not None:
        kind = 1
    elif _dt.is_int(out_dt):
        kind = 2
    elif _dt.is_bool(out_dt):
        kind = 3
    elif out_el in _FLOAT_ELEMENTS:
        kind = 0
    else:
        raise _Decline(f"dot_general result {out_el}")
    return ([len(lperm), *lperm, len(rperm), *rperm,
             _prod(batch), _prod(m), _prod(k), _prod(n),
             lo._dtype_code(out_el), len(out_shape), *out_shape,
             kind, chunk or 0], None)


def _lower_shift(lo, o):
    """ops/elementwise._shift_guard's static-amount peephole, resolved here.

    XLA defines a shift by >= the operand's bit width as 0 (logical/left) or
    the sign fill (arithmetic); Metal's shifts are mod-width. The guard is a
    compare and a select — except when the amount is a compile-time splat,
    which it almost always is, and then only one arm is emitted. That
    question is pure IR, so it is answered once, at lowering: `[1, amount]`
    means "the amount is statically `amount`", `[0, 0]` means the runtime
    select. Which side of the width the amount falls on stays in C++, where
    the operand's byte width is known.
    """
    from metaljax.ops.elementwise import _static_splat_int
    c = _static_splat_int(o.operands[1])
    if c is not None and c >= 0:
        return [1, c], None
    return [0, 0], None


def _lower_bitcast_convert(lo, o):
    """ops/shape.py _bitcast_convert, the byte-multiple arm.

    Which arm runs is a static property of the two element widths, and the
    other arm cannot be reached at all: it exists for i4/ui4, whose device
    storage is a wider dtype and which the dtype table therefore declines
    (with every other emulated type, for the same reason — a bitcast reads
    BITS, and theirs do not exist on this device).
    """
    from metaljax import dtypes as _dt
    out_el = lo._element(o.results[0])
    src = _dt.mx_dtype_for(_ir.tensor_type(o.operands[0]).element_type)
    dst = _dt.mx_dtype_for(ir.RankedTensorType(o.results[0].type).element_type)
    if dst.size == src.size:
        kind = 0
    elif dst.size < src.size:
        kind = 1
    else:
        kind = 2
    return [lo._dtype_code(out_el), kind], None


def _lower_dynamic_slice(lo, o):
    """ops/shape.py _dynamic_slice: XLA clamps the start indices so the
    window stays inside the operand, and both the sizes and the clamp
    bounds are shape arithmetic — resolved here, once."""
    sizes = _ir.i64_list(o, "slice_sizes")
    src = lo._shape(o.operands[0])
    if len(o.operands) - 1 != len(sizes):
        raise _Decline("dynamic_slice index arity mismatch")
    bounds = [d - s for d, s in zip(src, sizes)]
    return [len(sizes), *bounds, *sizes], None


def _lower_dynamic_update_slice(lo, o):
    sizes = lo._shape(o.operands[1])
    src = lo._shape(o.operands[0])
    if len(o.operands) - 2 != len(sizes):
        raise _Decline("dynamic_update_slice index arity mismatch")
    bounds = [d - s for d, s in zip(src, sizes)]
    return [len(sizes), *bounds], None


_HANDLERS = {
    "stablehlo.compare": _lower_compare,
    "stablehlo.bitcast_convert": _lower_bitcast_convert,
    "stablehlo.dynamic_slice": _lower_dynamic_slice,
    "stablehlo.dynamic_update_slice": _lower_dynamic_update_slice,
    "stablehlo.shift_left": _lower_shift,
    "stablehlo.shift_right_logical": _lower_shift,
    "stablehlo.shift_right_arithmetic": _lower_shift,
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

# A bitcast and a dynamic slice read their operand's storage too (mx.view
# and mx.slice are views), so a constant reaches an output through either.
_EXTRA_VIEW_OPS = ("stablehlo.bitcast_convert", "stablehlo.dynamic_slice")

# Ops whose result may be a VIEW of an operand rather than new storage.
# Used only to keep a constant's buffer out of an output position, where
# the coarse answer costs a decline and the precise one would cost a rule
# per op per MLX version. (Splat constants are the case that matters: they
# are one-element buffers a broadcast_in_dim spreads over a whole shape.)
_VIEW_OPS = frozenset(_IDENTITY_CHECKS) | frozenset(_EXTRA_VIEW_OPS)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def configure(native):
    """Hand the native engine the runtime cadences.

    Every one of them is parsed, documented and defended in the Python
    module that owns it; copying them across (rather than reading the
    environment again in C++) is what keeps the two engines from drifting
    on numbers the command-buffer lottery is pinned to. Re-read on every
    lowering, so a test that moves a cadence moves it for both engines.
    """
    from metaljax import interpreter
    from metaljax.ops import control
    native.configure(
        eager_flush_bytes=interpreter.FLUSH_BYTES,
        flush_sync_every=interpreter._FLUSH_SYNC_EVERY,
        flush_clear_bytes=interpreter._FLUSH_CLEAR_BYTES,
        loop_clear_cost=control._LOOP_CLEAR_COST,
        debug=interpreter._DEBUG,
        memdbg=control._MEMDBG,
    )


def lower(interp, compile_main=False):
    """The interpreter's main block as a native Program, or None.

    None means "run this on the Python engine" and is never an error: the
    caller caches it per executable and stops asking.

    `compile_main` is the Python engine's own compile decision for this
    program (engine.MetalExecutable.can_compile). The tape does not second-
    guess it: the estimators behind it — op cost, traced bytes, purity —
    live in ops/control.py and stay there.
    """
    from metaljax import engine
    native = engine.NATIVE
    if native is None:
        return None
    try:
        with interp.context:
            configure(native)
            return _Lowering(interp, native).run(compile_main=compile_main)
    except _Decline as e:
        if _DEBUG:
            print(f"[metaljax] native tape declined: {e}", flush=True)
        return None
