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

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir
from metaljax.interpreter import _SKIP, free_values, value_bytes

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

# Ops whose region is a BODY the lowering reads structurally rather than
# executing: a reduce's monoid, a sort's comparator.
_REGION_BODY_OPS = ("stablehlo.reduce", "stablehlo.sort")

# Ops a handler may give more than one result. Everything else has to
# produce exactly one array, since that is all the tape can bind.
_MULTI_RESULT_OPS = _REGION_BODY_OPS + ("chlo.top_k",)


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
        # M4: the packed quantized weights, threaded in as trailing
        # arguments exactly as engine.MetalExecutable.runner threads them
        # into an mx.compile trace (never captured — mx.compile would bake
        # the first pack by value and never see a repack). Filled by
        # `lower_block`; a region that holds a pack-reading root gets the
        # parent's slots as extra captures (see `_region`).
        self.pack_slots: list = []
        # metaljax.sdpa._mask_array's per-block cache, as tape entries: one
        # kSdpaMask per distinct mask key, read by every root that shares it.
        self.masks: dict = {}
        self._absorbed = None    # slot of the stand-in below, once built
        self._skip = None        # the recognizers' absorbed-op set, memoized

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

    def _skipped(self):
        """Every recognizer's absorbed-op set (their first results)."""
        if self._skip is None:
            from metaljax import sdpa as _sdpa
            skip: set = set()
            for st in (self.interp._qmm, _sdpa.analyze(self.interp)):
                if st is not None and st.active:
                    skip |= st.skip
            self._skip = skip
        return self._skip

    def _absorbed_slot(self) -> int:
        """ops/control.py's `_ABSORBED`: the stand-in a region is handed for
        a capture that was absorbed. Never read — see `_region`."""
        if self._absorbed is None:
            s = self._new_slot()
            self._emit(self.opcodes["stablehlo.constant"], [], [s], [],
                       mx.zeros((), mx.uint8), None)
            self.const_view.add(s)
            self._absorbed = s
        return self._absorbed

    def _new_slot(self) -> int:
        """A slot with no SSA value behind it (a pack array, or one of the
        intermediates a recognizer's `emit` builds)."""
        s = self.nslots
        self.nslots += 1
        return s

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

    def lower_block(self, block, captures=(), npacks=0):
        """Walk `block` into this frame; return (Program, output slots).

        `captures` are the values the block reads from enclosing scopes,
        already resolved to parent slots by the caller: they become extra
        arguments of the Program, after the block's own. `npacks` more
        arguments follow those — the packed quantized weights, which are
        arrays rather than SSA values and so are bound to bare slots.
        """
        returned = None
        for v in list(block.arguments) + list(captures):
            self._element(v)          # gates tokens and unsupported dtypes
            self._shape(v)
            s = self._bind(v)
            self.arg_alias[s] = frozenset({s})
        for _ in range(npacks):
            s = self._new_slot()
            self.arg_alias[s] = frozenset({s})
            self.pack_slots.append(s)
        # What each op DOES (metaljax.interpreter._rewrite_plan): an absorbed
        # op is not executed and gets no slot, a root executes its
        # recognizer's `emit` instead of itself.
        for o, entry in self._walk(block):
            if o.name in _TERMINATORS:
                returned = list(o.operands)
                break
            if entry is _SKIP:
                continue          # absorbed: no entry, no slot, no drops
            if entry is not None:
                self._root(o, entry[1])
                continue
            self._op(o)
        if returned is None:
            raise _Decline("block without a terminator")
        outputs = [self._slot(v) for v in returned]
        nargs = len(list(block.arguments)) + len(captures) + npacks
        prog = self._build(nargs, outputs)
        return prog, outputs

    def _build(self, nargs, outputs):
        drops = self._liveness(outputs)
        prog = self.native.Program(num_slots=self.nslots, num_args=nargs)
        for (opcode, ins, outs, attrs, payload, regions, nbytes,
             fattrs), drop in zip(self.entries, drops):
            prog.add(opcode=opcode, operands=ins, results=outs, attrs=attrs,
                     payload=payload, drops=drop, regions=regions,
                     bytes=nbytes, fattrs=fattrs)
        # Region programs never need output copies: their results are a
        # loop's carries or a branch's values, which stay inside this
        # engine. Only a whole program's outputs cross to a caller, and
        # `run` sets those.
        prog.set_outputs(slots=outputs)
        return prog

    def run(self, compile_main=False, npacks=0):
        func = self.interp.main
        if len(func.regions[0].blocks) != 1:
            raise _Decline("multi-block function")
        block = self.interp._main_block()

        args = list(block.arguments)
        nargs = len(args)
        arg_slots = set(range(nargs))
        argset = set(args)

        prog, outputs = self.lower_block(block, npacks=npacks)

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
        if o.regions and name not in _REGION_BODY_OPS:
            raise _Decline(f"op {name} carries a region")
        nres = len(o.results)
        if nres != 1 and name not in _MULTI_RESULT_OPS:
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

    # --- recognizer roots ---------------------------------------------
    #
    # A root does not run itself: it runs its recognizer's `emit`, on the
    # values `emit_reads` names, and the ops it absorbed never run at all.
    # The lowerings below are the same rewrite, resolved statically — every
    # branch of `emit` that depends only on the match becomes an attribute,
    # and the arrays it reads out of `env` (or out of the packing prologue)
    # become the entry's operands. Anything a lowering cannot express
    # declines the whole program, which then runs on the Python engine with
    # the identical rewrite.

    def _root(self, o, m):
        from metaljax import moe as _moe, qmm as _qmm, sdpa as _sdpa
        if isinstance(m, _qmm.Match):
            lower = _lower_qmm
        elif isinstance(m, _moe.Match):
            lower = _lower_moe
        elif isinstance(m, _sdpa.Match):
            lower = _lower_sdpa
        else:
            raise _Decline(f"unknown recognizer match {type(m).__name__}")
        if len(o.results) != 1:
            raise _Decline("recognizer root with several results")
        for v in o.results:
            self._dtype_code(self._element(v))
            self._shape(v)
        lower(self, o, m)

    def _pack_ins(self, m):
        """Slots of `m`'s packed arrays (qmm.pack_arrays, by index)."""
        if m.slot < 0 or m.nvals <= 0:
            raise _Decline("quantized match has no pack")
        if not self.pack_slots:
            raise _Decline("packed weights are not in scope")
        want = 2 + (1 if m.mode == "affine" else 0) + (1 if m.has_perm else 0)
        if m.nvals != want:
            # `emit` reads vals[0..2] and vals[-1] positionally; a pack that
            # does not have that shape would be read wrong rather than
            # rejected, so say so here.
            raise _Decline(f"pack arity {m.nvals} does not match "
                           f"mode={m.mode} perm={m.has_perm}")
        if m.slot + m.nvals > len(self.pack_slots):
            raise _Decline("pack slot out of range")
        return list(self.pack_slots[m.slot:m.slot + m.nvals])

    def _emit(self, opcode, ins, outs, attrs, payload, o, regions=(),
              fattrs=()):
        """Append one tape entry, charged the result bytes the eager flush
        cadence meters (interpreter.eager_plan's `out_bytes`: the plain
        value_bytes sum, splat corrections deliberately absent — see the
        note there on why a cadence over-counts rather than under-counts).

        `o` may be None for the intermediate entries a recognizer's `emit`
        expands into: eager_plan charges a ROOT its declared result and
        nothing else, so the whole expansion's bytes ride on its last entry
        and the flush lands where the Python engine's lands.
        """
        nbytes = 0 if o is None else sum(
            value_bytes(r, self._tcache) for r in o.results)
        self.entries.append((opcode, ins, outs, attrs, payload, list(regions),
                             nbytes, list(fattrs)))

    def _opcode(self, name) -> int:
        code = self.opcodes.get(name)
        if code is None:
            raise _Decline(f"op {name}")
        return code

    def _mx_code(self, dt) -> int:
        """Dtype code for an mx.Dtype a recognizer match recorded."""
        name = _mx_name(dt)
        if name is None:
            raise _Decline(f"dtype {dt}")
        return self._dtype_code(name)

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
        # ops/control.py _captures, transliterated: `free_values` is purely
        # syntactic, so an op the enclosing block ABSORBED still shows up as
        # a capture of this region — and it has no slot, because it never
        # runs. Every consumer of it inside the region is absorbed too, so
        # nothing can read it; a stand-in keeps the region's arity what
        # `_underived_outputs` and the Python engine both see. A free value
        # that is missing for any OTHER reason still declines, as it must.
        cap_slots = []
        for v in free:
            s = self.slots.get(v)
            if s is None and v in self._skipped():
                s = self._absorbed_slot()
            elif s is None:
                raise _Decline("region capture is defined outside the block")
            cap_slots.append(s)
        # A region holding a root that reads packed weights needs them too,
        # and they are not SSA values, so they ride as extra captures after
        # the free ones. Only when something inside actually reads them:
        # every capture is an input of the region's Program, and a compiled
        # body pays for inputs it never uses.
        npacks = 0
        if self.pack_slots and _uses_packs(self.interp, block):
            npacks = len(self.pack_slots)
            cap_slots = cap_slots + list(self.pack_slots)
        child = _Lowering(self.interp, self.native)
        prog, outputs = child.lower_block(block, free, npacks)
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

    def _walk(self, block):
        """(operation, rewrite entry) for each op of `block`, in order.

        The plan is resolved by POSITION, exactly as run_block replays it —
        and it is consulted here rather than only in `lower_block` because a
        callee's ops are spliced in by `_inline`, and a recognizer's root or
        an absorbed op is just as likely to live inside one.
        """
        plan = dict(self.interp._rewrite_plan(block))
        for i, op in enumerate(block.operations):
            yield op.operation, plan.get(i)

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
        for io, entry in self._walk(block):
            if io.name in _TERMINATORS:
                rets = [self._slot(v) for v in io.operands]
                break
            if entry is _SKIP:
                continue
            if entry is not None:
                self._root(io, entry[1])
                continue
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


def _lower_sort(lo, o):
    """ops/sort.py `_sort`, the arm whose comparator IS a compare.

    jax lowers every top_k as `sort(values, iota)` under a strict TOTALORDER
    GT, and a plain jnp.sort/argsort the same way with LT — the comparator's
    two sides are the (lhs, rhs) block-argument pair themselves, so the sort
    key is an operand and there is nothing to evaluate. The other arms of
    the Python handler (a comparator that computes a key first, the complex
    and multi-key lexicographic select trees) mean running block code on
    arrays, which is a plan this opcode does not carry: they decline.
    """
    from metaljax.ops.elementwise import _comparison_direction
    dim = _ir.int_attr(o, "dimension")
    block = o.regions[0].blocks[0]
    body = [x.operation for x in block.operations]
    if len(body) != 2 or body[0].name != "stablehlo.compare":
        raise _Decline("sort comparator is not a bare compare")
    cmp, ret = body
    if (ret.name != "stablehlo.return" or len(ret.operands) != 1
            or ret.operands[0] != cmp.results[0]):
        raise _Decline("sort comparator does not return its compare")
    d = _comparison_direction(cmp)
    if d not in ("LT", "GT"):
        raise _Decline(f"sort: non-strict compare {d}")
    args = list(block.arguments)
    if len(args) != 2 * len(o.operands) or len(o.results) != len(o.operands):
        raise _Decline("sort arity")
    pair = []
    for v in cmp.operands:
        if not isinstance(v, ir.BlockArgument) or v not in args:
            raise _Decline("sort comparator computes a key")
        pair.append(args.index(v))
    li, ri = pair
    if ri != li + 1 or li % 2:
        raise _Decline("sort comparator args are not an (lhs, rhs) pair")
    k = li // 2
    el = lo._element(o.operands[k])
    if el in _FLOAT_ELEMENTS:
        kind = 1
    elif el == "i1":
        kind = 2
    else:
        kind = 0
    return [dim, 1 if d == "GT" else 0, k, kind], None


def _lower_top_k(lo, o):
    """ops/sort.py `_top_k`. Survives a direct jax lowering; a portable
    artifact decomposes it into the sort above, so both forms are lowered."""
    el = lo._element(o.operands[0])
    if el in _FLOAT_ELEMENTS:
        kind = 1
    elif el == "i1":
        kind = 2
    else:
        kind = 0
    if len(o.results) != 2:
        raise _Decline("top_k does not return (values, indices)")
    return [_ir.int_attr(o, "k"), kind], None


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
    "stablehlo.sort": _lower_sort,
    "chlo.top_k": _lower_top_k,
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
# recognizer emits (M4)
# --------------------------------------------------------------------------
#
# One lowering per `emit`, transliterated branch for branch. What the emit
# decides from the MATCH (a permutation, a reshape, a mode, a dtype) is
# static and becomes an attribute; what it reads out of `env` becomes an
# operand; what it reads out of the packing prologue becomes a pack slot.
# Anything else declines the program, which then runs the identical rewrite
# on the Python engine.

# mx.Dtype -> the MLIR element name the tape's dtype table is keyed by. Only
# the types whose device storage IS their own bits; a match recording
# anything else declines (as the IR-side gate does for the ops).
_MX_NAMES = [
    (mx.bool_, "i1"), (mx.int8, "i8"), (mx.int16, "i16"), (mx.int32, "i32"),
    (mx.int64, "i64"), (mx.uint8, "ui8"), (mx.uint16, "ui16"),
    (mx.uint32, "ui32"), (mx.uint64, "ui64"), (mx.float16, "f16"),
    (mx.float32, "f32"), (mx.bfloat16, "bf16"),
]


def _mx_name(dt):
    for d, name in _MX_NAMES:
        if d == dt:
            return name
    return None


def _uses_packs(interp, block, seen=None) -> bool:
    """Whether `block`, or anything it calls, holds a pack-reading root."""
    from metaljax import moe as _moe, qmm as _qmm
    if seen is None:
        seen = set()
    for _i, entry in interp._rewrite_plan(block):
        if entry is _SKIP:
            continue
        m = entry[1]
        if isinstance(m, _qmm.Match) and m.nvals > 0:
            return True
        if isinstance(m, _moe.Match) and m.packs:
            return True
    for op in block.operations:
        o = op.operation
        attr = _CALL_OPS.get(o.name)
        if attr is not None:
            name = ir.FlatSymbolRefAttr(o.attributes[attr]).value
            fn = interp.funcs.get(name)
            if fn is not None and name not in seen:
                seen.add(name)
                if _uses_packs(interp, fn.regions[0].blocks[0], seen):
                    return True
        for region in o.regions:
            for b in region.blocks:
                if _uses_packs(interp, b, seen):
                    return True
    return False


# The counters the recognizers keep inside their `emit` (moe's per-dot
# gather kind, sdpa's fused count). The native tape emits ONCE, at lowering,
# and replays; so it ticks them there. Diagnostics either way — the analysis
# counters (recognized, packs, verified, fallbacks) are unaffected, since
# the analysis and the packing prologue are shared code.
def _count(mod, key):
    mod.STATS[key] += 1


def _lower_qmm(lo, o, m):
    """metaljax.qmm.emit: one mx.quantized_matmul for the whole chain."""
    if m.mode not in ("affine", "mxfp4"):
        raise _Decline(f"quantized matmul mode {m.mode}")
    x = m.lhs
    lo._dtype_code(lo._element(x))
    rank = len(lo._shape(x))
    lperm = list(m.lperm)
    if len(lperm) != rank or sorted(lperm) != list(range(rank)):
        raise _Decline("quantized matmul lhs permutation")
    ins = [lo._slot(x)] + lo._pack_ins(m)
    attrs = [1 if lperm != list(range(rank)) else 0, len(lperm), *lperm,
             1 if m.bshape else 0, m.B, m.M, m.K,
             m.gs, m.bits, 0 if m.mode == "affine" else 1,
             1 if m.has_perm else 0, 1 if m.swapped else 0,
             lo._mx_code(m.out_dtype),
             len(m.bshape), *m.bshape,
             len(m.mshape), *m.mshape,
             len(m.nshape), *m.nshape]
    lo._emit(lo._opcode("metaljax.qmm"), ins, [lo._bind(o.results[0])],
             attrs, None, o)


def _rec_attrs(rec):
    """metaljax.sdpa._apply's (perm, shape) recipe, as attributes."""
    perm, shape = rec
    out = [0, 0] if perm is None else [1, len(perm), *perm]
    out += [0, 0] if shape is None else [1, len(shape), *shape]
    return out


def _lower_sdpa(lo, o, m):
    """metaljax.sdpa.emit: one mx.fast.scaled_dot_product_attention.

    The mask's additive form is built by an entry of its own, emitted at the
    first root that needs it — `_mask_array`'s cache, whose whole point is
    that one causal mask is shared by every layer of a transformer and costs
    a full [.., .., Tq, Tk] tensor to build.
    """
    ins = [lo._slot(m.q), lo._slot(m.k), lo._slot(m.v)]
    dt = lo._mx_code(m.dtype)
    attrs = (_rec_attrs(m.q_rec) + _rec_attrs(m.k_rec) + _rec_attrs(m.v_rec)
             + [dt, lo._mx_code(m.out_dtype)])
    if m.mask is None:
        attrs += [0] + _rec_attrs((None, None))
    else:
        kind, base, const, mul = m.mask
        if kind not in ("add", "select"):
            raise _Decline(f"attention mask kind {kind}")
        key = (base, kind, const, mul, m.dtype)
        slot = lo.masks.get(key)
        if slot is None:
            lo._dtype_code(lo._element(base))
            slot = lo._new_slot()
            lo._emit(lo._opcode("metaljax.sdpa.mask"), [lo._slot(base)],
                     [slot], [0 if kind == "select" else 1, dt,
                              1 if float(mul) != 1.0 else 0],
                     None, None,
                     fattrs=[float(const) * float(mul), float(mul)])
            lo.masks[key] = slot
        ins.append(slot)
        attrs += [1] + _rec_attrs(m.mask_rec)
    pre, perm, post = m.out_rec
    attrs += ([0, 0] if pre is None else [1, len(pre), *pre])
    attrs += ([0, 0] if perm is None else [1, len(perm), *perm])
    attrs += ([0, 0] if post is None else [1, len(post), *post])
    lo._emit(lo._opcode("metaljax.sdpa"), ins, [lo._bind(o.results[0])],
             attrs, None, o, fattrs=[float(m.scale)])
    from metaljax import sdpa as _sdpa
    _count(_sdpa, "fused")


def _moe_node(lo, m, node, vals, eidx, tidx):
    """One node of metaljax.moe's pair-space plan, as one tape entry."""
    from metaljax import moe as _moe

    if isinstance(node, _moe._Ext):
        if node.op is not None:
            # A nullary op bound inside a callee (a constant or an iota):
            # `_emit_ext` re-runs its handler on an empty environment, so
            # the tape emits it as the op it is.
            name = node.op.name
            handler = _HANDLERS.get(name)
            res = handler(lo, node.op) if handler else ([], None)
            if len(res) != 2:
                raise _Decline(f"moe: {name} needs a different opcode")
            s = lo._new_slot()
            lo._emit(lo._opcode(name), [], [s], res[0], res[1], None)
            if name == "stablehlo.constant":
                lo.const_view.add(s)
            return s
        lo._dtype_code(lo._element(node.value))
        src = lo._slot(node.value)
        s = lo._new_slot()
        lo._emit(lo._opcode("metaljax.moe.gather"), [src, eidx, tidx], [s],
                 [1 if node.ea is not None else 0, node.ea or 0,
                  1 if node.ta is not None else 0, node.ta or 0, m.P],
                 None, None)
        if node.ea is None and node.ta is None:
            # The handler hands the array straight back, so the slot may BE
            # an argument's (or a constant's) storage; carry the taints.
            alias = lo.arg_alias.get(src)
            if alias:
                lo.arg_alias[s] = alias
            if src in lo.const_view:
                lo.const_view.add(s)
        return s

    if isinstance(node, _moe._Elem):
        op = node.op
        for v in list(op.operands) + list(op.results):
            lo._dtype_code(lo._element(v))
        ins = [vals[id(x)] for x in node.srcs]
        s = lo._new_slot()
        if op.name == "stablehlo.concatenate":
            d = _ir.int_attr(op, "dimension")
            axis = 1 + _moe._tpos(node.shape, node.ea, node.ta, d)
            lo._emit(lo._opcode("metaljax.moe.concat"), ins, [s], [axis],
                     None, None)
            return s
        handler = _HANDLERS.get(op.name)
        res = handler(lo, op) if handler else ([], None)
        if len(res) != 2:
            raise _Decline(f"moe: {op.name} needs a different opcode")
        lo._emit(lo._opcode(op.name), ins, [s], res[0], res[1], None)
        return s

    if isinstance(node, _moe._View):
        attrs = [1 if node.src.paired() else 0]
        kind, arg = node.kind, node.arg
        if kind == "bcast":
            keep, trailing = arg
            attrs += [0, len(keep)]
            for d in keep:
                attrs += [0, 0] if d is None else [1, d]
            attrs += [len(trailing), *trailing]
        elif kind == "perm":
            attrs += [1, len(arg), *arg]
        elif kind == "reshape":
            attrs += [2, len(arg), *arg]
        elif kind == "slice":
            attrs += [3, len(arg)]
            for a, b, c in arg:
                attrs += [a, b, c]
        elif kind == "dotperm":
            perm, trailing = arg
            attrs += [4, 1 if list(perm) != sorted(perm) else 0,
                      len(perm), *perm, len(trailing), *trailing]
        else:
            raise _Decline(f"moe view {kind}")
        s = lo._new_slot()
        lo._emit(lo._opcode("metaljax.moe.view"), [vals[id(node.src)]], [s],
                 attrs, None, None)
        return s

    if isinstance(node, _moe._Dot):
        data = node.data
        shortcut = (isinstance(data, _moe._Ext) and data.ea is None
                    and data.op is None)
        if shortcut:
            if data.ta is None:
                raise _Decline("moe: ungathered dot operand has no token axis")
            lo._dtype_code(lo._element(data.value))
            ins = [lo._slot(data.value), eidx, tidx]
            ta = data.ta
        else:
            ins = [vals[id(data)], eidx]
            ta = 0
        pack = node.pack
        if pack is not None:
            if pack.mode not in ("affine", "mxfp4"):
                raise _Decline(f"moe: packed mode {pack.mode}")
            ins = ins + lo._pack_ins(pack)
            gs, bits = pack.gs, pack.bits
            mode = 0 if pack.mode == "affine" else 1
            has_perm = pack.has_perm
        else:
            lo._dtype_code(lo._element(node.weight))
            ins = ins + [lo._slot(node.weight)]
            gs, bits, mode, has_perm = 0, 0, 0, False
        attrs = [1 if shortcut else 0, 1 if data.paired() else 0, ta,
                 1 if pack is not None else 0, m.E,
                 1 if node.n_first else 0, gs, bits, mode,
                 1 if has_perm else 0,
                 m.P, node.M, node.K, node.N, lo._mx_code(node.out_dtype),
                 len(node.mshape), *node.mshape,
                 len(node.nshape), *node.nshape]
        s = lo._new_slot()
        lo._emit(lo._opcode("metaljax.moe.dot"), ins, [s], attrs, None, None)
        _count(_moe, "gather_qmm" if pack is not None else "gather_mm")
        return s

    raise _Decline(f"moe node {type(node).__name__}")


def _lower_moe(lo, o, m):
    """metaljax.moe.emit: the gathered expert dispatch.

    `emit` is an interpreter over a plan of pair-space nodes, so the plan
    becomes a run of tape entries in its own dependency order. Only the
    tail is charged the root's bytes (see `_emit`), which keeps the eager
    flush landing where the Python engine's lands.
    """
    r = m.router
    lo._dtype_code(lo._element(r.indices))
    lo._dtype_code(lo._element(r.weights))
    eidx = lo._new_slot()
    lo._emit(lo._opcode("metaljax.moe.eidx"), [lo._slot(r.indices)], [eidx],
             [m.P], None, None)
    tidx = lo._new_slot()
    lo._emit(lo._opcode("metaljax.moe.tidx"), [], [tidx], [m.T, m.K],
             None, None)
    vals: dict = {}
    for node in m.order:
        vals[id(node)] = _moe_node(lo, m, node, vals, eidx, tidx)
    if id(m.out) not in vals:
        raise _Decline("moe: the plan's output is not in its order")
    attrs = [1 if m.out.paired() else 0, m.P, m.T, m.K,
             lo._mx_code(m.sum_dtype), lo._mx_code(m.out_dtype),
             m.out_axis, len(m.out_shape), *m.out_shape]
    lo._emit(lo._opcode("metaljax.moe.tail"),
             [vals[id(m.out)], lo._slot(r.weights)],
             [lo._bind(o.results[0])], attrs, None, o)


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


def pack_count(interp) -> int:
    """How many packed arrays this program threads in as trailing inputs.

    The packing prologue owns them (metaljax.qmm.prologue, run eagerly by
    engine.execute before anything is lowered or traced), and the Python
    engine passes exactly this list after the program's own arguments —
    see engine.MetalExecutable.runner. The tape does the same, so a repack
    is picked up by both engines and neither ever bakes one in.
    """
    st = getattr(interp, "_qmm", None)
    return 0 if st is None else len(st.values)


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
            return _Lowering(interp, native).run(
                compile_main=compile_main, npacks=pack_count(interp))
    except _Decline as e:
        if _DEBUG:
            print(f"[metaljax] native tape declined: {e}", flush=True)
        return None
