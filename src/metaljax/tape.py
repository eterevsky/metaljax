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

# dtypes.token_value: what an ordered-effect token IS at run time on both
# engines — an empty bool array, whose only job is to be somewhere in the
# data flow so effects keep their order.
_TOKEN_ELEMENT = "i1"
_TOKEN_SHAPE = (0,)

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

# The one complex type this device has (dtypes._MLIR_TO_MX). Several
# handlers branch on `x.dtype == mx.complex64` in Python; here the element
# type answers the same question statically.
_COMPLEX = "complex<f32>"

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
# executing: a reduce's monoid, a sort's comparator, a scatter's combiner.
_REGION_BODY_OPS = ("stablehlo.reduce", "stablehlo.sort", "stablehlo.scatter",
                    "stablehlo.reduce_window")

# Ops a handler may give more than one result. Everything else has to
# produce exactly one array, since that is all the tape can bind.
_MULTI_RESULT_OPS = _REGION_BODY_OPS + ("chlo.top_k",
                                        "stablehlo.rng_bit_generator")

# Ops whose handler computes on the host through numpy/scipy: they lower to
# a host-call entry (see `_host`), never to a native handler.
_HOST_OPS = ("stablehlo.triangular_solve", "stablehlo.cholesky")


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
        try:
            t = ir.RankedTensorType(v.type)
        except Exception:
            if _ir.is_token(v.type):
                return list(_TOKEN_SHAPE)
            raise _Decline(
                f"value is not a ranked tensor: {v.type}") from None
        shape = list(t.shape)
        for d in shape:
            if d < 0:
                # Shape-poly export: the tape bakes shapes in, so a
                # symbolic dim has nothing to bake.
                raise _Decline("dynamic dimension")
        return shape

    def _element(self, v) -> str:
        # An ordered-effect token is not a tensor and carries no data; it
        # crosses the runtime boundary — and lives in the environment of
        # both engines — as the empty bool array dtypes.token_value builds.
        # Sequencing is the tape's own order, so a token is an ordinary
        # value here and the ops that make one are ordinary entries.
        # Asked only when the tensor cast fails: this runs for every operand
        # and result of every op, and a whole-model main has tens of
        # thousands of them.
        try:
            return str(ir.RankedTensorType(v.type).element_type)
        except Exception:
            pass
        if _ir.is_token(v.type):
            return _TOKEN_ELEMENT
        raise _Decline(f"value is not a ranked tensor: {v.type}")

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
             fattrs, msl, host), drop in zip(self.entries, drops):
            prog.add(opcode=opcode, operands=ins, results=outs, attrs=attrs,
                     payload=payload, drops=drop, regions=regions,
                     bytes=nbytes, fattrs=fattrs, msl=msl, host=host)
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
        if name == "stablehlo.scatter" and _scatter_noop(self, o):
            # Empty updates or a zero-size operand: ops/gather.py returns the
            # operand ARRAY, so this is an alias like the ones above and not
            # an entry that happens to compute nothing.
            self._alias(o.results[0], self._slot(o.operands[0]))
            return
        if name in _REGION_OPS:
            self._control(o)
            return
        if name == "stablehlo.custom_call":
            self._custom_call(o)
            return
        if name in _HOST_OPS:
            self._host(o)
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
        # A handler may name a different opcode than the op does, and one
        # whose op carries a BODY it cannot read structurally (a general
        # reduce combiner) lowers that body into a sub-Program, whose
        # captures ride as extra operands after the op's own.
        regions, extra = (), []
        if len(res) == 5:
            attrs, payload, opname, regions, extra = res
        elif len(res) == 3:
            attrs, payload, opname = res
        else:
            attrs, payload = res
            opname = None
        if opname is not None:
            opcode = self.opcodes.get(opname)
            if opcode is None:
                raise _Decline(f"op {opname}")

        ins = [self._slot(v) for v in o.operands] + list(extra)
        outs = [self._bind(v) for v in o.results]
        if regions or name in _TAINTING_OPS:
            # A body that returns one of its own arguments hands an
            # operand's array straight back (a degenerate combiner does
            # exactly that), so every result inherits every operand's
            # taints. Conservative: the cost of being wrong here is an
            # output aliasing an argument or a constant across calls.
            src = frozenset().union(
                *[self.arg_alias.get(s, frozenset()) for s in ins])
            for s in outs:
                if src:
                    self.arg_alias[s] = src
                if any(i in self.const_view for i in ins):
                    self.const_view.add(s)
        elif name == "stablehlo.constant":
            self.const_view.add(outs[0])
        elif nres == 1:
            if any(s in self.const_view for s in ins) and name in _VIEW_OPS:
                self.const_view.add(outs[0])
            if self._is_identity(name, o):
                src = frozenset().union(
                    *[self.arg_alias.get(s, frozenset()) for s in ins])
                if src:
                    self.arg_alias[outs[0]] = src
        self._emit(opcode, ins, outs, attrs, payload, o, regions=regions)

    # --- host ops (M5b) -----------------------------------------------
    #
    # Some handlers compute on the HOST: the LAPACK family through
    # numpy/scipy (ops/lapack.py), and the in-process callbacks
    # jax.debug.print / pure_callback / io_callback lower to
    # (ops/callbacks.py). There is nothing to port — a native LU would be a
    # different LU — so the tape marks the SITE and the C++ interpreter
    # reacquires the GIL there and nowhere else. Everything around them,
    # which is the whole rest of an impure program, still runs natively.
    #
    # Ordering is the tape's own order, which is the block's order, which is
    # what sequences effects on the Python engine too: an ordered-effect
    # token is an ordinary (empty) value threaded through the entries, and
    # the entries run one after another.

    def _custom_call(self, o):
        """ops/control.py's `_custom_call`, resolved at lowering time."""
        try:
            target = _ir.str_attr(o, "call_target_name")
        except Exception:
            raise _Decline("custom_call without a target name") from None
        if target in ("Sharding", "annotate_device_placement"):
            if len(o.results) != len(o.operands):
                raise _Decline(f"custom_call {target} is not an alias")
            for r, v in zip(o.results, o.operands):
                self._alias(r, self._slot(v))
            return
        if target == "shape_assertion":
            # `return []`: nothing is computed and nothing is bound.
            if len(o.results):
                raise _Decline("shape_assertion with results")
            return
        from metaljax.ops import lapack
        if target in lapack.TARGETS:
            self._host(o)
            return
        raise _Decline(f"custom_call target {target!r}")

    def _host(self, o):
        """Bind this op's Python handler as a host-call entry.

        The closure is the handler, the op and the interpreter — the same
        three things `run_block` brings together — with the MLIR context
        entered around the call, since a host handler reads the op's
        attributes and result types to size its outputs. `env` is empty on
        purpose: no host handler reads it (they take their inputs as
        arrays), and passing the tape's environment would mean rebuilding a
        dict of MLIR values the tape does not keep.
        """
        from metaljax.interpreter import REGISTRY
        fn = REGISTRY.get(o.name)
        if fn is None:
            raise _Decline(f"op {o.name} has no handler")
        for v in list(o.operands) + list(o.results):
            self._dtype_code(self._element(v))
            self._shape(v)
        interp, op = self.interp, o

        def call(ins):
            with interp.context:
                res = fn(interp, op, list(ins), {})
            if res is None:
                return []
            if isinstance(res, mx.array):
                return [res]
            return list(res)

        ins = [self._slot(v) for v in o.operands]
        outs = [self._bind(v) for v in o.results]
        self._emit(self._opcode("metaljax.host_call"), ins, outs, [], None, o,
                   host=call)

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
              fattrs=(), msl=None, host=None):
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
                             nbytes, list(fattrs), msl, host))

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

        # M5b: a loop msl_scan planned into one generated kernel takes the
        # kernel here too, and the question is asked through the SAME
        # `_msl_plan_for` (same cache, same eligibility) that ops/control.py
        # asks, so the two engines can never disagree about which loops get
        # kernels. A plan that cannot be expressed on the tape declines the
        # whole program rather than running the loop op by op: that would
        # be a second answer to a question only one of them may answer.
        plan = control._msl_plan_for(self.interp, o)
        if plan is not None:
            opcode = self._opcode("metaljax.msl_scan")

        cond_block = o.regions[0].blocks[0]
        body_block = o.regions[1].blocks[0]
        ncarry = len(o.operands)
        if len(list(body_block.arguments)) != ncarry:
            raise _Decline("while body arity mismatch")
        if len(list(cond_block.arguments)) != ncarry:
            raise _Decline("while cond arity mismatch")

        # The loop op by op, as regions. With an msl plan these are the
        # FALLBACK: the kernel is what runs, and the interpreted loop is
        # what the native engine retires to if Metal's shader compiler
        # rejects the generated source (the C++ half of `disable_msl`). A
        # body outside the native op set is no reason to lose the kernel,
        # so with a plan the regions are optional — without them a build
        # failure hands the whole program back to the Python engine, which
        # runs its own recovery.
        try:
            cond_prog, cond_free, cond_caps, cond_outs, _ = self._region(
                cond_block)
            if len(cond_outs) != 1:
                raise _Decline("while cond does not return one value")
            body_prog, body_free, body_caps, body_outs, body_lo = self._region(
                body_block)
            if len(body_outs) != ncarry:
                raise _Decline("while body result count mismatch")
        except _Decline:
            if plan is None:
                raise
            if _DEBUG:
                print("[metaljax] msl_scan loop lowered without an "
                      "interpreted fallback", flush=True)
            cond_prog = body_prog = None
            cond_free = body_free = []
            cond_caps = body_caps = []
            body_outs, body_lo = [], None

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
        if body_compile_max and body_prog is not None:
            body_prog.set_compile(
                True,
                anchors=control._underived_outputs(body_block, body_free),
                max_repeat=body_compile_max)

        attrs = [ncarry, len(cond_caps), len(body_caps), is_counted, k,
                 bound_kind, bound, cost, period, chunkable, kmax,
                 body_compile_max]
        ins = [self._slot(v) for v in o.operands] + cond_caps + body_caps
        msl = None
        if plan is not None:
            # The kernel's own inputs ride after the loop's: the arrays it
            # reads are the loop's carries and values from the enclosing
            # scope, which are slots here like any other.
            msl, msl_ins, msl_taints = _lower_msl(self, o, plan)
            ins = ins + msl_ins
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
        if body_lo is not None:
            taints = self._region_taints(body_lo, body_outs, body_parents)
        else:
            # No interpreted fallback: the kernel is the only thing that
            # runs, and the only carry it can hand back untouched is one it
            # classified as pass-through (see `_lower_msl`).
            taints = msl_taints
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
        regions = [] if cond_prog is None else [cond_prog, body_prog]
        self._emit(opcode, ins, outs, attrs, None, o, regions=regions,
                   msl=msl)

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
    # IEEE totalOrder compares the order-preserving integer keys instead of
    # the raw floats (dtypes.total_order_key, which C++ already carries for
    # sort). The Python handler asks the operand's DTYPE; the element type
    # is static, so the answer is resolved here.
    total = ("compare_type" in o.attributes
             and "TOTALORDER" in str(o.attributes["compare_type"])
             and lo._element(o.operands[0]) in _FLOAT_ELEMENTS)
    return [code, 1 if total else 0], None


def _lower_convert(lo, o):
    # `_convert`'s complex arm: XLA's complex -> real convert keeps the
    # real part, and mx.astype alone would not.
    real_part = (lo._element(o.operands[0]) == _COMPLEX
                 and lo._element(o.results[0]) != _COMPLEX)
    return ([lo._dtype_code(lo._element(o.results[0])),
             1 if real_part else 0], None)


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
    # MLX has no bool or complex arange: ramp in i32 and cast, which is
    # what the Python handler's `ramp_dt` picks for both.
    ramp = "i32" if el in ("i1", _COMPLEX) else el
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
    arbitrary combiners — is _generic_reduce, which runs the body block on
    whole arrays; the tape lowers that body into a sub-Program and the C++
    handler calls it once per halving round.
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

    # _generic_reduce: any associative body, any arity. The reduced dims
    # move to one trailing axis and the body combines the two halves of it
    # until one element is left, then folds the init in. Everything the
    # Python version derives from the arrays is static here; the BODY is
    # the one thing that is not, so it becomes a sub-Program.
    if len(o.results) != n:
        raise _Decline("generic reduce result count")
    return _generic_reduce_attrs(lo, o, n, dims, body)


def _generic_reduce_attrs(lo, o, n, dims, body, rank=None):
    """(attrs, payload, opcode name, regions) for a generic reduce body.

    Shared with reduce_window, whose variadic and non-monoid arms hand
    their window axis to the same routine (ops/reduction.py calls
    `_generic_reduce` there too) — with `rank` the WINDOW rank, since what
    that call reduces is the extracted window view, not the operand.
    """
    if rank is None:
        rank = len(lo._shape(o.operands[0]))
    if any(d < 0 or d >= rank for d in dims):
        raise _Decline("reduce dimension out of range")
    keep = [i for i in range(rank) if i not in dims]
    args = list(body.arguments)
    if len(args) != 2 * n:
        raise _Decline("reduce body arity")
    prog, _free, cap_slots, outs, _child = lo._region(body)
    if len(outs) != n:
        raise _Decline("reduce body result count")
    return ([n, len(keep), *keep, len(dims), *dims, len(cap_slots)], None,
            "stablehlo.reduce.generic", [prog], cap_slots)


def _lower_rng(lo, o):
    """ops/rng.py `_rng_bit_generator`, whose whole schedule is static.

    How many philox blocks are consumed, where a threefry output shape
    splits in half, which halves are sliced back down, whether the state
    arrives as four u32 words or two u64 ones — all of it follows from the
    result type and the state's type, so it is resolved here and the C++
    handler is the arithmetic only. Bit-exactness against the Python engine
    (and so against XLA) is the whole point of the family, so nothing here
    rounds a shape differently: every expression below is copied from the
    handler.
    """
    try:
        algo = str(o.attributes["rng_algorithm"])
    except Exception:
        algo = "DEFAULT"
    if len(o.results) != 2:
        raise _Decline("rng_bit_generator result count")
    st_el = lo._element(o.operands[0])
    st_shape = lo._shape(o.operands[0])
    if st_el == "ui32" and st_shape == [4]:
        state_u32 = 1
    elif st_el == "ui64" and st_shape == [2]:
        state_u32 = 0
    else:
        raise _Decline(f"rng_bit_generator state {st_el}{st_shape}")

    out_el = lo._element(o.results[1])
    out_shape = lo._shape(o.results[1])
    out_code = lo._dtype_code(out_el)
    width = {"ui8": 8, "i8": 8, "ui16": 16, "i16": 16, "ui32": 32, "i32": 32,
             "ui64": 64, "i64": 64}.get(out_el)
    if width is None:
        raise _Decline(f"rng_bit_generator output {out_el}")
    unsigned = {8: "ui8", 16: "ui16", 32: "ui32", 64: "ui64"}[width]

    n = _prod(out_shape)
    head = [state_u32, out_code, lo._dtype_code(unsigned),
            len(out_shape), *out_shape]

    if "THREE_FRY" in algo:
        if width > 32:
            raise _Decline("rng_bit_generator THREE_FRY 64-bit")
        # _threefry_bits: one block per half-element pair, the output shape
        # split at the first even dim (else the largest).
        dims = list(out_shape) or [1]
        scalar = 1 if not out_shape else 0
        split = next((i for i, d in enumerate(dims) if d % 2 == 0), None)
        if split is None:
            split = max(range(len(dims)), key=lambda i: dims[i])
        half = list(dims)
        half[split] = -(-dims[split] // 2)
        n_half = _prod(half)
        h = half[:split + 1] + [1] + half[split + 1:]
        rounded = list(dims)
        rounded[split] = half[split] * 2
        return ([1, *head, n_half, split, scalar,
                 1 if rounded[split] != dims[split] else 0,
                 len(h), *h, len(rounded), *rounded, len(dims), *dims],
                None, "stablehlo.rng_bit_generator")

    num_u32 = n * 2 if width == 64 else n
    nv4 = -(-num_u32 // 4)
    return ([0, *head, n, width, num_u32, nv4], None,
            "stablehlo.rng_bit_generator")


_CUM_KINDS = {"stablehlo.add": 0, "stablehlo.maximum": 1,
              "stablehlo.minimum": 2, "stablehlo.multiply": 3}

# reduce_window's own materialization cap (ops/reduction.py raises above it).
_WINDOW_MAX = 200_000_000


def _opt_i64_list(o, name, default):
    return _ir.i64_list(o, name) if name in o.attributes else list(default)


def _window_plan(rank, src, wd, strides, bdil, wdil, pad):
    """The static half of ops/reduction.py `_extract_windows`.

    Base dilation, then padding, then the strided window view — every
    shape of which follows from the attributes. Returns (attrs, out_sizes,
    wflat); the arrays never enter into it.
    """
    attrs = []
    shape = list(src)
    dil = [(ax, b) for ax, b in enumerate(bdil) if b != 1]
    attrs.append(len(dil))
    for ax, b in dil:
        if b < 1:
            raise _Decline("reduce_window base dilation")
        shape[ax] = shape[ax] * b
        attrs += [ax, b, len(shape), *shape]
        shape[ax] = shape[ax] - (b - 1)
        attrs.append(shape[ax])
    if any(p != 0 for row in pad for p in row):
        if any(p < 0 for row in pad for p in row):
            # ops/reduction.py raises here (mx.pad has no negative widths).
            raise _Decline("reduce_window negative padding")
        attrs += [1, rank, *[int(p[0]) for p in pad],
                  rank, *[int(p[1]) for p in pad]]
        shape = [s + int(p[0]) + int(p[1]) for s, p in zip(shape, pad)]
    else:
        attrs += [0, 0, 0]

    out_sizes = []
    for i in range(rank):
        span = (wd[i] - 1) * wdil[i] + 1
        out_sizes.append(max(0, (shape[i] - span) // strides[i] + 1))
    wflat = _prod(wd)
    if wflat * _prod(out_sizes) > _WINDOW_MAX:
        raise _Decline("reduce_window materialization too large")
    elem = [1] * rank
    for i in range(rank - 2, -1, -1):
        elem[i] = elem[i + 1] * shape[i + 1]
    view_shape = out_sizes + list(wd)
    view_strides = ([elem[i] * strides[i] for i in range(rank)]
                    + [elem[i] * wdil[i] for i in range(rank)])
    flat = out_sizes + [wflat]
    attrs += [len(view_shape), *view_shape, len(view_strides), *view_strides,
              len(flat), *flat]
    empty = any(s == 0 for s in out_sizes) or wflat == 0
    attrs += [1 if empty else 0, len(out_sizes), *out_sizes]
    return attrs, out_sizes, wflat


def _lower_reduce_window(lo, o):
    """ops/reduction.py `_reduce_window`, with its three arms resolved.

    The cum-op peephole (jax lowers cumsum and friends as a full-width
    window with prefix padding), the windowed reduction over an as_strided
    view, and — where the body is neither a monoid nor the single compare
    of select_and_gather_add — the generic pairwise reduce over the window
    axis, whose body becomes a sub-Program.
    """
    import numpy as _np
    n = len(o.operands) // 2
    if len(o.results) != n:
        raise _Decline("reduce_window result count")
    src = lo._shape(o.operands[0])
    rank = len(src)
    wd = _opt_i64_list(o, "window_dimensions", [1] * rank)
    strides = _opt_i64_list(o, "window_strides", [1] * rank)
    bdil = _opt_i64_list(o, "base_dilations", [1] * rank)
    wdil = _opt_i64_list(o, "window_dilations", [1] * rank)
    if "padding" in o.attributes:
        pad = _np.array(ir.DenseIntElementsAttr(
            o.attributes["padding"])).reshape(rank, 2).tolist()
    else:
        pad = [[0, 0]] * rank
    if (len(wd) != rank or len(strides) != rank or len(bdil) != rank
            or len(wdil) != rank):
        raise _Decline("reduce_window attribute rank")
    if any(s < 1 for s in strides) or any(d < 1 for d in wdil):
        raise _Decline("reduce_window stride or dilation")
    body = o.regions[0].blocks[0]
    body_ops = [x.operation for x in body.operations]

    # The cumulative pattern, recognized exactly as the handler does.
    if (n == 1 and len(body_ops) == 2 and body_ops[0].name in _CUM_KINDS
            and all(s == 1 for s in strides)
            and all(d == 1 for d in bdil + wdil)):
        big = [i for i, w in enumerate(wd) if w > 1]
        if len(big) == 1:
            ax = big[0]
            size = src[ax]
            others = all(wd[i] == 1 and pad[i][0] == 0 and pad[i][1] == 0
                         for i in range(rank) if i != ax)
            if others and wd[ax] == size:
                kind = _CUM_KINDS[body_ops[0].name]
                if pad[ax][0] == size - 1 and pad[ax][1] == 0:
                    return [0, kind, ax, 0], None
                if pad[ax][0] == 0 and pad[ax][1] == size - 1:
                    return [0, kind, ax, 1], None

    for i in range(1, n):
        if lo._shape(o.operands[i]) != src:
            raise _Decline("reduce_window inputs of different shapes")
    plan, out_sizes, _wflat = _window_plan(
        rank, src, wd, strides, bdil, wdil, pad)
    attrs = [1, n, *plan]

    from metaljax.ops.elementwise import _comparison_direction
    if n >= 2:
        # select_and_gather_add: one compare on the first pair plus selects
        # picks a single window element for every output.
        cmps = [x for x in body_ops if x.name == "stablehlo.compare"]
        nsel = sum(x.name == "stablehlo.select" for x in body_ops)
        if len(cmps) == 1 and nsel >= n:
            d = _comparison_direction(cmps[0])
            return (attrs + [1, 1 if d in ("GE", "GT") else 0], None,
                    "stablehlo.reduce_window")
    elif len(body_ops) == 2:
        table = (_BOOL_REDUCE_KINDS if lo._element(o.operands[0]) == "i1"
                 else _REDUCE_KINDS)
        kind = table.get(body_ops[0].name)
        if kind is not None:
            return attrs + [0, kind], None, "stablehlo.reduce_window"

    # Everything else folds the window axis with the body itself.
    rest, payload, _name, regions, extra = _generic_reduce_attrs(
        lo, o, n, [len(out_sizes)], body, rank=len(out_sizes) + 1)
    return (attrs + [2, *rest], payload, "stablehlo.reduce_window",
            regions, extra)


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
    elif el == _COMPLEX:
        # `_sort_key`'s complex arm packs canonicalized (re, im) order keys
        # into one u64. Not a `kind` this opcode carries — and taking the
        # integer arm would sort by raw complex values, so it declines.
        raise _Decline("sort on complex")
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
    elif el == _COMPLEX:
        raise _Decline("top_k on complex")  # see _lower_sort
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
    elif out_el in _FLOAT_ELEMENTS or out_el == _COMPLEX:
        # complex takes the matmul arm exactly as floats do (the Python
        # handler's last `else`).
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


def _lower_pad(lo, o):
    """ops/shape.py `_pad`: interior dilation, then edge pads, then crops.

    All three stages are shape arithmetic on static attributes, so which of
    them runs — and the exact shapes each produces — is resolved here. The
    C++ handler reads three (flag, vectors) groups in this order and applies
    the ones whose flag is set, which is what the Python handler's three
    `if`s do.
    """
    low = _ir.i64_list(o, "edge_padding_low")
    high = _ir.i64_list(o, "edge_padding_high")
    interior = _ir.i64_list(o, "interior_padding")
    src = lo._shape(o.operands[0])
    rank = len(src)
    if not (len(low) == len(high) == len(interior) == rank):
        raise _Decline("pad attributes do not match the operand rank")
    if any(i < 0 for i in interior):
        raise _Decline("negative interior padding")

    attrs = []
    shape = list(src)
    if any(i > 0 for i in interior):
        # Holes between elements: a pad-valued array of the dilated shape
        # with the operand written into its strided slice.
        shape = [(s - 1) * (i + 1) + 1 if s > 0 else 0
                 for s, i in zip(src, interior)]
        attrs += [1, len(shape), *shape, rank, *[i + 1 for i in interior]]
    else:
        attrs += [0, 0, 0]

    pos = [(max(0, lo_), max(0, hi)) for lo_, hi in zip(low, high)]
    if any(p != (0, 0) for p in pos):
        attrs += [1, rank, *[p[0] for p in pos], rank, *[p[1] for p in pos]]
        shape = [s + p[0] + p[1] for s, p in zip(shape, pos)]
    else:
        attrs += [0, 0, 0]

    if any(lo_ < 0 or hi < 0 for lo_, hi in zip(low, high)):
        # Negative pads CROP: the Python handler slices the padded array.
        begin = [-lo_ if lo_ < 0 else 0 for lo_ in low]
        end = [s + hi if hi < 0 else s for s, hi in zip(shape, high)]
        attrs += [1, rank, *begin, rank, *end]
    else:
        attrs += [0, 0, 0]
    return attrs, None


def _lower_reverse(lo, o):
    """ops/shape.py `_reverse`: one descending take per reversed dim.

    A dim of extent 0 or 1 is skipped there (mx.take chokes on empties),
    and both the dims and their extents are static.
    """
    shape = lo._shape(o.operands[0])
    dims = _ir.i64_list(o, "dimensions")
    if any(d < 0 or d >= len(shape) for d in dims):
        raise _Decline("reverse dimension out of range")
    takes = [(d, shape[d]) for d in dims if shape[d] > 1]
    attrs = [len(takes)]
    for d, n in takes:
        attrs += [d, n]
    return attrs, None


def _lower_fft(lo, o):
    """ops/elementwise.py `_fft`, with its two workarounds intact.

    Both are MLX bugs the Python handler documents: a transform of an empty
    input is the (typed) empty result rather than an MLX error, and a unit
    LAST length on a real transform silently drops the transforms over the
    remaining axes — so that case is spelled out as the identity on the DC
    bin plus an ordinary complex transform over the leading axes. The
    barrier before an eager transform (MLX's FFT kernels can read an input
    whose producing copy is still in flight) rides on the same `in_trace`
    flag the C++ interpreter already threads.
    """
    kind = str(o.attributes["fft_type"])
    length = _ir.i64_list(o, "fft_length")
    in_shape = lo._shape(o.operands[0])
    out_shape = lo._shape(o.results[0])
    out_code = lo._dtype_code(lo._element(o.results[0]))
    rank = len(in_shape)
    if not length or len(length) > rank:
        raise _Decline("fft length rank")
    axes = list(range(rank - len(length), rank))
    s = [int(v) for v in length]

    if _prod(in_shape) == 0 or any(d == 0 for d in s):
        return [0, out_code, len(out_shape), *out_shape], None

    for name, code in (("IRFFT", 3), ("RFFT", 2), ("IFFT", 1)):
        if name in kind:
            break
    else:
        code = 0
    if s[-1] == 1 and code in (2, 3):
        # The unit-length rewrite: `1` for the real axis, the leading axes
        # transformed (or not, when there are none).
        lead_s, lead_axes = s[:-1], axes[:-1]
        return ([2 if code == 3 else 3, 1 if lead_axes else 0,
                 len(lead_s), *lead_s, len(lead_axes), *lead_axes], None)
    return [1, code, len(s), *s, len(axes), *axes], None


def _lower_popcnt(lo, o):
    """ops/elementwise._popcnt / _clz, whose SWAR width is the operand's.

    `_as_unsigned` views a signed operand as its unsigned twin and casts a
    bool to uint8; `_popcount` then computes in u64 for 64-bit operands and
    in u32 for everything else. Both choices are dtype questions the element
    type answers statically, so the tape carries the answers and the C++
    handler carries the arithmetic.
    """
    el = lo._element(o.operands[0])
    code = lo._dtype_code(el)
    unsigned = {"i8": "ui8", "i16": "ui16", "i32": "ui32", "i64": "ui64",
                "i1": "ui8"}.get(el, el)
    if not unsigned.startswith("ui"):
        raise _Decline(f"popcnt/clz on {el}")
    # `_as_unsigned` VIEWS a signed operand (same bits) but CASTS a bool.
    view = 0 if el == "i1" else (1 if el != unsigned else 2)
    wide = unsigned == "ui64"
    bits = {"ui8": 8, "ui16": 16, "ui32": 32, "ui64": 64}[unsigned]
    if el == "i1":
        bits = 8  # mx.bool_.size * 8, which is what the handler uses
    return ([lo._dtype_code(unsigned), view, 1 if wide else 0, bits, code],
            None)


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


def _index_plan(lo, dims, idx_shape, op_shape, sizes, smap, name):
    """The index arrays a gather/scatter hands to the MLX primitive.

    Both StableHLO ops carry the same two index sources: the COMPONENTS of
    the start-index vector (one per mapped operand dim, clamped so the whole
    window fits — XLA's rule, and MLX's own primitives clamp nothing: gather
    wraps a negative index like `take` and reads past the end otherwise, and
    scatter does no bounds checking at all, which for a write is memory
    corruption rather than a wrong number; both measured) and the implicit
    BATCHING coordinates, which are an iota along the index dim paired with
    the operand dim.

    Returns (entries, batch shape, split?, index_vector_dim). Each entry is
    (kind, a, b, operand axis): kind 0 is index-vector component `a` clamped
    to [0, b], kind 1 an iota of length `a` living at batch position `b`,
    kind 2 a constant zero (an op that indexes nothing at all — see below).
    The ORDER of the entries is free — MLX pairs `indices[i]` with `axes[i]`
    and lays the slice dims out in operand order whatever order the axes
    come in (probed) — so it is simply the IR's.
    """
    batching_key = ("start_indices_batching_dims" if name == "gather"
                    else "scatter_indices_batching_dims")
    ivd = dims["index_vector_dim"]
    if ivd > len(idx_shape):
        raise _Decline(f"{name} index_vector_dim out of range")
    batch_dims_idx = [i for i in range(len(idx_shape)) if i != ivd]
    batch_shape = [idx_shape[i] for i in batch_dims_idx]
    # index_vector_dim == rank means the whole array IS one component, which
    # is the shape ops/gather.py's `index_arrays` builds too.
    split = ivd != len(idx_shape)
    k = idx_shape[ivd] if split else 1
    if len(smap) != k:
        raise _Decline(f"{name} index map does not match the index vector")

    entries = []
    for j, dim in enumerate(smap):
        if not 0 <= dim < len(op_shape):
            raise _Decline(f"{name} index map dim out of range")
        entries.append((0, j, max(op_shape[dim] - sizes[dim], 0), dim))
    for op_dim, i_dim in zip(dims["operand_batching_dims"],
                             dims[batching_key]):
        if i_dim not in batch_dims_idx or not 0 <= op_dim < len(op_shape):
            raise _Decline(f"{name} batching dims out of range")
        entries.append((1, idx_shape[i_dim], batch_dims_idx.index(i_dim),
                        op_dim))
    axes = [e[3] for e in entries]
    if len(set(axes)) != len(axes):
        raise _Decline(f"{name} indexes an operand dim twice")
    if not entries:
        # An op that maps NO index component and carries no batching dim:
        # every start index is 0, so the whole thing is a static write at
        # the origin (shape-polymorphic LU emits three of them). Both
        # primitives want at least one index array, so synthesize the
        # constant zero — the clamp bound is 0 too, so there is nothing it
        # could ever be out of.
        if not op_shape:
            raise _Decline(f"{name} on a rank-0 operand with no indices")
        entries.append((2, 0, 0, 0))
    return entries, batch_shape, split, ivd


def _index_attrs(entries):
    out = [len(entries)]
    for e in entries:
        out.extend(int(x) for x in e)
    return out


def _lower_gather(lo, o):
    """ops/gather.py `_gather`, as ONE mx::gather.

    StableHLO's gather IS MLX's, modulo three static rearrangements: the
    coordinate vector is split into one index array per mapped operand dim
    (all of the index batch shape, which makes MLX's broadcast rule a
    no-op); `slice_sizes` crosses verbatim, since MLX starts an unindexed
    axis at 0 and an indexed one at its index — exactly XLA's rule, windows
    on indexed dims included; and the result is reshaped past the collapsed
    (extent-1) dims, then transposed into the offset_dims interleaving.

    The Python handler reaches the same values through numpy-style advanced
    indexing, which expands every window into an explicit arange axis and
    pre-slices the free dims. Same data movement, different composition —
    and a gather moves bits rather than computing on them, so the
    differential has no tolerance to hide behind.
    """
    from metaljax.ops.gather import _gather_dims

    d = _gather_dims(o)
    slice_sizes = _ir.i64_list(o, "slice_sizes")
    op_shape = lo._shape(o.operands[0])
    idx_shape = lo._shape(o.operands[1])
    out_shape = lo._shape(o.results[0])
    out_dt = lo._dtype_code(lo._element(o.results[0]))
    if len(slice_sizes) != len(op_shape):
        raise _Decline("gather slice_sizes rank mismatch")
    if any(s == 0 for s in out_shape):
        # The Python handler's short-circuit: nothing to gather, and the
        # decomposition mis-shapes empty batches. The attrs stop here.
        return [1, out_dt, len(out_shape), *out_shape], None
    if any(not 0 <= s <= op_shape[i] for i, s in enumerate(slice_sizes)):
        raise _Decline("gather slice does not fit the operand")

    entries, batch_shape, split, ivd = _index_plan(
        lo, d, idx_shape, op_shape, slice_sizes, d["start_index_map"],
        "gather")

    collapsed = (set(d["collapsed_slice_dims"])
                 | set(d["operand_batching_dims"]))
    if any(not 0 <= i < len(op_shape) or slice_sizes[i] != 1
           for i in collapsed):
        raise _Decline("gather collapses a dim whose slice is not 1")
    uncollapsed = [i for i in range(len(op_shape)) if i not in collapsed]
    mid = batch_shape + [slice_sizes[i] for i in uncollapsed]

    offset_dims = d["offset_dims"]
    out_rank = len(out_shape)
    out_batch = [i for i in range(out_rank) if i not in offset_dims]
    if (len(out_batch) != len(batch_shape)
            or len(offset_dims) != len(uncollapsed)
            or any(not 0 <= i < out_rank for i in offset_dims)):
        raise _Decline("gather output rank does not match its dimension "
                       "numbers")
    perm = [0] * out_rank
    for cur, pos in enumerate(out_batch):
        perm[pos] = cur
    for cur, pos in enumerate(offset_dims):
        perm[pos] = len(batch_shape) + cur
    # What the rearrangement lands on must be what the IR declares. Free
    # here, and it turns any future disagreement about a dimension-numbers
    # corner into a decline rather than a wrong answer.
    if [mid[p] for p in perm] != out_shape:
        raise _Decline("gather result shape does not follow from its dims")

    return ([0, out_dt, len(out_shape), *out_shape,
             len(batch_shape), *batch_shape, 1 if split else 0, ivd,
             len(slice_sizes), *slice_sizes] + _index_attrs(entries)
            + [len(mid), *mid, len(perm), *perm], None)


# ops/gather.py `_scatter_combiner`'s method names, in the order
# native/tape.cc switches on. "apply" — an arbitrary elementwise body, which
# jax's scatter_apply emits — is deliberately absent: running it means
# evaluating block code on the gathered current values, and with duplicate
# indices doing so one update at a time, which is a plan this opcode does
# not carry. Those programs decline and run the identical body in Python.
_SCATTER_METHODS = {"set": 0, "add": 1, "multiply": 2, "maximum": 3,
                    "minimum": 4, "subtract": 5}


def _scatter_noop(lo, o) -> bool:
    """Whether ops/gather.py `_scatter` hands the operand straight back.

    Empty updates apply nothing, and a zero-size operand drops every update
    as out of bounds. The result IS the operand array, so the tape aliases
    the slot (and the taints ride along) rather than emitting an entry.
    """
    if len(o.operands) != 3 or len(o.results) != 1:
        return False
    return (any(s == 0 for s in lo._shape(o.operands[2]))
            or any(s == 0 for s in lo._shape(o.operands[0])))


def _lower_scatter(lo, o):
    """ops/gather.py `_scatter`, as one mx::scatter/_add/_prod/_max/_min.

    The combiner picks the primitive; everything else is index and update
    preprocessing, resolved here:

    * the start vector splits into one clamped index array per mapped
      operand dim, plus an iota per batching dim (`_index_plan`);
    * the updates are transposed into [index batch dims, window dims in
      OPERAND order] and reshaped to insert an extent-1 axis for every
      inserted window dim — MLX wants `updates.ndim == indices.ndim +
      operand.ndim`, and its update slice starts at the index on an indexed
      axis and at 0 elsewhere, which is XLA's window rule for the windowed
      dims and for the partial windows on free dims alike;
    * XLA's OOB-DROP semantics, of which MLX has none, through the same two
      strategies ops/gather.py picks between, chosen by the same static
      sizes so the two engines cannot pick differently (adding a neutral 0
      and not adding it disagree in the bits at -0.0): redirect dropped
      updates into a pad on one indexed axis — required for "set", where
      neutralizing would race a genuine duplicate write at the clamped slot
      — or rewrite them to the combiner's identity.
    """
    from metaljax import dtypes as _dt
    from metaljax.interpreter import UnsupportedOpError
    from metaljax.ops.gather import (_combiner_neutral, _scatter_combiner,
                                     _scatter_dims)

    if len(o.operands) != 3 or len(o.results) != 1:
        raise _Decline("variadic scatter")
    if lo._element(o.operands[0]) == _COMPLEX:
        # MLX has no complex GPU scatter kernels at all: the Python handler
        # scatters the two parts separately and recombines them, which is a
        # different composition from the single primitive this entry calls.
        raise _Decline("scatter on complex")
    op_shape = lo._shape(o.operands[0])
    idx_shape = lo._shape(o.operands[1])
    upd_shape = lo._shape(o.operands[2])
    d = _scatter_dims(o)
    try:
        method = _scatter_combiner(o)
    except UnsupportedOpError as e:
        raise _Decline(f"scatter body: {e}") from None
    code = _SCATTER_METHODS.get(method)
    if code is None:
        raise _Decline(f"scatter combiner {method}")

    inserted = set(d["inserted_window_dims"]) | set(d["operand_batching_dims"])
    uwd = d["update_window_dims"]
    window_dims = [i for i in range(len(op_shape)) if i not in inserted]
    if len(uwd) != len(window_dims):
        raise _Decline("scatter window rank mismatch")
    if any(not 0 <= w < len(upd_shape) for w in uwd):
        raise _Decline("scatter update_window_dims out of range")
    uwd_of = {dim: w for w, dim in zip(uwd, window_dims)}
    # The update slice, per operand dim: an inserted window dim has an
    # implicit extent of 1, every other dim takes its update axis.
    sshape = [1 if i in inserted else upd_shape[uwd_of[i]]
              for i in range(len(op_shape))]
    if any(not 0 < s <= op_shape[i] for i, s in enumerate(sshape)):
        raise _Decline("scatter window does not fit the operand")

    entries, batch_shape, split, ivd = _index_plan(
        lo, d, idx_shape, op_shape, sshape,
        d["scatter_dims_to_operand_dims"], "scatter")

    upd_batch = [i for i in range(len(upd_shape)) if i not in uwd]
    uperm = upd_batch + [uwd_of[dim] for dim in window_dims]
    if ([upd_shape[p] for p in uperm]
            != batch_shape + [sshape[dim] for dim in window_dims]):
        raise _Decline("scatter updates do not match its dimension numbers")
    ushape = batch_shape + sshape

    # XLA drops an update whose start is out of bounds in ANY component, and
    # only the mapped components can be (a batching iota never is).
    payload = None
    strategy = 0
    extra: list = []
    if d["scatter_dims_to_operand_dims"]:
        if method == "set" or _prod(op_shape) < _prod(upd_shape):
            strategy = 2
            # Pad the LOWEST indexed axis, which is the one the Python
            # handler pads (it transposes that dim to the front and
            # concatenates there). Any of them would do; agreeing keeps the
            # two engines' allocation shapes comparable.
            axes = [e[3] for e in entries]
            pos = min(range(len(axes)), key=lambda i: axes[i])
            extra = [pos, sshape[axes[pos]], op_shape[axes[pos]]]
        else:
            strategy = 1
            dt = _dt.mx_dtype_for(
                ir.RankedTensorType(o.operands[0].type).element_type)
            try:
                payload = _combiner_neutral(method, dt)
            except UnsupportedOpError as e:
                raise _Decline(f"scatter neutral: {e}") from None
            # The mask broadcasts against the updates: one axis per batch
            # dim, then an extent-1 axis per operand dim.
            extra = [len(batch_shape) + len(op_shape),
                     *batch_shape, *([1] * len(op_shape))]

    # `strategy` rides second, ahead of everything variable-length, so a
    # reader (the C++ handler, a test) can see which drop rule this scatter
    # took without walking the whole vector.
    return ([code, strategy, len(batch_shape), *batch_shape,
             1 if split else 0, ivd] + _index_attrs(entries)
            + [len(sshape), *sshape, len(uperm), *uperm, len(ushape), *ushape]
            + extra, payload)


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
    "stablehlo.reduce_window": _lower_reduce_window,
    "stablehlo.sort": _lower_sort,
    "chlo.top_k": _lower_top_k,
    "stablehlo.dot_general": _lower_dot_general,
    "stablehlo.gather": _lower_gather,
    "stablehlo.scatter": _lower_scatter,
    "stablehlo.pad": _lower_pad,
    "stablehlo.reverse": _lower_reverse,
    "stablehlo.popcnt": _lower_popcnt,
    "stablehlo.count_leading_zeros": _lower_popcnt,
    "stablehlo.rng_bit_generator": _lower_rng,
    "stablehlo.fft": _lower_fft,
}

# Ops whose result may be a VIEW of an operand rather than a value computed
# from it: rng's empty-output arm hands the state array straight back
# (through two mx.view calls), and a generic reduce's body may return one of
# its own arguments. Their results inherit every operand's aliasing taints,
# which costs an output copy in the cases where nothing actually aliases.
_TAINTING_OPS = ("stablehlo.rng_bit_generator",)


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
# msl_scan plans (M5b)
# --------------------------------------------------------------------------
#
# metaljax.msl_scan turns a counted loop whose body is elementwise (plus
# small matvecs) into ONE generated Metal kernel: a thread per lane, the
# state in registers, the whole time loop inside the shader. Planning it —
# the IR pattern match, the mode choice, the MSL source generation — is
# compile-time work and stays exactly where it is, in Python. What the tape
# carries is the LAUNCH: the generated source, the binding order, the
# geometry, and the little post-kernel recipe `Plan.run` applies to turn the
# kernel's outputs back into the loop's carries.
#
# So this is a transliteration of `Plan.run`, resolved statically:
#
#   * its `bufs` loop becomes a list of slots (one per plan source) plus a
#     weight-normalization recipe per source;
#   * its `feed` becomes the unpacked-source list and the per-dtype pools of
#     the 0.4.3 input packing, in the order the kernel's `input_names` names;
#   * its `vals` assembly becomes four static lists (pass-through carries,
#     affine counters, stacked outputs, final states) plus the accumulator
#     recipes of loop fission, encoded as little trees.
#
# Anything the recipe cannot express declines the PROGRAM (never just the
# loop): both engines ask `_msl_plan_for`, so a loop that has a plan must
# take it on whichever engine runs, or the two would compute a carry by
# different arithmetic.


def _msl_vec(out, xs):
    """A length-prefixed vector, the shape native/tape.cc's Cursor reads."""
    out.append(len(xs))
    out.extend(int(x) for x in xs)


def _msl_value_slot(lo, v):
    """The slot holding `v` here, hoisting its defining ops if it has none.

    `Plan.run`'s `hoisted` reads the value out of the enclosing environment
    when it is bound there, and otherwise RE-EVALUATES its defining op with
    the ordinary handlers — which is what happens to a loop-invariant op the
    analyzer lifted out of the body (`_try_hoist`), and to a value the
    enclosing block never computed because a recognizer absorbed it. Both
    are transliterated here by lowering that op into this frame, ahead of
    the loop: the same ops on the same arrays, in the same order.
    """
    s = lo.slots.get(v)
    if s is not None:
        return s
    if not isinstance(v, ir.OpResult):
        raise _Decline("msl: source is defined outside the block")
    op = v.owner
    op = getattr(op, "operation", op)
    if op.regions:
        raise _Decline(f"msl: hoisted op {op.name} carries a region")
    for x in op.operands:
        _msl_value_slot(lo, x)
    lo._op(op)
    s = lo.slots.get(v)
    if s is None:
        raise _Decline("msl: hoisted op bound no slot")
    return s


def _msl_spec(lo, spec, out, source_index):
    """One accumulator node of `Plan.run`'s `stacked`, as a little tree.

    kinds: 0 a hidden per-step stack (a kernel output), 1 a slice of a
    device buffer, 2 a post-kernel sum, 3 a post-kernel batched matmul.
    """
    kind = spec[0]
    if kind == "hidden":
        out.extend([0, int(spec[2])])
    elif kind == "buffer":
        leaf = spec[1]
        idx = getattr(leaf, "idx", None)
        a = getattr(idx, "a", None)
        if a not in (1, -1):
            raise _Decline("msl: accumulator buffer with a non-unit stride")
        out.extend([1, source_index(leaf.source), int(a), int(idx.b)])
        _msl_vec(out, leaf.shape)
    elif kind == "red":
        _, inner, dims, perm, shape = spec
        out.append(2)
        _msl_spec(lo, inner, out, source_index)
        _msl_vec(out, dims)
        _msl_vec(out, perm)
        _msl_vec(out, shape)
    elif kind == "dot":
        _, lspec, rspec, dims, perm, lshape, rshape = spec
        if len(dims) != 4:
            raise _Decline("msl: accumulator dot without four dim lists")
        out.append(3)
        _msl_spec(lo, lspec, out, source_index)
        _msl_spec(lo, rspec, out, source_index)
        for d in dims:
            _msl_vec(out, d)
        _msl_vec(out, perm)
        _msl_vec(out, lshape)
        _msl_vec(out, rshape)
    else:
        raise _Decline(f"msl: accumulator node {kind}")


def _lower_msl(lo, o, plan):
    """(MslPlan, extra input slots, per-carry taints) for a planned loop."""
    from metaljax import msl_scan

    if msl_scan._VOLATILE == "tmap":
        # The table-driven loop counter binds an extra input the plan builds
        # per call (`_tmap`). A retest knob, never the shipped path.
        raise _Decline("msl: METALJAX_MSL_VOLATILE=tmap")
    if plan.mode not in ("scalar", "vector", "coop"):
        raise _Decline(f"msl: mode {plan.mode}")

    carries = [lo._slot(v) for v in o.operands]
    ncarry = len(carries)

    # Sources, in the plan's own order: the sids the generated source, the
    # packing groups and the weight norms are all keyed by. Accumulator
    # recipes may name a buffer the kernel itself never reads (a stacked
    # input consumed whole, post-kernel); those append after.
    slots: list = []
    seen: dict = {}

    def key_of(src):
        return ("carry", src[1]) if src[0] == "carry" else ("value", src[1])

    def slot_of(src):
        if src[0] == "carry":
            j = src[1]
            if not 0 <= j < ncarry:
                raise _Decline("msl: source carry out of range")
            return carries[j]
        return _msl_value_slot(lo, src[1])

    for src in plan.sources:
        seen.setdefault(key_of(src), len(slots))
        slots.append(slot_of(src))

    def source_index(src):
        k = key_of(src)
        got = seen.get(k)
        if got is not None:
            return got
        seen[k] = len(slots)
        slots.append(slot_of(src))
        return seen[k]

    # `Plan.run`'s output_shapes / output_dtypes, in binding order.
    out_shapes, out_dtypes = [], []
    for pos, _idx, _val in plan.stacked:
        out_shapes.append(list(plan.arg_shapes[pos]))
        out_dtypes.append(lo._dtype_code(plan.arg_dtypes[pos]))
    for h, _nm in plan.hidden:
        out_shapes.append([plan.trip, *h.shape])
        out_dtypes.append(lo._dtype_code("f32"))
    for pos, _expr in plan.states:
        out_shapes.append(list(plan.arg_shapes[pos]))
        out_dtypes.append(lo._dtype_code(plan.arg_dtypes[pos]))

    # Every carry position must be produced by exactly one rule: the kernel
    # writes the states and the stacked outputs, the post-kernel arithmetic
    # does the counters and the accumulators, and a pass-through carry is
    # handed back as it came. A position claimed twice (or not at all) would
    # silently take whichever rule ran last.
    acc_pos = [pos for pos, _ in plan.acc_plans]
    covered = (list(plan.passthrough) + [p for p, _ in plan.counters]
               + [p for p, _, _ in plan.stacked] + [p for p, _ in plan.states]
               + acc_pos)
    if sorted(covered) != list(range(ncarry)):
        raise _Decline("msl: carries are not covered exactly once")

    # The accumulator recipes go FIRST, because encoding one may name a
    # buffer the kernel itself never reads and so APPEND a source — and the
    # source count, the weight norms and the pack groups are all written
    # against the final list.
    acc: list = []
    for pos, specs in plan.acc_plans:
        acc.extend([int(pos), len(specs)])
        for spec in specs:
            _msl_spec(lo, spec, acc, source_index)

    tg = plan.F if plan.mode == "coop" else min(plan.N, 256)
    layout = [plan.N, tg, plan.trip, plan.start, len(slots),
              len(plan.hidden), ncarry]
    for sid in range(len(slots)):
        norm = plan._weight_norms.get(sid) if sid < len(plan.sources) else None
        if norm is None:
            layout.append(0)
            continue
        shape, strides, offset, perm = norm
        layout.append(1)
        _msl_vec(layout, shape)
        _msl_vec(layout, strides)
        layout.append(int(offset))
        _msl_vec(layout, perm)
    _msl_vec(layout, plan._unpacked)
    layout.append(len(plan._pack_groups))
    for _nm, dt, sids in plan._pack_groups:
        layout.append(lo._dtype_code(dt))
        _msl_vec(layout, sids)
    _msl_vec(layout, [pos for pos, _ in plan.states])
    _msl_vec(layout, [pos for pos, _, _ in plan.stacked])
    _msl_vec(layout, plan.passthrough)
    layout.append(len(plan.counters))
    for pos, delta in plan.counters:
        layout.extend([int(pos), int(delta)])
    layout.append(len(plan.acc_plans))
    layout.extend(acc)

    names_in = ([f"inp{i}" for i in plan._unpacked]
                + [nm for nm, _, _ in plan._pack_groups]
                + [f"init{j}" for j in range(len(plan.states))])
    names_out = ([f"out{q}" for q in range(len(plan.stacked))]
                 + [nm for _, nm in plan.hidden]
                 + [f"fin{j}" for j in range(len(plan.states))])
    msl = lo.native.MslPlan(
        name=plan.kernel_name, source=plan.source,
        header=msl_scan._KERNEL_HEADER, input_names=names_in,
        output_names=names_out, out_shapes=out_shapes,
        out_dtypes=out_dtypes, layout=layout)

    # Only a pass-through carry can be the very array an input holds; every
    # other position is a kernel output or post-kernel arithmetic.
    through = set(plan.passthrough)
    taints = [lo._tainted(carries[j]) if j in through
              else (frozenset(), False) for j in range(ncarry)]
    return msl, slots, taints


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
        while_pipeline=control._WHILE_PIPELINE,
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
