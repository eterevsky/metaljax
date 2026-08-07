"""Control flow, calls, composites, and pass-through ops."""

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes, qmm, sdpa
from metaljax.interpreter import (
    COMPILE_ENABLED,
    UnsupportedOpError,
    free_values,
    op_bytes,
    register,
    value_bytes,
)


def _callee_name(op, attr_name):
    return ir.FlatSymbolRefAttr(op.attributes[attr_name]).value


@register("func.call")
def _call(interp, op, ins, env):
    return interp.run_func(interp.funcs[_callee_name(op, "callee")], ins)


@register("stablehlo.composite")
def _composite(interp, op, ins, env):
    # Composites (e.g. jax.nn ops) carry a reference to a decomposition func.
    return interp.run_func(interp.funcs[_callee_name(op, "decomposition")], ins)


def _splat_int(op) -> int | None:
    """Value of a scalar integer constant op, else None."""
    if op.name != "stablehlo.constant":
        return None
    t = ir.RankedTensorType(op.results[0].type)
    if list(t.shape):
        return None
    try:
        return int(_ir.dense_to_np(op.attributes["value"], t))
    except Exception:
        return None


def _cond_has_effects(interp, block) -> bool:
    """True if a while-cond region performs host-visible side effects
    (callbacks / debug prints), directly or through a callee."""
    for op in block.operations:
        o = op.operation
        if o.name == "stablehlo.custom_call":
            try:
                if ir.BoolAttr(o.attributes["has_side_effect"]).value:
                    return True
            except (KeyError, ValueError):
                pass
        if o.name in ("func.call", "stablehlo.composite"):
            attr = "callee" if o.name == "func.call" else "decomposition"
            fn = interp.funcs.get(_callee_name(o, attr))
            if fn is not None and _cond_has_effects(
                    interp, fn.regions[0].blocks[0]):
                return True
        for region in o.regions:
            for b in region.blocks:
                if _cond_has_effects(interp, b):
                    return True
    return False


def _analyze_counted(interp, op):
    """Detect the canonical counted loop JAX emits for scan/fori_loop.

    cond:  %n = constant N            (or N captured from outside)
           %p = compare LT, %arg_k, %n
           return %p
    body:  ... return ..., add(%arg_k, 1), ...

    Returns (k, bound) where bound is an int or the ir.Value holding N,
    or None if the loop doesn't match.
    """
    cond_block = op.regions[0].blocks[0]
    cached = interp._counted_cache.get(cond_block, "miss")
    if cached != "miss":
        return cached
    if _cond_has_effects(interp, cond_block):
        # The counted fast path never executes the cond region, so a cond
        # with host-visible effects must take the dynamic path (which runs
        # the cond trip+1 times, matching XLA).
        interp._counted_cache[cond_block] = None
        return None

    result = None
    try:
        cond_ops = [o.operation for o in cond_block.operations]
        ret = cond_ops[-1]
        cmp_val = ret.operands[0]
        if isinstance(cmp_val, ir.OpResult):
            cmp = cmp_val.owner.operation
            if (
                cmp.name == "stablehlo.compare"
                and "LT" in str(cmp.attributes["comparison_direction"])
            ):
                lhs, rhs = cmp.operands
                args = list(cond_block.arguments)
                if isinstance(lhs, ir.BlockArgument) and lhs in args:
                    k = args.index(lhs)
                    bound = None
                    if isinstance(rhs, ir.OpResult):
                        n = _splat_int(rhs.owner.operation)
                        if n is not None:
                            bound = n
                    if bound is None and not isinstance(rhs, ir.OpResult):
                        if rhs in args:
                            # Bound carried in the loop state (e.g. lbfgs
                            # carries maxiter): counted iff the body
                            # forwards it unchanged.
                            j = args.index(rhs)
                            body_blk = op.regions[1].blocks[0]
                            fwd = list(body_blk.operations)[-1].operation
                            if fwd.operands[j] == list(body_blk.arguments)[j]:
                                bound = ("carry", j)
                        else:
                            bound = rhs  # captured from an enclosing scope
                    if bound is not None:
                        # body must return arg_k + 1 at position k
                        body_block = op.regions[1].blocks[0]
                        body_ret = list(body_block.operations)[-1].operation
                        inc = body_ret.operands[k]
                        ok = False
                        if isinstance(inc, ir.OpResult):
                            add = inc.owner.operation
                            if add.name == "stablehlo.add":
                                a, b = add.operands
                                barg = list(body_block.arguments)[k]
                                for x, y in ((a, b), (b, a)):
                                    if x == barg and isinstance(y, ir.OpResult):
                                        if _splat_int(y.owner.operation) == 1:
                                            ok = True
                        if ok:
                            result = (k, bound)
    except Exception:
        result = None
    interp._counted_cache[cond_block] = result
    return result


import gc
import os

# Max ops a single mx.compile trace may contain (counted loops unrolled).
# MLX retains every intermediate of a trace, so oversized traces exhaust the
# Metal live-buffer limit (~500k); ~20k ops is comfortably inside it.
# Governs: which loops may unroll into an enclosing trace, which loop
# bodies get compiled, and whether the whole program is compiled.
_TRACE_BUDGET = int(os.environ.get("METALJAX_TRACE_BUDGET", "20000"))
# METALJAX_BODY_COMPILE=0 keeps everything compiled EXCEPT while-loop
# bodies. This is the targeted mitigation for the MLX command-buffer
# corruption (notes/mlx-command-buffer-split.md): the corruption bites
# REPLAYED compiled bodies (first call clean, replays diverge), while
# compiled mains/prefill measure deterministic — and full
# METALJAX_COMPILE=0 also un-compiles the load path, whose eager
# conversion transients balloon (8B load: 67 GB in ~30 s vs ~18 GB
# compiled). Bodies run uncompiled; msl kernels and everything else
# keep their normal paths.
_BODY_COMPILE = os.environ.get("METALJAX_BODY_COMPILE", "1") != "0"
# Max loop iterations unrolled per compiled chunk in the eager-loop path.
# Bounds both trace time and MLX's fused-kernel argument count (long
# unrolled elementwise chains can exhaust Metal kernel argument buffers).
_CHUNK_MAX = int(os.environ.get("METALJAX_CHUNK_MAX", "16"))
# Only chunk small bodies: chunking amortizes the ~50us per-replay input
# evaluation, which is irrelevant for big bodies — while inflating trace
# size and loosening the flush cadence that keeps pending buffers bounded.
_CHUNK_MAX_COST = int(os.environ.get("METALJAX_CHUNK_MAX_COST", "1500"))
_DEBUG = os.environ.get("METALJAX_DEBUG", "") == "1"
# Metal caps LIVE buffers per device at ~499k while MLX's buffer cache is
# bounded by BYTES only — long-running loops over small models accumulate
# tiny cached buffers until metal::malloc dies mid-training (seen after
# ~5k steps of an S32768 run). Clear the cache after roughly this many
# op-units of loop work have been flushed; the engine-level clears (compile
# boundaries / 50k executes) are far too coarse for multi-hour single
# executes. Cadence: the observed worst case accumulated ~0.06 cached
# buffers per op-unit, so 500k units/window grows ~30k buffers — far under
# the limit — while clearing rarely enough to cost <2% on big models
# (100k measured 6-11% slower on emb.1024 configs: every clear forces
# re-mallocs of the working set). The clear-and-retry at the flush sites
# remains the hard backstop for pathological workloads.
_LOOP_CLEAR_COST = int(os.environ.get("METALJAX_LOOP_CLEAR_COST", "500000"))
_MEMDBG = os.environ.get("METALJAX_MEMDBG", "") == "1"
_flushed_cost = 0


def _loop_flush(arrays, cost_units):
    """Sync point inside a loop: evaluate pending work, keep the Metal
    buffer count bounded, and recover once if the limit is hit anyway."""
    global _flushed_cost
    try:
        mx.eval(*arrays)
    except RuntimeError as e:
        if "Resource limit" not in str(e):
            raise
        if _DEBUG:
            print("[metaljax] Metal buffer limit hit at loop flush; "
                  "clearing cache and retrying", flush=True)
        gc.collect()  # dead refcycles pin buffers clear_cache can't free
        mx.clear_cache()
        mx.eval(*arrays)
    _flushed_cost += cost_units
    if _flushed_cost >= _LOOP_CLEAR_COST:
        _flushed_cost = 0
        mx.clear_cache()
        if _MEMDBG:
            print(f"[metaljax-mem] loop clear: "
                  f"active={mx.get_active_memory()}B "
                  f"cache={mx.get_cache_memory()}B", flush=True)


# Iterations of an eagerly replayed loop body per sync point. Deliberately
# a function, and the ONE place the formula lives: the native engine's
# lowering (metaljax.tape) reads it too, and a loop's sync points are where
# MLX's command-buffer split bug bites (notes/mlx-command-buffer-split.md),
# so the two engines drifting apart here would be a correctness question,
# not a performance one. tests/test_command_buffer.py patches it to compare
# cadences -- which is only a detector because both engines take it from
# here.
_PERIOD_MAX = 64


def _flush_period(cost: int) -> int:
    return max(1, min(_PERIOD_MAX, 25_000 // max(cost, 1)))


def _static_start(op, k):
    """Static initial counter value of a while op, else None."""
    v = op.operands[k]
    if isinstance(v, ir.OpResult):
        return _splat_int(v.owner.operation)
    return None


def _scatter_cost(interp, op) -> int:
    """Cost of a scatter's update-computation region. A computed body with
    possibly-duplicate indices is replayed once per update (the sequential
    apply path in ops/gather.py), so charge the whole expansion instead of
    a single body. Must never raise — fall back to one body."""
    body = 0
    for region in op.regions:
        for b in region.blocks:
            body += _block_cost(interp, b)
    try:
        from metaljax.ops.gather import _scatter_combiner, _scatter_dims
        if ("unique_indices" in op.attributes
                and "true" in str(op.attributes["unique_indices"])):
            return body
        if _scatter_combiner(op) != "apply":
            return body
        ivd = _scatter_dims(op)["index_vector_dim"]
        n = 1
        for i, s in enumerate(_ir.tensor_type(op.operands[1]).shape):
            if i != ivd:
                n *= s
        return max(n, 1) * body
    except Exception:
        return body


def _block_cost(interp, block) -> int:
    """Approximate op count when this block is traced (loops unrolled)."""
    cached = interp._cost_cache.get(block)
    if cached is not None:
        return cached
    interp._cost_cache[block] = 1  # break cycles defensively
    cost = 0
    # Ops absorbed into a recognized quantized matmul are never executed, and
    # the fused dot is one kernel. Charging the literal dequant chain (~83
    # units per matmul) is what pushed LLM decode bodies over the budget.
    # A recognized fused attention is ~15 ops of logits-sized work (dot,
    # scale, mask, max, subtract, exp, sum, divide, dot, plus their
    # broadcasts) collapsing to one kernel, so charging it literally would
    # keep prefill and decode bodies out of the trace budget for no reason.
    # Both recognizers are merged into one skip set and one cost table --
    # they are disjoint by construction, and one lookup beats four.
    skip = set()
    roots = {}
    for st, unit in ((interp._qmm, 2), (sdpa.analyze(interp), 3)):
        if st is None or not st.active:
            continue
        skip |= st.skip
        roots.update(dict.fromkeys(st.roots, unit))
    for op in block.operations:
        o = op.operation
        if roots and len(o.results):
            key = o.results[0]
            if key in skip:
                continue
            unit = roots.get(key)
            if unit is not None:
                cost += unit
                continue
        cost += 1
        if o.name == "stablehlo.while":
            if _msl_plan_for(interp, o) is not None:
                cost += 8  # a single generated kernel
                continue
            counted = _analyze_counted(interp, o)
            body = o.regions[1].blocks[0]
            # Unknown trip counts must be PESSIMISTIC: a dynamic-bound
            # 2048-step loop once cost-counted as one body, so the
            # enclosing block passed the trace budget and mx.compile
            # tried to trace the whole thing (buffer exhaustion, and a
            # worker livelocked retrying it).
            trip = 1024
            if counted is not None and isinstance(counted[1], int):
                start = _static_start(o, counted[0])
                if start is not None:
                    trip = max(counted[1] - start, 1)
            cost += trip * _block_cost(interp, body)
        elif o.name == "stablehlo.scatter":
            cost += _scatter_cost(interp, o)
        elif o.name in ("func.call", "stablehlo.composite"):
            attr = "callee" if o.name == "func.call" else "decomposition"
            fn = interp.funcs.get(_callee_name(o, attr))
            if fn is not None:
                cost += _block_cost(interp, fn.regions[0].blocks[0])
        else:
            for region in o.regions:
                for b in region.blocks:
                    cost += _block_cost(interp, b)
    interp._cost_cache[block] = cost
    return cost


def _block_bytes(interp, block) -> int:
    """Approximate device bytes materialized when this block is TRACED.

    The byte analogue of `_block_cost`, and deliberately the same walk:
    loops unrolled (trip x body, pessimistic trip=1024 when the bound is not
    static), func.call/composite callees recursed into, ops a recognizer
    absorbs charged nothing because they never execute -- but its ROOT
    charged what the emission in their place really does build
    (`<recognizer>.emit_bytes`) -- and a loop that became one generated msl
    kernel charged only its outputs (its per-timestep state lives in
    registers, not in buffers).

    A program's own pass-through outputs are NOT here; they belong to
    `program_bytes`, which is what a whole-program decision uses.

    Why a second metric at all: `_block_cost` bounds how many OPS a trace may
    contain, which bounds the Metal live-BUFFER count -- it says nothing about
    how much memory those buffers hold. A program can sit far inside the op
    budget and still make the compiled path hold gigabytes, and the compiled
    peak grows with the program while the eager one does not. One jitted
    `random.normal` -- 365 ops, the maxtext/keras load shape -- at four sizes,
    peak MLX memory compiled vs the same program refused:

        output   4 MB    16 MB    64 MB   256 MB
        compiled 0.18 GB  0.72 GB  2.87 GB 10.00 GB
        eager       --    0.80 GB  1.19 GB  3.25 GB

    That last column is the 256 MB init the eager-memory note had to leave at
    "9.75 GB compiled either way": bounding it is what this gate is for.

    This is a TRAFFIC estimate (every result the block produces), not a
    live-set one, so it reads 3-6x higher than what actually lands: 5.0x on
    every size of that init, 2.8x on three texmo training chunks (11.92 vs
    4.26 GB, 35.87 vs 12.31, 41.90 vs 14.82). mx.compile fusing elementwise
    chains and MLX making broadcasts/reshapes/transposes/slices as views is
    where the slack comes from; modelling that is not attempted, and the
    threshold is calibrated against this estimator rather than against ground
    truth. What the estimate must NOT do is invent bytes that exist in no
    regime -- see `interpreter.op_bytes` on splat constants -- because those
    turn compilation off for programs that are perfectly small.
    """
    cached = interp._bytes_cache.get(block)
    if cached is not None:
        return cached
    interp._bytes_cache[block] = 0  # break cycles defensively
    total = 0
    # Both recognizers' absorbed ops, exactly as _block_cost skips them: they
    # are not executed and never materialize a result.
    #
    # A ROOT is charged its own result AND whatever its `emit` builds on the
    # way there (`<recognizer>.emit_bytes`). Charging it the result alone was
    # a three-orders-of-magnitude hole on rewrite-heavy graphs: on gpt-oss's
    # decode program 156 of 783 ops are absorbed and they carry 99.97% of the
    # block's declared result bytes (28.9 of 28.9 GB), so the estimate came
    # back at 10.4 MB -- a number no threshold can act on. A recognizer that
    # declares no emit_bytes keeps the old accounting, which is right where
    # the fused op really does materialize only its output (sdpa: the whole
    # point of mx.fast.scaled_dot_product_attention is that the [B,H,T,T]
    # scores it absorbs are never written).
    skip = set()
    extra: dict = {}
    for st, mod in ((interp._qmm, qmm), (sdpa.analyze(interp), sdpa)):
        if st is None or not st.active:
            continue
        skip |= st.skip
        fn = getattr(mod, "emit_bytes", None)
        if fn is None:
            continue
        for key, match in st.roots.items():
            try:
                extra[key] = extra.get(key, 0) + fn(match)
            except Exception:
                # A recognizer whose plan cannot be sized must not zero the
                # block; leave the root charged its result and say so.
                if _DEBUG:
                    print("[metaljax] byte estimate: emit_bytes failed for a "
                          f"{type(match).__name__} root", flush=True)
    tcache: dict = {}  # ir.Type -> bytes, per block (see value_bytes)
    for op in block.operations:
        o = op.operation
        if skip and len(o.results) and o.results[0] in skip:
            continue
        total += op_bytes(o, tcache)
        if extra and len(o.results):
            total += extra.get(o.results[0], 0)
        if o.name == "stablehlo.while":
            if _msl_plan_for(interp, o) is not None:
                continue
            counted = _analyze_counted(interp, o)
            body = o.regions[1].blocks[0]
            trip = 1024
            if counted is not None and isinstance(counted[1], int):
                start = _static_start(o, counted[0])
                if start is not None:
                    trip = max(counted[1] - start, 1)
            total += trip * _block_bytes(interp, body)
        elif o.name in ("func.call", "stablehlo.composite"):
            attr = "callee" if o.name == "func.call" else "decomposition"
            fn = interp.funcs.get(_callee_name(o, attr))
            if fn is not None:
                total += _block_bytes(interp, fn.regions[0].blocks[0])
        else:
            # if/case: charge every branch (which one runs is data).
            for region in o.regions:
                for b in region.blocks:
                    total += _block_bytes(interp, b)
    interp._bytes_cache[block] = total
    return total


def _passthrough_bytes(block) -> int:
    """Bytes of the block's results that no operation in it produces.

    A program whose output IS one of its inputs still materializes that
    output: XLA's no-alias contract makes engine.execute copy it (see the
    `forwarded_outputs` pass there). The op walk cannot see this, because
    there is no op -- `jit_stage`, the keras streaming loader's per-variable
    move, is one `func.return %arg0` and was estimated at 0.0 MB while every
    call really did copy 1.1 GB of embedding.

    Charged at the WHOLE-PROGRAM level only, never inside `_block_bytes`'
    recursion: a loop body returns its carries, most of which are block
    arguments passed through untouched, and charging those per iteration
    (x trip) would inflate every scan in texmo and move the chunk sizing
    the command-buffer lottery is pinned to.
    """
    args = set(block.arguments)
    if not args:
        return 0
    total = 0
    tcache: dict = {}
    for op in block.operations:
        o = op.operation
        if not o.name.endswith(".return") or o.results:
            continue
        # Counted once PER RETURNED POSITION, not per distinct value: two
        # outputs may not share a buffer either, so returning one argument
        # twice really is two copies.
        for v in o.operands:
            if v in args:
                total += value_bytes(v, tcache)
    return total


def program_bytes(interp, block) -> int:
    """`_block_bytes` for a whole program: the walk plus pass-through outputs."""
    return _block_bytes(interp, block) + _passthrough_bytes(block)


# Bytes a single mx.compile trace may materialize. The op-count budget above
# bounds the live-buffer COUNT; this bounds their SIZE, and the two are
# independent: every tensor of a command buffer stays allocated until it
# completes, so a program well inside 20k ops can still hold tens of GB.
# The class this catches is high-bytes/low-op-count: big-model load, warmup
# and conversion programs (a 30B+ checkpoint conversion is a few thousand
# converts moving 70+ GB; a jitted init lowers 64 MB of weights into ~15 GB
# of traffic at 365 ops) and diffusion sampler bodies. Declining to compile
# hands the program to the eager path, whose peak IS bounded -- by last-use
# pruning and the byte-denominated flush (notes/eager-memory-2026-08.md).
#
# THE DEFAULT IS MEASURED, and it is much larger than it looks like it should
# be, because this estimator counts TRAFFIC (see _block_bytes) while what
# lands is the live set -- 3-6x less, depending on shape. 64 GB here means a
# compiled program may hold roughly 10 GB (init-shaped) to 23 GB (texmo
# training chunk) before it is pushed onto the eager path.
#
# The distribution it was picked from -- every compile decision metaljax says
# YES to today, logged at all four gate sites while the workload ran:
#
#   pytest suite (589 tests)                       max   1.19 GB
#   tests/data qwen3 prefill (shrunk, 8 layers)          1.99 GB
#   gpt-oss-20b decode/prefill main, 2 layers     0.03-0.21 GB (see below)
#   texmo whole-main, 8-step chunk (gate harness)  max   6.11 GB
#     the same, scaled to the 20-step benchmark chunk   ~15.3 GB (derived)
#   texmo training step, inner loop unrolled       max  27.4 GB (lstm.512 b32)
#   texmo chunked replay at K=16                   max  41.9 GB (db17 b256 L512)
#   ---- 64 GB ----
#   Qwen3-8B prefill, 36 layers                        139.8 GB
#   Qwen3 maxtext parameter init scan                  695.5 GB
#
# The gpt-oss row is in that list because it is where the estimator was found
# to be UNDER-reporting rather than over: its expert-gather rewrite absorbs
# 99.97% of the block's declared result bytes, and until the roots were
# charged their emission (see _block_bytes) the whole program read 10.4 MB
# against a measured 55.8 MB first-call trace. It now reads 34.1 MB there and
# 214.0 MB at a 64-token prompt against 107.1 MB measured -- above the truth,
# which is the side of it a gate can be built on. And the number stays small:
# the memory wave that guard-kills that row is the qmm PACKING PROLOGUE
# (~8 GB per weight, MLX's cache never cleared between matches), which runs
# before any compile decision and which no trace budget can bound.
#
# (The last two are over the op budget as well, so they were already eager;
# they are here as the scale of what must never be traced. The class this
# gate adds cover for is the one the op budget misses -- see above.)
#
# 64 GB is 1.5x above the largest program metaljax compiles today and 2.1x
# below the smallest it must refuse. Headroom on the low side is not
# politeness: firing the gate on a texmo body would also change K, and with
# it where a loop's sync points fall -- which is a correctness lottery on
# MLX 0.32, not just a perf question (notes/mlx-command-buffer-split.md).
# For the record, tightening is CHEAP where it was measured (db17 K 16->4:
# 7.271 -> 7.257 ms/step, i.e. free; lstm.512 b32 with its whole training
# step refused: 29.60 -> 29.85 ms/step, +0.8%), so a workload that needs a
# tighter bound can have one -- but it should be measured, not assumed, and
# both tests/test_command_buffer.py cases rerun. 0 disables the gate.
_COMPILE_BYTES_MB = int(os.environ.get("METALJAX_COMPILE_BYTES_MB", "65536"))
_COMPILE_BYTES = max(_COMPILE_BYTES_MB, 0) * 2**20


def _bytes_ok(interp, block, mult=1, what="block", whole=False) -> bool:
    """Whether tracing `mult` copies of `block` fits the byte budget.

    Every compile decision asks this: the whole-main compile (mult=1,
    whole=True), a compiled while body (mult=repeat), unrolling a counted
    loop into an enclosing trace (mult=trip), and how many iterations one
    chunked replay may hold. All four multiply the SAME per-block estimate,
    which is what keeps them coherent -- a body that may be compiled per step
    is not thereby licensed to be unrolled a thousand times into a trace.
    `whole` adds the pass-through outputs, which only a program has.
    """
    if _COMPILE_BYTES <= 0:
        return True
    per = program_bytes(interp, block) if whole else _block_bytes(interp, block)
    nb = mult * per
    if nb <= _COMPILE_BYTES:
        return True
    _note_gated(interp, block, nb, mult, what)
    return False


def _bytes_chunks(interp, block) -> int:
    """How many copies of `block` a single trace may hold (>=1 always: the
    single-step path is gated by `_bytes_ok` in _body_fn instead)."""
    if _COMPILE_BYTES <= 0:
        return 1 << 30
    return max(1, _COMPILE_BYTES // max(_block_bytes(interp, block), 1))


def _note_gated(interp, block, nb, mult, what):
    """Record a fired gate on the interpreter (see Interpreter._bytes_gated)
    and report it once per block, not once per call."""
    seen = interp._bytes_gated
    first = block not in seen
    seen.add(block)
    if _DEBUG and first:
        print(f"[metaljax] compile bytes gate: {what} would trace "
              f"{nb / 2**20:.0f} MB (x{mult}) > {_COMPILE_BYTES_MB} MB "
              f"(METALJAX_COMPILE_BYTES_MB); running eagerly", flush=True)


def _msl_plan_for(interp, op):
    """Build (and cache) an msl_scan Plan for a statically-counted loop,
    or None. Used by purity/cost analysis and by _while dispatch."""
    counted = _analyze_counted(interp, op)
    if counted is None or not isinstance(counted[1], int):
        if _DEBUG and counted is not None:
            print(f"[metaljax] msl_scan: skipped (captured bound "
                  f"{type(counted[1]).__name__})", flush=True)
        return None
    k, bound = counted
    start = _static_start(op, k)
    if start is None:
        if _DEBUG:
            print("[metaljax] msl_scan: skipped (dynamic start)", flush=True)
        return None
    trip = bound - start
    if trip <= 0:
        return None
    from metaljax import msl_scan
    if not msl_scan.ENABLED or interp._no_msl:
        return None
    body = op.regions[1].blocks[0]
    key = (body, trip, start)
    if any(_ir.is_token(a.type) for a in body.arguments):
        # ordered-effect carries: not kernel-representable, and the
        # analyzer would raise deep inside on the non-tensor type
        interp._msl_cache[key] = None
        return None
    plan = interp._msl_cache.get(key, "miss")
    if plan == "miss":
        try:
            plan = msl_scan.build_plan(interp, body, k, trip, start)
            if msl_scan._DEBUG:
                print(f"[metaljax] msl_scan: compiled plan trip={trip} "
                      f"mode={plan.mode} lanes={plan.N} "
                      f"states={len(plan.states)} "
                      f"stacked={len(plan.stacked)} "
                      f"packed={len(plan._packed)}", flush=True)
        except Exception as e:
            if msl_scan._DEBUG:
                print(f"[metaljax] msl_scan: not eligible ({e})", flush=True)
            plan = None
        interp._msl_cache[key] = plan
    return plan


def _while_traceable(interp, op) -> bool:
    """True when this while can be unrolled inside an mx.compile trace:
    statically counted, pure body, and small enough for the trace budget."""
    body_block = op.regions[1].blocks[0]
    cached = interp._traceable_cache.get(body_block)
    if cached is not None:
        return cached
    interp._traceable_cache[body_block] = False  # break recursion
    if _msl_plan_for(interp, op) is not None:
        interp._traceable_cache[body_block] = True
        return True
    ok = False
    counted = _analyze_counted(interp, op)
    if counted is not None and isinstance(counted[1], int):
        k, bound = counted
        start = _static_start(op, k)
        if start is not None:
            trip = max(bound - start, 0)
            if (
                trip * _block_cost(interp, body_block) <= _TRACE_BUDGET
                and interp.block_is_pure(body_block)
                # trip copies of the body land in the enclosing trace, and
                # the tracer holds every one of their intermediates. Asked
                # LAST, so a gate that fires means "everything else said
                # compile" -- which is what makes the debug line, and the
                # `_bytes_gated` record, mean something.
                and _bytes_ok(interp, body_block, trip, "unroll-in-trace")
            ):
                ok = True
    interp._traceable_cache[body_block] = ok
    return ok


# Let the interpreter's purity analysis see through unrollable loops.
from metaljax.interpreter import Interpreter as _Interpreter
_Interpreter.while_traceable_hook = staticmethod(_while_traceable)


def _underived_outputs(block, free):
    """Indices of terminator operands with NO data path from any block arg
    or capture. When traced, such outputs are baked by MLX's compiler into
    a constants table KEYED BY VALUE — two equal-valued constant outputs
    collide and the compiled call dies with unordered_map::at. (Repro:
    mx.compile(lambda x: (x+1, mx.array(.9), mx.array(.9))) fails.)"""
    ops = [o.operation for o in block.operations]
    derived = set(block.arguments) | set(free)
    for o in ops[:-1]:
        dep = any(v in derived for v in o.operands)
        if not dep:
            for region in o.regions:
                for b in region.blocks:
                    if any(v in derived for v in free_values(b)):
                        dep = True
                        break
                if dep:
                    break
        if dep:
            for r in o.results:
                derived.add(r)
    return [i for i, v in enumerate(ops[-1].operands) if v not in derived]


def _anchor_outputs(outs, args, underived):
    """Give constant outputs a bitwise-exact data dependency on an input:
    where(x==x, out, out) == out for every bit pattern (both branches are
    `out`; a NaN anchor merely flips which identical branch is taken), but
    the result is a computed node, not a bakeable constant."""
    if not underived or not args:
        return outs
    anchor = None
    for a in args:
        if a.size:
            anchor = mx.reshape(a, (-1,))[:1] == mx.reshape(a, (-1,))[:1]
            break
    if anchor is None:
        return outs
    outs = list(outs)
    for i in underived:
        if i < len(outs):
            outs[i] = mx.where(mx.reshape(anchor, ()), outs[i], outs[i])
    return tuple(outs)


# Stand-in for a capture the quantized-matmul rewrite absorbed (see below).
_ABSORBED = mx.zeros((), mx.uint8)


def _captures(interp, env, free):
    """A body's captured values, plus the packed weights of any quantized
    matmul it contains. The packed arrays MUST travel as explicit inputs:
    mx.compile bakes arrays captured from an enclosing scope by value, so a
    repack would never reach an already-compiled body."""
    # EVERY recognizer's skip set, not just qmm's: `free_values` is purely
    # syntactic, so an op absorbed in the enclosing block still shows up as
    # a capture of the body, and reading it from `env` would raise.
    skip = set()
    for st in (interp._qmm, sdpa.analyze(interp)):
        if st is not None and st.active:
            skip |= st.skip
    caps = []
    for v in free:
        a = env.get(v)
        if a is None and v in skip:
            # A reconstruction op that jax hoisted out of the loop: it was
            # skipped in the enclosing block, and every consumer of it
            # inside the body is skipped too, so nothing can read this.
            # Substituting keeps the compiled body's arity stable.
            a = _ABSORBED
        elif a is None:
            a = env[v]  # missing for some other reason: raise as before
        caps.append(a)
    caps.extend(qmm.values(interp))
    return caps


def _body_fn(interp, body_block, compile_body, repeat=1):
    """Cached executor for `repeat` iterations of a while body:
    fn(*vals, *captures) -> vals."""
    key = (body_block, compile_body, repeat)
    entry = interp._body_cache.get(key)
    if entry is None:
        free = free_values(body_block)
        nvals = len(list(body_block.arguments))
        nfree = nvals + len(free)

        def raw(*flat):
            vals = list(flat[:nvals])
            captures = dict(zip(free, flat[nvals:nfree]))
            token = qmm.push(interp, flat[nfree:])
            try:
                for _ in range(repeat):
                    vals = interp.run_block(body_block, vals, captures)
            finally:
                qmm.pop(interp, token)
            return tuple(vals)

        fn = raw
        compiled = False
        if (
            compile_body
            and COMPILE_ENABLED
            and _BODY_COMPILE
            and body_block not in interp._no_body_compile
            and interp.block_is_pure(body_block)
            and repeat * _block_cost(interp, body_block) <= _TRACE_BUDGET
            # `repeat` iterations are traced into ONE graph, so the byte
            # budget covers the whole chunk, not one iteration.
            and _bytes_ok(interp, body_block, repeat, "while body")
        ):
            underived = _underived_outputs(body_block, free)

            def traced(*flat):
                prev = interp._in_trace
                interp._in_trace = True
                try:
                    return _anchor_outputs(raw(*flat), flat, underived)
                finally:
                    interp._in_trace = prev

            fn = mx.compile(traced)
            compiled = True
        entry = (fn, free, compiled)
        interp._body_cache[key] = entry
    return entry


class _BodyRunner:
    """Runs a while body, recovering from the ways MLX's compiled path can
    fail at CALL time. Every recovery simply redoes the iteration: bodies
    are pure, and a failed call leaves the caller's carry untouched.

    - "Resource limit": Metal ran out of buffer handles mid-body. Purge the
      cache and retry. BOUNDED: after two failures force the uncompiled
      body (retrying an oversized mx.compile trace forever once livelocked
      a worker for hours); if even that cannot fit, surface the error.
    - anything else out of a COMPILED call (e.g. unordered_map::at on
      graphs with unused inputs): fall back to the uncompiled body for the
      rest of this program's life.

    A compiled call only BUILDS the graph; MLX generates and compiles the
    fused Metal kernels at eval, so a whole class of failures ("Too many
    inputs/outputs fused ...": a fused elementwise chain over more than
    Metal's 31 argument buffers) lands at the next sync point instead of
    at the call — by which time the carry has advanced past the iteration
    a redo could repair. The first call of each freshly bound compiled
    body is therefore evaluated synchronously, so those failures surface
    HERE, with `vals` still the caller's untouched input.
    """

    def __init__(self, interp, body_block, env, repeat=1):
        self.interp = interp
        self.block = body_block
        self.env = env
        self.repeat = repeat
        self._limit_retries = 0
        self._bind()

    def _bind(self):
        self.fn, free, self.compiled = _body_fn(
            self.interp, self.block, compile_body=True, repeat=self.repeat)
        self.captures = _captures(self.interp, self.env, free)
        # Probe the freshly bound compiled body once (see class docstring).
        # One sync per loop ENTRY: the shapes are fixed for the life of the
        # loop, so one buildable call proves every later one. Uncompiled
        # bodies build no kernels of their own and need no probe.
        self._probe = self.compiled

    def _drop_compiled(self):
        # _body_fn consults _no_body_compile, so rebinding now yields the
        # uncompiled body (and drops the compiled entry that failed).
        self.interp._no_body_compile.add(self.block)
        self.interp._body_cache.pop((self.block, True, self.repeat), None)
        self._bind()

    def step(self, vals):
        """One iteration: the next carry, or None when it must be redone."""
        try:
            out = list(self.fn(*vals, *self.captures))
            if self._probe:
                # Inside the try on purpose: this is where a body MLX
                # cannot generate kernels for reports itself. Synchronous
                # (not async_eval) — a Metal build error raised on a
                # worker thread aborts the process.
                mx.eval(*out)
                self._probe = False
            self._limit_retries = 0
            return out
        except (RuntimeError, IndexError, ValueError) as e:
            if isinstance(e, RuntimeError) and "Resource limit" in str(e):
                if _DEBUG:
                    print("[metaljax] Metal buffer limit hit in while "
                          "body; clearing cache and retrying", flush=True)
                gc.collect()
                mx.clear_cache()
                self._limit_retries += 1
                if self._limit_retries == 2 and self.compiled:
                    self._drop_compiled()
                elif self._limit_retries > 3:
                    raise
                return None
            if not self.compiled:
                raise
            if _DEBUG:
                print("[metaljax] compiled while body failed; "
                      "retrying eagerly", flush=True)
            self._drop_compiled()
            return None

    def run_one(self, vals):
        """One iteration, retrying in place until it lands or gives up."""
        while True:
            out = self.step(vals)
            if out is not None:
                return out


@register("stablehlo.while")
def _while(interp, op, ins, env):
    cond_block = op.regions[0].blocks[0]
    body_block = op.regions[1].blocks[0]

    counted = _analyze_counted(interp, op)
    if counted is not None:
        k, bound = counted
        if isinstance(bound, int):
            n = bound
        elif isinstance(bound, tuple):  # ("carry", j): invariant loop state
            n = int(ins[bound[1]].item())
        elif bound in env:
            n = int(env[bound].item())
        else:
            # Bound defined beyond our capture scope (deeply nested
            # regions): treat as a dynamic loop rather than KeyError.
            counted = None
    if counted is not None:
        start = int(ins[k].item())
        trip = max(n - start, 0)
        from metaljax import msl_scan
        res = msl_scan.try_run(interp, op, ins, env, trip, start, k)
        if res is not None:
            return res
        if interp._in_trace:
            if trip > 64:
                # Reachable only when a loop promised as traceable (an
                # MSL plan existed at cost time) fell through at run
                # time: unrolling thousands of iterations into the
                # enclosing trace exhausts Metal's buffer budget. Abort
                # the trace; the engine falls back to the eager path.
                raise RuntimeError(
                    f"metaljax: refusing to unroll trip={trip} inside a "
                    "trace (msl plan fell through)")
            # An enclosing mx.compile is tracing us (only possible for
            # unrollable loops): inline the iterations into that graph.
            if _DEBUG:
                print(f"[metaljax] while(unroll-in-trace): trip={trip} "
                      f"cost={_block_cost(interp, body_block)}", flush=True)
            fn, free, _ = _body_fn(interp, body_block, compile_body=False)
            captures = _captures(interp, env, free)
            vals = list(ins)
            for _ in range(trip):
                vals = list(fn(*vals, *captures))
            return vals
        # Eager loop. Chained replays are expensive (~50us each: a compiled
        # call evaluates its inputs), so unroll as many iterations as the
        # trace budget allows into each compiled chunk, replaying trip/K
        # chunks instead of trip single steps. Deferring too much work can
        # exhaust MLX's live-buffer limit (each pending replay pins its
        # internal buffers), so flush the graph often enough that pending
        # buffers stay bounded; not in a trace here, so eval is safe.
        cost = _block_cost(interp, body_block)
        chunkable = (
            COMPILE_ENABLED
            and cost <= _CHUNK_MAX_COST
            and interp.block_is_pure(body_block)
            and body_block not in interp._no_chunk
        )
        K = 1
        if chunkable:
            # ...and no more iterations than the byte budget allows either:
            # a chunk's intermediates are all live until it completes. K
            # falling to 1 hands the body to the single-step path, whose
            # own compile is gated in _body_fn.
            K = max(1, min(trip, _TRACE_BUDGET // max(cost, 1), _CHUNK_MAX,
                           _bytes_chunks(interp, body_block)))
        period = _flush_period(cost)
        if _DEBUG:
            print(f"[metaljax] while: trip={trip} cost={cost} K={K} "
                  f"period={period} pure={interp.block_is_pure(body_block)} "
                  f"bytes/iter={_block_bytes(interp, body_block) / 2**20:.1f}MB",
                  flush=True)
        if K > 1:
            try:
                return _run_chunked(interp, body_block, env, ins, trip, K, cost)
            except RuntimeError as e:
                # MLX's compiler can reject big fused traces (e.g. "Too many
                # inputs/outputs fused..." on long elementwise chains from
                # linear recurrences). Fall back to single-step replays.
                if _DEBUG:
                    print(f"[metaljax] chunked loop failed ({e}); "
                          f"falling back to single-step", flush=True)
                interp._no_chunk.add(body_block)
        runner = _BodyRunner(interp, body_block, env)
        vals = list(ins)
        for i in range(1, trip + 1):
            vals = runner.run_one(vals)
            if i % period == 0:
                _loop_flush(vals, period * cost)
        return vals

    # Dynamic (non-counted) loop: evaluate the condition each iteration.
    # The BODY still gets compiled — a data-dependent trip count says
    # nothing about the body, and interpreting it op by op is what made
    # LLM decode (one keras sampler step = a whole transformer forward)
    # Python-dispatch-bound. The cond stays eager: it ends in .item().
    dyn_cost = _block_cost(interp, body_block)
    runner = _BodyRunner(interp, body_block, env)
    if _DEBUG:
        print(f"[metaljax] while(fallback-dynamic): cost={dyn_cost} "
              f"compiled={runner.compiled}", flush=True)
    vals = list(ins)
    while True:
        (pred,) = interp.run_block(cond_block, vals, env)
        if not bool(pred.item()):
            return vals
        vals = runner.run_one(vals)
        # Flush the carry, not nothing: the cond only forces the values it
        # reads, so anything else in the carry (a KV cache) would pile up
        # as unevaluated graph across iterations.
        _loop_flush(vals, dyn_cost)


def _run_chunked(interp, body_block, env, ins, trip, K, cost):
    fn, free, _ = _body_fn(interp, body_block, compile_body=True, repeat=K)
    captures = _captures(interp, env, free)
    vals = list(ins)
    # Async-flush each chunk (a blocking sync per chunk serializes CPU and
    # GPU); block only often enough to bound pending buffers (~4-5 live
    # buffers per traced op, keep well under Metal's ~500k cap).
    sync_every = max(1, 75_000 // max(K * cost, 1))
    for i in range(trip // K):
        vals = list(fn(*vals, *captures))
        if (i + 1) % sync_every == 0:
            _loop_flush(vals, sync_every * K * cost)
        else:
            mx.async_eval(*vals)
    rem = trip % K
    if rem:
        fn1, free1, _ = _body_fn(interp, body_block, compile_body=True)
        captures1 = _captures(interp, env, free1)
        for _ in range(rem):
            vals = list(fn1(*vals, *captures1))
    _loop_flush(vals, (trip % max(sync_every * K, 1)) * cost)
    return vals


@register("stablehlo.if")
def _if(interp, op, ins, env):
    (pred,) = ins
    region = op.regions[0] if bool(pred.item()) else op.regions[1]
    return interp.run_block(region.blocks[0], [], env)


@register("stablehlo.case")
def _case(interp, op, ins, env):
    (index,) = ins
    i = int(index.item())
    i = min(max(i, 0), len(op.regions) - 1)
    return interp.run_block(op.regions[i].blocks[0], [], env)


@register("stablehlo.optimization_barrier")
def _optimization_barrier(interp, op, ins, env):
    return list(ins)


# Single-device: sharding ops are identities.
@register("sdy.sharding_constraint", "sdy.reshard")
def _sharding_identity(interp, op, ins, env):
    return list(ins)


@register("stablehlo.custom_call")
def _custom_call(interp, op, ins, env):
    target = _ir.str_attr(op, "call_target_name")
    if target in ("Sharding", "annotate_device_placement"):
        return list(ins)
    if target == "shape_assertion":
        return []
    if target == "ApproxTopK":
        from metaljax.ops import sort as _sort_mod
        return _sort_mod.approx_top_k(op, ins)
    from metaljax.ops import lapack
    res = lapack.run_target(interp, op, ins, env)
    if res is not None:
        return res
    raise UnsupportedOpError(f"custom_call target {target!r} not implemented")
