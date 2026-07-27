"""Control flow, calls, composites, and pass-through ops."""

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.interpreter import (
    COMPILE_ENABLED,
    UnsupportedOpError,
    free_values,
    register,
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


import os

# Max ops a single mx.compile trace may contain (counted loops unrolled).
# MLX retains every intermediate of a trace, so oversized traces exhaust the
# Metal live-buffer limit (~500k); ~20k ops is comfortably inside it.
# Governs: which loops may unroll into an enclosing trace, which loop
# bodies get compiled, and whether the whole program is compiled.
_TRACE_BUDGET = int(os.environ.get("METALJAX_TRACE_BUDGET", "20000"))
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
        mx.clear_cache()
        mx.eval(*arrays)
    _flushed_cost += cost_units
    if _flushed_cost >= _LOOP_CLEAR_COST:
        _flushed_cost = 0
        mx.clear_cache()


def _static_start(op, k):
    """Static initial counter value of a while op, else None."""
    v = op.operands[k]
    if isinstance(v, ir.OpResult):
        return _splat_int(v.owner.operation)
    return None


def _block_cost(interp, block) -> int:
    """Approximate op count when this block is traced (loops unrolled)."""
    cached = interp._cost_cache.get(block)
    if cached is not None:
        return cached
    interp._cost_cache[block] = 1  # break cycles defensively
    cost = 0
    for op in block.operations:
        o = op.operation
        cost += 1
        if o.name == "stablehlo.while":
            if _msl_plan_for(interp, o) is not None:
                cost += 8  # a single generated kernel
                continue
            counted = _analyze_counted(interp, o)
            body = o.regions[1].blocks[0]
            trip = 1
            if counted is not None and isinstance(counted[1], int):
                start = _static_start(o, counted[0])
                if start is not None:
                    trip = max(counted[1] - start, 1)
            cost += trip * _block_cost(interp, body)
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
    if not msl_scan.ENABLED:
        return None
    body = op.regions[1].blocks[0]
    key = (body, trip, start)
    plan = interp._msl_cache.get(key, "miss")
    if plan == "miss":
        try:
            plan = msl_scan.Plan(interp, body, k, trip, start)
            if msl_scan._DEBUG:
                print(f"[metaljax] msl_scan: compiled plan trip={trip} "
                      f"lanes={plan.N} states={len(plan.states)} "
                      f"stacked={len(plan.stacked)}", flush=True)
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


def _body_fn(interp, body_block, compile_body, repeat=1):
    """Cached executor for `repeat` iterations of a while body:
    fn(*vals, *captures) -> vals."""
    key = (body_block, compile_body, repeat)
    entry = interp._body_cache.get(key)
    if entry is None:
        free = free_values(body_block)
        nvals = len(list(body_block.arguments))

        def raw(*flat):
            vals = list(flat[:nvals])
            captures = dict(zip(free, flat[nvals:]))
            for _ in range(repeat):
                vals = interp.run_block(body_block, vals, captures)
            return tuple(vals)

        fn = raw
        compiled = False
        if (
            compile_body
            and COMPILE_ENABLED
            and body_block not in interp._no_body_compile
            and interp.block_is_pure(body_block)
            and repeat * _block_cost(interp, body_block) <= _TRACE_BUDGET
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
            # An enclosing mx.compile is tracing us (only possible for
            # unrollable loops): inline the iterations into that graph.
            if _DEBUG:
                print(f"[metaljax] while(unroll-in-trace): trip={trip} "
                      f"cost={_block_cost(interp, body_block)}", flush=True)
            fn, free, _ = _body_fn(interp, body_block, compile_body=False)
            captures = [env[v] for v in free]
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
            K = max(1, min(trip, _TRACE_BUDGET // max(cost, 1), _CHUNK_MAX))
        period = max(1, min(64, 25_000 // max(cost, 1)))
        if _DEBUG:
            print(f"[metaljax] while: trip={trip} cost={cost} K={K} "
                  f"period={period} pure={interp.block_is_pure(body_block)}",
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
        fn, free, compiled = _body_fn(interp, body_block, compile_body=True)
        captures = [env[v] for v in free]
        vals = list(ins)
        i = 0
        while i < trip:
            try:
                vals = list(fn(*vals, *captures))
            except (RuntimeError, IndexError, ValueError) as e:
                if isinstance(e, RuntimeError) and "Resource limit" in str(e):
                    # Metal ran out of buffer handles mid-body; `vals` is
                    # only rebound on success, so purge the cache and redo
                    # the iteration on the same path.
                    if _DEBUG:
                        print("[metaljax] Metal buffer limit hit in while "
                              "body; clearing cache and retrying", flush=True)
                    mx.clear_cache()
                    continue
                # MLX's compiled path can fail at call time (e.g.
                # unordered_map::at on graphs with unused inputs). The body
                # is pure and the call left `vals` untouched: redo this
                # iteration with the uncompiled body.
                if not compiled:
                    raise
                if _DEBUG:
                    print("[metaljax] compiled while body failed; "
                          "retrying eagerly", flush=True)
                interp._no_body_compile.add(body_block)
                interp._body_cache.pop((body_block, True, 1), None)
                fn, free, compiled = _body_fn(
                    interp, body_block, compile_body=True)
                captures = [env[v] for v in free]
                continue
            i += 1
            if i % period == 0:
                _loop_flush(vals, period * cost)
        return vals

    # Dynamic (non-counted) loop: evaluate the condition each iteration.
    dyn_cost = _block_cost(interp, body_block)
    if _DEBUG:
        print(f"[metaljax] while(fallback-dynamic): cost={dyn_cost}",
              flush=True)
    vals = list(ins)
    while True:
        (pred,) = interp.run_block(cond_block, vals, env)
        if not bool(pred.item()):
            return vals
        vals = interp.run_block(body_block, vals, env)
        _loop_flush((), dyn_cost)


def _run_chunked(interp, body_block, env, ins, trip, K, cost):
    fn, free, _ = _body_fn(interp, body_block, compile_body=True, repeat=K)
    captures = [env[v] for v in free]
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
        captures1 = [env[v] for v in free1]
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
    from metaljax.ops import lapack
    res = lapack.run_target(interp, op, ins, env)
    if res is not None:
        return res
    raise UnsupportedOpError(f"custom_call target {target!r} not implemented")
