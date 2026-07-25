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


def _while_traceable(interp, op) -> bool:
    """True when this while can be unrolled inside an mx.compile trace:
    statically counted, pure body, and small enough for the trace budget."""
    body_block = op.regions[1].blocks[0]
    cached = interp._traceable_cache.get(body_block)
    if cached is not None:
        return cached
    interp._traceable_cache[body_block] = False  # break recursion
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
        if (
            compile_body
            and COMPILE_ENABLED
            and interp.block_is_pure(body_block)
            and repeat * _block_cost(interp, body_block) <= _TRACE_BUDGET
        ):
            def traced(*flat):
                prev = interp._in_trace
                interp._in_trace = True
                try:
                    return raw(*flat)
                finally:
                    interp._in_trace = prev

            fn = mx.compile(traced)
        entry = (fn, free)
        interp._body_cache[key] = entry
    return entry


@register("stablehlo.while")
def _while(interp, op, ins, env):
    cond_block = op.regions[0].blocks[0]
    body_block = op.regions[1].blocks[0]

    counted = _analyze_counted(interp, op)
    if counted is not None:
        k, bound = counted
        n = bound if isinstance(bound, int) else int(env[bound].item())
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
            fn, free = _body_fn(interp, body_block, compile_body=False)
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
        fn, free = _body_fn(interp, body_block, compile_body=True)
        captures = [env[v] for v in free]
        vals = list(ins)
        for i in range(trip):
            vals = list(fn(*vals, *captures))
            if (i + 1) % period == 0:
                mx.eval(*vals)
        return vals

    # Dynamic (non-counted) loop: evaluate the condition each iteration.
    if _DEBUG:
        print(f"[metaljax] while(fallback-dynamic): "
              f"cost={_block_cost(interp, body_block)}", flush=True)
    vals = list(ins)
    while True:
        (pred,) = interp.run_block(cond_block, vals, env)
        if not bool(pred.item()):
            return vals
        vals = interp.run_block(body_block, vals, env)


def _run_chunked(interp, body_block, env, ins, trip, K, cost):
    fn, free = _body_fn(interp, body_block, compile_body=True, repeat=K)
    captures = [env[v] for v in free]
    vals = list(ins)
    # Async-flush each chunk (a blocking sync per chunk serializes CPU and
    # GPU); block only often enough to bound pending buffers (~4-5 live
    # buffers per traced op, keep well under Metal's ~500k cap).
    sync_every = max(1, 75_000 // max(K * cost, 1))
    for i in range(trip // K):
        vals = list(fn(*vals, *captures))
        if (i + 1) % sync_every == 0:
            mx.eval(*vals)
        else:
            mx.async_eval(*vals)
    rem = trip % K
    if rem:
        fn1, free1 = _body_fn(interp, body_block, compile_body=True)
        captures1 = [env[v] for v in free1]
        for _ in range(rem):
            vals = list(fn1(*vals, *captures1))
    mx.eval(*vals)
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
    raise UnsupportedOpError(f"custom_call target {target!r} not implemented")
