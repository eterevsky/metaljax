"""jax_plugins entry for the metaljax Metal PJRT plugin.

jax discovers this via the `jax_plugins` namespace package and calls
initialize() at backend-discovery time.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _library_path() -> Path | None:
    env = os.environ.get("METALJAX_PLUGIN_PATH")
    if env:
        return Path(env)
    # Packaged location (works for wheel and editable installs alike).
    try:
        import metaljax
        p = Path(metaljax.__file__).parent / "lib" / "libmetal_pjrt.dylib"
        if p.exists():
            return p
    except ImportError:
        pass
    # Repo layout fallback: <root>/src/jax_plugins/metal/__init__.py
    root = Path(__file__).resolve().parents[3]
    p = root / "plugin" / "build" / "libmetal_pjrt.dylib"
    return p if p.exists() else None


def initialize():
    path = _library_path()
    if path is None or not path.exists():
        logger.warning("metaljax plugin dylib not found; run plugin/build.sh")
        return
    import jax._src.xla_bridge as xb

    # Keep CPU (priority 0) as the default backend; select metal explicitly
    # via JAX_PLATFORMS=metal / jax.config.update('jax_platforms', 'metal')
    # or jax.devices('metal').
    xb.register_plugin("metal", priority=-1, library_path=str(path))
    _register_linalg_lowerings()


def _register_linalg_lowerings():
    """eigh/svd/eig have no generic StableHLO lowering — jax only
    registers per-platform rules, and those reject bf16/f16 outright
    (LAPACK routine tables). Emit our own custom_calls instead, with the
    primitive's declared result types; metaljax.ops.lapack implements
    them on the host (upcasting halves to f32)."""
    try:
        from jax._src.interpreters import mlir
        from jax._src.lax import linalg as ll

        def emit(ctx, target, operands, config=""):
            out_types = [mlir.aval_to_ir_type(ctx.module_context, a)
                         for a in ctx.avals_out]
            kwargs = {}
            if any(not isinstance(d, int)
                   for a in ctx.avals_out for d in a.shape):
                # symbolic dims (jax.export shape polymorphism): declare
                # result shapes so the refine-polymorphic-shapes pass can
                # staticize our custom calls
                kwargs["result_shapes"] = [
                    mlir.eval_dynamic_shape_as_tensor(ctx, a.shape)
                    for a in ctx.avals_out]
            op = mlir.custom_call(
                target, result_types=out_types, operands=operands,
                backend_config=config, **kwargs)
            return op.results

        def eigh_rule(ctx, operand, *, lower, sort_eigenvalues,
                      subset_by_index, algorithm=None, **_):
            n = ctx.avals_in[0].shape[-1]
            if not (subset_by_index is None or subset_by_index == (0, n)):
                raise NotImplementedError("eigh subset_by_index on metal")
            return emit(ctx, "metaljax_eigh", [operand],
                        "L" if lower else "U")

        def svd_rule(ctx, operand, *, full_matrices, compute_uv, **_):
            return emit(ctx, "metaljax_svd", [operand])

        def eig_rule(ctx, operand, *, compute_left_eigenvectors,
                     compute_right_eigenvectors, **_):
            cfg = ("L" if compute_left_eigenvectors else "") + \
                  ("R" if compute_right_eigenvectors else "")
            return emit(ctx, "metaljax_eig", [operand], cfg)

        def schur_rule(ctx, operand, **_):
            return emit(ctx, "metaljax_schur", [operand])

        def hessenberg_rule(ctx, operand, **_):
            return emit(ctx, "metaljax_hessenberg", [operand])

        def tridiagonal_rule(ctx, operand, *, lower=True, **_):
            return emit(ctx, "metaljax_tridiagonal", [operand],
                        "L" if lower else "U")

        def tri_solve_rule(ctx, a, b, *, left_side, lower, transpose_a,
                           conjugate_a, unit_diagonal,
                           perturb_singular=False, **_):
            cfg = (("L" if left_side else "R") + ("l" if lower else "u")
                   + ("t" if transpose_a else "n")
                   + ("c" if conjugate_a else "-")
                   + ("1" if unit_diagonal else "0")
                   + ("p" if perturb_singular else "-"))
            return emit(ctx, "metaljax_triangular_solve", [a, b], cfg)

        mlir.register_lowering(ll.triangular_solve_p, tri_solve_rule,
                               platform="metal")

        def tridiag_solve_rule(ctx, dl, d, du, b, *,
                               perturb_singular=False, **_):
            return emit(ctx, "metaljax_tridiagonal_solve", [dl, d, du, b],
                        "p" if perturb_singular else "-")

        mlir.register_lowering(ll.tridiagonal_solve_p, tridiag_solve_rule,
                               platform="metal")
        mlir.register_lowering(ll.schur_p, schur_rule, platform="metal")
        mlir.register_lowering(ll.hessenberg_p, hessenberg_rule,
                               platform="metal")
        mlir.register_lowering(ll.tridiagonal_p, tridiagonal_rule,
                               platform="metal")
        mlir.register_lowering(ll.eigh_p, eigh_rule, platform="metal")
        mlir.register_lowering(ll.svd_p, svd_rule, platform="metal")
        mlir.register_lowering(ll.eig_p, eig_rule, platform="metal")
        _register_callback_lowerings(mlir)

        # Buffer donation: jax only sets up input-output aliasing for
        # platforms in this list. With metal added, donate_argnums marks
        # arguments in the lowered module (tf.aliasing_output /
        # jax.buffer_donor) and the plugin invalidates those buffers
        # after execute — the standard XLA contract. Reusing a donated
        # array afterwards raises, exactly as on cpu/cuda/tpu.
        mlir._platforms_with_donation.append("metal")

        # jax.export refuses to serialize custom calls without a
        # registered stability guarantee; ours are versioned with the
        # plugin itself, so they are as stable as the platform.
        from jax._src.export import _export
        _export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE |= {
            "metaljax_eigh", "metaljax_svd", "metaljax_eig",
            "metaljax_schur", "metaljax_hessenberg",
            "metaljax_tridiagonal", "metaljax_triangular_solve",
            "metaljax_tridiagonal_solve", "metaljax_callback",
        }
    except Exception as e:  # jax internals moved; degrade to unsupported
        logger.warning("metaljax: linalg lowering registration failed: %s", e)


def _register_callback_lowerings(mlir):
    """jax.debug.print / debug_callback / pure_callback on metal: stash
    the Python callable in metaljax's registry and emit a
    metaljax_callback custom call with its index (we run in-process, so
    the interpreter can just call it)."""
    from functools import partial
    from jax._src import debugging

    def emit_callback(ctx, args, callback, with_results=True):
        from metaljax.ops import callbacks as cb_mod
        idx = cb_mod.register_callback(callback)
        out_types = ([mlir.aval_to_ir_type(ctx.module_context, a)
                      for a in ctx.avals_out] if with_results else [])
        op = mlir.custom_call(
            "metaljax_callback", result_types=out_types,
            operands=list(args), backend_config=str(idx),
            has_side_effect=True)
        try:
            if ctx.tokens_in:
                ctx.set_tokens_out(ctx.tokens_in)
        except Exception:
            pass
        return list(op.results)

    def debug_cb_rule(ctx, *args, effect, partitioned, callback, **params):
        # Route through the primitive's impl like jax's own lowering does:
        # it device_puts the operands onto CPU, so the user callback sees
        # jax Arrays (not raw numpy) exactly as on other backends.
        def run(*flat_args):
            debugging.debug_callback_p.impl(
                *flat_args, effect=effect, partitioned=partitioned,
                callback=callback, **params)
            return ()
        return emit_callback(ctx, args, run, with_results=False)

    def debug_print_rule(ctx, *dyn_args, fmt, ordered, partitioned,
                         in_tree, static_args, np_printoptions,
                         has_placeholders, logging_record):
        callback = partial(
            debugging._format_print_callback, fmt, dict(np_printoptions),
            has_placeholders, logging_record)
        callback = debugging._make_flat_callback(
            in_tree, callback, static_args)

        def run(*flat_args):
            debugging.debug_callback_p.impl(
                *flat_args, effect=debugging.debug_effect,
                partitioned=partitioned, callback=callback)
            return ()
        return emit_callback(ctx, dyn_args, run, with_results=False)

    mlir.register_lowering(debugging.debug_callback_p, debug_cb_rule,
                           platform="metal")
    mlir.register_lowering(debugging.debug_print_p, debug_print_rule,
                           platform="metal")
    try:
        from jax._src import callback as jcb

        def pure_cb_rule(ctx, *args, callback, sharding=None,
                         vectorized=None, vmap_method=None, **params):
            return emit_callback(ctx, args, callback)

        mlir.register_lowering(jcb.pure_callback_p, pure_cb_rule,
                               platform="metal")
        mlir.register_lowering(jcb.io_callback_p, pure_cb_rule,
                               platform="metal")
    except Exception:
        pass
