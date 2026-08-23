"""jax_plugins entry for the metaljax Metal PJRT plugin.

jax discovers this via the `jax_plugins` namespace package and calls
initialize() at backend-discovery time.  The plugin is the fully-native
dylib (an xla::PjRtClient; plugin-native/): since the Stage-1 retirement
(0.11.6) there is no other engine, and nothing here imports the `metaljax`
package -- the loader locates its lib/ directory with find_spec, so no
Python is pulled into the process beyond this module.
"""

import ctypes
import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_NATIVE_DYLIB = "libmetal_pjrt_native.dylib"


def _library_path() -> Path | None:
    env = os.environ.get("METALJAX_PLUGIN_PATH")
    if env:  # an explicit override always wins (measurements pin builds this
        p = Path(env)             # way, often under a frozen file name)
        return p if p.exists() else None
    # Packaged location (wheel and editable installs alike), found WITHOUT
    # importing metaljax.
    try:
        spec = importlib.util.find_spec("metaljax")
    except (ImportError, ValueError):
        spec = None
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if locations:
        p = Path(next(iter(locations))) / "lib" / _NATIVE_DYLIB
        if p.exists():
            return p
    # Repo layout fallback: <root>/src/jax_plugins/metal/__init__.py, with
    # the plugin freshly built by bazel.
    root = Path(__file__).resolve().parents[3]
    p = root / "plugin-native" / "bazel-bin" / "metal" / _NATIVE_DYLIB
    return p if p.exists() else None


def initialize():
    path = _library_path()
    if path is None:
        logger.warning(
            "metaljax native plugin dylib not found; install the wheel or "
            "build it (cd plugin-native && bazel build "
            "//metal:libmetal_pjrt_native.dylib)")
        return
    # The MLX precision default, repeated from the plugin's own static
    # initializer (plugin-native/metal/metal_client.cc, which owns the
    # numbers): MLX reads it when it initializes its Metal device, and
    # setting it before jax dlopens the dylib below keeps the ordering
    # obvious from the Python side too.
    if os.environ.get("METALJAX_MATMUL_PRECISION", "highest") == "highest":
        os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g16g")
    import jax._src.xla_bridge as xb

    # Keep CPU (priority 0) as the default backend; select metal explicitly
    # via JAX_PLATFORMS=metal / jax.config.update('jax_platforms', 'metal')
    # or jax.devices('metal').
    xb.register_plugin("metal", priority=-1, library_path=str(path))
    _register_linalg_lowerings(callbacks=_install_native_callbacks(path),
                               donation=True)


# --------------------------------------------------------------------------
# the plugin's callback bridge (P13)
# --------------------------------------------------------------------------
#
# jax's debug.print / debug.callback / pure_callback / io_callback lower, on
# every backend, to something that eventually calls the user's Python. The
# native plugin holds no interpreter, so the callable stays HERE, in the
# registry below, and the dylib reaches it through one C function pointer.
#
# ctypes is what makes that safe: a CFUNCTYPE callback acquires the GIL for the
# duration of the call and releases it after, so the GIL enters the native
# engine only inside a user callback.
_NATIVE_CALLBACKS: list = []
# The ctypes callback object: freeing it would leave the dylib holding a
# dangling function pointer.
_NATIVE_TRAMPOLINE = None


class _HostBuffer(ctypes.Structure):
    """`MetaljaxHostBuffer` (plugin-native/runtime/host_callback.h)."""

    _fields_ = [("data", ctypes.c_void_p),
                ("dtype", ctypes.c_int32),
                ("rank", ctypes.c_int32),
                ("dims", ctypes.POINTER(ctypes.c_int64))]


_TRAMPOLINE_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(_HostBuffer), ctypes.c_int32, ctypes.POINTER(_HostBuffer),
    ctypes.POINTER(ctypes.c_char), ctypes.c_int32)


def _host_dtypes():
    """`MetaljaxHostDtype` -> numpy, in the enum's order."""
    import ml_dtypes
    import numpy as np
    return [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64, np.float16, ml_dtypes.bfloat16,
            np.float32, np.complex64]


def _install_native_callbacks(path: Path):
    """Install the trampoline, and return the registrar the lowerings use.

    Returns None when the dylib predates the bridge (or ctypes cannot open
    it), which leaves the callback lowerings unregistered -- so jax refuses
    debug.print at TRACE time with its own message rather than the plugin
    failing at execute.
    """
    global _NATIVE_TRAMPOLINE
    import numpy as np

    try:
        lib = ctypes.CDLL(str(path))
        setter = lib.metaljax_native_set_callback_trampoline
    except (OSError, AttributeError) as e:
        logger.warning("metaljax: no callback bridge in the plugin: %s", e)
        return None

    dtypes = _host_dtypes()

    def view(buf):
        """The numpy array over one host buffer -- a VIEW, never a copy: an
        output is written through it."""
        dt = np.dtype(dtypes[buf.dtype])
        shape = tuple(buf.dims[i] for i in range(buf.rank))
        n = 1
        for d in shape:
            n *= d
        if n == 0 or not buf.data:
            return np.empty(shape, dt)
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf.data, ctypes.POINTER(ctypes.c_ubyte)),
            shape=(n * dt.itemsize,))
        return raw.view(dt).reshape(shape)

    def trampoline(index, nin, ins, nout, outs, error, error_len):
        try:
            fn = _NATIVE_CALLBACKS[index]
            outputs = fn(*[view(ins[i]) for i in range(nin)])
            if outputs is None:
                outputs = []
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            if len(outputs) != nout:
                raise ValueError(
                    f"callback returned {len(outputs)} values, "
                    f"the program declares {nout}")
            for i in range(nout):
                dst = view(outs[i])
                # The declared shape and dtype win.
                dst[...] = np.asarray(outputs[i]).reshape(dst.shape)
            return 0
        except BaseException as e:   # noqa: BLE001 - it becomes a PJRT error
            msg = f"{type(e).__name__}: {e}".encode()[:max(error_len - 1, 0)]
            ctypes.memmove(error, msg, len(msg))
            error[len(msg)] = b"\0"
            return 1

    _NATIVE_TRAMPOLINE = _TRAMPOLINE_TYPE(trampoline)
    setter.argtypes = [ctypes.c_void_p]
    setter.restype = None
    setter(ctypes.cast(_NATIVE_TRAMPOLINE, ctypes.c_void_p))

    def register(fn) -> int:
        _NATIVE_CALLBACKS.append(fn)
        return len(_NATIVE_CALLBACKS) - 1

    return register


def _register_linalg_lowerings(callbacks=None, donation=True):
    """eigh/svd/eig have no generic StableHLO lowering — jax only
    registers per-platform rules, and those reject bf16/f16 outright
    (LAPACK routine tables). Emit our own custom_calls instead, with the
    primitive's declared result types; the plugin implements them on
    Accelerate's LAPACK (plugin-native/runtime/host_lapack.cc)."""
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

        # LU: jax's generic (non-cpu/gpu/tpu) lowering is a *python* blocked
        # factorization — `for k in range(0, min(m, n), 128)` — so a symbolic
        # *matrix* dimension cannot enter it ('_DimExpr' object cannot be
        # interpreted as an integer, and min() on two unrelated symbols is
        # inconclusive). Only those shapes go to a host getrf, whose result
        # shapes travel with the call. Symbolic BATCH dimensions are fine
        # (they are vmap axes, not loop bounds) and keep the on-device
        # factorization, as do fully static shapes: the two algorithms round
        # differently, and at bfloat16 that difference is visible, so the
        # host path must not take work the device path can do.
        try:
            generic_lu = mlir.lower_fun(ll._lu_python, multiple_results=True)

            def lu_rule(ctx, operand, **_):
                m_n = ctx.avals_in[0].shape[-2:]
                if all(isinstance(d, int) for d in m_n):
                    return generic_lu(ctx, operand)
                return emit(ctx, "metaljax_lu", [operand])

            mlir.register_lowering(ll.lu_p, lu_rule, platform="metal")
        except AttributeError:
            # jax renamed its private generic LU; leaving lu_p unregistered
            # keeps static-shape LU working (jax's own default rule).
            logger.warning("metaljax: shape-polymorphic LU unavailable")

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
        if callbacks is not None:
            # `callbacks` is the plugin's registrar, installed by
            # _install_native_callbacks; None (no bridge in the dylib)
            # leaves the callback lowerings unregistered so jax refuses
            # debug.print at TRACE time with its own message.
            _register_callback_lowerings(mlir, callbacks)

        # Buffer donation: jax only sets up input-output aliasing for
        # platforms in this list. With metal added, donate_argnums marks
        # arguments in the lowered module (tf.aliasing_output /
        # jax.buffer_donor) and the plugin invalidates those buffers
        # after execute — the standard XLA contract. Reusing a donated
        # array afterwards raises, exactly as on cpu/cuda/tpu.
        if donation:
            mlir._platforms_with_donation.append("metal")

        # jax.export refuses to serialize custom calls without a
        # registered stability guarantee; ours are versioned with the
        # plugin itself, so they are as stable as the platform.
        from jax._src.export import _export
        _export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE |= {
            "metaljax_eigh", "metaljax_svd", "metaljax_eig",
            "metaljax_lu", "metaljax_schur", "metaljax_hessenberg",
            "metaljax_tridiagonal", "metaljax_triangular_solve",
            "metaljax_tridiagonal_solve", "metaljax_callback",
        }
    except Exception as e:  # jax internals moved; degrade to unsupported
        logger.warning("metaljax: linalg lowering registration failed: %s", e)


def _register_callback_lowerings(mlir, register):
    """jax.debug.print / debug_callback / pure_callback on metal: stash
    the Python callable in this module's registry and emit a
    metaljax_callback custom call with its index (we run in-process, so
    whoever holds the registry can just call it)."""
    from functools import partial
    from jax._src import debugging

    def emit_callback(ctx, args, callback, with_results=True):
        idx = register(callback)
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
