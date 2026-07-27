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
    """eigh/svd have no generic StableHLO lowering — jax only registers
    per-platform rules. Reuse the CPU rules for platform 'metal': they
    emit lapack_*_ffi custom_calls, which metaljax's interpreter
    implements on the host (metaljax.ops.lapack)."""
    try:
        from functools import partial
        from jax._src.interpreters import mlir
        from jax._src.lax import linalg as ll
        for prim, rule in ((ll.eigh_p, ll._eigh_cpu_gpu_lowering),
                           (ll.svd_p, ll._svd_cpu_gpu_lowering)):
            mlir.register_lowering(
                prim, partial(rule, target_name_prefix="cpu"),
                platform="metal")
        mlir.register_lowering(ll.eig_p, ll._eig_cpu_lowering,
                               platform="metal")
    except Exception as e:  # jax internals moved; degrade to unsupported
        logger.warning("metaljax: linalg lowering registration failed: %s", e)
