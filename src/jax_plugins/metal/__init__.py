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
    # Editable/source layout: <root>/src/jax_plugins/metal/__init__.py
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
