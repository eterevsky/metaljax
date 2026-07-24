"""metaljax: a Metal backend for JAX (StableHLO interpreter on MLX)."""

import os as _os

# On M5-generation GPUs (applegpu_g17*), MLX routes float32 GEMM through the
# neural accelerators with ~bf16 input precision (~4e-3 error). Our default is
# CPU-matching accuracy, so pin MLX's kernel arch to the previous generation
# unless the user opts into the fast path. Must happen before mlx.core
# initializes its Metal device.
if _os.environ.get("METALJAX_MATMUL_PRECISION", "highest") == "highest":
    _os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g16g")

from metaljax.interpreter import Interpreter, UnsupportedOpError
from metaljax import ops as _ops  # noqa: F401  (registers all op handlers)

__version__ = "0.1.1"
__all__ = ["Interpreter", "UnsupportedOpError"]
