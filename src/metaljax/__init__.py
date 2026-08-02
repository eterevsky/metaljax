"""metaljax: a Metal backend for JAX (StableHLO interpreter on MLX)."""

import os as _os

# On M5-generation GPUs (applegpu_g17*), MLX routes float32 GEMM through the
# neural accelerators with ~bf16 input precision (~4e-3 error). Our default is
# CPU-matching accuracy, so pin MLX's kernel arch to the previous generation
# unless the user opts into the fast path. Must happen before mlx.core
# initializes its Metal device.
if _os.environ.get("METALJAX_MATMUL_PRECISION", "highest") == "highest":
    _os.environ.setdefault("MLX_METAL_GPU_ARCH", "applegpu_g16g")

# How much work MLX packs into one Metal command buffer, by kernel count and
# by bytes. Both are raised well above MLX's defaults (10 ops / 40 MB):
#
#  - ops: measured ~20% faster on launch-bound (small-kernel) workloads.
#  - bytes: CORRECTNESS. Splitting a single eval of an mx.compile'd graph into
#    many command buffers corrupts results in MLX 0.32 -- silently, and
#    differently on every call (see notes/mlx-command-buffer-split.md). The
#    byte budget is what bites: a transformer layer's intermediates are
#    megabytes, so 40 MB commits every few kernels, and an LLM-sized compiled
#    main (maxtext qwen3 decode) came out as garbage tokens. Corruption
#    appears once splits are ~4 kernels apart and disappears by ~8; keeping
#    the byte budget far above any realistic single graph leaves the kernel
#    count (400) as the only splitter, orders of magnitude inside the safe
#    region.
#
# Override either by exporting a different value before importing metaljax.
_os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "400")
_os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "16384")

from metaljax.interpreter import Interpreter, UnsupportedOpError
from metaljax import ops as _ops  # noqa: F401  (registers all op handlers)

__version__ = "0.11.1"
__all__ = ["Interpreter", "UnsupportedOpError"]
