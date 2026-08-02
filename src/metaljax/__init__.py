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
#  - bytes: BOUNDED both ways.
#    Floor -- CORRECTNESS: splitting a single eval of an mx.compile'd graph
#    into many command buffers corrupts results in MLX 0.32 (silently,
#    differently on every call; see notes/mlx-command-buffer-split.md).
#    Corruption appears with splits <=80 MB apart and is gone by 160 MB on
#    the shipped repro; 512 keeps a 3x margin, and perf is flat from
#    128 MB up (big15 55.5-56.1 ms across 128..16384).
#    Ceiling -- STABILITY: every intermediate tensor of a command buffer
#    stays allocated until the buffer completes, so an unbounded budget let
#    one commit accumulate ~90 GB of transient attention logits (SD3.5
#    MMDiT at 1024^2, ~400 kernels in one buffer) as unpageable wired
#    memory -- the kernel starved and the MACHINE PANICKED (twice; flight
#    recorder showed 18 GB -> 90+ GB in the final second at pressure
#    level 1). 512 MB restores intermediate recycling per commit.
#
# Override either by exporting a different value before importing metaljax.
_os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "400")
_os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "512")

from metaljax.interpreter import Interpreter, UnsupportedOpError
from metaljax import ops as _ops  # noqa: F401  (registers all op handlers)

__version__ = "0.11.1"
__all__ = ["Interpreter", "UnsupportedOpError"]
