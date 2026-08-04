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
#  - ops: 400 was measured ~20% faster than MLX's 10 on launch-bound
#    (small-kernel) workloads, but 400 exactly is a CORRUPTING alignment:
#    maxtext's qwen3 parameter init (a 28-layer scan interpreted op by op,
#    flushed every 5 iterations) computed one layer's RNG key wrong there,
#    so training started from different weights -- first-step loss 208.78
#    where jax-CPU and our own compiled path both say 247.78. The same
#    program is clean at 200, 800, 1600 and 5000 kernels per buffer, and at
#    400 with any other flush cadence: it is where the boundaries land that
#    decides, not how far apart they are (see the byte note below -- same
#    bug, other budget). 800 costs ~2-3% against 400 (mid08 4.83 -> 4.93 ms,
#    big15 55.9 -> 57.5, qwen3 decode 15.6 -> 16.0 ms/tok) and is pinned by
#    tests/test_command_buffer.py, which replays that init scan.
#  - bytes: BOUNDED both ways.
#    Floor -- CORRECTNESS: splitting a single eval of an mx.compile'd graph
#    into many command buffers corrupts results in MLX 0.32 (silently,
#    differently on every call; see notes/mlx-command-buffer-split.md).
#    Corruption appears with splits <=80 MB apart and is gone by 160 MB on
#    the shipped repro -- but the safe band scales with TENSOR SIZE, not
#    absolute bytes: at Qwen3-8B shapes (100 MB weight slices, ~8-10 GB per
#    flush window) 512 MB cuts every ~2-5 kernels and corrupts decode
#    replays (2026-08-03: bf16 AND qwix int8 8B both garbage at 512/768/
#    1024/1536, correct at 2048; per-layer KV deltas uncorrelated vs CPU at
#    512, textbook-benign at 2048; repro
#    notes/data/qwen3_8b_prefill_36layer.mlir corrupts in 0.3 s). Perf is
#    flat from 128 MB up (big15 55.5-56.1 ms across 128..16384; 8B decode
#    within 5% between 512 and 2048).
#    Ceiling -- STABILITY: every intermediate tensor of a command buffer
#    stays allocated until the buffer completes, so an unbounded budget let
#    one commit accumulate ~90 GB of transient attention logits (SD3.5
#    MMDiT at 1024^2, ~400 kernels in one buffer) as unpageable wired
#    memory -- the kernel starved and the MACHINE PANICKED (twice; flight
#    recorder showed 18 GB -> 90+ GB in the final second at pressure
#    level 1). 512 MB restores intermediate recycling per commit.
#
# THE 8B-CLASS BIND (2026-08-03): no safe budget exists for Qwen3-8B-shaped
# maxtext workloads. 512 corrupts their compiled decode replays (bf16 AND
# qwix int8; per-layer KV deltas uncorrelated vs CPU; correct under
# METALJAX_COMPILE=0). 2048 gives correct output in short-prefill probes --
# but has NO stability margin: a 4096 probe wedged an 8B checkpoint load at
# a 120 GB wired footprint, and a subsequent 8B load at 2048 (same config
# that had just succeeded) kernel-panicked the machine (watchdog timeout).
# Raising the default is a nondeterministic machine-killer; the default
# stays 512 and the 8B class is unsupported until MLX fixes the split
# corruption upstream. Repro: notes/data/qwen3_8b_prefill_36layer.mlir
# (weights as args, corrupts in 0.3 s at 512, clean at 2048).
#
# Override either by exporting a different value before importing metaljax.
_os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "800")
_os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "512")

from metaljax.interpreter import Interpreter, UnsupportedOpError
from metaljax import ops as _ops  # noqa: F401  (registers all op handlers)

__version__ = "0.11.3"
__all__ = ["Interpreter", "UnsupportedOpError"]
