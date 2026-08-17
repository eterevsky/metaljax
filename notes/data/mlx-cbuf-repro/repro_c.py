"""Minimal, self-contained reproducer for the MLX command-buffer corruption.

Needs only `mlx`.  Runs in ~2 s and ~1 GB, on MLX's own default budgets.

    python repro_c.py                       # exits 1 when the update is misplaced

`mx.slice_update(target, update, start, axes)` with an ARRAY start lowers to
`compute_dynamic_offset` (mlx/backend/metal/slicing.cpp).  When `start` is
donatable that helper aliases the caller's buffer for its `offset` output
(`offset.copy_shared_buffer(indices)`) and still registers it as a command
encoder temporary; `CommandEncoder::end_encoding()` erases every temporary
from the cross-encoder fence bookkeeping ("temporaries do not cross command
encoder boundaries"), so the wait on the encoder that PRODUCED `start` is
dropped.  When a command-buffer boundary falls between the producer and the
slice — which is what the op/byte budgets decide — the offset kernel reads the
buffer before the producer's writes have landed and the update is written at a
stale offset.  No error, no warning.

`start` must stay donatable, which is why it is not bound to a Python name:
holding a reference makes MLX allocate a separate `offset` buffer and the bug
disappears.  This is also why the failure needs `mx.compile` (or any graph
where intermediates have a single reference) in larger programs.

Expected: the 7 lands at [1, 1].
Actual on mlx 0.32.0 / macOS 26.5 / M5 Max: on the FIRST evaluation in a fresh
process it lands at [0, 0], or nowhere at all (3 of 3 fresh processes).

Fixed upstream by ml-explore/mlx@7e8b4ccc ("Fix fence tracking for donated
dynamic slice offsets", #4099), which is not in any released version as of
v0.32.0.
"""

import os
import sys

import mlx.core as mx


def build():
    # A producer with enough work in front of it to still be executing when
    # the next command buffer starts.
    source = mx.concatenate(
        [mx.zeros((2, 1 << 26), mx.int32), mx.ones((2, 1 << 26), mx.int32)],
        axis=1,
    )
    target = mx.zeros((4, 4), mx.int32)
    update = mx.full((1, 1), 7, mx.int32)
    mx.eval(source, target, update)
    # NB no Python reference to the start array: it must stay donatable.
    return mx.slice_update(target, update, mx.max(source, axis=1), (0, 1))


def main():
    reps = int(os.environ.get("REPS", "20"))
    bad = []
    for i in range(reps):
        out = build()
        mx.eval(out)
        got = out.tolist()
        if got[1][1] != 7 or got[0][0] != 0:
            bad.append((i, got))
    print(
        f"mlx {mx.__version__} "
        f"ops/buffer={os.environ.get('MLX_MAX_OPS_PER_BUFFER', '<default>')} "
        f"mb/buffer={os.environ.get('MLX_MAX_MB_PER_BUFFER', '<default>')}: "
        f"{len(bad)}/{reps} evaluations put the update at the wrong offset"
    )
    for i, got in bad[:3]:
        print(f"  eval {i}: {got}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
