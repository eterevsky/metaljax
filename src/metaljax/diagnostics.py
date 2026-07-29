"""Operational diagnostics for long-lived metaljax processes."""

import gc

import mlx.core as mx


def live_buffer_floor(limit=499000):
    """Measure how many Metal buffers the current process has pinned.

    Metal caps LIVE buffer allocations per device (~499k, `device_info()`
    resource_limit) and MLX exposes no allocation count, so this probes
    directly: allocate 1-element buffers until the limit, count how many
    fit, free them. Returns `limit - fitted` — the floor pinned by
    everything else alive (executables, caches, leaked cycles), or -1 if
    the limit was never hit (floor is below probe resolution, ~5000).

    SIDE EFFECTS: briefly drives the process to the buffer limit (do not
    run concurrently with GPU work) and clears MLX's buffer cache.
    """
    gc.collect()
    mx.clear_cache()
    bufs = []
    hit = False
    try:
        while len(bufs) <= limit + 20000:
            batch = [mx.array([0.0]) for _ in range(5000)]
            mx.eval(*batch)
            bufs.extend(batch)
    except RuntimeError as e:
        if "Resource limit" not in str(e):
            raise
        hit = True
    n = len(bufs)
    bufs.clear()
    mx.clear_cache()
    return (limit - n) if hit else -1
