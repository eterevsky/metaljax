# Branch `fix/command-buffer-split` — **no upstream PR needed**

This branch is a **backport, not a proposal**: it cherry-picks
`ml-explore/mlx@7e8b4ccc` — *"Fix fence tracking for donated dynamic slice
offsets"* (PR #4099, merged 2026-08-10, author Michael Ellis, co-authored by
Cheng Zhao) — onto the `v0.32.0` tag, with authorship preserved
(`git commit -C`).

The fix is one line in `mlx/backend/metal/slicing.cpp`: only register the
`offset` array as a command-encoder temporary when it *owns* its buffer, never
when `offset.copy_shared_buffer(indices)` aliased the caller's.

```diff
     offset.set_data(allocator::malloc(offset.itemsize()));
+    compute_encoder.add_temporary(offset);
   }
-  compute_encoder.add_temporary(offset);
```

We carry it because **it is not in any released MLX**: `v0.32.0` is still the
newest tag and it predates the fix by ~230 commits.

## What Oleg may want to send upstream anyway

Not a PR — a **comment on #4099 / a release request**, because upstream almost
certainly does not know how large this one is. Suggested content, all of it
measured on this machine (M5 Max, macOS 26.5, mlx built from the v0.32.0 tag):

> This fixed considerably more than a dynamic-slice corner for us. We maintain
> a JAX (PJRT) backend on top of MLX; every transformer we run writes its KV
> cache with a dynamic update slice, so `compute_dynamic_offset` runs once per
> layer per step, and the dropped fence meant a stale offset whenever a
> command-buffer boundary fell between the index's producer and the slice.
>
> * A 36-layer Qwen3-8B prefill returned a **different answer on most
>   evaluations** and lost 5 of 15 outputs entirely (`max_norm_err` 1.000).
>   With 7e8b4ccc it returns the same values at 40, 512 and 2048 MB per
>   command buffer — the byte budget stops mattering at all.
> * A Qwen3-8B decode emitted 9 distinct first tokens in 10 prefills of the
>   same parameters; with the fix, 10/10 identical and the text is correct.
> * A 28-layer parameter-init scan (no `mx.compile`, corrupted through
>   `MLX_MAX_OPS_PER_BUFFER` instead) started training from the wrong weights.
>   Same fix, also cured.
>
> Minimal repro, pure MLX, MLX's own default budgets, wrong on the first
> evaluation in 3 of 3 fresh processes on 0.32.0 (the start index must not be
> bound to a Python name, or it is not donatable and the bug disappears):
>
> ```python
> import mlx.core as mx
> source = mx.concatenate([mx.zeros((2, 1 << 26), mx.int32),
>                          mx.ones((2, 1 << 26), mx.int32)], axis=1)
> target, update = mx.zeros((4, 4), mx.int32), mx.full((1, 1), 7, mx.int32)
> mx.eval(source, target, update)
> out = mx.slice_update(target, update, mx.max(source, axis=1), (0, 1))
> mx.eval(out)
> print(out)   # expected 7 at [1,1]; on 0.32.0 it lands at [0,0] or vanishes
> ```
>
> Given the failure is silent and hits any KV-cache workload, a release
> carrying 7e8b4ccc would be worth a lot to downstream users.

## Verification on this branch

| check | result |
|---|---|
| `notes/data/mlx-cbuf-repro/repro_c.py` | 0/60 evaluations wrong (was 3/3 processes wrong) |
| 8B canary, 3 draws at 512 MB | `9.277e-03 / 4.097e-02`, identical every draw (was `1.085e+04 / 1.000`) |
| 8B canary at 40 MB | identical to the 512 MB numbers |
| `tests/test_command_buffer.py` (Python engine) | 4 correctness tests pass; both corruption canaries can no longer find a corrupting budget |
| upstream's own `tests/gpu_tests.cpp` case | carried by the cherry-pick |
