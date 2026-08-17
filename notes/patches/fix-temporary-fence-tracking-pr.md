# Branch `fix/temporary-fence-tracking` — draft upstream PR

*Draft for Oleg to review and open against ml-explore/mlx. Everything below
the rule is the PR body; this preamble is not part of it.*

Base: the branch is cut from `v0.32.0` and touches only
`mlx/backend/metal/device.cpp` (18 insertions, 7 deletions). Before opening it,
rebase onto `main` — `#4099` (the instance this generalizes) is already merged
there, so the PR should be framed as hardening, not as a duplicate fix.
Local verification is in `notes/mlx-patch-diagnosis.md` §3.

---

**Title:** Metal: keep a temporary in the fence bookkeeping when a previous
command encoder wrote its buffer

## The invariant that does not hold

`CommandEncoder::end_encoding()` erases every temporary's buffer from
`all_inputs_`/`all_outputs_` before it emits the cross-command-encoder fence
waits:

```cpp
// - Temporaries are a special case as they do not cross command encoder
//   boundaries. These can be removed early from the encoders inputs and
//   outputs since they don't need synchronization.
for (auto& t : temporaries_) {
  all_outputs_.erase(t.buffer().ptr());
  all_inputs_.erase(t.buffer().ptr());
}
```

Those two sets are the *only* input to the fence protocol, so an erased buffer
loses its dependency on whatever previous encoder wrote it.

The premise holds for a temporary **array** but not for its **buffer**:

1. `eval_gpu` helpers alias a caller's buffer into a temporary.
   `compute_dynamic_offset` did exactly this — `offset.copy_shared_buffer(
   indices)` followed by `add_temporary(offset)` — which is the bug fixed in
   #4099. The same shape is available to any helper that pairs
   `copy_shared_buffer` with `add_temporary`.
2. `MetalAllocator::free()` recycles a buffer into its cache immediately, with
   no regard for command buffers still in flight, so a *fresh* temporary can
   land on the buffer a previous encoder wrote.

Nothing else orders the two: buffers are allocated
`MTL::ResourceHazardTrackingModeUntracked`, and the command buffers come from
`commandBufferWithUnretainedReferences()`. So the kernel reads stale data,
silently, and only when a command-buffer boundary happens to fall between the
producer and the consumer — i.e. depending on `MLX_MAX_OPS_PER_BUFFER` /
`MLX_MAX_MB_PER_BUFFER` and on the size of the tensors.

## The change

Keep the entry when `prev_ce_outputs_` knows the buffer; erase it otherwise, so
the map still stays small for genuinely fresh temporaries. The loop moves
inside the existing `outputs_mtx_` critical section, which is where
`prev_ce_outputs_` may be read.

```cpp
std::lock_guard lk(outputs_mtx_);
for (auto& t : temporaries_) {
  auto ptr = t.buffer().ptr();
  if (prev_ce_outputs_.find(ptr) != prev_ce_outputs_.end()) {
    continue;   // a previous encoder wrote it: it needs the wait below
  }
  all_outputs_.erase(ptr);
  all_inputs_.erase(ptr);
}
```

Cost: at most one extra `waitForFence` per encoder boundary (waits are already
deduplicated by fence), and a few more entries in `prev_ce_outputs_`, each
removed by the command buffer's own completion handler. On a Qwen3-8B prefill
(760 encoder boundaries) it keeps 172 entries that were previously dropped and
does not change measured throughput: 404-408 ms per prefill across seven runs
of the unpatched and patched builds alike, with no ordering between them.

## Evidence

Measured on an M5 Max, macOS 26.5, Metal 4, mlx built from the `v0.32.0` tag.
The workload is a 36-layer Qwen3-8B prefill graph (bf16, 16.4 GB of weights)
lowered from JAX, checked against a CPU reference.

| build | `MLX_MAX_MB_PER_BUFFER` | result |
|---|---|---|
| v0.32.0 | 512 | wrong on 2 of 4 evaluations; 5 of 15 outputs total loss of signal (`max_norm_err` 1.000) |
| v0.32.0 | 40 (default for this GPU) | wrong |
| this patch | 40 / 512 / 2048 | **identical outputs at all three budgets, every run** |

Diagnostic counters (a scratch build, not part of this PR) attribute it: 144 of
760 encoder boundaries dropped a fence wait, one per transformer layer per
execution, every one of them in the encoder holding the dynamic-slice offset
computation.

The same patch also fixes an unrelated-looking failure on the *uncompiled*
path — a 28-layer parameter-init scan whose RNG keys came out wrong when the
kernel-count budget cut the eval in a particular place.

## Relationship to #4099

#4099 fixed the one instance that was reachable through
`compute_dynamic_offset` (this patch is not a substitute for it: keeping a
*non-temporary* array's buffer out of `temporaries_` is right on its own
merits). This patch removes the class: if a temporary ever shares a buffer with
work a previous encoder did, the fence protocol now sees it.

I am happy to split the `prev_ce_inputs_` write-after-read diagnostic out of my
scratch build into a follow-up if it would be useful — on my workloads it never
fires, which is evidence that the input-retention in `gpu::eval` is doing its
job and this erase was the only hole.
