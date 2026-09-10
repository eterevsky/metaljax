# Branch `fix/skip-empty-finalize` — draft upstream PR

*Draft for Oleg to review and open against ml-explore/mlx. Everything below
the rule is the PR body; this preamble is not part of it.*

Base: cut from our `vendor/0.32.0`. Touches
`mlx/backend/metal/{device.h,eval.cpp}` only (+44/-1) and depends on nothing
else in the fork — it applies to upstream `main` as written. Patch:
`notes/patches/0006-skip-empty-finalize.patch`. Local verification (the
microbenchmark, the row-10 A/B, the suites) is in
`~/.cache/metaljax-bench/logs/r10-cbuf/findings.md`.

---

**Title:** Metal: don't commit an empty command buffer from `gpu::finalize`

## What happens

`gpu::eval` commits the current command buffer whenever
`CommandEncoder::needs_commit()` trips — the `MLX_MAX_OPS_PER_BUFFER` /
`MLX_MAX_MB_PER_BUFFER` cadence — and tells the scheduler about it:

```c++
  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    encoder.commit([s]() { scheduler::notify_task_completion(s); });
  }
```

`commit()` immediately replaces `buffer_` with a fresh command buffer. Control
returns to `eval_impl`, whose very next statement is the back-pressure check:

```c++
    if (scheduler::n_active_tasks() > MAX_ACTIVE_TASKS ||
        (get_active_memory() > get_memory_limit() &&
         scheduler::n_active_tasks() > 0)) {
      // Commit any open streams
      for (auto& s : open_streams) {
        if (s.device == Device::gpu) {
          gpu::finalize(s);
        }
      }
      scheduler::wait_for_one();
```

Once ten command buffers are in flight — routine for any eval whose cadence
trips faster than the GPU drains it — that check is true, and `gpu::finalize`
commits the command buffer created microseconds ago on the line above. Nothing
has been encoded on it. `wait_for_one()` then blocks until one buffer
completes, which brings the count back to ten, so the pattern repeats on the
*next* cadence commit: **one empty command buffer per cadence commit, for the
whole eval.**

## Why it costs something

Metal schedules a committed command buffer whether or not it holds a compute
pass, and an empty one sits in the queue in front of the next real buffer.
With per-buffer GPU timestamps on an M5 Max, about 40 % of these empty buffers
are given `GPUStartTime`/`GPUEndTime` at all, and each of those costs 35–50 µs
of device idle between the buffers around it; the rest cost a host commit and
a completion handler.

A minimal reproduction, no model needed — a 40-step chain of 1536×1536 f32
matmuls under `MLX_MAX_OPS_PER_BUFFER=1`:

| | command buffers | of them empty |
|---|---|---|
| before | 71 | **31** |
| after | 41 | 1 |

The one that remains is structural and correct: `eval_impl` signals the
stream's `Event` after the last primitive, so when the cadence has just
committed, that signal lands on a fresh buffer. It has `ops == 0` but it is
not empty — every waiter depends on it.

The workload that found this is a decode step of DeepSeek-V2-Lite (bf16, 26
MoE layers, one token). Its resident routed-expert weights are a single 9.6 GB
array, so the byte cadence trips on every routed `gather_mm`: **68.8 empty
command buffers and 1.08 ms of device idle per token**, the largest single
idle item in that step and about 5 % of its wall time.

## The fix

`CommandEncoder::is_empty()` asks whether anything has been encoded on the
current command buffer since it was created — no dispatches, no open compute
encoder, no encoder that has already ended into it, and no signal or wait
event — and `gpu::finalize` returns without committing when that holds.

It cannot stall the back-pressure it sits in. `finalize` passes no completion
handler, so a commit there books no scheduler task; and the branch is only
reached with `n_active_tasks() > MAX_ACTIVE_TASKS > 0`, so `wait_for_one()`
always has an in-flight buffer left to complete. The inner
`while (get_active_memory() > get_memory_limit() && n_active_tasks() > 0)`
loop is guarded the same way.

Nothing else can be waiting on such a buffer either: a caller that needs a
specific command buffer to complete uses `CommandEncoder::synchronize()`,
which commits and waits on the buffer itself, and cross-stream ordering goes
through `Event`s, which are encoded on the buffer and therefore make it
non-empty.

## Verification

* the reproduction above, before/after;
* the full plugin differential suite against the CPU backend, plus 512
  pytest cases and an ingest suite — all pass, and the two suites are
  bit-identical to the unpatched arm;
* a new contract group asserts that a per-op cadence adds command buffers
  without adding empty ones, and that answers are identical at cadences
  1 / 8 / 800 / off;
* the DeepSeek-V2-Lite decode row A/B, both orders, with identical generated
  text.

## Note on the remaining empty buffer

`CommandEncoder::synchronize()` still commits an empty buffer and waits for it
when a stream is already idle (`cbuf = buffer_; end_encoding(); commit();
cbuf->waitUntilCompleted()`). Retaining the last committed buffer and waiting
on *that* instead would remove one more GPU round trip, but it was not needed
for the workload above and is left out to keep this change to one behaviour.
