# Branch `fix/donate-through-stream-pins` — draft upstream PR

*Draft for Oleg to review and open against ml-explore/mlx. Everything below
the rule is the PR body; this preamble is not part of it.*

Base: cut from our `vendor/0.32.0` (v0.32.0 + the fence fix + the gemv
occupancy floor + the dispatch-stats diagnostic). Touches `mlx/array.{h,cpp}`,
`mlx/backend/metal/{device.{h,cpp},eval.cpp,indexing.cpp,metal.h}` and
`mlx/backend/gpu/primitives.cpp` (+372/-26). The diagnostic counters it adds
to `metal::dispatch_stats()` depend on the dispatch-stats branch; for an
upstream PR drop the two `DispatchStats` fields and the `donation_counters()`
reads (six lines) or rebase both together. Local verification is in
`~/.cache/metaljax-bench/logs/b1-kvcarry/findings.txt`.

---

**Title:** Metal: let slice updates donate an operand whose only other
holders are this stream's command-buffer pins

## The copy

`SliceUpdate::eval_gpu` and `DynamicSliceUpdate::eval_gpu` copy the whole
operand into the output before writing the window unless the operand is
donatable (`copy_gpu` → `set_copy_output_data` → `is_donatable`). On Metal an
operand that ANY kernel of a still-live command buffer has read is never
donatable: `gpu::eval` captures a `shared_ptr<Data>` of every input of every
primitive it encodes in the command buffer's completion handler, so the
use-count test sees a second holder until the buffer completes.

For a decode loop's KV cache that is every update. The cache is read by the
layer's attention, written by the next layer's update in the same eval, and
read again by the next token's attention in the next one — each write of an
8 KiB row costs a copy of the whole cache (20 MiB on a 28-layer 0.6B model at
T=179; 56 such writes per token when the cache is one stacked array). The
same pin makes a loop carry updated at a chunk boundary of an `async_eval`
chain copy once per boundary (the Synchronizer's inputs are pinned too).

## Why the pin is not an observer

A pin keeps the buffer alive while the GPU may still read it. On the stream
that encoded those reads it does not observe the buffer's contents, because
a kernel encoded LATER on the same stream is ordered after them:

* within one encoder, `register_output_array` already inserts a memory
  barrier when the output buffer was an input of an earlier dispatch
  (write-after-read);
* across encoders — a previous encoder of the same command buffer, or a
  previous command buffer still in flight — ordering needs a fence, which
  this patch adds for exactly the encoders that donate this way.

A pin from ANOTHER stream is a different queue and stays an observer.

## The change

* `array::Data` gains one 64-bit pin word — pins outstanding, pins from the
  open encoder generation, that generation, the stream — maintained by
  compare-and-swap (`detail::pin_data` / `unpin_data`): the eval thread pins,
  a completion thread unpins.
* `array::is_donatable(bool through_stream_pins)`. With the flag, a buffer
  whose extra holders are exactly the pins of the calling stream is
  donatable; if some of those pins belong to encoders that have ended, the
  per-thread eval context is marked so the encoder can fence. The no-arg
  `is_donatable()` and every primitive that calls it are unchanged. Pins from
  a second stream, or more encoders than the word counts (255), saturate into
  the strict answer until every pin is released.
* `gpu::eval` sets the eval context (stream index, encoder generation) around
  `eval_gpu` and pins through the encoder: `CommandEncoder::pin` keeps one
  pin per encoder per buffer in a set, and `commit` moves the sets of the
  buffer's encoders into ONE completion handler (instead of one handler per
  primitive), which releases them before the scheduler and dispatch
  accounting notices run.
* `end_encoding` records the encoder's fence as in flight (pruned by the
  command buffer's completion) and, when the encoder donated through pins of
  ended encoders, waits on every in-flight fence first. Fence waits apply to
  the whole compute pass, so this orders the pass after everything committed
  ahead of it; it costs nothing when those buffers have already completed
  (the previous token's, at the decode loop's blocking condition read).
* `SliceUpdate::eval_gpu` (Metal) and `DynamicSliceUpdate::eval_gpu` (gpu,
  shared with CUDA) donate a row-contiguous, same-dtype operand through the
  pins; `metal::dispatch_stats()` reports the donated / copied counts.

The CUDA backend never sets the eval context, so `is_donatable(true)` is the
strict test there and its behavior is unchanged.

## Evidence

Measured on an M5 Max, macOS 26.5, from the metaljax plugin's tape (a JAX
StableHLO program lowered onto MLX). On a 12-step decode loop with a stacked
[1,3,2,8,2,4] cache and four `slice_update`s per step, `dispatch_stats()`
reads `slice_update_donated=47 slice_update_copied=1` (the one copy is the
first iteration, whose carry the loop still holds) against `0 / 48` before;
results bit-identical. The model-row numbers (Qwen3-0.6B / gemma4-E2B /
gpt-oss-20b keras-hub decode) are in the findings file named above.
