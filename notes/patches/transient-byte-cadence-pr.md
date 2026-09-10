# Branch `fix/transient-byte-cadence` — draft upstream PR

*Draft for Oleg to review and open against ml-explore/mlx. Everything below
the rule is the PR body; this preamble is not part of it.*

Base: cut from our `vendor/0.32.0`, independent of
`fix/skip-empty-finalize` (they merge cleanly; the combined tree is what the
row-10 measurements used). Touches `mlx/utils.h` and
`mlx/backend/metal/{device.h,device.cpp}` (+59/-1). Patch:
`notes/patches/0007-transient-byte-cadence.patch`.

**Not on by default anywhere.** metaljax pins `MLX_MAX_MB_PER_BUFFER=512` as a
NO-PANIC bound (`plugin-native/metal/metal_client.cc`), and this changes what
that number counts. The plugin exposes it as `METALJAX_CBUF_BYTES=transient`,
default `inputs`; flipping the default is Oleg's decision and wants the
cadence canary battery plus the big-model rows re-run first.

---

**Title:** Metal: charge the per-command-buffer size budget for what a buffer
allocates, not what it reads

## What the budget counts today

`CommandEncoder::set_input_array` adds each distinct input's whole
`data_size()` to `buffer_sizes_`:

```c++
  if (all_inputs_.insert(a.buffer().ptr()).second) {
    buffer_sizes_ += a.data_size();
  }
```

and `needs_commit()` ends the command buffer when that crosses
`MLX_MAX_MB_PER_BUFFER`. So the budget is charged for **reading** a buffer,
not for holding one.

For a program whose weights are small relative to its intermediates that is a
fine proxy. For a program whose weights are resident and large it is not: the
budget then splits command buffers at every weight read.

Measured on a DeepSeek-V2-Lite decode step (bf16, one token, 26 MoE layers):
the routed-expert weights are one `[26, 64, 2048, 1408]` bf16 array — 9.6 GB —
so every routed `gather_mm` that touches it trips the budget on its own. **52
of the 149 command buffers per token are these splits**, two per layer holding
a single dot and nothing else, costing about 0.59 ms/token of device idle plus
the per-buffer overhead. The slab was resident before the buffer opened and is
still resident after it completes. Nothing about reading it is transient.

## What the budget is for

The quantity worth bounding is how much **unpageable memory one in-flight
command buffer is responsible for**: the intermediates it allocates and holds
until it completes. (metaljax sized its 512 against exactly that — above ~2048
a diffusion model at 1024² panicked the machine.)

That quantity is the arrays the command buffer **writes**, not the ones it
reads.

## The change

`buffer_out_sizes_` sums the `data_size()` of the distinct buffers a command
buffer writes (`register_output_array`, so outputs and written temporaries),
deduplicated per **command buffer** rather than per encoder, and
`MLX_MAX_MB_TRANSIENT_ONLY=1` applies the same numeric budget to it.

The bound on transients is unchanged: a command buffer still commits after
`max_mb` of intermediates. What goes away is splitting on inputs the buffer
neither allocates nor frees. An output that donated its operand's storage is
charged even though nothing new was allocated — over-counting is the safe
direction for a budget.

Off by default: a caller who sized `MLX_MAX_MB_PER_BUFFER` against the old
meaning should opt in rather than have the bound they chose loosened
underneath them.

## Verification

Two shapes differing only in which side of a dot is big, at
`MLX_MAX_MB_PER_BUFFER=8` (8 mega-elements, so a 4096×4096 f32 array trips it
alone):

| | reads six 16-Melem weights, writes 8×4096 | reads 8×4096, writes six 16-Melem outer products |
|---|---|---|
| `MLX_MAX_MB_TRANSIENT_ONLY=0` | 7 command buffers | 13 command buffers |
| `MLX_MAX_MB_TRANSIENT_ONLY=1` | **1** | **7** |

The resident-input splits are gone; the transient splits are not. Results are
bit-identical in both modes, as are the plugin's differential suite (all cases
vs the CPU backend), 512 pytest cases, the ingest suite, and the
command-buffer corruption canary at 40 MB and 512 MB in both modes.

## Aside: the unit is elements, not bytes

`array::data_size()` is in elements, so `(buffer_sizes_ >> 20) > max_mb`
compares mega-**elements** against a number named MB. The effective byte
budget is `max_mb` × itemsize — 1 GB at 512 for bf16, 2 GB for f32. This patch
does not change that (both counters keep the same unit, so the two modes stay
comparable), but anyone tuning either one should know.
