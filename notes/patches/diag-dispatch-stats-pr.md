# Branch `diag/dispatch-counter` — draft upstream PR

*Draft for Oleg to review and open against ml-explore/mlx. Everything below
the rule is the PR body; this preamble is not part of it.*

Base: the branch is cut from our `vendor/0.32.0` (= v0.32.0 + the fence fix
+ the gemv occupancy floor) and touches `mlx/backend/metal/device.cpp`,
`mlx/backend/metal/metal.h`, `mlx/backend/metal/no_metal.cpp` (one commit,
`notes/patches/0004-diag-dispatch-stats.patch`). It applies to upstream
`main` unchanged in intent; rebase before opening. Local verification is in
`~/.cache/metaljax-bench/logs/t3-dispatch/findings.txt` (the metaljax side
reads the stats under `METALJAX_DEBUG=1`; the Instruments cross-check is
there too).

Not a fix: a diagnostic that costs nothing when unread. Upstream may prefer
it behind a build flag; the patch keeps it unconditional because the hot
path is untouched (see "Cost").

---

**Title:** Metal: expose a process-wide dispatch counter and per-command-buffer
GPU timestamps (`metal::dispatch_stats()`)

## Why

An embedder that lowers whole programs into MLX graphs (a JAX/PJRT plugin in
our case) can count the *nodes* it builds but not the *kernels* the device
runs: `mx::compile` fuses some of them, copies and contiguity fix-ups add
others, and views cost nothing. Estimates of "kernels per token" for the
same decode loop differed by 2-3x depending on how the graph was read. The
number already exists inside the backend — `CommandEncoder::buffer_ops_`
counts every `dispatchThreads` / `dispatchThreadgroups` for the commit
cadence — it just is not visible.

The same goes for GPU time: `MTLCommandBuffer` carries `GPUStartTime` /
`GPUEndTime`, so "how much of this window was the device actually busy, and
how long did each buffer wait between commit and start" is one completion
handler away from being a number rather than an argument.

## The change

`metal::dispatch_stats()` returns a cumulative, process-wide snapshot:

| field | meaning |
|---|---|
| `dispatches` | compute dispatches committed (`buffer_ops_` summed at every `commit`) |
| `command_buffers`, `empty_command_buffers` | command buffers committed / of which held no dispatch |
| `completed_command_buffers` | buffers whose completion handler has run (the GPU fields below cover exactly these) |
| `gpu_busy_ns` | Σ (`GPUEndTime` − `GPUStartTime`) |
| `gpu_span_ns` | the same with overlapping buffers merged: device occupancy |
| `gpu_gap_ns` | Σ idle between one buffer's GPU end and the next buffer's GPU start |
| `queue_ns` | Σ (`GPUStartTime` − host commit time) |
| `now_ns` | host clock at the snapshot, same base as the GPU timestamps |

Everything is on the `CLOCK_UPTIME_RAW` base (mach_absolute_time in ns, the
base `CACurrentMediaTime` / the Metal timestamps use), so two snapshots
bracket a window: `wall = now₂ − now₁`, `device idle = wall − Δgpu_span`.
`MLX_DISPATCH_TRACE=1` additionally prints one line per command buffer as it
completes (`cb=<n> ops=<k> queue=<us> gpu=<us> gap=<us>`).

The struct lives in `metal.h` with only `<cstdint>` behind it, so a consumer
never sees a Metal header; `no_metal.cpp` returns zeros.

## Cost

* The dispatch hot path is **untouched**: `dispatch_threads` /
  `dispatch_threadgroups` still do `buffer_ops_++` and nothing else. The
  accounting is one relaxed `fetch_add` of that count per command buffer
  at `commit`, plus one `clock_gettime_nsec_np` read.
* The completion handler reads two timestamps and takes one uncontended
  mutex (the span merge has to be ordered across queues). Completion
  handlers already run off the encoding thread.
* Snapshots are a handful of relaxed loads. Nothing is allocated per
  buffer; the accounting object is immortal by design (a completion
  handler can outlive static destruction — the first build died with
  `mutex lock failed` at exit for exactly that reason).

Measured on the consumer side (metaljax, this binary vs the same tree on
the unpatched fork, `METALJAX_DEBUG=0` so nothing reads the stats), see
findings.txt §4: rows 4 and 11 within noise, token streams identical.

## Semantics worth knowing

* `dispatches` is accumulated at commit, so a snapshot lags the *open*
  command buffer by at most `MLX_MAX_OPS_PER_BUFFER` dispatches; at any
  blocking eval it is exact (the eval commits).
* The GPU fields are booked by completion handlers a few microseconds
  after the host wakes from an event wait, so a snapshot taken right after
  `eval` can be one buffer short: `completed_command_buffers <
  command_buffers` says so. The metaljax reader waits (≤ 5 ms) for the
  two to meet before printing.
* `gpu_gap_ns` counts the idle before a buffer since the *previous* buffer
  in the process, so the first buffer of a window carries the idle that
  preceded the window.

## Evidence

Measured on an M5 Max, macOS 26.5, Xcode 26.6, mlx built from our
`vendor/0.32.0` (v0.32.0 + two fixes), read from a JAX program through
metaljax's `METALJAX_DEBUG=1` narration; one decode loop per row:

| row | dispatches/token | command buffers/token | GPU busy / wall per token |
|---|---:|---:|---|
| (filled in from findings.txt §2 once the cells are in) | | | |

Cross-check: an Instruments *Metal System Trace* of the row-4 run
(findings.txt §3) — the counter and Instruments' compute-dispatch count
agree within the ±10 % the diagnosis asked for / disagree by X (state which).
