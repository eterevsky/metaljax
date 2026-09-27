# The loop in-flight window (METALJAX_LOOP_INFLIGHT_MB), 2026-09-27

## The report

A texmo search worker (Mac mini M4, 32 GB, metaljax 0.11.8, jax 0.11.0) died with
`RESOURCE_EXHAUSTED ... at flush ... Needed 0.0 GB more ... this process holds 5.9 GB` on the jitted
training chunk of a 10,097-weight recurrent LM: an outer `lax.scan` over 256 training steps (forward,
`value_and_grad`, optax Adam), inner `lax.scan`s over 128 timesteps (mgru.32, rglru.16, lstm.16).
The largest tensor in the program is 32 MB. Kit (module, repro, lowered-ceiling script):
`~/.cache/metaljax-bench/logs/metal-mem/kit/`.

## Diagnosis (M5 Max, release binary frozen-x918main-e2ffcc75)

- Fresh process, first call: footprint 0.3 -> 5.5 GB (5.9 on the M4). jax 0.11.0 and 0.11.2 alike.
  Not per-call growth: six consecutive calls stay at 5.5 GB, and a 16-step variant of the module
  reaches 5.1 GB.
- `footprint`: the 5.2 GB is `IOAccelerator (graphics)`, i.e. MLX buffers. MLX's own counters: peak
  ACTIVE 5.16 GB during a call, then a 5.1 GB buffer CACHE between calls (cache limit 121.6 GB here).
- A 1 ms trace of MLX active memory: 0.4 -> 4.8 GB within 20 ms, draining in ~0.8 GB steps over the
  next ~100 ms, repeating every ~14 outer iterations. With `METALJAX_MSL=0` (inner scans as compiled
  per-timestep bodies) peak active is 0.29 GB, at 10x the time.
- Cause: the outer loop runs through run_while's single-step arm, which submits `period` iterations
  per `mx::async_eval` and blocks every `hard_every`. `period = min(64, 25000 / cost)` counts ops only
  (cost 1674 -> 14). MLX allocates every intermediate of a submission when it ENCODES it and frees it
  when the command buffer completes; with msl_scan kernels nothing inside an iteration waits for the
  device, so the host queues all 14 iterations (~0.3 GB each) before the GPU finishes the first.
- The refusal: the governor's hard-line reclaim ran while those buffers were in flight, and its 5 s
  stall never reclaims again. Traced through the stall: MLX active 0.03 GB, cache 5.12 GB, then the
  refusal "this process holds 5.4 GB" -- almost all of it reclaimable cache. Reproduced by setting
  `METALJAX_MEM_SYS_MB` to this machine's baseline + 1.5 / 3 / 4.5 GB: all three refused. (The
  governor's side is fixed separately: past a hard line it now settles the device and clears MLX's
  cache before the stall and at every tick of it -- runtime/memory.cc `reclaim_hard`.)

## The fix (runtime/control.cc `LoopWindow`)

A counted loop whose op-count cadence could queue more than METALJAX_LOOP_INFLIGHT_MB (default 1024;
0 = the old cadence exactly) measures what one submission holds -- the growth of
`mx::get_active_memory()` across it, on a settled device -- and keeps only as many submissions in
flight as fit, waiting on the OLDEST so newer work keeps the device busy. The lowering passes its
per-iteration estimate (the while gate's `real=`) as attrs[12]; it only decides whether to measure.
Single-step arm: `run_single_bounded`. Chunked arm: the fixed `chunk_inflight` ring becomes a window
(`keep` <= METALJAX_CHUNK_INFLIGHT). The pipelined dynamic-while arm (decode loops) is unchanged: it
holds at most two iterations by construction. Interpreted bodies big enough for their own eager flush
keep the old cadence (SD3.5's diffusion loops, the init loops of rows 10/15/20).

## Results

| module | binary | MLX peak active | footprint | s/call (steady) | first call |
|---|---|---:|---:|---:|---:|
| train chunk, 256 steps | release | 5.16 GB | 5.50 GB | 0.83 | 0.89 |
| train chunk, 256 steps | window | 0.76 GB | 1.09 GB | 0.69 | 1.03 |
| 16-step variant | release | 4.79 GB | 5.13 GB | 0.05 | 0.18 |
| 16-step variant | window | 0.76 GB | 1.08 GB | 0.05 | 0.15 |

- GPU idle per call 238 -> ~3 ms: the old cadence drained the device at every blocking flush.
- Lowered ceilings (+1.5 / 3 / 4.5 GB): release refuses all three; the window passes all three.
- texmo 223-config sweep vs the 0.11.8 release run: geomean 1.016x faster; the 20 single-step configs
  1.11-1.61x faster; the one config past -5 % (tc027-w47) re-drawn standalone: release 3.75, window
  3.73 ms (the release run's anchor was the outlier).
- Correctness, both arms: execute_test (new P54 contract `_p54_loop_window`), ingest_test, pytest
  559 + 1 xfail, texmo_gate 106/106.
- Model rows (A/B): row 14 26.4 / 26.4 ms/tok with peak footprint 25 -> 10 GB (maxtext init loop);
  row 19 362.5 / 362.3 ms/step; row 11 maxtext arm 8.37 / 8.47 (its per-token graph runs no changed
  code: knob off vs on on the window binary 8.41 / 8.45). Rows 10 and 15 engage the chunked window
  but keep the full ring (`keep=4`): row 10 19.69 (off) / 19.67 (on); row 15, three arms interleaved
  with 60 s cool-downs, two rounds, release / window-off / window-on 265.6, 265.3 / 265.2, 265.6 /
  265.4, 265.6 (an earlier pass with 30 s cool-downs read ~310-320 on every arm alike). Decode rows 1-9, 11
  keras, 12, 13, 20, 21 run only pipelined loops; 16 and 18 have no while loops; 17's loops are
  self-settling.

## Known limits

- A program's FIRST execute that traces msl kernels submits synchronously until they are proven, and
  its groups are sized from the static estimate (3-4x the real size): first calls of the single-step
  texmo configs are 0.1-0.2 s slower (warmup geomean 0.91x on those 20). Validating plans right after
  BodyRunner's probe would remove it; it touches the msl proving protocol.
- The chunked arm's measurement can read ~0 MB (rows 10 and 15 did): an under-reading only loosens
  the window, never below the old ring, so it cannot cost time -- but it can miss a bound.
- Engaged single-step loops no longer pass through the blocking `loop_flush` every `hard_every`, so
  its clear-and-retry on Metal's buffer limit is not reached there (the loop-clear cadence and
  governor admits still run at the old iterations).
- Readings are process-wide (`mx::get_active_memory`): noisy under METALJAX_CONCURRENT_EXECUTE=1.
