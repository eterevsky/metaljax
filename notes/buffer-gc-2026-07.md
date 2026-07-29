# Long-lived worker buffer GC (2026-07-29)

Report (Oleg): a search worker on v0.4.3 died with `[metal::malloc]
Resource limit (499000) exceeded` ~1100 steps into
`tokens.32.raw_fold.emb.4|split.add(mingru.4-rmsnorm, pass)-matlstm.4`
(4x256, replay path, ~105 ms/step) — AFTER the engine's clear-and-retry
ran and failed identically twice. Workers run for days, evaluating
thousands of models; the crash config's own working set is fine.

## Measurement (new tool: metaljax.diagnostics.live_buffer_floor)

MLX has no buffer-count API and byte telemetry rounds to zero at these
array sizes (the limit is a COUNT), so the probe allocates 1-element
buffers to the limit and counts: floor = 499000 - fitted (resolution
~5k). Findings on v0.4.3, this machine:

- Fresh process, his exact config: survives 7+ chunks. Fresh floor 4000.
- 9 varied configs run-and-released + 2 matlstm chunks: floor stays
  4000 after every one — NO leak in the release chain (plugin
  UnrefExecutable -> Py_XDECREF under GIL works; jax executable caching
  is weakref'd on the jitted fn).
- 9 configs RETAINED (simulating dead-but-uncollected managers): floor
  4000 -> 9000 (~500 pinned buffers per config: mx.compile trace
  constants, plan arrays, weights/opt state). drop + gc.collect():
  floor back to 4000 — fully recoverable.

## Story

Python's cycle collector triggers on allocation counts, which
array-heavy workloads barely tick; texmo managers sit in reference
cycles. Dead-but-uncollected configs pin ~500 buffers each, so a
worker that has churned through ~1000 configs' worth of uncollected
state hits Metal's 499k live cap — and mx.clear_cache() (all our
recovery did) cannot free REFERENCED buffers, which is exactly why the
engine's retry failed twice identically. gc.collect() frees them.

## Fixes (HEAD)

- gc.collect() before mx.clear_cache() in EVERY resource-limit
  recovery path: engine.execute retry, _loop_flush recovery, the
  while-body limit handler.
- gc.collect() at compile boundaries (new config = dead configs can
  release first; compiles are rare, cost irrelevant) and in the
  periodic execute backstop (every METALJAX_CLEAR_PERIOD=50k executes).
- plan._last_bufs (debug retention of last-call kernel inputs, real
  device arrays pinned per plan for the executable lifetime) now only
  stored under MJDBG_VERIFY_MSL.
- metaljax.diagnostics.live_buffer_floor() shipped for operational
  monitoring (NOT concurrent with GPU work).

Worker-side belt-and-braces (Oleg's call, not required):
- periodic gc.collect() between configs in the client loop;
- jax.clear_caches() occasionally if worker RSS matters.

Unproven remainder: the exact composition of his worker's 495k floor
(hours of history, eval programs, data queues). The fixes attack the
recoverable class generically; if a worker still dies with these in
place, capture live_buffer_floor() before/after gc.collect() at the
failure — that split (recoverable vs live) pinpoints the rest.
