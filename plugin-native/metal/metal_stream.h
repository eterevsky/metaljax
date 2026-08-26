/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The MLX stream discipline every PJRT entry point runs under.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_STREAM_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_STREAM_H_

#include <mutex>

namespace metaljax {

// Give this thread a cross-thread-evaluable default MLX stream, once.
//
// Stage 1 engine.py's `bind_thread` (deleted 0.11.6, ef5774d),
// transliterated, and for the same
// reason: MLX streams are thread-LOCAL and the default one is thread-BOUND, so
// a graph built on one thread can only be evaluated there -- and if the owning
// thread has exited, MLX throws where nothing catches and std::terminate takes
// the process down.  jax dispatches an execute from whatever thread the caller
// is on and reads the results back from another, so producer and consumer
// routinely differ.  `new_thread_unsafe_stream` is MLX's answer: a stream that
// may be passed to and evaluated anywhere.
//
// One stream PER THREAD, emphatically not one shared: "thread unsafe" means
// concurrent use is the caller's problem, and two threads encoding into one
// stream's command buffer is a Metal assertion, not a soft failure.
//
// Every entry point that touches MLX calls this first: transfers in, transfers
// out, and execute.
void BindThread();

// Serialize MLX submission across threads.  Per-thread streams make graphs
// EVALUABLE from any thread, but MLX 0.32 routes every stream through one
// process-wide command-encoder map, and eight genuinely concurrent evals
// segfault inside metal::get_command_encoder ~5% of runs (measured 4/74 at
// the pinned command-buffer budgets; the Stage 1 plugin never sees this only
// because the GIL serializes its entries).  Until MLX's map is safe, every
// path that drives evaluation -- execute, host transfers in -- holds this
// lock.  METALJAX_CONCURRENT_EXECUTE=1 lifts it, for measuring the day an
// MLX fix lands; a returned lock that owns nothing is that flag, not an
// error.
//
// RECURSIVE, since P13: a host callback runs the USER's Python in the middle
// of an execute that holds this lock, and that Python may perfectly well
// touch a metal array -- a `device_put`, or another jitted call.  Same-thread
// re-entry is not the concurrency this lock exists to prevent (it is exactly
// what a single-threaded process does anyway, one submission after another),
// and a plain mutex would answer it with a deadlock.
std::unique_lock<std::recursive_mutex> SubmissionLock();

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_STREAM_H_
