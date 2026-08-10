/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The MLX stream discipline every PJRT entry point runs under.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_STREAM_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_STREAM_H_

namespace metaljax {

// Give this thread a cross-thread-evaluable default MLX stream, once.
//
// src/metaljax/engine.py's `bind_thread`, transliterated, and for the same
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

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_STREAM_H_
