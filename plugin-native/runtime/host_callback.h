// metaljax native engine — jax's host callbacks (P13).
//
// `jax.debug.print`, `jax.debug.callback`, `pure_callback` and `io_callback`
// all lower, on platform `metal`, to a `metaljax_callback` custom call whose
// `backend_config` is an INDEX into a registry of Python callables
// (src/jax_plugins/metal/__init__.py).  That callable is the USER'S code: it
// cannot be ported, and every PJRT backend needs a way to reach it, so this is
// the one host op whose other side is an interpreter.
//
// The line drawn here is the same one `host_lapack.h` draws, one level lower.
// This file knows arrays and a C function pointer; it does not know what a
// Python object is, and neither does anything else in the runtime.  The
// embedder (the plugin's Python-side registration) installs a TRAMPOLINE of
// the C signature below and owns everything on the other side of it —
// including the GIL, which a ctypes callback acquires for the duration of the
// call and for nothing else.  With no trampoline installed a callback program
// declines at LOWERING time rather than failing at execute.

#ifndef METALJAX_HOST_CALLBACK_H_
#define METALJAX_HOST_CALLBACK_H_

#include <cstdint>
#include <vector>

#include "program.h"

namespace metaljax {

// The element type of one callback buffer, as the ABI spells it.  These codes
// are part of the contract with the embedder and are appended to, never
// renumbered.  They are deliberately NOT the tape's dtype codes: those come
// from a runtime registry keyed by MLIR names and may be reordered by any
// milestone that adds a type, where this list crosses a process boundary.
enum MetaljaxHostDtype : int32_t {
  kHostBool = 0,
  kHostInt8 = 1,
  kHostInt16 = 2,
  kHostInt32 = 3,
  kHostInt64 = 4,
  kHostUint8 = 5,
  kHostUint16 = 6,
  kHostUint32 = 7,
  kHostUint64 = 8,
  kHostFloat16 = 9,
  kHostBfloat16 = 10,
  kHostFloat32 = 11,
  kHostComplex64 = 12,
};

// One host-side buffer: row-major, contiguous, `rank` dimensions.  The memory
// belongs to the CALLER for the duration of the call — an input is the
// operand's own (evaluated, unified-memory) storage and an output is a staging
// block the handler is about to hand to MLX.
struct MetaljaxHostBuffer {
  void* data;
  int32_t dtype;   // MetaljaxHostDtype
  int32_t rank;
  const int64_t* dims;
};

// The embedder's entry point.  Returns 0 on success; on failure it writes a
// NUL-terminated message into `error` (at most `error_len` bytes) and returns
// nonzero, which becomes a `std::runtime_error` here and a PJRT error at the
// call site.  It must not throw.
using CallbackTrampoline = int32_t (*)(int32_t index, int32_t nin,
                                       const MetaljaxHostBuffer* ins,
                                       int32_t nout,
                                       const MetaljaxHostBuffer* outs,
                                       char* error, int32_t error_len);

// Install (or clear, with nullptr) the trampoline.  Called once per process,
// from the embedder's registration, before any program is compiled.
void SetCallbackTrampoline(CallbackTrampoline fn);
bool HasCallbackTrampoline();

// One declared result of a callback, read off the custom call's result type.
struct CallbackSpec {
  std::vector<int64_t> shape;
  mx::Dtype dtype;
};

// The handler for one call site.  Throws `std::invalid_argument` at BIND time
// for anything this file cannot carry (no trampoline, a dtype outside the ABI
// above), so the plugin declines the program by name instead of failing at
// execute.
HostFn MakeHostCallback(int32_t index, std::vector<CallbackSpec> results);

}  // namespace metaljax

#endif  // METALJAX_HOST_CALLBACK_H_
