/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_stream.h"

#include <cstdlib>
#include <string>

#include "mlx/mlx.h"

namespace metaljax {

namespace mx = mlx::core;

void BindThread() {
  // The initializer of a function-local thread_local runs once per thread, on
  // the thread's first arrival -- which is exactly the "bind once" the Python
  // engine gets from its threading.local flag.
  static thread_local const bool bound = [] {
    mx::set_default_stream(mx::new_thread_unsafe_stream(mx::default_device()));
    return true;
  }();
  (void)bound;
}

std::unique_lock<std::mutex> SubmissionLock() {
  static std::mutex mu;
  static const bool concurrent = [] {
    const char* v = std::getenv("METALJAX_CONCURRENT_EXECUTE");
    return v != nullptr && std::string(v) == "1";
  }();
  if (concurrent) return std::unique_lock<std::mutex>();
  return std::unique_lock<std::mutex>(mu);
}

}  // namespace metaljax
