/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_C_PJRT_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_C_PJRT_H_

#include "xla/pjrt/c/pjrt_c_api.h"

extern "C" {

// The single symbol jaxlib dlsym()s out of the plugin dylib.  Ownership of the
// returned PJRT_Api is NOT transferred.  The visibility attribute makes the
// export independent of whatever -fvisibility the build happens to use.
__attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi();
}

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_C_PJRT_H_
