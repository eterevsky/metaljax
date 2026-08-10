/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The strings jaxlib and user code see.  They must match the hand-rolled C-API
plugin in metaljax/plugin/metal_pjrt.cc, except for the platform version, which
is deliberately distinguishable so a test can tell which plugin answered.

Their own header because both the client and the executable report them, and
those two are on opposite sides of a build dependency.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_NAMES_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_NAMES_H_

#include "absl/strings/string_view.h"

namespace metaljax {

inline constexpr absl::string_view kPlatformName = "metal";
inline constexpr absl::string_view kDeviceKind = "Apple GPU";
inline constexpr absl::string_view kMemoryKind = "device";
// The sentinel names the PLUGIN, not the phase it has reached: the test
// harnesses match on it to prove which dylib answered, so it stays put as the
// plugin grows.
inline constexpr absl::string_view kPlatformVersion = "metaljax-native-p0";

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_NAMES_H_
