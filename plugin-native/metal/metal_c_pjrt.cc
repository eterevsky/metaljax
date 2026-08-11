/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

The whole PJRT C API surface is manufactured by XLA's
pjrt_c_api_wrapper_impl from our C++ xla::PjRtClient subclass; this file is
only the entry point and the three create-hooks the wrapper cannot infer.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_c_pjrt.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "host_callback.h"
#include "metal/metal_client.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"

namespace metaljax {
namespace {

const PJRT_Api* GetMetalPjrtApi();

PJRT_Error* MetalClientCreate(PJRT_Client_Create_Args* args) {
  std::unique_ptr<xla::PjRtClient> client = CreateMetalClient();
  args->client = pjrt::CreateWrapperClient(GetMetalPjrtApi(), std::move(client));
  return nullptr;
}

PJRT_Error* MetalExecuteContextCreate(PJRT_ExecuteContext_Create_Args* args) {
  return pjrt::StatusToPjRtError(absl::UnimplementedError(
      "metaljax: PJRT execute contexts are not supported."));
}

PJRT_Error* MetalTopologyCreate(PJRT_TopologyDescription_Create_Args* args) {
  // The wording carries information: jax's own AOT tests catch this refusal
  // and skip on the substring "topology not implemented" (aot_test's
  // `test_get_topology_from_devices`, `test_topology_jit_serialize`), so
  // spelling it that way is the difference between two reported failures and
  // two honest skips.  There is nothing to describe here -- a topology is a
  // machine shape compiled for WITHOUT the machine, and this plugin compiles
  // against the one GPU in the process.
  return pjrt::StatusToPjRtError(absl::UnimplementedError(
      "metaljax: topology not implemented (ahead-of-time compilation against "
      "a described machine; metaljax compiles for the attached GPU)."));
}

const PJRT_Api* GetMetalPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);
  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      MetalClientCreate, MetalExecuteContextCreate, MetalTopologyCreate,
      pjrt::PJRT_Plugin_Initialize_NoOp, &layouts_extension.base,
      pjrt::PJRT_Plugin_Attributes_Xla);
  return &pjrt_api;
}

}  // namespace
}  // namespace metaljax

const PJRT_Api* GetPjrtApi() { return metaljax::GetMetalPjrtApi(); }

// The second symbol this dylib exports, and the whole of P13's callback
// bridge: jax's `debug.print` / `debug.callback` / `pure_callback` /
// `io_callback` lower to a `metaljax_callback` custom call whose
// `backend_config` indexes a registry of Python callables that lives where the
// lowerings are registered (src/jax_plugins/metal/__init__.py).  That module
// installs a ctypes callback here, so the GIL is taken by ctypes for the
// duration of one user callback and nowhere else in this plugin.  Installed
// before any program is compiled; a callback program lowered without one
// declines by name (runtime/host_callback.h).
extern "C" void metaljax_native_set_callback_trampoline(void* fn) {
  metaljax::SetCallbackTrampoline(
      reinterpret_cast<metaljax::CallbackTrampoline>(fn));
}
