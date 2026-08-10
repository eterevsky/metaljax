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
  return pjrt::StatusToPjRtError(absl::UnimplementedError(
      "metaljax: ahead-of-time topology compilation is not supported."));
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
