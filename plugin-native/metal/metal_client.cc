/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_client.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal/metal_buffer.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlx/array.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/runtime/device_id.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace metaljax {

namespace mx = mlx::core;

// ------------------------------------------------------------- description

MetalDeviceDescription::MetalDeviceDescription(int id, int process_index)
    : id_(id),
      process_index_(process_index),
      debug_string_(absl::StrFormat("MetalDevice(id=%d)", id)),
      to_string_(absl::StrFormat("MetalDevice(id=%d)", id)) {
  memory_space_ptrs_.push_back(&memory_space_);
}

// ------------------------------------------------------------ memory space

MetalMemorySpace::MetalMemorySpace(xla::PjRtClient* client, int id)
    : client_(client),
      id_(id),
      debug_string_(absl::StrFormat("MetalMemory(id=%d)", id)) {}

absl::Span<xla::PjRtDevice* const> MetalMemorySpace::devices() const {
  return client_->devices();
}

// ------------------------------------------------------------------ device

MetalDevice::MetalDevice(xla::PjRtClient* client, int id, int process_index)
    : client_(client), description_(id, process_index) {}

absl::Status MetalDevice::TransferToInfeed(const xla::LiteralSlice& literal) {
  return absl::UnimplementedError("metaljax does not support infeed.");
}

absl::Status MetalDevice::TransferFromOutfeed(
    xla::MutableBorrowingLiteral literal) {
  return absl::UnimplementedError("metaljax does not support outfeed.");
}

absl::Span<xla::PjRtMemorySpace* const> MetalDevice::memory_spaces() const {
  return client_->memory_spaces();
}

absl::StatusOr<xla::PjRtMemorySpace*> MetalDevice::default_memory_space()
    const {
  absl::Span<xla::PjRtMemorySpace* const> spaces = client_->memory_spaces();
  if (spaces.empty()) {
    return absl::InternalError("metaljax: device has no memory space.");
  }
  return spaces.front();
}

absl::StatusOr<xla::PjRtMemorySpace*> MetalDevice::memory_space_by_kind(
    absl::string_view memory_space_kind) const {
  for (xla::PjRtMemorySpace* space : client_->memory_spaces()) {
    if (space->kind() == memory_space_kind) {
      return space;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("metaljax: no memory space of kind ", memory_space_kind));
}

// ------------------------------------------------------------------ client

MetalClient::MetalClient() {
  // One GPU, one unified memory space.  Apple silicon exposes a single
  // integrated GPU per process; multi-device is out of scope.
  owned_memory_spaces_.push_back(
      std::make_unique<MetalMemorySpace>(this, /*id=*/0));
  owned_devices_.push_back(
      std::make_unique<MetalDevice>(this, /*id=*/0, /*process_index=*/0));
  for (auto& space : owned_memory_spaces_) {
    memory_spaces_.push_back(space.get());
  }
  for (auto& device : owned_devices_) {
    devices_.push_back(device.get());
  }
}

xla::PjRtPlatformId MetalClient::platform_id() const {
  static const uint64_t kId = tsl::Fingerprint64(kPlatformName);
  return kId;
}

absl::StatusOr<xla::PjRtDevice*> MetalClient::LookupDevice(
    xla::GlobalDeviceId global_device_id) const {
  for (xla::PjRtDevice* device : devices_) {
    if (device->global_device_id() == global_device_id) {
      return device;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "metaljax: no device with global id ", global_device_id.value()));
}

absl::StatusOr<xla::PjRtDevice*> MetalClient::LookupAddressableDevice(
    xla::LocalDeviceId local_device_id) const {
  for (xla::PjRtDevice* device : devices_) {
    if (device->local_device_id() == local_device_id) {
      return device;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "metaljax: no addressable device with local id ",
      local_device_id.value()));
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>>
MetalClient::BufferFromHostBuffer(
    const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    xla::PjRtMemorySpace* memory_space, const xla::Layout* device_layout) {
  if (type != xla::F32) {
    return absl::UnimplementedError(
        absl::StrCat("metaljax-native P0 handles f32 only, got ",
                     xla::PrimitiveType_Name(type)));
  }
  if (device_layout != nullptr &&
      !xla::LayoutUtil::IsMonotonicWithDim0Major(*device_layout)) {
    return absl::UnimplementedError(
        "metaljax-native P0 handles row-major layouts only.");
  }
  // Reject any host buffer that is not densely packed row-major: MLX would
  // otherwise read the wrong elements.
  if (byte_strides.has_value()) {
    if (byte_strides->size() != dims.size()) {
      return absl::InvalidArgumentError(
          "metaljax: byte_strides rank does not match dims.");
    }
    int64_t expected = sizeof(float);
    for (int64_t i = dims.size() - 1; i >= 0; --i) {
      if ((*byte_strides)[i] != expected) {
        return absl::UnimplementedError(
            "metaljax-native P0 handles densely packed host buffers only.");
      }
      expected *= dims[i];
    }
  }

  mx::Shape shape;
  shape.reserve(dims.size());
  for (int64_t d : dims) {
    if (d < 0 || d > std::numeric_limits<mx::ShapeElem>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("metaljax: dimension out of range: ", d));
    }
    shape.push_back(static_cast<mx::ShapeElem>(d));
  }

  // The iterator constructor copies, so the host buffer is ours to release as
  // soon as this returns; eval() materialises the array, which is what makes
  // the buffer honestly "ready".
  mx::array array(static_cast<const float*>(data), shape, mx::float32);
  array.eval();
  if (on_done_with_host_buffer) {
    std::move(on_done_with_host_buffer)();
  }

  xla::PjRtMemorySpace* space =
      memory_space != nullptr ? memory_space : memory_spaces_.front();
  return std::make_unique<MetalBuffer>(
      this, space, devices_.front(), xla::ShapeUtil::MakeShape(type, dims),
      std::move(array));
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
MetalClient::CompileAndLoad(xla::MaybeOwningMlirModule module,
                            xla::CompileOptions options) {
  // P0 checkpoint 4: report what the C-API wrapper actually handed us.  If the
  // op names below are `stablehlo.*` (and not `vhlo.*`), the wrapper has
  // already parsed the portable artifact and run the VHLO->StableHLO upgrade
  // for us -- i.e. a native engine never has to touch serialization.
  mlir::ModuleOp mlir_module = module.mlir_module();
  std::vector<std::string> ops;
  if (mlir_module) {
    mlir_module->walk([&](mlir::Operation* op) {
      ops.push_back(op->getName().getStringRef().str());
    });
  }
  std::fprintf(stderr,
               "[metaljax-native] CompileAndLoad(mlir): %zu ops: %s\n",
               ops.size(), absl::StrJoin(ops, " ").c_str());
  std::fflush(stderr);
  return absl::UnimplementedError(absl::StrCat(
      "metaljax-native P0: received a parsed MLIR module with ", ops.size(),
      " ops; no executor yet."));
}

std::unique_ptr<xla::PjRtClient> CreateMetalClient() {
  return std::make_unique<MetalClient>();
}

}  // namespace metaljax
