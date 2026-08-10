/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

A PjRtLoadedExecutable that IS a lowered tape: `Execute` unwraps the argument
buffers into mlx arrays, replays the `metaljax::Program` the compile built, and
wraps the results back up.

The pairing of the two classes below is not decoration, it is what makes the
metadata answerable.  `PjRtLoadedExecutable` is not a `PjRtExecutable`: it owns
one, and the C-API wrapper asks THAT object for the output types and dimensions
it hands jax.  XLA's default answer derives them from `GetHloModules`, which a
plugin holding a tape instead of an HLO module cannot give -- and the default
forwarder would bounce the question back to the loaded executable, so
overriding it there is both a recursion and an answer nobody asks for.  So
`MetalExecutable` is a real `PjRtExecutable` that answers from the MLIR
function's own signature, recorded at compile time, and `GetExecutable()`
returns it.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_EXECUTABLE_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal/metal_lowering.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace metaljax {

class MetalExecutable final : public xla::PjRtExecutable {
 public:
  MetalExecutable(std::shared_ptr<const LoweredProgram> lowered,
                  xla::CompileOptions options, std::string name)
      : lowered_(std::move(lowered)),
        options_(std::move(options)),
        name_(std::move(name)) {}

  int num_replicas() const override { return 1; }
  int num_partitions() const override { return 1; }
  int64_t SizeOfGeneratedCodeInBytes() const override { return -1; }
  absl::string_view name() const override { return name_; }

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override {
    return absl::UnimplementedError(
        "metaljax-native: the executable is a tape, not an HLO module.");
  }

  // Answered from the function signature rather than from an HloModule, which
  // is the whole reason this class exists.
  absl::StatusOr<std::vector<xla::Shape>> GetOutputShapes() const override;
  absl::StatusOr<std::vector<std::vector<xla::PrimitiveType>>>
  GetOutputElementTypes() const override;
  absl::StatusOr<std::vector<std::vector<xla::DimensionVector>>>
  GetOutputDimensions() const override;
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override;
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetParameterMemoryKinds() const override;
  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  absl::StatusOr<struct xla::CompileOptions> GetCompileOptions()
      const override {
    return options_;
  }

 private:
  std::shared_ptr<const LoweredProgram> lowered_;
  xla::CompileOptions options_;
  std::string name_;
};

class MetalLoadedExecutable final : public xla::PjRtLoadedExecutable {
 public:
  MetalLoadedExecutable(xla::PjRtClient* client, xla::PjRtDevice* device,
                        xla::PjRtMemorySpace* memory_space,
                        LoweredProgram lowered, xla::CompileOptions options,
                        std::string name);

  xla::PjRtClient* client() const override { return client_; }
  xla::PjRtExecutable* GetExecutable() const override {
    return executable_.get();
  }
  const xla::DeviceAssignment& device_assignment() const override {
    return device_assignment_;
  }
  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return logical_ids_;
  }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return devices_;
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<xla::PjRtBuffer*>> argument_handles,
          const xla::ExecuteOptions& options,
          std::optional<std::vector<xla::Future<>>>& returned_futures)
      const override;

  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> ExecuteSharded(
      absl::Span<xla::PjRtBuffer* const> argument_handles,
      xla::PjRtDevice* device, const xla::ExecuteOptions& options,
      std::optional<xla::Future<>>& returned_future,
      bool fill_future) const override;

  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> ExecutePortable(
      absl::Span<xla::PjRtBuffer* const> argument_handles,
      xla::PjRtDevice* device, const xla::ExecuteOptions& options,
      std::optional<xla::Future<>>& returned_future,
      bool fill_future) const override;

  void Delete() override { deleted_ = true; }
  bool IsDeleted() const override { return deleted_; }

 private:
  // One replay: check the arguments against what the tape expects, run it,
  // settle the outputs and wrap them.  Everything Execute* does.
  absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> RunOnce(
      absl::Span<xla::PjRtBuffer* const> argument_handles) const;

  xla::PjRtClient* client_;
  xla::PjRtDevice* device_;
  xla::PjRtMemorySpace* memory_space_;
  std::shared_ptr<const LoweredProgram> lowered_;
  std::string name_;
  std::unique_ptr<MetalExecutable> executable_;
  xla::DeviceAssignment device_assignment_;
  std::vector<LogicalDeviceIds> logical_ids_;
  std::vector<xla::PjRtDevice*> devices_;
  bool deleted_ = false;
};

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_EXECUTABLE_H_
