/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_CLIENT_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_CLIENT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "metal/metal_names.h"
#include "xla/layout.h"
#include "xla/xla_data.pb.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"

namespace metaljax {

class MetalDeviceDescription final : public xla::PjRtDeviceDescription {
 public:
  MetalDeviceDescription(int id, int process_index);

  int id() const override { return id_; }
  int process_index() const override { return process_index_; }
  absl::string_view device_kind() const override { return kDeviceKind; }
  absl::string_view DebugString() const override { return debug_string_; }
  absl::string_view ToString() const override { return to_string_; }

  const absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::Span<const xla::PjRtMemorySpaceDescription* const> memory_spaces()
      const override {
    return memory_space_ptrs_;
  }

  absl::StatusOr<const xla::PjRtMemorySpaceDescription*> default_memory_space()
      const override {
    return &memory_space_;
  }

 private:
  int id_;
  int process_index_;
  std::string debug_string_;
  std::string to_string_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
  xla::PjRtMemorySpaceDescription memory_space_{kMemoryKind, /*kind_id=*/0};
  xla::PjRtMemorySpaceDescription host_memory_space_{kHostMemoryKind,
                                                    /*kind_id=*/1};
  std::vector<const xla::PjRtMemorySpaceDescription*> memory_space_ptrs_;
};

// Unified memory: on Apple silicon the GPU and CPU share one physical pool.
// The client still exposes TWO spaces -- "device" and "pinned_host" -- because
// the kind is something jax asks about and reports back to the user, and one
// space meant every host placement was silently answered with "device".  They
// name the same pool, so a copy between them is the identity and no allocation
// differs; what differs is that the plugin now says which one a buffer is on
// rather than assuming the question away.
class MetalMemorySpace final : public xla::PjRtMemorySpace {
 public:
  MetalMemorySpace(xla::PjRtClient* client, int id, absl::string_view kind,
                   int kind_id);

  xla::PjRtClient* client() const override { return client_; }
  absl::Span<xla::PjRtDevice* const> devices() const override;
  int id() const override { return id_; }
  absl::string_view kind() const override { return kind_; }
  int kind_id() const override { return kind_id_; }
  absl::string_view DebugString() const override { return debug_string_; }
  absl::string_view ToString() const override { return debug_string_; }
  PJRT_Memory* ToCApiPtr() override { return capi_delegator_.ToCApiPtr(); }

 private:
  xla::PjRtClient* client_;
  int id_;
  absl::string_view kind_;
  int kind_id_;
  std::string debug_string_;
  xla::PjRtMemorySpaceCApiDelegator capi_delegator_{this};
};

class MetalDevice final : public xla::PjRtDevice {
 public:
  MetalDevice(xla::PjRtClient* client, int id, int process_index);

  xla::PjRtClient* client() const override { return client_; }
  bool IsAddressable() const override { return true; }
  const MetalDeviceDescription& description() const override {
    return description_;
  }

  xla::LocalDeviceId local_device_id() const override {
    return xla::LocalDeviceId(description_.id());
  }
  xla::LocalChipId local_hardware_id() const override {
    return xla::LocalChipId(description_.id());
  }

  std::unique_ptr<xla::ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    // metaljax has no asynchronous device-activity tracking to expose.
    return nullptr;
  }

  absl::Status TransferToInfeed(const xla::LiteralSlice& literal) override;
  absl::Status TransferFromOutfeed(
      xla::MutableBorrowingLiteral literal) override;

  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override;
  absl::StatusOr<xla::PjRtMemorySpace*> default_memory_space() const override;
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override;

 private:
  xla::PjRtClient* client_;
  MetalDeviceDescription description_;
};

class MetalClient final : public xla::PjRtClient {
 public:
  MetalClient();
  ~MetalClient() override = default;

  absl::string_view platform_name() const override { return kPlatformName; }
  absl::string_view platform_version() const override {
    return kPlatformVersion;
  }
  xla::PjRtPlatformId platform_id() const override;

  int process_index() const override { return 0; }
  int device_count() const override { return devices_.size(); }
  int addressable_device_count() const override { return devices_.size(); }
  absl::Span<xla::PjRtDevice* const> devices() const override {
    return devices_;
  }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return devices_;
  }
  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }

  absl::StatusOr<xla::PjRtDevice*> LookupDevice(
      xla::GlobalDeviceId global_device_id) const override;
  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
      xla::LocalDeviceId local_device_id) const override;

  // P0 checkpoint 3: host -> device transfers into MLX arrays.  Dense,
  // contiguous, row-major f32 only; everything else is refused rather than
  // silently mis-transferred.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHostBuffer(
      const void* data, xla::PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      xla::PjRtMemorySpace* memory_space,
      const xla::Layout* device_layout) override;

  // The layout of a shape on this backend, which is the only one it has:
  // dense, row-major, no tiling.  jax asks for it through
  // `jax.Array.format`/`.layout` and through its memory profiler, so leaving
  // it Unimplemented (PjRtClient's default) turned three suite files red on
  // an attribute read that has nothing to do with computing.
  absl::StatusOr<xla::Layout> GetDefaultLayout(
      xla::PrimitiveType element_type,
      absl::Span<const int64_t> dims) override;

  // P0 checkpoint 4: prove that the C-API wrapper hands us a *parsed* MLIR
  // module (StableHLO, already upgraded from the VHLO portable artifact), then
  // fail cleanly.
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileAndLoad(
      xla::MaybeOwningMlirModule module,
      xla::CompileOptions options) override;

 private:
  std::vector<std::unique_ptr<MetalDevice>> owned_devices_;
  std::vector<std::unique_ptr<MetalMemorySpace>> owned_memory_spaces_;
  std::vector<xla::PjRtDevice*> devices_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;
};

std::unique_ptr<xla::PjRtClient> CreateMetalClient();

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_CLIENT_H_
