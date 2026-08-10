/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

A PjRtBuffer backed by an mlx::core::array living in Apple unified memory.

P0 scope: dense, contiguous, row-major f32 only.  Anything else is rejected
with Unimplemented rather than silently mis-transferred.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_BUFFER_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlx/array.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace metaljax {

class MetalBuffer final : public xla::PjRtBuffer {
 public:
  MetalBuffer(xla::PjRtClient* client, xla::PjRtMemorySpace* memory_space,
              xla::PjRtDevice* device, xla::Shape shape,
              mlx::core::array array);

  const xla::Shape& on_device_shape() const override { return shape_; }
  xla::PjRtMemorySpace* memory_space() const override { return memory_space_; }
  xla::PjRtDevice* device() const override { return device_; }
  xla::PjRtClient* client() const override { return client_; }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  xla::Future<> ToLiteral(xla::MutableLiteralBase* literal) override;
  xla::Future<> LazyToLiteral(
      absl::AnyInvocable<xla::Future<xla::MutableLiteralBase*>() &&> generator)
      override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  xla::Future<> CopyRawToHost(void* dst, int64_t offset,
                              int64_t transfer_size) override;

  void Delete() override;
  bool IsDeleted() const override { return deleted_; }

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      xla::PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(xla::Future<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> Bitcast(
      xla::PrimitiveType element_type, absl::Span<const int64_t> dims,
      const xla::Layout* device_layout) override;

  // The array is materialised (mlx::core::eval) before the buffer is handed
  // out, so it is genuinely ready at construction time.
  xla::Future<> GetReadyFuture() override;

  // Unified memory is host-visible, but the buffer is device-resident from
  // PJRT's point of view: reporting true would let jax bypass transfers and
  // read MLX's storage directly.
  bool IsOnCpu() const override { return false; }

 private:
  // Bytes of the (contiguous) payload; fails if the buffer was deleted.
  absl::StatusOr<size_t> CopyOut(void* dst, size_t dst_size,
                                 size_t src_offset) const;

  xla::PjRtClient* client_;
  xla::PjRtMemorySpace* memory_space_;
  xla::PjRtDevice* device_;
  xla::Shape shape_;
  std::optional<mlx::core::array> array_;
  bool deleted_ = false;
};

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_BUFFER_H_
