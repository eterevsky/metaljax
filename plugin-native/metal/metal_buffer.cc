/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mlx/array.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace metaljax {

namespace mx = mlx::core;

MetalBuffer::MetalBuffer(xla::PjRtClient* client,
                         xla::PjRtMemorySpace* memory_space,
                         xla::PjRtDevice* device, xla::Shape shape,
                         mx::array array)
    : client_(client),
      memory_space_(memory_space),
      device_(device),
      shape_(std::move(shape)),
      array_(std::move(array)) {}

absl::StatusOr<size_t> MetalBuffer::CopyOut(void* dst, size_t dst_size,
                                            size_t src_offset) const {
  if (deleted_ || !array_.has_value()) {
    return absl::FailedPreconditionError(
        "metaljax: buffer has been deleted.");
  }
  // The array was evaluated when the buffer was created, so this is a plain
  // read of already-materialised unified memory.
  const size_t src_size = array_->nbytes();
  if (src_offset > src_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "metaljax: offset %d beyond buffer of %d bytes", src_offset, src_size));
  }
  const size_t available = src_size - src_offset;
  if (dst_size > available) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "metaljax: asked for %d bytes, buffer has %d", dst_size, available));
  }
  const auto* src = static_cast<const std::byte*>(
      static_cast<const void*>(array_->data<float>()));
  std::memcpy(dst, src + src_offset, dst_size);
  return dst_size;
}

xla::Future<> MetalBuffer::ToLiteral(xla::MutableLiteralBase* literal) {
  if (!literal->shape().IsArray()) {
    return xla::Future<>(absl::UnimplementedError(
        "metaljax: only array-shaped literals are supported."));
  }
  const size_t dst_size = literal->size_bytes();
  absl::StatusOr<size_t> copied =
      CopyOut(literal->untyped_data(), dst_size, /*src_offset=*/0);
  if (!copied.ok()) {
    return xla::Future<>(copied.status());
  }
  return xla::Future<>(absl::OkStatus());
}

xla::Future<> MetalBuffer::LazyToLiteral(
    absl::AnyInvocable<xla::Future<xla::MutableLiteralBase*>() &&> generator) {
  // The contents are already materialised, so there is nothing to defer.
  xla::Future<xla::MutableLiteralBase*> future = std::move(generator)();
  const absl::StatusOr<xla::MutableLiteralBase*>& literal = future.Await();
  if (!literal.ok()) {
    return xla::Future<>(literal.status());
  }
  return ToLiteral(*literal);
}

absl::StatusOr<size_t> MetalBuffer::GetOnDeviceSizeInBytes() const {
  if (deleted_ || !array_.has_value()) {
    return absl::FailedPreconditionError("metaljax: buffer has been deleted.");
  }
  return array_->nbytes();
}

xla::Future<> MetalBuffer::CopyRawToHost(void* dst, int64_t offset,
                                         int64_t transfer_size) {
  if (offset < 0 || transfer_size < 0) {
    return xla::Future<>(
        absl::InvalidArgumentError("metaljax: negative offset/size."));
  }
  absl::StatusOr<size_t> copied = CopyOut(
      dst, static_cast<size_t>(transfer_size), static_cast<size_t>(offset));
  if (!copied.ok()) {
    return xla::Future<>(copied.status());
  }
  return xla::Future<>(absl::OkStatus());
}

void MetalBuffer::Delete() {
  array_.reset();
  deleted_ = true;
}

xla::Future<> MetalBuffer::GetReadyFuture() {
  if (deleted_) {
    return xla::Future<>(
        absl::FailedPreconditionError("metaljax: buffer has been deleted."));
  }
  return xla::Future<>(absl::OkStatus());
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
MetalBuffer::AcquireExternalReference() {
  return absl::UnimplementedError(
      "metaljax: external buffer references are not supported yet.");
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
MetalBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  return absl::UnimplementedError(
      "metaljax: device memory ownership transfer is not supported yet.");
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> MetalBuffer::CopyToMemorySpace(
    xla::PjRtMemorySpace* dst_memory_space) {
  return absl::UnimplementedError(
      "metaljax: cross-memory-space copies are not supported yet.");
}

void MetalBuffer::CopyToRemoteDevice(
    xla::Future<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  std::move(on_done)(
      absl::UnimplementedError("metaljax: remote sends are not supported."),
      /*sends_were_enqueued=*/false);
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> MetalBuffer::Bitcast(
    xla::PrimitiveType element_type, absl::Span<const int64_t> dims,
    const xla::Layout* device_layout) {
  return absl::UnimplementedError(
      "metaljax: buffer bitcast is not supported yet.");
}

}  // namespace metaljax
