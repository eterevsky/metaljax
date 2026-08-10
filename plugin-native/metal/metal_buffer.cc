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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "metal/metal_dtypes.h"
#include "metal/metal_stream.h"
#include "mlx/mlx.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

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

absl::StatusOr<mx::array> MetalBuffer::Settled() const {
  if (deleted_ || !array_.has_value()) {
    return absl::FailedPreconditionError("metaljax: buffer has been deleted.");
  }
  BindThread();
  // Settle the array ITSELF and build a `contiguous` only for a layout that
  // needs gathering.  Wrapping every result in a fresh `contiguous` node and
  // evaluating THAT costs a full round trip through the stream -- 20us,
  // measured, per output, even though the copy short-circuits to a shared
  // buffer -- where `eval` on a value that is already available returns in
  // 0.1us (notes/cpp-migration-plan.md, the M5c deficit).
  mx::array c = *array_;
  c.eval();
  if (!c.flags().row_contiguous ||
      c.data_size() != static_cast<size_t>(c.size())) {
    c = mx::contiguous(c);
    c.eval();
  }
  return c;
}

absl::StatusOr<size_t> MetalBuffer::CopyOut(void* dst, size_t dst_size,
                                            size_t src_offset) const {
  ASSIGN_OR_RETURN(mx::array c, Settled());
  const size_t src_size = c.nbytes();
  if (src_offset > src_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "metaljax: offset %d beyond buffer of %d bytes", src_offset, src_size));
  }
  const size_t available = src_size - src_offset;
  if (dst_size > available) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "metaljax: asked for %d bytes, buffer has %d", dst_size, available));
  }
  if (dst_size == 0) {
    // MLX 0.32 can hand out null-backed zero-size buffers whose data pointer
    // segfaults on access, so a zero-length read never touches one.
    return size_t{0};
  }
  std::memcpy(dst, c.data<char>() + src_offset, dst_size);
  return dst_size;
}

xla::Future<> MetalBuffer::ToLiteral(xla::MutableLiteralBase* literal) {
  if (!literal->shape().IsArray()) {
    return xla::Future<>(absl::UnimplementedError(
        "metaljax: only array-shaped literals are supported."));
  }
  std::optional<WireType> wire = WireTypeOf(shape_.element_type());
  if (!wire.has_value()) {
    return xla::Future<>(absl::UnimplementedError(absl::StrCat(
        "metaljax: no host transfer for ",
        xla::PrimitiveType_Name(shape_.element_type()))));
  }
  const size_t dst_size = literal->size_bytes();
  if (!wire->widen) {
    absl::StatusOr<size_t> copied =
        CopyOut(literal->untyped_data(), dst_size, /*src_offset=*/0);
    if (!copied.ok()) return xla::Future<>(copied.status());
    return xla::Future<>(absl::OkStatus());
  }
  // f64/c128: the device holds f32/c64, so widen on the way out.  Same
  // float->double conversion numpy's astype would do.
  absl::StatusOr<mx::array> settled = Settled();
  if (!settled.ok()) return xla::Future<>(settled.status());
  const int64_t scalars = settled->size() * wire->lanes;
  if (dst_size != static_cast<size_t>(scalars) * sizeof(double)) {
    return xla::Future<>(absl::InvalidArgumentError(absl::StrFormat(
        "metaljax: literal wants %d bytes, the buffer widens to %d", dst_size,
        static_cast<size_t>(scalars) * sizeof(double))));
  }
  if (scalars != 0) {
    const float* narrow = settled->data<float>();
    auto* wide = static_cast<double*>(literal->untyped_data());
    for (int64_t i = 0; i < scalars; i++) wide[i] = narrow[i];
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
  // The WIRE size, which is what the Stage 1 plugin reports too: for a widened
  // dtype the device really does hold less, but a caller sizing a host
  // transfer from this number would come up short.
  return static_cast<size_t>(xla::ShapeUtil::ByteSizeOf(shape_));
}

xla::Future<> MetalBuffer::CopyRawToHost(void* dst, int64_t offset,
                                         int64_t transfer_size) {
  if (offset < 0 || transfer_size < 0) {
    return xla::Future<>(
        absl::InvalidArgumentError("metaljax: negative offset/size."));
  }
  std::optional<WireType> wire = WireTypeOf(shape_.element_type());
  if (wire.has_value() && wire->widen) {
    // A raw copy of a widened buffer would hand out the f32 storage under an
    // f64 shape.  ToLiteral is the path that widens; this one declines.
    return xla::Future<>(absl::UnimplementedError(
        "metaljax: raw host copies of f64/c128 buffers are not supported "
        "(their device storage is f32/c64)."));
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

namespace {

// A hold on the array's unified-memory storage.  Keeping the mx::array by
// value is the hold: MLX frees a buffer when the last handle on it dies, so
// while this object lives the pointer stays valid.
class MetalExternalReference final : public xla::PjRtBuffer::ExternalReference {
 public:
  explicit MetalExternalReference(mx::array array) : array_(std::move(array)) {
    data_ptr_ = array_.size() == 0 ? nullptr
                                   : static_cast<void*>(array_.data<char>());
  }
  ~MetalExternalReference() override = default;

 private:
  mx::array array_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
MetalBuffer::AcquireExternalReference() {
  // Settled first: an unevaluated array has no storage to point at, and this
  // is what jax reads through `unsafe_buffer_pointer` to check XLA's no-alias
  // contract -- two buffers that share storage must come back equal here, and
  // two that do not must not.
  ASSIGN_OR_RETURN(mx::array settled, Settled());
  return std::make_unique<MetalExternalReference>(std::move(settled));
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
