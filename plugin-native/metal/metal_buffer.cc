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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "metal/metal_dtypes.h"
#include "metal/metal_stream.h"
#include "mlx/mlx.h"
#include "program.h"
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
  std::unique_lock<std::recursive_mutex> submission = SubmissionLock();
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
    // ...and KEEP it.  A buffer whose array is a view (every `jnp.ones` is a
    // broadcast of one element) would otherwise gather afresh on every read,
    // which costs the gather each time and, worse, gives
    // `unsafe_buffer_pointer` a NEW address per call -- where jax reads it as
    // the buffer's identity and asserts on it (`testArrayCopy`, and XLA's
    // no-alias contract generally).  The values are the same values; from
    // PJRT's point of view this buffer always held its own dense storage.
    array_ = c;
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
  if (wire->emulated >= 0) {
    // An emulated grid: the device holds VALUES in a wider dtype, so the wire
    // byte has to be re-encoded per element.  Rounding is RNE and overflow is
    // the format's own rule (an infinity, a NaN, or saturation), which is what
    // llvm::APFloat implements and what ml_dtypes -- the reference the CPU
    // backend's answers come from -- implements too.
    absl::StatusOr<mx::array> settled = Settled();
    if (!settled.ok()) return xla::Future<>(settled.status());
    const int64_t n = settled->size();
    if (dst_size != static_cast<size_t>(n)) {
      return xla::Future<>(absl::InvalidArgumentError(absl::StrFormat(
          "metaljax: literal wants %d bytes, the buffer holds %d elements",
          dst_size, n)));
    }
    auto* out = static_cast<uint8_t*>(literal->untyped_data());
    const mx::Dtype dt = settled->dtype();
    for (int64_t i = 0; i < n; i++) {
      float v;
      if (dt == mx::int8) {
        v = static_cast<float>(settled->data<int8_t>()[i]);
      } else if (dt == mx::uint8) {
        v = static_cast<float>(settled->data<uint8_t>()[i]);
      } else if (dt == mx::float32) {
        v = settled->data<float>()[i];
      } else {
        llvm::APFloat h(llvm::APFloat::IEEEhalf(),
                        llvm::APInt(16, settled->data<uint16_t>()[i]));
        bool lost = false;
        h.convert(llvm::APFloat::IEEEsingle(),
                  llvm::APFloat::rmNearestTiesToEven, &lost);
        v = h.convertToFloat();
      }
      out[i] = EmulatedEncode(wire->emulated, v);
    }
    return xla::Future<>(absl::OkStatus());
  }
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
  if (wire.has_value() && (wire->widen || wire->emulated >= 0)) {
    // A raw copy would hand out the STORAGE under the wire's shape: f32 under
    // an f64 one, or a float16 grid value under a one-byte f8 one.  ToLiteral
    // is the path that converts; this one declines.
    return xla::Future<>(absl::UnimplementedError(
        "metaljax: raw host copies of f64/c128 or emulated sub-byte buffers "
        "are not supported (their device storage is not the wire's bits)."));
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
  // Unified memory: this client has exactly ONE memory space, so there is no
  // space to cross -- but there is still a copy to make.  jax asks for this
  // whenever it wants a buffer that provably shares nothing with the source
  // (`jax.device_put(x, may_alias=False)`, and the donating device_put, which
  // deletes the source right afterwards), so a handle on the same storage
  // would be the wrong answer, not a fast one.  `mx::copy` is the one that
  // really allocates: `contiguous`/`astype` short-circuit on an array that is
  // already both.
  if (dst_memory_space != nullptr && dst_memory_space->client() != client_) {
    return absl::UnimplementedError(
        "metaljax: copies to another client's memory space are not "
        "supported.");
  }
  ASSIGN_OR_RETURN(mx::array settled, Settled());
  // The governor gates this bulk path too (the no-panic contract): a loop of
  // `device_put(x, may_alias=False)` moves a model's worth of bytes, and the
  // gate has to be BEFORE the copy that makes them resident.
  try {
    governor_admit(static_cast<int64_t>(settled.nbytes()), MemWhere::kIngest);
  } catch (const std::exception& e) {
    return absl::ResourceExhaustedError(e.what());
  }
  mx::array fresh = mx::copy(settled);
  {
    BindThread();
    std::unique_lock<std::recursive_mutex> submission = SubmissionLock();
    fresh.eval();   // honestly ready, like every buffer this plugin hands out
    // The second bulk-ingest entry: a loop of `device_put(x, may_alias=False)`
    // moves as many bytes as one of BufferFromHostBuffer, and unlike that one
    // it allocates through MLX -- so its garbage really does land in the
    // buffer cache the ingest cadence clears (runtime.cc `ingest_account`).
    ingest_account(static_cast<int64_t>(fresh.nbytes()));
  }
  return std::make_unique<MetalBuffer>(
      client_,
      dst_memory_space != nullptr ? dst_memory_space : memory_space_, device_,
      shape_, std::move(fresh));
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
