/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_executable.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "metal/metal_buffer.h"
#include "metal/metal_names.h"
#include "metal/metal_stream.h"
#include "mlx/mlx.h"
#include "program.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace metaljax {

namespace {

namespace mx = mlx::core;

bool SameShape(const mx::array& a, const ValueSpec& spec) {
  if (a.shape().size() != spec.dims.size()) return false;
  for (size_t i = 0; i < spec.dims.size(); i++)
    if (static_cast<int64_t>(a.shape()[i]) != spec.dims[i]) return false;
  return true;
}

std::string ShapeString(const mx::array& a) {
  std::string s = "[";
  for (size_t i = 0; i < a.shape().size(); i++)
    absl::StrAppend(&s, i ? "," : "", a.shape()[i]);
  return absl::StrCat(s, "]");
}

// METALJAX_DEBUG=1 reports what one execute spent of the runtime's cadences.
// The counters are the executor's own (native/program.h `g_stats`), so this
// is the only window a process with no interpreter in it has on them -- and
// the loop counters are what say the flush discipline a long loop depends on
// really engaged.  Process-wide and unsynchronized, like the counters
// themselves: with two executes in flight the delta below attributes some of
// the other one's work, which is a diagnostic's business and not a run's.
const bool kDebug = [] {
  const char* v = std::getenv("METALJAX_DEBUG");
  return v != nullptr && std::string(v) == "1";
}();

std::string StatsDelta(const Stats& before, const Stats& after) {
  return absl::StrFormat(
      "flushes=%d(+clear %d) loop_flushes=%d(+clear %d) limit_retries=%d "
      "serial_loops=%d pipelined_loops=%d pipelined_steps=%d "
      "compiles=%d compiled_calls=%d unrolls=%d drops=%d/%d",
      after.flushes - before.flushes,
      after.cache_clears - before.cache_clears,
      after.loop_flushes - before.loop_flushes,
      after.loop_clears - before.loop_clears,
      after.limit_retries - before.limit_retries,
      after.serial_loops - before.serial_loops,
      after.pipelined_loops - before.pipelined_loops,
      after.pipelined_steps - before.pipelined_steps,
      // P5: the compiled path is otherwise invisible in a process with no
      // interpreter -- `compiles` counts traces BUILT, `compiled_calls`
      // replays (main, a while body, one chunk), and `drops` the two ways a
      // compiled path retires (compile_drops / chunk_drops).
      after.compiles - before.compiles,
      after.compiled_calls - before.compiled_calls,
      after.unrolls - before.unrolls,
      after.compile_drops - before.compile_drops,
      after.chunk_drops - before.chunk_drops);
}

}  // namespace

MetalLoadedExecutable::MetalLoadedExecutable(
    xla::PjRtClient* client, xla::PjRtDevice* device,
    xla::PjRtMemorySpace* memory_space, LoweredProgram lowered,
    xla::CompileOptions options, std::string name)
    : client_(client),
      device_(device),
      memory_space_(memory_space),
      lowered_(std::make_shared<const LoweredProgram>(std::move(lowered))),
      name_(std::move(name)),
      executable_(std::make_unique<MetalExecutable>(lowered_,
                                                    std::move(options), name_)),
      device_assignment_(/*replica_count=*/1, /*computation_count=*/1) {
  device_assignment_(0, 0) = device->id();
  logical_ids_.push_back(LogicalDeviceIds{/*replica=*/0, /*partition=*/0});
  devices_.push_back(device);
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
MetalLoadedExecutable::RunOnce(
    absl::Span<xla::PjRtBuffer* const> argument_handles) const {
  if (deleted_) {
    return absl::FailedPreconditionError(
        "metaljax-native: the executable has been deleted.");
  }
  if (argument_handles.size() != lowered_->parameters.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "metaljax-native: %s takes %d arguments, got %d", name_,
        lowered_->parameters.size(), argument_handles.size()));
  }
  BindThread();
  std::unique_lock<std::mutex> submission = SubmissionLock();

  std::vector<mx::array> inputs;
  inputs.reserve(argument_handles.size());
  for (size_t i = 0; i < argument_handles.size(); i++) {
    xla::PjRtBuffer* handle = argument_handles[i];
    if (handle == nullptr || handle->client() != client_) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "metaljax-native: argument %d is not a buffer of this client", i));
    }
    // Safe without RTTI: this client hands out nothing but MetalBuffers, and
    // the client identity above is what says the buffer came from here.
    const auto* buffer = static_cast<const MetalBuffer*>(handle);
    if (!buffer->array().has_value()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "metaljax-native: argument %d has been deleted", i));
    }
    const mx::array& a = *buffer->array();
    const ValueSpec& spec = lowered_->parameters[i];
    if (a.dtype() != spec.dtype || !SameShape(a, spec)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "metaljax-native: argument %d is %s, the program expects %s", i,
          ShapeString(a), xla::ShapeUtil::HumanString(spec.shape())));
    }
    inputs.push_back(a);
  }

  std::vector<mx::array> outs;
  const Stats before = g_stats;
  try {
    outs = lowered_->program->run(std::move(inputs));
    // P2 executes SYNCHRONOUSLY: correctness before pipelining.  Settling
    // here is what makes every buffer this returns honestly ready, and it is
    // the one thing to revisit first when the plugin is measured -- the Stage
    // 1 engine hands out lazy arrays and submits with async_eval instead.
    mx::eval(outs);
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("metaljax-native: ", name_, " failed: ", e.what()));
  }
  if (kDebug) {
    std::fprintf(stderr, "[metaljax-native] %s: %s\n", name_.c_str(),
                 StatsDelta(before, g_stats).c_str());
    std::fflush(stderr);
  }

  if (outs.size() != lowered_->results.size()) {
    return absl::InternalError(absl::StrFormat(
        "metaljax-native: %s returned %d values, the module declares %d",
        name_, outs.size(), lowered_->results.size()));
  }
  std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
  buffers.reserve(outs.size());
  for (size_t j = 0; j < outs.size(); j++) {
    const ValueSpec& spec = lowered_->results[j];
    // A mismatch here would be a lowering bug handing back the wrong bytes
    // under the right shape, which is the one failure mode this plugin must
    // never have: check it on every call rather than trusting the tape.
    if (outs[j].dtype() != spec.dtype || !SameShape(outs[j], spec)) {
      return absl::InternalError(absl::StrFormat(
          "metaljax-native: %s result %d came back as %s, the module declares "
          "%s",
          name_, j, ShapeString(outs[j]),
          xla::ShapeUtil::HumanString(spec.shape())));
    }
    buffers.push_back(std::make_unique<MetalBuffer>(
        client_, memory_space_, device_, spec.shape(), outs[j]));
  }
  return buffers;
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>
MetalLoadedExecutable::Execute(
    absl::Span<const std::vector<xla::PjRtBuffer*>> argument_handles,
    const xla::ExecuteOptions& options,
    std::optional<std::vector<xla::Future<>>>& returned_futures) const {
  if (argument_handles.size() != 1) {
    return absl::UnimplementedError(absl::StrFormat(
        "metaljax-native: one device, got %d argument lists",
        argument_handles.size()));
  }
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers,
                   RunOnce(argument_handles[0]));
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> result;
  result.push_back(std::move(buffers));
  if (returned_futures.has_value()) {
    // The run above already settled: the execution really is complete, so a
    // ready future is the truth and not a shortcut.
    returned_futures->clear();
    returned_futures->push_back(xla::Future<>(absl::OkStatus()));
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
MetalLoadedExecutable::ExecuteSharded(
    absl::Span<xla::PjRtBuffer* const> argument_handles,
    xla::PjRtDevice* device, const xla::ExecuteOptions& options,
    std::optional<xla::Future<>>& returned_future, bool fill_future) const {
  if (device != device_) {
    return absl::InvalidArgumentError(
        "metaljax-native: the executable is loaded on a different device.");
  }
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers,
                   RunOnce(argument_handles));
  if (fill_future) returned_future = xla::Future<>(absl::OkStatus());
  return buffers;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
MetalLoadedExecutable::ExecutePortable(
    absl::Span<xla::PjRtBuffer* const> argument_handles,
    xla::PjRtDevice* device, const xla::ExecuteOptions& options,
    std::optional<xla::Future<>>& returned_future, bool fill_future) const {
  return ExecuteSharded(argument_handles, device, options, returned_future,
                        fill_future);
}

// --------------------------------------------------------------------------
// the metadata half (see the header: the C-API wrapper asks THIS object)
// --------------------------------------------------------------------------

absl::StatusOr<std::vector<xla::Shape>> MetalExecutable::GetOutputShapes()
    const {
  std::vector<xla::Shape> shapes;
  shapes.reserve(lowered_->results.size());
  for (const ValueSpec& spec : lowered_->results) shapes.push_back(spec.shape());
  // One program, whose result is the TUPLE of the function's results: that is
  // the shape XLA's own implementations read out of an HloModule, and what the
  // element-type and dimension queries below are consistent with.
  return std::vector<xla::Shape>{xla::ShapeUtil::MakeTupleShape(shapes)};
}

absl::StatusOr<std::vector<std::vector<xla::PrimitiveType>>>
MetalExecutable::GetOutputElementTypes() const {
  std::vector<xla::PrimitiveType> types;
  types.reserve(lowered_->results.size());
  for (const ValueSpec& spec : lowered_->results) types.push_back(spec.type);
  return std::vector<std::vector<xla::PrimitiveType>>{std::move(types)};
}

absl::StatusOr<std::vector<std::vector<xla::DimensionVector>>>
MetalExecutable::GetOutputDimensions() const {
  std::vector<xla::DimensionVector> dims;
  dims.reserve(lowered_->results.size());
  for (const ValueSpec& spec : lowered_->results)
    dims.push_back(xla::DimensionVector(spec.dims.begin(), spec.dims.end()));
  return std::vector<std::vector<xla::DimensionVector>>{std::move(dims)};
}

namespace {

std::vector<std::shared_ptr<const xla::PjRtLayout>> DefaultLayouts(
    const std::vector<ValueSpec>& specs) {
  // Dense row-major, which is the only layout this backend has: MLX's storage
  // is major-to-minor and the transfer paths refuse anything else.
  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
  layouts.reserve(specs.size());
  for (const ValueSpec& spec : specs) {
    layouts.push_back(std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::GetDefaultLayoutForShape(spec.shape())));
  }
  return layouts;
}

}  // namespace

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
MetalExecutable::GetParameterLayouts() const {
  return DefaultLayouts(lowered_->parameters);
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
MetalExecutable::GetOutputLayouts() const {
  return DefaultLayouts(lowered_->results);
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
MetalExecutable::GetParameterMemoryKinds() const {
  return std::vector<std::vector<absl::string_view>>{
      std::vector<absl::string_view>(lowered_->parameters.size(), kMemoryKind)};
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
MetalExecutable::GetOutputMemoryKinds() const {
  return std::vector<std::vector<absl::string_view>>{
      std::vector<absl::string_view>(lowered_->results.size(), kMemoryKind)};
}

}  // namespace metaljax
