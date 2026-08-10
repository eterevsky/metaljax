/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_dtypes.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlx/mlx.h"
#include "program.h"
#include "xla/xla_data.pb.h"

namespace metaljax {

namespace mx = mlx::core;

std::optional<WireType> WireTypeOf(xla::PrimitiveType type) {
  switch (type) {
    case xla::PRED: return WireType{mx::bool_, 1, false, 1};
    case xla::S8:   return WireType{mx::int8, 1, false, 1};
    case xla::S16:  return WireType{mx::int16, 2, false, 1};
    case xla::S32:  return WireType{mx::int32, 4, false, 1};
    case xla::S64:  return WireType{mx::int64, 8, false, 1};
    case xla::U8:   return WireType{mx::uint8, 1, false, 1};
    case xla::U16:  return WireType{mx::uint16, 2, false, 1};
    case xla::U32:  return WireType{mx::uint32, 4, false, 1};
    case xla::U64:  return WireType{mx::uint64, 8, false, 1};
    case xla::F16:  return WireType{mx::float16, 2, false, 1};
    case xla::F32:  return WireType{mx::float32, 4, false, 1};
    // Metal has no doubles: an f64 buffer is STORED as f32 and widened back
    // on egress, which is the pass-through half of the STRICT policy.  An
    // f64 computation still declines -- the lowering finds no dtype code.
    case xla::F64:  return WireType{mx::float32, 8, true, 1};
    case xla::BF16: return WireType{mx::bfloat16, 2, false, 1};
    case xla::C64:  return WireType{mx::complex64, 8, false, 2};
    case xla::C128: return WireType{mx::complex64, 16, true, 2};
    default: return std::nullopt;
  }
}

std::optional<std::string> TapeElementName(mlir::Type type) {
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(type)) {
    // StableHLO spells signed integers signless and unsigned ones unsigned;
    // a genuinely `si`-typed element is not something this backend has ever
    // seen, so it declines rather than being mapped to its signless twin.
    const unsigned w = it.getWidth();
    if (it.isSigned()) return std::nullopt;
    if (w == 1) return std::string("i1");
    if (w != 8 && w != 16 && w != 32 && w != 64) return std::nullopt;
    return absl::StrCat(it.isUnsigned() ? "ui" : "i", w);
  }
  if (mlir::isa<mlir::Float16Type>(type)) return std::string("f16");
  if (mlir::isa<mlir::BFloat16Type>(type)) return std::string("bf16");
  if (mlir::isa<mlir::Float32Type>(type)) return std::string("f32");
  if (mlir::isa<mlir::Float64Type>(type)) return std::string("f64");
  if (auto ct = mlir::dyn_cast<mlir::ComplexType>(type)) {
    if (mlir::isa<mlir::Float32Type>(ct.getElementType()))
      return std::string("complex<f32>");
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<int> TapeDtypeCode(mlir::Type type) {
  static const auto* codes = [] {
    auto* m = new absl::flat_hash_map<std::string, int>();
    for (const std::pair<std::string, int>& kv : dtype_codes())
      m->emplace(kv.first, kv.second);
    return m;
  }();
  std::optional<std::string> name = TapeElementName(type);
  if (!name.has_value()) return std::nullopt;
  auto it = codes->find(*name);
  if (it == codes->end()) return std::nullopt;
  return it->second;
}

std::optional<mx::Dtype> MxDtypeOf(mlir::Type type) {
  std::optional<int> code = TapeDtypeCode(type);
  if (!code.has_value()) return std::nullopt;
  return dtype_of(*code);
}

std::optional<xla::PrimitiveType> PrimitiveTypeOf(mlir::Type type) {
  std::optional<std::string> name = TapeElementName(type);
  if (!name.has_value()) return std::nullopt;
  const std::string& n = *name;
  if (n == "i1") return xla::PRED;
  if (n == "i8") return xla::S8;
  if (n == "i16") return xla::S16;
  if (n == "i32") return xla::S32;
  if (n == "i64") return xla::S64;
  if (n == "ui8") return xla::U8;
  if (n == "ui16") return xla::U16;
  if (n == "ui32") return xla::U32;
  if (n == "ui64") return xla::U64;
  if (n == "f16") return xla::F16;
  if (n == "f32") return xla::F32;
  if (n == "f64") return xla::F64;
  if (n == "bf16") return xla::BF16;
  if (n == "complex<f32>") return xla::C64;
  return std::nullopt;
}

}  // namespace metaljax
