/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "metal/metal_dtypes.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlx/mlx.h"
#include "program.h"
#include "xla/xla_data.pb.h"

namespace metaljax {

namespace mx = mlx::core;

namespace {

// The emulated element types, in src/metaljax/dtypes.py `EMULATED` order --
// which is also the order of the runtime's own grid table, though nothing
// depends on that: both sides key by NAME.  `sem` is nullptr for the two
// integer grids, whose encoding is the low nibble and needs no float
// machinery.
struct EmulatedWire {
  const char* name;
  xla::PrimitiveType wire;
  int bits;
  const llvm::fltSemantics& (*sem)();
  bool integer;   // i4 / ui4
  bool is_signed;
};

const EmulatedWire kEmulated[] = {
    {"f8E4M3FN", xla::F8E4M3FN, 8, &llvm::APFloat::Float8E4M3FN, false, false},
    {"f8E5M2", xla::F8E5M2, 8, &llvm::APFloat::Float8E5M2, false, false},
    {"f8E4M3", xla::F8E4M3, 8, &llvm::APFloat::Float8E4M3, false, false},
    {"f8E3M4", xla::F8E3M4, 8, &llvm::APFloat::Float8E3M4, false, false},
    {"f8E8M0FNU", xla::F8E8M0FNU, 8, &llvm::APFloat::Float8E8M0FNU, false,
     false},
    {"f8E4M3B11FNUZ", xla::F8E4M3B11FNUZ, 8, &llvm::APFloat::Float8E4M3B11FNUZ,
     false, false},
    {"f8E5M2FNUZ", xla::F8E5M2FNUZ, 8, &llvm::APFloat::Float8E5M2FNUZ, false,
     false},
    {"f8E4M3FNUZ", xla::F8E4M3FNUZ, 8, &llvm::APFloat::Float8E4M3FNUZ, false,
     false},
    {"f6E2M3FN", xla::F6E2M3FN, 6, &llvm::APFloat::Float6E2M3FN, false, false},
    {"f6E3M2FN", xla::F6E3M2FN, 6, &llvm::APFloat::Float6E3M2FN, false, false},
    {"f4E2M1FN", xla::F4E2M1FN, 4, &llvm::APFloat::Float4E2M1FN, false, false},
    {"i4", xla::S4, 4, nullptr, true, true},
    {"ui4", xla::U4, 4, nullptr, true, false},
};
constexpr int kNumEmulated = sizeof(kEmulated) / sizeof(kEmulated[0]);

}  // namespace

int EmulatedKindOfName(const std::string& name) {
  for (int i = 0; i < kNumEmulated; i++)
    if (name == kEmulated[i].name) return i;
  return -1;
}

int EmulatedKindOfPrimitive(xla::PrimitiveType type) {
  for (int i = 0; i < kNumEmulated; i++)
    if (kEmulated[i].wire == type) return i;
  return -1;
}

int EmulatedBits(int kind) {
  if (kind < 0 || kind >= kNumEmulated) return 0;
  return kEmulated[kind].bits;
}

float EmulatedDecode(int kind, uint8_t code) {
  const EmulatedWire& w = kEmulated[kind];
  const uint8_t masked =
      w.bits >= 8 ? code : static_cast<uint8_t>(code & ((1u << w.bits) - 1));
  if (w.integer) {
    if (!w.is_signed) return static_cast<float>(masked);
    // Two's complement in the low nibble: 0x8..0xF are -8..-1.
    return static_cast<float>(masked >= 8 ? static_cast<int>(masked) - 16
                                          : static_cast<int>(masked));
  }
  if (w.wire == xla::F8E8M0FNU) {
    // Exponent only, unsigned, no zero: code c is 2^(c-127) and 0xFF is the
    // single NaN.  APFloat carries the semantics but converting THROUGH it
    // returns an infinity for the NaN code, so the format is spelled out --
    // it is two lines and ml_dtypes, which is what the CPU backend's answers
    // come from, says exactly this.
    if (masked == 0xFF) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(masked) - 127);
  }
  llvm::APFloat v(w.sem(), llvm::APInt(w.bits, masked));
  bool lost = false;
  v.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven,
            &lost);
  return v.convertToFloat();
}

uint8_t EmulatedEncode(int kind, float value) {
  const EmulatedWire& w = kEmulated[kind];
  if (w.integer) {
    // 4-bit wrap, which is what the device-side grid already applied; doing
    // it again here is what keeps a value that never met the grid (a host
    // buffer built from a wider numpy array) honest.
    const int v = static_cast<int>(std::llrint(static_cast<double>(value)));
    return static_cast<uint8_t>(static_cast<unsigned>(v) & 0x0Fu);
  }
  if (w.wire == xla::F8E8M0FNU) {
    // Anything that is not a positive power of two in [2^-127, 2^127] --
    // zero, a negative, an infinity, a NaN, an exponent out of range -- is
    // the NaN code.  ml_dtypes' rule, measured.
    if (!(value > 0) || std::isinf(value)) return 0xFF;
    const int e = static_cast<int>(std::nearbyint(std::log2(
        static_cast<double>(value))));
    if (e < -127 || e > 127) return 0xFF;
    return static_cast<uint8_t>(e + 127);
  }
  llvm::APFloat v(value);
  bool lost = false;
  if (std::isnan(value) &&
      w.sem().nonFiniteBehavior == llvm::fltNonfiniteBehavior::FiniteOnly) {
    // The OCP FP4/FP6 formats have no NaN to convert to, and ml_dtypes maps
    // one to a ZERO whose sign is the OPPOSITE of the NaN's (measured: +NaN
    // becomes -0, -NaN becomes +0).  Reproduced rather than reasoned about:
    // the CPU backend's answers come from that cast, and APFloat -- which
    // has no rule to follow here -- returns +0 for both.
    const bool neg = std::signbit(value);
    return neg ? 0u : static_cast<uint8_t>(1u << (w.bits - 1));
  }
  v.convert(w.sem(), llvm::APFloat::rmNearestTiesToEven, &lost);
  const llvm::APInt bits = v.bitcastToAPInt();
  return static_cast<uint8_t>(bits.getZExtValue() &
                              ((w.bits >= 8) ? 0xFFu : ((1u << w.bits) - 1)));
}

std::optional<WireType> WireTypeOf(xla::PrimitiveType type) {
  // The emulated grids first: one wire byte per element (XLA's default layout
  // gives a sub-byte type a whole byte -- `primitive_util::ByteWidth` rounds
  // up and nothing sets `element_size_in_bits` here), decoded to a value in
  // the wider storage the device holds.
  if (int kind = EmulatedKindOfPrimitive(type); kind >= 0) {
    const mx::Dtype device =
        kEmulated[kind].integer
            ? (kEmulated[kind].is_signed ? mx::int8 : mx::uint8)
            : (type == xla::F8E8M0FNU ? mx::float32 : mx::float16);
    return WireType{device, 1, false, 1, kind};
  }
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
    if (w != 4 && w != 8 && w != 16 && w != 32 && w != 64) return std::nullopt;
    return absl::StrCat(it.isUnsigned() ? "ui" : "i", w);
  }
  // The emulated float grids, spelled out one by one for the reason this
  // whole function is spelled out: a type nobody has thought about must fall
  // out as nullopt rather than match by accident.
  if (mlir::isa<mlir::Float8E4M3FNType>(type)) return std::string("f8E4M3FN");
  if (mlir::isa<mlir::Float8E5M2Type>(type)) return std::string("f8E5M2");
  if (mlir::isa<mlir::Float8E4M3Type>(type)) return std::string("f8E4M3");
  if (mlir::isa<mlir::Float8E3M4Type>(type)) return std::string("f8E3M4");
  if (mlir::isa<mlir::Float8E8M0FNUType>(type))
    return std::string("f8E8M0FNU");
  if (mlir::isa<mlir::Float8E4M3B11FNUZType>(type))
    return std::string("f8E4M3B11FNUZ");
  if (mlir::isa<mlir::Float8E5M2FNUZType>(type))
    return std::string("f8E5M2FNUZ");
  if (mlir::isa<mlir::Float8E4M3FNUZType>(type))
    return std::string("f8E4M3FNUZ");
  if (mlir::isa<mlir::Float6E2M3FNType>(type)) return std::string("f6E2M3FN");
  if (mlir::isa<mlir::Float6E3M2FNType>(type)) return std::string("f6E3M2FN");
  if (mlir::isa<mlir::Float4E2M1FNType>(type)) return std::string("f4E2M1FN");
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
  if (int kind = EmulatedKindOfName(n); kind >= 0) return kEmulated[kind].wire;
  return std::nullopt;
}

}  // namespace metaljax
