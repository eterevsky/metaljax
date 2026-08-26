/* metaljax: fully-native PJRT plugin for Apple-silicon GPUs (Stage 2).

One element-type table, read three ways: by the transfer path (PJRT wire type
-> device storage), by the lowering (MLIR element type -> the tape's dtype
code) and by the executable's result metadata (MLIR element type -> PJRT wire
type).  There is deliberately no second table: the tape's codes come from
`metaljax::dtype_codes()` at run time, keyed by the same names Stage 1's
src/metaljax/tape.py gated on, so a type the runtime cannot hold declines the
program instead of being guessed at here.

Licensed under the Apache License, Version 2.0.
==============================================================================*/

#ifndef METALJAX_PLUGIN_NATIVE_METAL_METAL_DTYPES_H_
#define METALJAX_PLUGIN_NATIVE_METAL_METAL_DTYPES_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/APFloat.h"
#include "mlir/IR/Types.h"
#include "mlx/dtype.h"
#include "xla/xla_data.pb.h"

namespace metaljax {

// How a PJRT element type crosses the wire and what holds it on the device.
// Ported from native/metaljax_native.cc's `type_info`, rule for rule: bf16 is
// its own bits in both places (never staged through f32 -- that doubled the
// transfer and canonicalized NaN payloads), and f64/c128 pass through STORED
// as f32/c64, which is what makes an f64 buffer legal while an f64 COMPUTE
// declines.
struct WireType {
  mlx::core::Dtype device;  // dtype stored on the device
  size_t host_item;         // bytes per element on the PJRT wire
  bool widen;               // wire is f64/c128, device is f32/c64
  int lanes;                // f32 scalars per element when widening (1 or 2)
  // An EMULATED element type (i4/ui4 and the f8/f6/f4 grids), whose device
  // storage holds the VALUE in a wider dtype: the wire byte is the type's own
  // encoding, so the transfer is a per-element CONVERSION rather than a copy.
  // `EmulatedKind` names which grid; -1 for every real type.
  int emulated = -1;
};

std::optional<WireType> WireTypeOf(xla::PrimitiveType type);

// The emulated grids, keyed by the same MLIR element names src/metaljax/
// dtypes.py's `EMULATED` used.  A "kind" is an index into that table; the
// encode/decode pair is llvm::APFloat's, which is the same specification
// ml_dtypes implements and is what the host transfer needs to agree with.
int EmulatedKindOfName(const std::string& name);
int EmulatedKindOfPrimitive(xla::PrimitiveType type);
// LOGICAL bits of one value (4 for i4/ui4/f4, 6 for the f6 pair, 8 for f8) --
// what XLA's memory layout counts and what a bitcast_convert reads, which is
// NOT the width of the storage the device holds it in.
int EmulatedBits(int kind);
// Wire byte -> value, and value -> wire byte (round to nearest, ties to even;
// overflow by the format's own rule -- an infinity, a NaN, or saturation).
// The byte is canonical: the bits above the format's width are ignored on
// decode and zero on encode.
float EmulatedDecode(int kind, uint8_t code);
uint8_t EmulatedEncode(int kind, float value);

// The element-type name src/metaljax/tape.py keyed the dtype table by -- the
// MLIR type as printed ("i1", "ui8", "bf16", "complex<f32>").  Spelled out
// rather than printed so that a type nobody thought about (a quantized or
// sub-byte element) falls out as nullopt instead of matching by accident.
std::optional<std::string> TapeElementName(mlir::Type type);

// The tape's dtype code for an MLIR element type, or nullopt when the runtime
// has no entry for it (f64, the emulated sub-byte grids, anything exotic).
std::optional<int> TapeDtypeCode(mlir::Type type);

// The device dtype for an MLIR element type; nullopt for the same set.
std::optional<mlx::core::Dtype> MxDtypeOf(mlir::Type type);

// The PJRT wire type for an MLIR element type, for result metadata.
std::optional<xla::PrimitiveType> PrimitiveTypeOf(mlir::Type type);

}  // namespace metaljax

#endif  // METALJAX_PLUGIN_NATIVE_METAL_METAL_DTYPES_H_
