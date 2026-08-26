// metaljax native engine — shape and layout ops (ported from Stage 1's
// src/metaljax/ops/shape.py, deleted 0.11.6, ef5774d).
//
// Reshape, transpose, broadcast, slice/pad/reverse, the dynamic slice pair
// and the bitcast -- every one of them a rearrangement whose shapes tape.py
// already resolved, so what is left here is the MLX calls in the Python
// handler's order. `stablehlo.constant` rides along: its value crossed the
// boundary once, at lowering, and the entry simply hands it out.

#include "program.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

bool Program::step_shape(const Entry& e,
                         std::vector<std::optional<mx::array>>& env,
                         bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    // --- shape (ops/shape.py) ---
    case kReshape:
      env[e.outs[0]] = mx::reshape(in(0), shape(at, 1, at[0]));
      break;
    case kTranspose:
      env[e.outs[0]] = mx::transpose(in(0), axes(at, 1, at[0]));
      break;
    case kBroadcastInDim: {
      // _broadcast_in_dim: unsorted broadcast_dimensions become a
      // transpose, then the operand reshapes to an interim shape with a
      // 1 in every dim it does not name and broadcasts out. The perm and
      // the interim shape are static, so tape.py resolved both.
      mx::array x = in(0);
      size_t p = 0;
      bool do_transpose = at[p++] != 0;
      int64_t in_rank = at[p++];
      if (do_transpose) x = mx::transpose(x, axes(at, p, in_rank));
      p += static_cast<size_t>(in_rank);
      int64_t out_rank = at[p++];
      mx::Shape interim = shape(at, p, out_rank);
      p += static_cast<size_t>(out_rank);
      mx::Shape out = shape(at, p, out_rank);
      env[e.outs[0]] = mx::broadcast_to(mx::reshape(x, interim), out);
      break;
    }
    case kSlice: {
      int64_t rank = at[0];
      env[e.outs[0]] = mx::slice(
          in(0), shape(at, 1, rank), shape(at, 1 + rank, rank),
          shape(at, 1 + 2 * rank, rank));
      break;
    }
    case kConcatenate: {
      std::vector<mx::array> parts;
      parts.reserve(e.ins.size());
      for (size_t i = 0; i < e.ins.size(); i++) parts.push_back(in(i));
      env[e.outs[0]] =
          mx::concatenate(std::move(parts), static_cast<int>(at[0]));
      break;
    }
    case kIota: {
      // _iota: ramp along `dim`, broadcast, cast. MLX has no bool arange,
      // so the ramp runs in int32 for a bool result (the Python handler's
      // complex arm is unreachable: complex declines).
      int dim = static_cast<int>(at[0]);
      mx::Dtype ramp_dt = dtype_of(at[1]);
      mx::Dtype dt = dtype_of(at[2]);
      int64_t rank = at[3];
      mx::Shape out = shape(at, 4, rank);
      mx::array ramp = mx::arange(static_cast<double>(out[dim]), ramp_dt);
      mx::Shape view(static_cast<size_t>(rank));
      for (int64_t i = 0; i < rank; i++) view[i] = 1;
      view[dim] = out[dim];
      env[e.outs[0]] =
          mx::astype(mx::broadcast_to(mx::reshape(ramp, view), out), dt);
      break;
    }

    case kPad: {
      // ops/shape.py _pad: interior dilation, then edge pads, then the
      // crop negative pads mean. tape.py resolved which stages run and
      // every shape they produce; each is read whether it runs or not so
      // the cursor stays aligned.
      Cursor c(at);
      mx::array x = in(0);
      mx::array fill = mx::astype(in(1), x.dtype());
      bool interior = c.flag();
      mx::Shape dilated = c.shp();
      std::vector<int> istrides = c.vec();
      if (interior) {
        // The Python handler materializes the broadcast before writing
        // into its strided slice (`mx.array(exp)`); mx::contiguous is
        // that materialization, and slice_update is the write.
        mx::array base = mx::contiguous(mx::broadcast_to(fill, dilated));
        mx::Shape start(dilated.size(), 0);
        mx::Shape strides(istrides.begin(), istrides.end());
        x = mx::slice_update(base, x, start, dilated, strides);
      }
      bool padded = c.flag();
      std::vector<int> lo_w = c.vec(), hi_w = c.vec();
      if (padded) {
        std::vector<int> ax(lo_w.size());
        for (size_t i = 0; i < ax.size(); i++) ax[i] = static_cast<int>(i);
        x = mx::pad(x, ax, mx::Shape(lo_w.begin(), lo_w.end()),
                    mx::Shape(hi_w.begin(), hi_w.end()), fill, "constant");
      }
      bool crop = c.flag();
      mx::Shape begin = c.shp(), end = c.shp();
      if (crop) x = mx::slice(x, begin, end);
      env[e.outs[0]] = x;
      break;
    }

    case kReverse: {
      // ops/shape.py _reverse: a descending take per reversed dim. Dims
      // of extent 0 or 1 are identity and tape.py already dropped them.
      mx::array x = in(0);
      for (int64_t i = 0; i < at[0]; i++) {
        int d = static_cast<int>(at[1 + 2 * i]);
        double n = static_cast<double>(at[2 + 2 * i]);
        x = mx::take(x, mx::arange(n - 1, -1.0, -1.0, mx::int32), d);
      }
      env[e.outs[0]] = x;
      break;
    }

    case kConstant:
      // Decoded once, at lowering, by the same rules the eager engine
      // applies (splat broadcast, the raw dense blob, the rank-0 literal
      // rule); the value crosses once and never again.
      //
      // attrs[0] is that rule's other half. MLX bakes a rank-0 constant
      // into generated Metal source as a `%.7g` literal, which costs an
      // f32 its last ULP, so the lowering left the ones that lose it as a
      // ONE-ELEMENT buffer for this reshape to hand out at rank 0. The
      // reshape belongs HERE and not at lowering because `eval` DETACHES
      // a reshape node into a leaf -- a rank-0 leaf is bakeable again, so
      // a payload reshaped once would go back to being a literal for
      // every trace built after the first eager pass over this entry.
      env[e.outs[0]] = at.empty() || at[0] == 0
                           ? *e.payload
                           : mx::reshape(*e.payload, mx::Shape{});
      break;

    case kBitcastConvert: {
      // ops/shape.py _bitcast_convert. The byte-multiple arms are a view:
      // MLX's storage IS the XLA layout there. A 4-bit end is not -- an
      // i4/ui4 value lives in a whole byte here and XLA packs two per byte
      // along the minor-most dimension, low nibble first -- so a row-major
      // flatten makes the packed stream contiguous and the pack or unpack
      // is one linear reinterpretation. Every emulated type OTHER than
      // i4/ui4 is declined at lowering: a value stored in a wider float has
      // no bit pattern on this device to read.
      mx::Dtype dt = dtype_of(at[0]);
      if (at[1] == 0) {
        env[e.outs[0]] = mx::view(in(0), dt);
        break;
      }
      if (at[1] == 1) {
        // Narrowing: the result gains a trailing dim of the size ratio,
        // and mx::view rescales the LAST axis -- so split a fresh unit
        // axis (which also makes a rank-0 input legal).
        env[e.outs[0]] = mx::view(mx::expand_dims(in(0), -1), dt);
        break;
      }
      if (at[1] == 2) {
        // Widening: the input's trailing ratio-sized dim collapses.
        env[e.outs[0]] = mx::squeeze(mx::view(in(0), dt), -1);
        break;
      }
      mx::Shape out_shape = shape(at, 3, at[2]);
      if (at[1] == 6) {   // nothing to reinterpret
        env[e.outs[0]] = mx::zeros(out_shape, dt);
        break;
      }
      auto u8 = [](int64_t v) { return mx::array(v, mx::uint8); };
      auto flat = [](const mx::array& a) {
        return mx::reshape(a, mx::Shape{-1});
      };
      if (at[1] == 3) {
        // i4 <-> ui4: reinterpret the nibble in place. The entry's regrid
        // turns the nibbles into the result type's storage values.
        env[e.outs[0]] = mx::reshape(
            mx::bitwise_and(mx::astype(flat(in(0)), mx::uint8), u8(0x0F)),
            out_shape);
        break;
      }
      if (at[1] == 4) {
        // Pack pairs into bytes, low nibble first, then read the byte
        // stream as the (byte-multiple) result type.
        mx::array n =
            mx::bitwise_and(mx::astype(flat(in(0)), mx::uint8), u8(0x0F));
        const mx::Shape stop{n.shape()[0]};
        mx::array lo = mx::slice(n, mx::Shape{0}, stop, mx::Shape{2});
        mx::array hi = mx::slice(n, mx::Shape{1}, stop, mx::Shape{2});
        env[e.outs[0]] = mx::reshape(
            mx::view(mx::bitwise_or(lo, mx::left_shift(hi, u8(4))), dt),
            out_shape);
        break;
      }
      // Unpack each byte into (low, high). Again the regrid does the last
      // step, from nibbles to the result type's storage values.
      mx::array b = flat(mx::view(mx::reshape(in(0), mx::Shape{-1, 1}),
                                  mx::uint8));
      mx::array lo = mx::bitwise_and(b, u8(0x0F));
      mx::array hi = mx::right_shift(b, u8(4));
      env[e.outs[0]] = mx::reshape(flat(mx::stack({lo, hi}, -1)), out_shape);
      break;
    }

    case kDynamicSlice:
    case kDynamicUpdateSlice: {
      // ops/shape.py _dynamic_slice / _dynamic_update_slice. XLA clamps
      // the start indices so the window stays inside the operand; the
      // clamp bounds are shape arithmetic, resolved at lowering.
      const bool update = e.op == kDynamicUpdateSlice;
      const size_t first = update ? 2 : 1;
      int64_t rank = at[0];
      std::vector<mx::array> parts;
      parts.reserve(static_cast<size_t>(rank));
      for (int64_t i = 0; i < rank; i++)
        parts.push_back(
            mx::reshape(mx::astype(in(first + static_cast<size_t>(i)),
                                   mx::int32),
                        mx::Shape{}));
      mx::Shape bounds = shape(at, 1, rank);
      std::vector<int> ax(static_cast<size_t>(rank));
      for (int64_t i = 0; i < rank; i++) ax[i] = static_cast<int>(i);
      mx::array starts =
          mx::clip(mx::stack(parts), mx::array(0, mx::int32),
                   mx::array(bounds.begin(),
                             mx::Shape{static_cast<int>(rank)}, mx::int32));
      if (update) {
        env[e.outs[0]] = mx::slice_update(in(0), in(1), starts, ax);
      } else {
        env[e.outs[0]] =
            mx::slice(in(0), starts, ax, shape(at, 1 + rank, rank));
      }
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
