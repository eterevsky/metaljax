// metaljax native engine — shape and layout ops (ported from Stage 1's
// src/metaljax/ops/shape.py, deleted 0.11.6, ef5774d).
//
// Reshape, transpose, broadcast, slice/pad/reverse, the dynamic slice pair
// and the bitcast -- every one of them a rearrangement whose shapes tape.py
// already resolved, so what is left here is the MLX calls in the Python
// handler's order. `stablehlo.constant` rides along: its value crossed the
// boundary once, at lowering, and the entry simply hands it out.

#include "program.h"

#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {
namespace {

// The kill switch for the dynamic-slice start plan, so it can be A/B'd
// against the same binary (the engine's convention: a perf rewrite is
// switchable off, and a measurement pins ONE binary).
bool StartPlanEnabled() {
  static const bool on = [] {
    const char* v = std::getenv("METALJAX_DS_PLAN");
    return v == nullptr || std::strcmp(v, "0") != 0;
  }();
  return on;
}

}  // namespace

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
    case kReshape: {
      const mx::array& x = in(0);
      mx::Shape want = shape(at, 1, at[0]);
      // A reshape that only inserts and/or removes size-1 axes is a
      // squeeze/expand_dims pair, and those are VIEWS where mx::reshape
      // copies any non-row-contiguous operand.  The case that matters is
      // jax's dynamic_index_in_dim — dynamic_slice + reshape — over a
      // transposed stack (scan-over-layers weights): the slice is a strided
      // view and the reshape re-materialized 100s of MB per layer per step
      // (row 10's DeepSeek MoE weights, 3 x 369 MB per decode token per
      // layer).  Zero-size arrays keep the plain reshape: their stride
      // bookkeeping has no view form worth special-casing.
      const mx::Shape& have = x.shape();
      bool view_ok = x.size() > 0;
      {
        size_t i = 0;
        for (auto d : have)
          if (d != 1) {
            while (i < want.size() && want[i] == 1) i++;
            if (i < want.size() && want[i] == d) {
              i++;
            } else {
              view_ok = false;
              break;
            }
          }
        if (view_ok)
          for (; i < want.size(); i++) view_ok = view_ok && want[i] == 1;
      }
      if (view_ok && have != want) {
        std::vector<int> drop;
        for (size_t i = 0; i < have.size(); i++)
          if (have[i] == 1) drop.push_back(static_cast<int>(i));
        std::vector<int> add;
        for (size_t i = 0; i < want.size(); i++)
          if (want[i] == 1) add.push_back(static_cast<int>(i));
        mx::array y = drop.empty() ? x : mx::squeeze(x, drop);
        if (!add.empty()) y = mx::expand_dims(y, add);
        env[e.outs[0]] = y;
      } else {
        env[e.outs[0]] = mx::reshape(x, want);
      }
      break;
    }
    case kTranspose:
      env[e.outs[0]] = mx::transpose(in(0), axes(at, 1, at[0]));
      break;
    case kBroadcastInDim: {
      // _broadcast_in_dim: unsorted broadcast_dimensions become a
      // transpose, then the operand reshapes to an interim shape with a
      // 1 in every dim it does not name and broadcasts out. The perm and
      // the interim shape are static, so tape.py resolved both.
      //
      // NOT DONE, and the reason is worth keeping. The interim reshape is
      // usually pure LEFT PADDING -- [2048] to [1,1,2048] on the way out --
      // and `mx::broadcast_to` right-aligns exactly like numpy, so in that
      // shape it could be skipped: MLX fuses `Broadcast` and does not fuse
      // `Reshape` (mlx/compile.cpp `is_fusable`), so the reshape splits the
      // fusion group and leaves a node for a compiled body to walk at every
      // replay. Skipping it MEASURED AS A THREAD HAZARD: with the reshape
      // gone the broadcast's operand is the payload/argument leaf itself,
      // and `execute_test`'s 32-executes-on-8-threads case then failed
      // "There is no Stream(gpu, 2) in current thread" in 3 of 8 runs
      // (0 of 8 with it, 0 of 4 on the base binary) -- the P30 hazard class
      // where a node carries the creating thread's thread-unsafe stream.
      // The reshape is what was keeping those leaves one node away from a
      // fused kernel's input list. Worth ~1 node per broadcast; not worth
      // that. Numbers in ~/.cache/metaljax-bench/logs/row10-bookkeeping.
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
      //
      // Only the axes in the lowering's START PLAN get a start at all
      // (metal_lowering.cc `AppendStartPlan` carries the reasoning): MLX
      // starts every axis outside `ax` at zero, and jax spells all but one
      // of a slice's starts as a rank-0 zero constant. Building those zeros
      // cost an `ExpandDims` each plus the `stack`'s `Concatenate`, and
      // none of those is in MLX's fusable set, so a compiled loop body
      // walked them again at every replay.
      //
      // The one-axis case -- every scanned-layer weight read and every KV
      // cache write -- passes the clamped index STRAIGHT THROUGH as the
      // rank-0 array it already is: `normalize_dynamic_slice_inputs` takes
      // a zero- or one-dimensional start, so there is nothing left to
      // build.
      const bool update = e.op == kDynamicUpdateSlice;
      const size_t first = update ? 2 : 1;
      const int64_t rank = at[0];
      const mx::Shape bounds = shape(at, 1, rank);
      const size_t plan_at =
          static_cast<size_t>(update ? 1 + rank : 1 + 2 * rank);
      // P49: is this thread tracing a POSITION-SPECIALIZED variant of the
      // loop body this entry belongs to, and did the position's start table
      // resolve this entry's starts?  Then the starts are host integers and
      // MLX's static spellings apply:
      //
      //   * `mx::slice(a, Shape, Shape)` is `shared_buffer_slice` -- a VIEW,
      //     no `compute_dynamic_offset` kernel and no row copy.  A
      //     leading-axis unit-stride slice of a row-contiguous stack is
      //     itself row-contiguous, so every downstream kernel sees the shape
      //     and stride class it sees today, at a different base offset.
      //   * `mx::slice_update(a, upd, Shape, Shape)` runs the SAME donation
      //     test (`row_contiguous && size == data_size &&
      //     is_donatable(through_stream_pins)`) and the SAME window
      //     `copy_gpu_inplace`; only the offset kernel goes.
      //
      // The values are the ones the dynamic spelling would have computed:
      // `loop_spec` evaluates the clip expression built below, on the same
      // operand, through the same handlers. Unplanned axes start at zero,
      // which is what clamping their constant-zero operand yields -- the
      // same fact `AppendStartPlan` resolves them on, so this is correct
      // with the start plan on or off.
      const std::vector<int32_t>* fold = nullptr;
      if (t_spec_frame != nullptr && t_spec_frame->prog == this &&
          t_spec_frame->spec != nullptr) {
        auto hit = t_spec_frame->spec->starts.find(&e);
        if (hit != t_spec_frame->spec->starts.end()) fold = &hit->second;
      }
      if (fold != nullptr) {
        const mx::array& src = in(0);
        const int64_t nstart = at[plan_at];
        mx::Shape lo(static_cast<size_t>(rank), 0);
        bool ok = static_cast<int64_t>(fold->size()) == nstart;
        for (int64_t j = 0; ok && j < nstart; j++) {
          const int64_t axis = at[plan_at + 1 + 3 * static_cast<size_t>(j)];
          if (axis < 0 || axis >= rank) { ok = false; break; }
          lo[static_cast<size_t>(axis)] =
              static_cast<mx::ShapeElem>((*fold)[static_cast<size_t>(j)]);
        }
        if (ok) {
          const mx::Shape win =
              update ? in(1).shape() : shape(at, 1 + rank, rank);
          mx::Shape hi(static_cast<size_t>(rank));
          for (int64_t i = 0; i < rank; i++)
            hi[static_cast<size_t>(i)] =
                lo[static_cast<size_t>(i)] + win[static_cast<size_t>(i)];
          if (update) {
            env[e.outs[0]] = mx::slice_update(src, in(1), lo, hi);
          } else {
            env[e.outs[0]] = mx::slice(src, lo, hi);
          }
          break;
        }
      }
      auto raw_start = [&](int64_t axis) {
        return mx::reshape(
            mx::astype(in(first + static_cast<size_t>(axis)), mx::int32),
            mx::Shape{});
      };
      std::vector<int> ax;
      std::optional<mx::array> starts;
      if (!StartPlanEnabled()) {
        // The pre-plan spelling, kept verbatim as the A/B arm: every axis
        // gets a start, they stack into one [rank] vector, and ONE clip
        // bounds the whole vector. Reproducing it exactly is the point --
        // an off-arm that differed by so much as a node would show up in
        // the measurement as part of the plan's win.
        std::vector<mx::array> parts;
        parts.reserve(static_cast<size_t>(rank));
        ax.resize(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; i++) {
          ax[static_cast<size_t>(i)] = static_cast<int>(i);
          parts.push_back(raw_start(i));
        }
        starts = mx::clip(
            mx::stack(parts), mx::array(0, mx::int32),
            mx::array(bounds.begin(), mx::Shape{static_cast<int>(rank)},
                      mx::int32));
      } else {
        const int64_t nstart = at[plan_at];
        std::vector<mx::array> parts;
        parts.reserve(static_cast<size_t>(nstart));
        ax.resize(static_cast<size_t>(nstart));
        for (int64_t j = 0; j < nstart; j++) {
          const size_t p = plan_at + 1 + 3 * static_cast<size_t>(j);
          const int64_t axis = at[p];
          ax[static_cast<size_t>(j)] = static_cast<int>(axis);
          // A `1` here says the operand was a constant and this is its
          // already-clamped value; otherwise clamp the data start against
          // this axis's bound, exactly as the vector clip above did.
          parts.push_back(
              at[p + 1] != 0
                  ? mx::array(static_cast<int>(at[p + 2]), mx::int32)
                  : mx::clip(raw_start(axis), mx::array(0, mx::int32),
                             mx::array(static_cast<int>(
                                           bounds[static_cast<size_t>(axis)]),
                                       mx::int32)));
        }
        // One start needs no vector at all: MLX takes a rank-0 start.
        starts = parts.size() == 1 ? parts[0] : mx::stack(parts);
      }
      if (update) {
        env[e.outs[0]] = mx::slice_update(in(0), in(1), *starts, ax);
      } else {
        env[e.outs[0]] =
            mx::slice(in(0), *starts, ax, shape(at, 1 + rank, rank));
      }
      break;
    }

    // --- the KV in-place rewrite (metal_lowering.cc `KvInplace`) ---
    case kKvStarts: {
      // Row i of the result is the start vector of the i-th cache update:
      // the constant part, plus for every dynamic cell its raw start
      // clipped to that cell's bound (XLA's clamp, in the SLAB's frame the
      // program spelled it in) -- all elementwise over one small int32
      // matrix, which MLX fuses into one kernel per token however many
      // updates share it.  D distinct raw starts, one column selector each.
      const int64_t nd = at[0];
      mx::array acc = in(0);
      const mx::array& bound = in(1);
      const mx::array& sel = in(2);
      for (int64_t d = 0; d < nd; d++) {
        mx::array raw = mx::reshape(
            mx::astype(in(3 + static_cast<size_t>(d)), mx::int32),
            mx::Shape{});
        mx::array clipped = mx::clip(mx::broadcast_to(raw, bound.shape()),
                                     mx::array(0, mx::int32), bound);
        acc = mx::add(
            acc, mx::where(mx::equal(sel, mx::array(static_cast<int>(d + 1),
                                                    mx::int32)),
                           clipped, mx::array(0, mx::int32)));
      }
      env[e.outs[0]] = acc;
      break;
    }
    case kKvUpdate: {
      // One in-place window write on the cache carry.  `starts` is the
      // kKvStarts matrix (or a constant one); this update's row is a view
      // of it.  Whether MLX writes the window into the operand's buffer or
      // copies the operand first is its `is_donatable(true)` call at eval
      // time -- the fork's donation through this stream's pins -- and the
      // answer is data movement only, never a value.
      const int row = static_cast<int>(at[0]);
      const int64_t rank = at[1];
      std::vector<int> ax(static_cast<size_t>(rank));
      for (int64_t i = 0; i < rank; i++)
        ax[static_cast<size_t>(i)] = static_cast<int>(at[2 + i]);
      // The row as a 1-D VIEW: flatten the (contiguous) matrix once and
      // slice the row out of the flat vector.  Slicing the 2-D matrix and
      // reshaping the [1, rank] row would cost a copy kernel per update
      // (MLX reshapes a strided slice by copying) -- measured: 56 extra
      // dispatches per token on row 11's first cut.
      const mx::array& starts = in(2);
      const int n = static_cast<int>(rank);
      mx::array flat = mx::reshape(starts, mx::Shape{-1});
      mx::array start =
          mx::slice(flat, mx::Shape{row * n}, mx::Shape{row * n + n});
      env[e.outs[0]] = mx::slice_update(in(0), in(1), start, ax);
      break;
    }
    case kDepends: {
      // The cache, ordered after the kernels that read its previous state:
      // MLX evaluates a dependency before the arrays that depend on it, so
      // by the time the next in-place update runs those readers are done
      // and their views of the buffer released -- which is what lets the
      // update donate.  A Depends node moves no data.
      std::vector<mx::array> deps;
      deps.reserve(e.ins.size() - 1);
      for (size_t i = 1; i < e.ins.size(); i++) deps.push_back(in(i));
      if (deps.empty()) {
        env[e.outs[0]] = in(0);
      } else {
        env[e.outs[0]] = mx::depends({in(0)}, deps)[0];
      }
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
