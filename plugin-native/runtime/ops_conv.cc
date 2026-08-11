// metaljax native engine — convolution (src/metaljax/ops/conv.py).
//
// One StableHLO op with four executions behind it, and which one runs is a
// static property of the result dtype and the spatial rank that the lowering
// already resolved: `mx::conv_general` for the float layouts (and for the
// four real convolutions a complex one decomposes into), an im2col view
// summed in int64 for the integers (MLX's convolution is float-only, and an
// f32 emulation would round), and a plain matmul when there are no spatial
// dims at all. XLA's feature and batch groups ride on top of whichever arm
// runs, expanded into one convolution per group wherever MLX's own `groups`
// cannot serve them.
//
// The two shape guards are not decoration. At a zero-size spatial extent XLA
// computes the dilated extent as 0 where MLX computes (0-1)*d+1, so MLX hands
// back a NARROWER array than the result type declares and whoever reads the
// result reads past its end, out of uninitialized device memory — the conv
// overread of CLAUDE.md item 20, a flaky wrong answer rather than a crash.
// So an empty operand never reaches MLX (the lowering folds it into a zeros
// entry), and everything MLX does produce is measured against what the IR
// declares before it is handed on.

#include "program.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace metaljax {

namespace {

// Python's `//`, which the im2col window count depends on. A kernel wider
// than its padded axis makes the numerator NEGATIVE, and C++'s truncation
// rounds it towards zero where Python's floors -- after the `+ 1` and the
// `max(0, ...)` that is one window versus none. (The same trap the
// reduce_window plan documents; here it is on the runtime side, because
// ops/conv.py computes these sizes from the array rather than from the IR.)
int floor_div(int a, int b) {
  int q = a / b, r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) q--;
  return q;
}

std::string shape_str(const mx::Shape& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); i++) {
    if (i) out += ", ";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

// ops/conv.py `_int_conv`: an exact integer convolution. MLX's kernels are
// float-only, so the windows are read out of ONE strided view (im2col) and
// the products are summed in int64 -- which wraps exactly like XLA's integer
// dot when the result dtype is narrower. Every shape here follows from the
// operand's, exactly as it does in Python, because the group splits above
// hand this the per-group shapes rather than the whole operand's.
mx::array int_conv(mx::array x, mx::array w, const std::vector<int>& strides,
                   const std::vector<int>& ldil, const std::vector<int>& rdil,
                   const std::vector<int>& lo, const std::vector<int>& hi,
                   bool flip, const mx::Dtype& out_dt) {
  const int rank = static_cast<int>(x.ndim()) - 2;
  for (int ax = 0; ax < rank; ax++) {
    const int b = ldil[static_cast<size_t>(ax)];
    if (b == 1) continue;
    // Input dilation: insert b-1 zero holes between every pair of elements,
    // then drop the trailing holes the last element does not need.
    mx::Shape shape = x.shape();
    shape[static_cast<size_t>(ax) + 1] *= b;
    mx::array holes = mx::zeros(shape, x.dtype());
    mx::Shape start(shape.size(), 0), st(shape.size(), 1);
    st[static_cast<size_t>(ax) + 1] = b;
    holes = mx::slice_update(holes, x, start, shape, st);
    mx::Shape stop = shape;
    stop[static_cast<size_t>(ax) + 1] =
        shape[static_cast<size_t>(ax) + 1] - (b - 1);
    x = mx::slice(holes, start, stop);
  }
  bool any_pad = false;
  for (int i = 0; i < rank; i++)
    any_pad = any_pad || lo[static_cast<size_t>(i)] != 0 ||
              hi[static_cast<size_t>(i)] != 0;
  if (any_pad) {
    // The batch and channel axes carry a (0, 0) width, as the Python
    // handler's `widths` does.
    std::vector<int> ax(x.ndim());
    mx::Shape low(x.ndim(), 0), high(x.ndim(), 0);
    for (size_t i = 0; i < ax.size(); i++) ax[i] = static_cast<int>(i);
    for (int i = 0; i < rank; i++) {
      low[static_cast<size_t>(i) + 1] = lo[static_cast<size_t>(i)];
      high[static_cast<size_t>(i) + 1] = hi[static_cast<size_t>(i)];
    }
    x = mx::pad(x, ax, low, high, mx::array(0, x.dtype()), "constant");
  }
  x = mx::contiguous(x);   // as_strided needs row-contiguous storage

  mx::Shape wd(w.shape().begin() + 1, w.shape().end() - 1);
  const int C = x.shape().back();
  mx::Shape out_sizes(static_cast<size_t>(rank));
  for (int i = 0; i < rank; i++) {
    const size_t u = static_cast<size_t>(i);
    const int span = (wd[u] - 1) * rdil[u] + 1;
    out_sizes[u] = std::max(
        0, floor_div(x.shape()[u + 1] - span, strides[u]) + 1);
  }
  // Row-major element strides of the padded operand: one window step is
  // `stride` of them on an output axis and `dilation` of them inside the
  // window, which is the whole of the view.
  std::vector<int64_t> es(x.ndim());
  int64_t acc = 1;
  for (size_t i = x.ndim(); i-- > 0;) {
    es[i] = acc;
    acc *= x.shape()[i];
  }
  mx::Shape view_shape{x.shape()[0]};
  view_shape.insert(view_shape.end(), out_sizes.begin(), out_sizes.end());
  view_shape.insert(view_shape.end(), wd.begin(), wd.end());
  view_shape.push_back(C);
  mx::Strides view_strides{es[0]};
  for (int i = 0; i < rank; i++)
    view_strides.push_back(es[static_cast<size_t>(i) + 1] *
                           strides[static_cast<size_t>(i)]);
  for (int i = 0; i < rank; i++)
    view_strides.push_back(es[static_cast<size_t>(i) + 1] *
                           rdil[static_cast<size_t>(i)]);
  view_strides.push_back(es.back());
  mx::array patches = mx::as_strided(x, view_shape, view_strides, 0);

  if (flip) {
    for (int ax = 0; ax < rank; ax++) {
      const int n = wd[static_cast<size_t>(ax)];
      if (n > 1) w = mx::take(w, mx::arange(n - 1, -1, -1), ax + 1);
    }
  }
  int K = C;
  for (int v : wd) K *= v;
  mx::Shape flat{x.shape()[0]};
  flat.insert(flat.end(), out_sizes.begin(), out_sizes.end());
  flat.push_back(K);
  mx::array p2 = mx::reshape(patches, flat);
  mx::array w2 = mx::reshape(w, mx::Shape{w.shape()[0], K});
  const mx::Dtype acc_t = x.dtype() == mx::bool_ ? mx::bool_ : mx::int64;
  // [N, *out, 1, K] against [C_out, K] broadcasts to [N, *out, C_out, K], so
  // the axis to fold is one PAST p2's last -- `p2.ndim() - 1` would sum the
  // output channels instead, which is why the suite has 2-D integer rows.
  const int last = static_cast<int>(p2.ndim());
  mx::array prod = mx::multiply(mx::astype(mx::expand_dims(p2, -2), acc_t),
                                mx::astype(w2, acc_t));
  mx::array out = acc_t == mx::bool_ ? mx::any(prod, last)
                                     : mx::sum(prod, last);
  return mx::astype(out, out_dt);
}

}  // namespace

// --------------------------------------------------------------------------
// stablehlo.convolution
// --------------------------------------------------------------------------
//
// Attribute layout (`kConv`), read with a Cursor:
//
//   [empty?]
//     1 -> [out dtype, [out shape]]        the result is zeros of that shape,
//                                          and the attrs stop right there
//     0 -> [out dtype, rank,
//           [lhs perm], [rhs perm], [out perm], fgc, bgc,
//           rank == 0 ? (nothing more)
//                     : [strides], [lhs dilation], [rhs dilation],
//                       [pad lo], [pad hi],
//                       crop?, [crop start], [crop stop],
//                       flip?, mode, native groups?, [want]]
//
// `rank` is the number of SPATIAL dims, and 0 selects the matmul arm (no
// window to slide, so the convolution is a contraction over features). The
// three perms are the transposes into and out of MLX's layouts -- input
// (N, *spatial, C_in), weight (C_out, *spatial, C_in) -- so every
// dimension-numbers layout XLA can spell arrives here as three permutations
// and nothing else. `crop` carries the negative-padding rewrite the lowering
// resolved (a slice of the operand, plus the non-negative remainder of the
// pad); `mode` is 0 float / 1 exact integer / 2 complex; `native groups?`
// says MLX's own `groups` can serve this feature grouping. `want` is the
// result shape in MLX's layout -- the guard, not a hint.
bool Program::step_conv(const Entry& e,
                        std::vector<std::optional<mx::array>>& env,
                        bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };

  switch (e.op) {
    case kConv: {
      Cursor c(e.attrs);
      if (c.flag()) {
        // An empty operand (or an empty result) means every output element
        // sums an empty set of products. Never handed to MLX: see the file
        // comment for what its narrower answer costs the reader.
        mx::Dtype dt = dtype_of(c.next());
        env[e.outs[0]] = mx::zeros(c.shp(), dt);
        break;
      }
      const mx::Dtype out_dt = dtype_of(c.next());
      const int64_t rank = c.next();
      std::vector<int> lperm = c.vec(), rperm = c.vec(), operm = c.vec();
      const int64_t fgc = c.next(), bgc = c.next();

      if (rank == 0) {
        // No spatial dims: a (grouped) matmul over features. Both operands
        // go through f32, which is what the Python handler does -- the arm
        // exists for the degenerate shapes jax's own conv wrappers produce,
        // not for exact integer arithmetic.
        mx::array x0 = mx::astype(mx::transpose(in(0), lperm), mx::float32);
        mx::array w0 = mx::astype(mx::transpose(in(1), rperm), mx::float32);
        auto mm = [](const mx::array& xg, const mx::array& wg) {
          return mx::matmul(xg, mx::transpose(wg));
        };
        std::optional<mx::array> out;
        if (bgc > 1 || fgc > 1) {
          const int n = static_cast<int>(bgc > 1 ? bgc : fgc);
          // Batch groups split the BATCH axis, feature groups the feature
          // axis; the weight splits along its output features either way.
          std::vector<mx::array> xs = mx::split(x0, n, bgc > 1 ? 0 : 1);
          std::vector<mx::array> ws = mx::split(w0, n, 0);
          std::vector<mx::array> parts;
          parts.reserve(xs.size());
          for (size_t g = 0; g < xs.size(); g++)
            parts.push_back(mm(xs[g], ws[g]));
          out = mx::concatenate(parts, -1);
        } else {
          out = mm(x0, w0);
        }
        env[e.outs[0]] = mx::transpose(mx::astype(*out, out_dt), operm);
        break;
      }

      std::vector<int> strides = c.vec(), ldil = c.vec(), rdil = c.vec();
      std::vector<int> lo = c.vec(), hi = c.vec();
      const bool crop = c.flag();
      mx::Shape cstart = c.shp(), cstop = c.shp();
      const bool flip = c.flag();
      const int64_t mode = c.next();
      const bool native_groups = c.flag();
      const mx::Shape want = c.shp();

      // MLX layouts: input (N, *spatial, C_in), weight (C_out, *spatial,
      // C_in).
      mx::array x = mx::transpose(in(0), lperm);
      mx::array w = mx::transpose(in(1), rperm);
      if (crop) {
        // XLA pads AFTER lhs dilation, so a negative pad crops the DILATED
        // array; MLX would crop the undilated operand instead. The lowering
        // rewrote each crop as dropping whole operand elements plus a
        // non-negative pad of the leftover holes, and this is that drop.
        x = mx::contiguous(mx::slice(x, cstart, cstop));
      }

      auto run = [&](const mx::array& xi, const mx::array& wi, int groups) {
        if (mode == 1)
          return int_conv(xi, wi, strides, ldil, rdil, lo, hi, flip, out_dt);
        return mx::conv_general(xi, wi, strides, lo, hi, rdil, ldil, groups,
                                flip);
      };
      auto feature_groups = [&](const mx::array& xi, const mx::array& wi) {
        if (fgc == 1) return run(xi, wi, 1);
        if (native_groups) return run(xi, wi, static_cast<int>(fgc));
        // XLA feature groups: the input features split into fgc groups and
        // output feature block g is computed from input group g. Expanded
        // into one ungrouped convolution per group for the cases MLX will
        // not take (its `groups` covers 1-D and 2-D floats only).
        const int n = static_cast<int>(fgc);
        std::vector<mx::array> xs = mx::split(xi, n, -1);
        std::vector<mx::array> ws = mx::split(wi, n, 0);
        std::vector<mx::array> parts;
        parts.reserve(xs.size());
        for (size_t g = 0; g < xs.size(); g++)
          parts.push_back(run(xs[g], ws[g], 1));
        return mx::concatenate(parts, -1);
      };
      auto conv_all = [&](const mx::array& xi, const mx::array& wi) {
        if (bgc <= 1) return feature_groups(xi, wi);
        // XLA batch groups: the batch and the kernel's output features split
        // into bgc groups, group i convolved with kernel group i, the
        // outputs concatenated along features (this is the grad-of-weights
        // shape).
        const int n = static_cast<int>(bgc);
        std::vector<mx::array> xs = mx::split(xi, n, 0);
        std::vector<mx::array> ws = mx::split(wi, n, 0);
        std::vector<mx::array> parts;
        parts.reserve(xs.size());
        for (size_t g = 0; g < xs.size(); g++)
          parts.push_back(feature_groups(xs[g], ws[g]));
        return mx::concatenate(parts, -1);
      };

      std::optional<mx::array> out;
      if (mode == 2) {
        // (ar + i*ai) conv (br + i*bi): four real convolutions.
        mx::array xc = mx::astype(x, mx::complex64);
        mx::array wc = mx::astype(w, mx::complex64);
        mx::array ar = mx::real(xc), ai = mx::imag(xc);
        mx::array br = mx::real(wc), bi = mx::imag(wc);
        out = make_complex(
            mx::subtract(conv_all(ar, br), conv_all(ai, bi)),
            mx::add(conv_all(ar, bi), conv_all(ai, br)));
      } else if (mode == 1) {
        out = conv_all(x, w);
      } else {
        if (x.dtype() != out_dt) x = mx::astype(x, out_dt);
        if (w.dtype() != out_dt) w = mx::astype(w, out_dt);
        out = conv_all(x, w);
      }

      // Guard against MLX sizing a window differently from XLA: handing back
      // a wrong-shaped buffer makes the caller read past its end.
      if (out->shape() != want)
        throw std::runtime_error("conv: produced " + shape_str(out->shape()) +
                                 ", XLA declares " + shape_str(want));
      env[e.outs[0]] = mx::transpose(*out, operm);
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
