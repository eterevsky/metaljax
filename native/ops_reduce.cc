// metaljax native engine — reductions (src/metaljax/ops/reduction.py).
//
// Four shapes of the same op. A monoid reduce goes straight to MLX; the
// (values, indices) pair jax lowers argmax/argmin to carries XLA's NaN rule,
// which is why it is not just an argmax call; a body neither table
// recognizes runs pairwise over whole arrays through `generic_reduce`; and
// reduce_window is the cumulative peephole or one strided view of every
// window fed to any of the three.

#include "program.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

namespace {

// Reduce monoids, resolved at lowering time from the body op and the input
// element type (ops/reduction.py picks _BOOL_REDUCERS vs _REDUCERS on the
// dtype, which is static in the IR).
mx::array reduce_apply(int64_t kind, const mx::array& x,
                       const std::vector<int>& axes) {
  switch (kind) {
    case 0: return mx::sum(x, axes);
    case 1: return mx::prod(x, axes);
    case 2: return mx::max(x, axes);
    case 3: return mx::min(x, axes);
    case 4: return mx::any(x, axes);
    case 5: return mx::all(x, axes);
    default: throw std::invalid_argument("tape: bad reduce kind");
  }
}

mx::array reduce_combine(int64_t kind, const mx::array& a,
                         const mx::array& b) {
  switch (kind) {
    case 0: return mx::add(a, b);
    case 1: return mx::multiply(a, b);
    case 2: return mx::maximum(a, b);
    case 3: return mx::minimum(a, b);
    case 4: return mx::logical_or(a, b);
    case 5: return mx::logical_and(a, b);
    default: throw std::invalid_argument("tape: bad reduce kind");
  }
}

// --------------------------------------------------------------------------
// reduce_window (src/metaljax/ops/reduction.py `_extract_windows`)
// --------------------------------------------------------------------------
//
// Base-dilate with the init, pad with the init, then read every window out
// of ONE strided view. Every shape here is static and tape.py resolved it;
// what is left is the four MLX calls, in the handler's order — including
// the materialization at the end, which is not an optimization but a
// workaround: MLX 0.32 reductions over a strided view read stale device
// memory once the reshape folds the window block into a non-unit stride.

struct WindowPlan {
  struct Dil {
    int axis;
    int b;
    mx::Shape shape;   // the array's shape with this axis dilated
    int end;           // ...and the extent kept after the holes are cut
  };
  std::vector<Dil> dils;
  bool padded = false;
  std::vector<int> lo, hi;
  mx::Shape view_shape;
  mx::Strides view_strides;
  mx::Shape flat;      // out_sizes + [prod(window)]
  bool empty = false;
  mx::Shape out_sizes;
};

WindowPlan read_window_plan(Cursor& c) {
  WindowPlan p;
  int64_t ndil = c.next();
  for (int64_t i = 0; i < ndil; i++) {
    WindowPlan::Dil d;
    d.axis = static_cast<int>(c.next());
    d.b = static_cast<int>(c.next());
    d.shape = c.shp();
    d.end = static_cast<int>(c.next());
    p.dils.push_back(std::move(d));
  }
  p.padded = c.flag();
  p.lo = c.vec();
  p.hi = c.vec();
  p.view_shape = c.shp();
  std::vector<int64_t> st = c.vec64();
  p.view_strides = mx::Strides(st.begin(), st.end());
  p.flat = c.shp();
  p.empty = c.flag();
  p.out_sizes = c.shp();
  return p;
}

mx::array extract_windows(const WindowPlan& p, mx::array x,
                          const mx::array& init) {
  for (const WindowPlan::Dil& d : p.dils) {
    // Holes of the init value, the operand written into every b-th slot.
    mx::array holes =
        mx::contiguous(mx::broadcast_to(mx::astype(init, x.dtype()), d.shape));
    mx::Shape start(d.shape.size(), 0), strides(d.shape.size(), 1);
    strides[static_cast<size_t>(d.axis)] = d.b;
    holes = mx::slice_update(holes, x, start, d.shape, strides);
    mx::Shape stop = d.shape;
    stop[static_cast<size_t>(d.axis)] = d.end;
    x = mx::slice(holes, start, stop);
  }
  if (p.padded) {
    std::vector<int> ax(p.lo.size());
    for (size_t i = 0; i < ax.size(); i++) ax[i] = static_cast<int>(i);
    x = mx::pad(x, ax, mx::Shape(p.lo.begin(), p.lo.end()),
                mx::Shape(p.hi.begin(), p.hi.end()),
                mx::astype(init, x.dtype()), "constant");
  }
  x = mx::contiguous(x);   // as_strided needs row-contiguous storage
  return mx::contiguous(
      mx::reshape(mx::as_strided(x, p.view_shape, p.view_strides, 0), p.flat));
}

}  // namespace

bool Program::step_reduce(const Entry& e,
                          std::vector<std::optional<mx::array>>& env,
                          bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    case kReduce: {
      // ops/reduction.py _reduce, single-operand monoid form.
      const mx::array& x = in(0);
      const mx::array& init = in(1);
      int64_t kind = at[0];
      int64_t ndims = at[1];
      std::vector<int> dims = axes(at, 2, ndims);
      bool empty = false;
      for (auto s : x.shape()) if (s == 0) empty = true;
      if (empty) {
        // MLX reducers crash on zero-size inputs (mx.max raises; a
        // zero-size uint32 sum aborts in a missing Metal kernel). An
        // empty fold is well defined: the init value.
        mx::Shape out;
        for (size_t i = 0; i < x.shape().size(); i++) {
          bool reduced = false;
          for (int d : dims) if (static_cast<size_t>(d) == i) reduced = true;
          if (!reduced) out.push_back(x.shape()[i]);
        }
        bool reduced_empty = false;
        for (int d : dims) if (x.shape()[d] == 0) reduced_empty = true;
        env[e.outs[0]] =
            reduced_empty
                ? mx::broadcast_to(mx::astype(init, x.dtype()), out)
                : mx::zeros(out, x.dtype());
        break;
      }
      mx::array out = dims.empty() ? x : reduce_apply(kind, x, dims);
      env[e.outs[0]] = reduce_combine(kind, out, init);
      break;
    }

    case kArgReduce: {
      // ops/reduction.py _reduce, the (values, indices) form jax lowers
      // argmax/argmin to. Ties resolve to the lowest index, which is
      // MLX's first-occurrence answer; the NaN rules are XLA's, not
      // MLX's, and are the reason this is not just an argmax call.
      const mx::array& x = in(0);
      const mx::array& ids = in(1);
      bool is_max = at[0] != 0;
      int d = static_cast<int>(at[1]);
      bool empty = false;
      for (auto s : x.shape()) if (s == 0) empty = true;
      if (empty) {
        // Only BATCH dims can be zero here (jax forbids argmax over an
        // empty reduced axis) and MLX's reducers crash on empties.
        mx::Shape out;
        for (size_t i = 0; i < x.shape().size(); i++)
          if (static_cast<int>(i) != d) out.push_back(x.shape()[i]);
        env[e.outs[0]] = mx::zeros(out, x.dtype());
        env[e.outs[1]] = mx::zeros(out, ids.dtype());
        break;
      }
      mx::array val = is_max ? mx::max(x, d) : mx::min(x, d);
      mx::array arg = is_max ? mx::argmax(x, d) : mx::argmin(x, d);
      if (is_float(x.dtype())) {
        // XLA/numpy: a NaN wins argmax AND argmin, and the FIRST one's
        // index is the answer. MLX skips NaNs entirely.
        mx::array nans = mx::isnan(x);
        mx::array has_nan = mx::any(nans, std::vector<int>{d});
        mx::array first_nan = mx::argmax(nans, d);
        arg = mx::where(has_nan, first_nan, arg);
        val = mx::where(
            has_nan,
            mx::array(std::numeric_limits<double>::quiet_NaN(), val.dtype()),
            val);
      }
      mx::array idx = mx::take_along_axis(ids, mx::expand_dims(arg, d), d);
      env[e.outs[0]] = val;
      env[e.outs[1]] = mx::squeeze(idx, d);
      break;
    }

    case kGenericReduce: {
      // ops/reduction.py _reduce's last arm. Operands are the n inputs,
      // the n inits, then the body's captures.
      Cursor c(at);
      size_t n = static_cast<size_t>(c.next());
      std::vector<int> keep = c.vec();
      std::vector<int> dims = c.vec();
      int64_t ncaps = c.next();
      std::vector<mx::array> inputs, inits, caps;
      for (size_t i = 0; i < n; i++) inputs.push_back(in(i));
      for (size_t i = 0; i < n; i++) inits.push_back(in(n + i));
      for (int64_t i = 0; i < ncaps; i++) caps.push_back(in(2 * n + i));
      std::vector<mx::array> outs = generic_reduce(
          inputs, inits, caps, keep, dims, e.regions[0].get(), in_trace);
      if (outs.size() != e.outs.size())
        throw std::runtime_error("tape: generic reduce result count");
      for (size_t i = 0; i < outs.size(); i++) env[e.outs[i]] = outs[i];
      break;
    }

    case kReduceWindow: {
      // ops/reduction.py _reduce_window: the cumulative peephole, or one
      // windowed reduction whose fold is a monoid, a single compare
      // (select_and_gather_add), or the body itself.
      Cursor c(at);
      if (c.next() == 0) {
        int64_t cum = c.next();
        int ax = static_cast<int>(c.next());
        bool rev = c.flag();
        const mx::array& x = in(0);
        switch (cum) {
          case 0: env[e.outs[0]] = mx::cumsum(x, ax, rev, true); break;
          case 1: env[e.outs[0]] = mx::cummax(x, ax, rev, true); break;
          case 2: env[e.outs[0]] = mx::cummin(x, ax, rev, true); break;
          default: env[e.outs[0]] = mx::cumprod(x, ax, rev, true); break;
        }
        break;
      }
      size_t n = static_cast<size_t>(c.next());
      WindowPlan plan = read_window_plan(c);
      if (plan.empty) {
        for (size_t i = 0; i < n; i++)
          env[e.outs[i]] = mx::broadcast_to(
              mx::astype(in(n + i), in(i).dtype()), plan.out_sizes);
        break;
      }
      std::vector<mx::array> wins;
      wins.reserve(n);
      for (size_t i = 0; i < n; i++)
        wins.push_back(extract_windows(plan, in(i), in(n + i)));
      int last = static_cast<int>(wins[0].ndim()) - 1;
      int64_t mode = c.next();
      if (mode == 0) {
        // Monoid: reduce the flattened window axis, then fold the init.
        int64_t kind = c.next();
        env[e.outs[0]] = reduce_combine(
            kind, reduce_apply(kind, wins[0], std::vector<int>{last}),
            in(n));
      } else if (mode == 1) {
        // select_and_gather_add: the compare picks ONE window element and
        // every output is read at that position.
        bool is_max = c.flag();
        mx::array arg =
            is_max ? mx::argmax(wins[0], last) : mx::argmin(wins[0], last);
        mx::array idx = mx::expand_dims(arg, last);
        for (size_t i = 0; i < n; i++)
          env[e.outs[i]] =
              mx::squeeze(mx::take_along_axis(wins[i], idx, last), last);
      } else {
        size_t bn = static_cast<size_t>(c.next());
        std::vector<int> keep = c.vec();
        std::vector<int> dims = c.vec();
        int64_t ncaps = c.next();
        std::vector<mx::array> inits, caps;
        for (size_t i = 0; i < bn; i++) inits.push_back(in(n + i));
        for (int64_t i = 0; i < ncaps; i++)
          caps.push_back(in(2 * n + static_cast<size_t>(i)));
        std::vector<mx::array> outs = generic_reduce(
            wins, inits, caps, keep, dims, e.regions[0].get(), in_trace);
        if (outs.size() != e.outs.size())
          throw std::runtime_error("tape: reduce_window result count");
        for (size_t i = 0; i < outs.size(); i++) env[e.outs[i]] = outs[i];
      }
      break;
    }

    default:
      return false;
  }
  return true;
}

// ops/reduction.py `_generic_reduce`: any associative body, any number of
// operands. The reduced dims move to one trailing axis and the body runs
// on the two halves of it until one element is left, padding an odd
// extent with the init; the init is folded in once at the end (XLA leaves
// the order unspecified, and this is the order the Python engine picks).
// The body is a sub-Program whose arguments are the 2n operands and whose
// trailing arguments are the values it captures from the enclosing block.
std::vector<mx::array> Program::generic_reduce(
    const std::vector<mx::array>& inputs, const std::vector<mx::array>& inits,
    const std::vector<mx::array>& caps, const std::vector<int>& keep,
    const std::vector<int>& dims, Program* body, bool in_trace) const {
  const size_t n = inputs.size();
  int64_t r = 1;
  for (int d : dims) r *= inputs[0].shape()[d];
  std::vector<int> perm = keep;
  perm.insert(perm.end(), dims.begin(), dims.end());
  mx::Shape kept;
  for (int i : keep) kept.push_back(inputs[0].shape()[i]);
  mx::Shape flat = kept;
  flat.push_back(static_cast<mx::ShapeElem>(r));
  std::vector<mx::array> xs;
  xs.reserve(n);
  for (const mx::array& x : inputs)
    xs.push_back(mx::reshape(mx::transpose(x, perm), flat));
  if (r == 0) {
    std::vector<mx::array> outs;
    outs.reserve(n);
    for (const mx::array& init : inits)
      outs.push_back(mx::broadcast_to(init, kept));
    return outs;
  }
  auto run_body = [&](std::vector<mx::array> args) {
    args.insert(args.end(), caps.begin(), caps.end());
    std::vector<mx::array> out = body->call(args, in_trace);
    if (out.size() != n)
      throw std::runtime_error("tape: reduce body result count mismatch");
    return out;
  };
  // x[..., off::2]
  auto half = [](const mx::array& x, int off) {
    mx::Shape start(x.ndim(), 0), stop = x.shape(), strides(x.ndim(), 1);
    start.back() = off;
    strides.back() = 2;
    return mx::slice(x, start, stop, strides);
  };
  while (r > 1) {
    if (r % 2) {
      for (size_t j = 0; j < n; j++) {
        mx::Shape s = xs[j].shape();
        s.back() = 1;
        xs[j] = mx::concatenate(
            {xs[j], mx::broadcast_to(mx::astype(inits[j], xs[j].dtype()), s)},
            -1);
      }
      r += 1;
    }
    std::vector<mx::array> args;
    args.reserve(2 * n);
    for (size_t j = 0; j < n; j++) args.push_back(half(xs[j], 0));
    for (size_t j = 0; j < n; j++) args.push_back(half(xs[j], 1));
    xs = run_body(std::move(args));
    r /= 2;
  }
  for (size_t j = 0; j < n; j++) xs[j] = mx::squeeze(xs[j], -1);
  std::vector<mx::array> args(inits.begin(), inits.end());
  args.insert(args.end(), xs.begin(), xs.end());
  return run_body(std::move(args));
}

}  // namespace metaljax
