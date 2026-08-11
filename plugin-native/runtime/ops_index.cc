// metaljax native engine — indexed reads and writes
// (src/metaljax/ops/gather.py, ops/sort.py).
//
// StableHLO's gather and scatter go STRAIGHT to MLX's primitives: one index
// array per indexed operand axis, all of the index batch shape, built by the
// plan tape.py resolved. XLA's out-of-bounds rules are the part MLX does not
// share -- a gather clamps, a scatter DROPS -- and the two drop strategies
// below are how a drop is spelled with primitives that cannot skip a write.
// Sort and top_k live here too: they end in the same take_along_axis, over
// indices an argsort produced.

#include "program.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

namespace {

// --------------------------------------------------------------------------
// gather / scatter index plans (src/metaljax/tape.py `_index_plan`)
// --------------------------------------------------------------------------
//
// One index array per indexed operand axis, all of the index BATCH shape so
// MLX's broadcast rule has nothing to do. Three sources, as the layout says:
// a component of the start-index vector, the implicit batching iota, or the
// constant zero an op that indexes nothing degenerates to.

// Component `j` of the coordinate vector. `split` is false when
// index_vector_dim == rank, where the whole array is the single component --
// ops/gather.py's `index_arrays` reads it exactly this way.
mx::array index_component(const mx::array& starts, bool split, int ivd,
                          int j) {
  if (!split) return starts;
  mx::Shape lo(starts.ndim(), 0), hi = starts.shape();
  lo[static_cast<size_t>(ivd)] = j;
  hi[static_cast<size_t>(ivd)] = j + 1;
  return mx::squeeze(mx::slice(starts, lo, hi), ivd);
}

// An index plan, read and built. `oob` collects XLA's drop predicate -- an
// update whose start is out of bounds in ANY component is dropped -- and
// stays unset when the caller does not want it (a gather CLAMPS instead,
// which is XLA's rule there).
void read_index_plan(Cursor& c, const mx::array& starts, bool split, int ivd,
                     const mx::Shape& bshape, std::vector<mx::array>& idxs,
                     std::vector<int>& axes, std::optional<mx::array>* oob) {
  int64_t n = c.next();
  idxs.reserve(static_cast<size_t>(n));
  axes.reserve(static_cast<size_t>(n));
  for (int64_t q = 0; q < n; q++) {
    int64_t kind = c.next(), a = c.next(), b = c.next(), axis = c.next();
    if (kind == 0) {
      mx::array raw = mx::astype(
          index_component(starts, split, ivd, static_cast<int>(a)), mx::int32);
      mx::array zero = mx::array(0, mx::int32);
      mx::array top = mx::array(static_cast<int>(b), mx::int32);
      if (oob) {
        mx::array bad = mx::logical_or(mx::less(raw, zero),
                                       mx::greater(raw, top));
        *oob = *oob ? mx::logical_or(**oob, bad) : bad;
      }
      idxs.push_back(mx::clip(raw, zero, top));
    } else if (kind == 2) {
      // The op indexes nothing: every start is 0, and the write (or read)
      // is the static one at the origin.
      idxs.push_back(mx::zeros(bshape, mx::int32));
    } else {
      // The batching coordinate: an iota along its own batch axis, spread
      // over the batch shape. Broadcast explicitly rather than leaning on
      // MLX's -- with only batching indices the broadcast of the views
      // alone would be narrower than the batch shape.
      mx::Shape view(bshape.size(), 1);
      view[static_cast<size_t>(b)] = static_cast<mx::ShapeElem>(a);
      idxs.push_back(mx::broadcast_to(
          mx::reshape(mx::arange(static_cast<int>(a), mx::int32), view),
          bshape));
    }
    axes.push_back(static_cast<int>(axis));
  }
}

// ops/gather.py's `.at[...]` methods, as the primitives they bottom out in.
// MLX has no scatter_subtract; its ArrayAt.subtract is add-of-negated, and
// a - b == a + (-b) for every IEEE bit pattern, so this is exact.
mx::array scatter_by(int64_t method, const mx::array& a,
                     const std::vector<mx::array>& idx, const mx::array& u,
                     const std::vector<int>& axes) {
  switch (method) {
    case 0: return mx::scatter(a, idx, u, axes);
    case 1: return mx::scatter_add(a, idx, u, axes);
    case 2: return mx::scatter_prod(a, idx, u, axes);
    case 3: return mx::scatter_max(a, idx, u, axes);
    case 4: return mx::scatter_min(a, idx, u, axes);
    case 5: return mx::scatter_add(a, idx, mx::negative(u), axes);
    default: throw std::invalid_argument("tape: bad scatter method");
  }
}

}  // namespace

bool Program::step_index(const Entry& e,
                         std::vector<std::optional<mx::array>>& env,
                         bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    case kSort: {
      // ops/sort.py _sort -> _gather_sorted, the arm whose comparator
      // compares an operand pair DIRECTLY (no key chain to evaluate).
      // That is the shape every top_k lowers to — jax's chlo.top_k
      // decomposition is `sort(values, iota)` under a strict TOTALORDER
      // GT — and it is what a plain jnp.sort/argsort emits too. A
      // comparator that computes a key first is declined in tape.py,
      // where the structural recognizer lives.
      // attrs [dim, descending?, key operand, key kind]
      //   key kind: 0 integer (already ordered), 1 float totalOrder,
      //             2 bool (widened to uint8, as the Python handler does)
      int dim = static_cast<int>(at[0]);
      bool descending = at[1] != 0;
      mx::array key = in(static_cast<size_t>(at[2]));
      if (at[3] == 1) {
        key = total_order_key(key);
      } else if (at[3] == 2) {
        key = mx::astype(key, mx::uint8);
      }
      // Bitwise NOT reverses the order for signed and unsigned alike.
      if (descending) key = mx::bitwise_invert(key);
      // mx::argsort reads the WRONG elements from a non-contiguous input
      // (MLX 0.32), and a non-last-axis sort arrives as a strided view.
      mx::array idx = mx::argsort(mx::contiguous(key), dim);
      for (size_t i = 0; i < e.outs.size(); i++)
        env[e.outs[i]] = mx::take_along_axis(in(i), idx, dim);
      break;
    }

    case kTopK: {
      // ops/sort.py _top_k: a stable DESCENDING argsort of the last axis,
      // cut to k. attrs [k, key kind] (kinds as kSort's).
      const mx::array& x = in(0);
      mx::array key = x;
      if (at[1] == 1) {
        key = total_order_key(key);
      } else if (at[1] == 2) {
        key = mx::astype(key, mx::uint8);
      }
      mx::array idx =
          mx::argsort(mx::contiguous(mx::bitwise_invert(key)), -1);
      mx::Shape lo(idx.shape().size(), 0), hi = idx.shape();
      hi.back() = static_cast<mx::ShapeElem>(at[0]);
      idx = mx::slice(idx, lo, hi);
      env[e.outs[0]] = mx::take_along_axis(x, idx, -1);
      env[e.outs[1]] = mx::astype(idx, mx::int32);
      break;
    }

    case kGather: {
      // ops/gather.py `_gather`, as ONE mx::gather. StableHLO's gather is
      // MLX's: index arrays keyed to operand axes, `slice_sizes` over the
      // whole operand rank, the slice starting at the index on an indexed
      // axis and at 0 elsewhere. What tape.py resolved is the split of
      // the coordinate vector, the clamp bounds (MLX clamps NOTHING --
      // it wraps a negative index like `take` and reads past the end
      // otherwise, measured), the reshape past the collapsed extent-1
      // dims and the offset_dims transpose.
      Cursor c(at);
      bool empty = c.flag();
      mx::Dtype odt = dtype_of(c.next());
      mx::Shape oshape = c.shp();
      if (empty) {
        env[e.outs[0]] = mx::zeros(oshape, odt);
        break;
      }
      mx::Shape bshape = c.shp();
      bool split = c.flag();
      int ivd = static_cast<int>(c.next());
      mx::Shape sizes = c.shp();
      std::vector<mx::array> idxs;
      std::vector<int> axes;
      read_index_plan(c, in(1), split, ivd, bshape, idxs, axes, nullptr);
      mx::array g = mx::gather(in(0), idxs, axes, sizes);
      g = mx::reshape(g, c.shp());
      std::vector<int> perm = c.vec();
      if (!is_identity_perm(perm)) g = mx::transpose(g, perm);
      env[e.outs[0]] = g;
      break;
    }

    case kScatter: {
      // ops/gather.py `_scatter`. The combiner picked the primitive at
      // lowering; here the indices are built and clamped, the updates are
      // put into MLX's [index batch dims, operand-rank slice] layout, and
      // XLA's OOB-DROP is applied by one of the two strategies tape.py
      // chose between (it chooses from static sizes, so both engines
      // choose alike).
      Cursor c(at);
      int64_t method = c.next();
      int64_t strategy = c.next();
      mx::Shape bshape = c.shp();
      bool split = c.flag();
      int ivd = static_cast<int>(c.next());
      std::vector<mx::array> idxs;
      std::vector<int> axes;
      std::optional<mx::array> oob;
      // A scatter always wants the drop predicate; it is unset only when
      // the op maps no index components at all, which is also the only
      // way `strategy` comes out 0.
      read_index_plan(c, in(1), split, ivd, bshape, idxs, axes, &oob);
      c.shp();  // the update slice shape, already folded into the reshape
      std::vector<int> uperm = c.vec();
      mx::array upd = in(2);
      if (!is_identity_perm(uperm)) upd = mx::transpose(upd, uperm);
      upd = mx::astype(mx::reshape(upd, c.shp()), in(0).dtype());
      if (strategy == 1) {
        // Neutral value: a dropped update becomes the combiner's
        // identity, so applying it is a no-op. Order- and duplicate-safe,
        // and it touches the UPDATES rather than the operand.
        if (!oob || !e.payload)
          throw std::runtime_error("tape: scatter neutral without a mask");
        upd = mx::where(mx::reshape(*oob, c.shp()), *e.payload, upd);
      } else if (strategy == 2) {
        // Dummy pad: one indexed axis grows by a window's worth of rows
        // and dropped updates are redirected there, then the pad is cut
        // off. Required for "set", where neutralizing an update would
        // race a genuine duplicate write at the clamped slot (a
        // fill_value == size index clamps onto the last real slot, which
        // is a systematic collision, not a rare one).
        if (!oob)
          throw std::runtime_error("tape: scatter pad without a mask");
        size_t pos = static_cast<size_t>(c.next());
        int pad = static_cast<int>(c.next());
        int extent = static_cast<int>(c.next());
        if (pos >= idxs.size())
          throw std::invalid_argument("tape: scatter pad position");
        int axis = axes[pos];
        mx::Shape padshape = in(0).shape();
        padshape[static_cast<size_t>(axis)] = pad;
        mx::array ext = mx::concatenate(
            {in(0), mx::zeros(padshape, in(0).dtype())}, axis);
        idxs[pos] = mx::where(*oob, mx::array(extent, mx::int32), idxs[pos]);
        mx::array full = scatter_by(method, ext, idxs, upd, axes);
        mx::Shape lo(full.ndim(), 0), hi = full.shape();
        hi[static_cast<size_t>(axis)] = extent;
        env[e.outs[0]] = mx::slice(full, lo, hi);
        break;
      }
      env[e.outs[0]] = scatter_by(method, in(0), idxs, upd, axes);
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
