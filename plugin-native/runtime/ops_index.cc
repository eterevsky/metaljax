// metaljax native engine — indexed reads and writes (ported from Stage 1's
// src/metaljax/ops/gather.py and ops/sort.py, deleted 0.11.6, ef5774d).
//
// StableHLO's gather and scatter go STRAIGHT to MLX's primitives: one index
// array per indexed operand axis, all of the index batch shape, built by the
// plan tape.py resolved. XLA's out-of-bounds rules are the part MLX does not
// share -- a gather clamps, a scatter DROPS -- and the two drop strategies
// below are how a drop is spelled with primitives that cannot skip a write.
// Sort and top_k live here too: they end in the same take_along_axis, over
// indices an argsort produced.

#include "program.h"

#include <limits>
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
// ops/gather.py's `index_arrays` read it exactly this way.
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

// ops/sort.py `_sort_key` and `_gather_sorted`'s key arms, as the `kind` a
// sort entry carries. The lowering picks the kind from the operand's element
// type and from WHICH arm of the recognizer produced the key:
//
//   0  integer -- already ordered, nothing to do
//   1  float, totalOrder key (`_gather_sorted`: jax's comparator did the
//      canonicalization itself, inside the key chain this frame evaluated)
//   2  bool, widened to uint8
//   3  complex, the (re, im) pair of canonicalized totalOrder keys packed
//      into one u64 -- which is what orders complex lexicographically
//   4  float, CANONICALIZED then totalOrder (`_sort_key`: the lexicographic
//      arm reads the raw operand, so -0 must tie with +0 and every NaN with
//      every other NaN here rather than in the comparator)
//
// The tie in 3 and 4 is not cosmetic: a -0 real part that splits a group
// would let the imaginary parts decide an order XLA does not have, and
// jnp.unique compares neighbours of a lexsort to count its uniques.
mx::array canon_float(const mx::array& x) {
  mx::array c = mx::add(x, weak(0.0, x));   // -0 + +0 == +0
  return mx::where(mx::isnan(c),
                   weak(std::numeric_limits<double>::quiet_NaN(), c), c);
}

mx::array sort_key(const mx::array& x, int64_t kind) {
  switch (kind) {
    case 1: return total_order_key(x);
    case 2: return mx::astype(x, mx::uint8);
    case 3: {
      mx::array re = mx::astype(total_order_key(canon_float(mx::real(x))),
                                mx::uint64);
      mx::array im = mx::astype(total_order_key(canon_float(mx::imag(x))),
                                mx::uint64);
      return mx::bitwise_or(
          mx::left_shift(re, mx::array(uint64_t{32}, mx::uint64)), im);
    }
    case 4: return total_order_key(canon_float(x));
    default: return x;
  }
}

// ops/sort.py `_argsort`: mx::argsort reads the WRONG elements from a
// non-contiguous input (MLX 0.32), and jax lowers a non-last-axis sort as
// transpose -> sort -> transpose, so the operand is routinely a strided view.
mx::array stable_argsort(const mx::array& key, int axis) {
  return mx::argsort(mx::contiguous(key), axis);
}

// ops/gather.py's `.at[...]` methods, as the primitives they bottom out in.
// MLX has no scatter_subtract; its ArrayAt.subtract is add-of-negated, and
// a - b == a + (-b) for every IEEE bit pattern, so this is exact.
mx::array scatter_by(int64_t method, const mx::array& a,
                     const std::vector<mx::array>& idx, const mx::array& u,
                     const std::vector<int>& axes) {
  switch (method) {
    case 0: return mx::scatter(a, idx, u, axes);
    case 1: return scatter_add_wide(a, idx, u, axes);
    case 2: return mx::scatter_prod(a, idx, u, axes);
    case 3: return mx::scatter_max(a, idx, u, axes);
    case 4: return mx::scatter_min(a, idx, u, axes);
    case 5: return scatter_add_wide(a, idx, mx::negative(u), axes);
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
      // comparator that computes a key first is declined at lowering
      // (tape.py then, metal_lowering.cc now), where the structural
      // recognizer lives.
      // attrs [dim, descending?, key operand, key kind] (kinds beside
      // `sort_key` above)
      int dim = static_cast<int>(at[0]);
      bool descending = at[1] != 0;
      mx::array key = sort_key(in(static_cast<size_t>(at[2])), at[3]);
      // Bitwise NOT reverses the order for signed and unsigned alike.
      if (descending) key = mx::bitwise_invert(key);
      mx::array idx = stable_argsort(key, dim);
      for (size_t i = 0; i < e.outs.size(); i++)
        env[e.outs[i]] = mx::take_along_axis(in(i), idx, dim);
      break;
    }

    case kLexSort: {
      // ops/sort.py `_lex_sorted`: a stable ASCENDING lexicographic sort by
      // the first `nkeys` operands, as successive stable argsorts from the
      // LAST key to the first, each one threaded through the permutation the
      // previous ones built. Stability is what makes that equal a single
      // lexicographic pass, and mx::argsort is stable (is_stable = true is
      // the only form jax emits).
      //
      // The comparator that means this is a select TREE over several key
      // pairs, which the lowering recognizes structurally; the tree itself is
      // never evaluated, so every canonicalization it would have applied is
      // in `sort_key` instead.
      // attrs [dim, nkeys, kind per key...]
      int dim = static_cast<int>(at[0]);
      const int64_t nkeys = at[1];
      if (nkeys < 1 || nkeys > static_cast<int64_t>(e.ins.size()))
        throw std::invalid_argument("tape: lex sort key count");
      std::optional<mx::array> perm;
      for (int64_t j = nkeys - 1; j >= 0; j--) {
        mx::array key = sort_key(in(static_cast<size_t>(j)),
                                 at[2 + static_cast<size_t>(j)]);
        if (perm) key = mx::take_along_axis(key, *perm, dim);
        mx::array idx = stable_argsort(key, dim);
        perm = perm ? mx::take_along_axis(*perm, idx, dim) : idx;
      }
      for (size_t i = 0; i < e.outs.size(); i++)
        env[e.outs[i]] = mx::take_along_axis(in(i), *perm, dim);
      break;
    }

    case kTopK: {
      // ops/sort.py _top_k: a stable DESCENDING argsort of the last axis,
      // cut to k. attrs [k, key kind] (kinds as kSort's).
      const mx::array& x = in(0);
      mx::array idx =
          stable_argsort(mx::bitwise_invert(sort_key(x, at[1])), -1);
      mx::Shape lo(idx.shape().size(), 0), hi = idx.shape();
      hi.back() = static_cast<mx::ShapeElem>(at[0]);
      idx = mx::slice(idx, lo, hi);
      env[e.outs[0]] = mx::take_along_axis(x, idx, -1);
      env[e.outs[1]] = mx::astype(idx, mx::int32);
      break;
    }

    case kApproxTopK: {
      // ops/sort.py `approx_top_k`: XLA's ApproxTopK custom call, answered
      // EXACTLY -- recall 1.0 satisfies any recall_target. Operands
      // (values, indices), results (values, indices) along `dim`; how many
      // to keep is resolved at lowering (aggregate_to_topk = false asks for
      // a WIDER result than backend_config's top_k, and under-filling it
      // would leave uninitialised device memory in the tail).
      // attrs [k, dim, key kind, is_max?]. The kind is the CANONICALIZING
      // one for floats (`_sort_key`, which `_top_k` does not need): -0 ties
      // with +0 and every NaN with every other NaN before the totalOrder key
      // is taken.
      const mx::array& vals = in(0);
      const int dim = static_cast<int>(at[1]);
      mx::array key = sort_key(vals, at[2]);
      if (at[3] != 0) key = mx::bitwise_invert(key);
      mx::array order = stable_argsort(key, dim);
      mx::Shape lo(order.shape().size(), 0), hi = order.shape();
      hi[static_cast<size_t>(dim)] = static_cast<mx::ShapeElem>(at[0]);
      mx::array top = mx::slice(order, lo, hi);
      env[e.outs[0]] = mx::take_along_axis(vals, top, dim);
      env[e.outs[1]] = mx::take_along_axis(in(1), top, dim);
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
      // chose between (it chose from static sizes, so both engines
      // chose alike).
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
      mx::Shape sshape = c.shp();  // the update slice, per operand dim
      std::vector<int> uperm = c.vec();
      mx::array upd = in(2);
      if (!is_identity_perm(uperm)) upd = mx::transpose(upd, uperm);
      upd = mx::astype(mx::reshape(upd, c.shp()), in(0).dtype());
      if (method == 6) {
        // A complex MULTIPLY, which is not componentwise: ops/gather.py
        // rewrote it into the apply path -- gather the current values,
        // combine, and SET the result -- and the lowering only takes this
        // arm when the op declares its indices unique, which is what makes
        // one write per slot the same as the combiner. The gather reads the
        // operand BEFORE any dummy pad, over the same clamped indices, so a
        // dropped update's product is garbage that the redirect discards.
        upd = mx::multiply(mx::gather(in(0), idxs, axes, sshape), upd);
        method = 0;
      } else if (method == 7) {
        // An APPLY body (jax's scatter_apply, and every `.at[i].apply(f)`):
        // the region is elementwise code over (current value, update), and
        // running it on the GATHERED values and SETTING the result equals
        // the scatter only while no two updates land on the same slot --
        // which is why the lowering takes this arm only under the op's own
        // `unique_indices`. Same read as the complex multiply above: before
        // any dummy pad, over the same clamped indices.
        const int64_t ncaps = c.next();
        std::vector<mx::array> args{mx::gather(in(0), idxs, axes, sshape),
                                    upd};
        for (int64_t k = 0; k < ncaps; k++)
          args.push_back(in(static_cast<size_t>(3 + k)));
        if (e.regions.size() != 1)
          throw std::runtime_error("tape: scatter apply without a body");
        std::vector<mx::array> res = e.regions[0]->call(args, in_trace);
        if (res.size() != 1)
          throw std::runtime_error("tape: scatter apply body result count");
        upd = mx::astype(res[0], in(0).dtype());
        method = 0;
      } else if (method == 8) {
        // The same body with NO uniqueness promise: XLA applies the updates
        // sequentially, and a computed body need be neither associative nor
        // idempotent, so two updates on one slot really do mean f(f(x)).
        // One update at a time, in row-major update order -- ops/gather.py's
        // arm, with its cap enforced at lowering. A dropped update leaves the
        // slot alone, which is why this arm takes no drop strategy: the
        // `where` below is per update, where a mask over the whole update
        // array could not say it once the body has run.
        const int64_t ncaps = c.next();
        if (e.regions.size() != 1)
          throw std::runtime_error("tape: scatter apply without a body");
        std::vector<mx::array> caps;
        for (int64_t k = 0; k < ncaps; k++)
          caps.push_back(in(static_cast<size_t>(3 + k)));
        int64_t nb = 1;
        for (mx::ShapeElem d : bshape) nb *= d;
        std::vector<mx::array> fidx;
        fidx.reserve(idxs.size());
        for (const mx::array& a : idxs)
          fidx.push_back(mx::reshape(a, mx::Shape{-1}));
        mx::Shape uflat{static_cast<mx::ShapeElem>(nb)};
        for (mx::ShapeElem d : sshape) uflat.push_back(d);
        mx::array uf = mx::reshape(upd, uflat);
        std::optional<mx::array> of;
        if (oob) of = mx::reshape(*oob, mx::Shape{-1});
        auto at_i = [](const mx::array& a, int64_t i) {
          return mx::reshape(mx::slice(a, mx::Shape{static_cast<mx::ShapeElem>(i)},
                                       mx::Shape{static_cast<mx::ShapeElem>(i + 1)}),
                             mx::Shape{});
        };
        // MLX has no complex scatter kernels, so a complex write goes by
        // parts here as it does everywhere else in this handler.
        auto put = [&](const mx::array& base, const std::vector<mx::array>& ix,
                       const mx::array& v) {
          if (!is_complex(base.dtype())) return mx::scatter(base, ix, v, axes);
          return make_complex(
              mx::scatter(mx::real(base), ix, mx::real(v), axes),
              mx::scatter(mx::imag(base), ix, mx::imag(v), axes));
        };
        mx::array out = in(0);
        for (int64_t i = 0; i < nb; i++) {
          std::vector<mx::array> one;
          one.reserve(fidx.size());
          for (const mx::array& a : fidx) one.push_back(at_i(a, i));
          mx::array old = mx::gather(out, one, axes, sshape);
          mx::Shape lo(uflat.size(), 0), hi = uflat;
          lo[0] = static_cast<mx::ShapeElem>(i);
          hi[0] = static_cast<mx::ShapeElem>(i + 1);
          mx::array ui = mx::reshape(mx::slice(uf, lo, hi), sshape);
          std::vector<mx::array> args{old, ui};
          args.insert(args.end(), caps.begin(), caps.end());
          std::vector<mx::array> res = e.regions[0]->call(args, in_trace);
          if (res.size() != 1)
            throw std::runtime_error("tape: scatter apply body result count");
          mx::array nv = mx::astype(res[0], out.dtype());
          if (of) nv = mx::where(at_i(*of, i), old, nv);
          out = put(out, one, nv);
        }
        env[e.outs[0]] = out;
        break;
      }
      std::optional<mx::array> mask;
      if (strategy == 1) {
        // Neutral value: a dropped update becomes the combiner's
        // identity, so applying it is a no-op. Order- and duplicate-safe,
        // and it touches the UPDATES rather than the operand.
        if (!oob || !e.payload)
          throw std::runtime_error("tape: scatter neutral without a mask");
        mask = mx::reshape(*oob, c.shp());
      }
      // Dummy pad: one indexed axis grows by a window's worth of rows
      // and dropped updates are redirected there, then the pad is cut
      // off. Required for "set", where neutralizing an update would
      // race a genuine duplicate write at the clamped slot (a
      // fill_value == size index clamps onto the last real slot, which
      // is a systematic collision, not a rare one). The redirection is
      // an index question, so it is done once, whatever the operand's
      // dtype makes of the write below.
      size_t pos = 0;
      int pad = 0, extent = 0, axis = 0;
      if (strategy == 2) {
        if (!oob)
          throw std::runtime_error("tape: scatter pad without a mask");
        pos = static_cast<size_t>(c.next());
        pad = static_cast<int>(c.next());
        extent = static_cast<int>(c.next());
        if (pos >= idxs.size())
          throw std::invalid_argument("tape: scatter pad position");
        axis = axes[pos];
        idxs[pos] = mx::where(*oob, mx::array(extent, mx::int32), idxs[pos]);
      }
      // The single-window SET (`metal_lowering.cc`'s strategies 3 and 4, the
      // decode cache append): the whole index batch is ONE coordinate, so the
      // op writes one contiguous window and `mx::slice_update` is the write --
      // one pass over the operand against the dummy pad's three. The starts are
      // the plan's, already clamped, stacked into the vector MLX's dynamic
      // slice takes.
      std::optional<mx::array> start;
      std::optional<mx::array> guard;
      if (strategy == 3 || strategy == 4) {
        std::vector<mx::array> parts;
        parts.reserve(idxs.size());
        for (const mx::array& a : idxs)
          parts.push_back(mx::reshape(a, mx::Shape{}));
        start = mx::stack(parts);
        if (strategy == 4) {
          // XLA DROPS an update whose start is out of bounds. The lowering
          // could not prove that impossible here, so the drop is spelled at
          // window size: the indices are clamped, and writing back what is
          // already at the clamped start leaves the operand exactly as it was.
          if (!oob)
            throw std::runtime_error("tape: scatter window guard without a "
                                     "mask");
          guard = mx::reshape(*oob, mx::Shape(sshape.size(), 1));
        }
      }
      // One write, over an operand and updates of a dtype MLX can scatter.
      auto write = [&](const mx::array& base, mx::array u) -> mx::array {
        if (mask) u = mx::where(*mask, *e.payload, u);
        if (start) {
          mx::array win = mx::reshape(u, sshape);
          if (guard)
            win = mx::where(
                *guard,
                mx::reshape(mx::gather(base, idxs, axes, sshape), sshape), win);
          return mx::slice_update(base, win, *start, axes);
        }
        if (strategy != 2) return scatter_by(method, base, idxs, u, axes);
        mx::Shape padshape = base.shape();
        padshape[static_cast<size_t>(axis)] = pad;
        mx::array ext = mx::concatenate(
            {base, mx::zeros(padshape, base.dtype())}, axis);
        mx::array full = scatter_by(method, ext, idxs, u, axes);
        mx::Shape lo(full.ndim(), 0), hi = full.shape();
        hi[static_cast<size_t>(axis)] = extent;
        return mx::slice(full, lo, hi);
      };
      if (is_complex(in(0).dtype())) {
        // MLX has no complex scatter kernels: write the two PARTS and
        // recombine, which ops/gather.py did by recursing on
        // `mx.real`/`mx.imag`. Exact for the componentwise combiners, and
        // the lowering declines the rest. The neutral in `payload` is the
        // part's (f32), for the same reason.
        env[e.outs[0]] =
            make_complex(write(mx::real(in(0)), mx::real(upd)),
                         write(mx::imag(in(0)), mx::imag(upd)));
      } else {
        env[e.outs[0]] = write(in(0), upd);
      }
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
