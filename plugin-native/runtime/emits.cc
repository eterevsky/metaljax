// metaljax native engine — the recognizer emits (M4).
//
// These are not StableHLO ops: src/metaljax/tape.py (Stage 1, deleted
// 0.11.6, ef5774d; now metal/metal_lowering.cc) resolved a recognizer's
// plan (metaljax.interpreter._rewrite_plan) at lowering time and asked for
// them by pseudo-name, and the chain of literal ops each one replaces never
// reaches the tape at all. Each handler is a transliteration of the `emit`
// in the Python module that owned the recognizer -- qmm, sdpa, moe -- and the
// differential tests compare output BYTES against those very functions, so a
// drift is a failure and not a tolerance question.

#include "program.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

namespace {

// metaljax.sdpa._apply: an optional transpose then an optional reshape.
struct Rec {
  bool has_perm = false;
  std::vector<int> perm;
  bool has_shape = false;
  mx::Shape shape;
};

Rec read_rec(Cursor& c) {
  Rec r;
  r.has_perm = c.flag();
  r.perm = c.vec();
  r.has_shape = c.flag();
  r.shape = c.shp();
  return r;
}

mx::array apply_rec(const Rec& r, mx::array x) {
  if (r.has_perm) x = mx::transpose(x, r.perm);
  if (r.has_shape) x = mx::reshape(x, r.shape);
  return x;
}

// MLX names its quantization modes with strings; the tape carries the code
// the Python match recorded.
const char* qmode(int64_t mode) { return mode == 0 ? "affine" : "mxfp4"; }

// moe._pair_lead: `x` with a leading pair axis (size P or 1).
mx::array pair_lead(const mx::array& x, bool paired) {
  if (paired) return x;
  mx::Shape s{1};
  for (auto d : x.shape()) s.push_back(d);
  return mx::reshape(x, s);
}

// moe._split_experts: a qmm pack's leading `E * N` rows as [E, N, ...].
mx::array split_experts(const mx::array& a, int64_t E) {
  if (a.ndim() >= 3) return a;
  return mx::reshape(a, mx::Shape{static_cast<mx::ShapeElem>(E),
                                  a.shape()[0] / static_cast<int>(E),
                                  a.shape()[1]});
}

}  // namespace

bool Program::step_emit(const Entry& e,
                        std::vector<std::optional<mx::array>>& env,
                        bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    // --- recognizer emits (M4) -------------------------------------
    //
    // Each is a transliteration of the `emit` in the Python module that
    // owns the recognizer; the analysis, the packing and every static
    // decision stay there, and what arrives here is the plan. The
    // differential tests compare output BYTES against those very
    // functions, so a drift is a failure and not a tolerance question.

    case kQmm: {
      // metaljax.qmm.emit. attrs:
      //   [transpose?, lrank, lperm..., batched?, B, M, K,
      //    group_size, bits, mode, perm?, swapped?, out_dtype,
      //    bshape, mshape, nshape]     (each shape: rank, dims...)
      // ins: [activation, w, scales, (biases if affine), (perm if any)]
      Cursor c(at);
      bool do_transpose = c.flag();
      std::vector<int> lperm = c.vec();
      bool batched = c.flag();
      int64_t B = c.next(), M = c.next(), K = c.next();
      int64_t gs = c.next(), bits = c.next(), mode = c.next();
      bool has_perm = c.flag();
      bool swapped = c.flag();
      mx::Dtype out_dt = dtype_of(c.next());
      mx::Shape bshape = c.shp(), mshape = c.shp(), nshape = c.shp();

      mx::array x = in(0);
      if (do_transpose) x = mx::transpose(x, lperm);
      x = mx::reshape(x, batched
                             ? mx::Shape{static_cast<mx::ShapeElem>(B),
                                         static_cast<mx::ShapeElem>(M),
                                         static_cast<mx::ShapeElem>(K)}
                             : mx::Shape{static_cast<mx::ShapeElem>(M),
                                         static_cast<mx::ShapeElem>(K)});
      if (!is_float(x.dtype())) x = mx::astype(x, out_dt);
      if (has_perm)
        x = mx::take(x, in(e.ins.size() - 1),
                     static_cast<int>(x.ndim()) - 1);
      std::optional<mx::array> biases;
      if (mode == 0) biases = in(3);
      mx::array y = mx::quantized_matmul(
          x, in(1), in(2), biases, /*transpose=*/true,
          static_cast<int>(gs), static_cast<int>(bits), qmode(mode));
      mx::Shape out;
      for (auto d : bshape) out.push_back(d);
      if (swapped) {
        // The dot had the weight on its LHS, so the result is [..., N, M].
        y = mx::swapaxes(y, static_cast<int>(y.ndim()) - 1,
                         static_cast<int>(y.ndim()) - 2);
        for (auto d : nshape) out.push_back(d);
        for (auto d : mshape) out.push_back(d);
      } else {
        for (auto d : mshape) out.push_back(d);
        for (auto d : nshape) out.push_back(d);
      }
      y = mx::reshape(y, out);
      if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
      env[e.outs[0]] = y;
      break;
    }

    case kSdpaMask: {
      // metaljax.sdpa._mask_array, the cache MISS body. The cache itself
      // is this entry: tape.py emitted one per distinct mask key, at the
      // first attention that reads it, and every later root reads the
      // slot — which is what `env`'s tuple-keyed entry did in Python.
      // attrs [kind (0 select, 1 additive), dtype, scaled?];
      // fattrs [const * mul, mul].
      int64_t kind = at[0];
      mx::Dtype dt = dtype_of(at[1]);
      mx::array mask = in(0);
      if (kind == 0) {
        // Additive, never boolean: MLX's boolean mask gives a
        // fully-masked row a finite value where the literal chain gives
        // NaN. `L + C == C` for a sentinel C, so this is exact.
        mask = mx::where(mask, mx::array(0.0, dt),
                         mx::array(e.fattrs[0], dt));
      } else {
        if (mask.dtype() != dt) mask = mx::astype(mask, dt);
        if (at[2]) mask = mx::multiply(mask, mx::array(e.fattrs[1],
                                                       mask.dtype()));
      }
      env[e.outs[0]] = mask;
      break;
    }

    case kSdpa: {
      // metaljax.sdpa.emit. attrs [q_rec, k_rec, v_rec, dtype, out_dtype,
      //   mask?, mask_rec, out_pre, out_perm, out_post]; fattrs [scale].
      // ins: [q, k, v, (mask)].
      Cursor c(at);
      Rec q_rec = read_rec(c), k_rec = read_rec(c), v_rec = read_rec(c);
      mx::Dtype dt = dtype_of(c.next());
      mx::Dtype out_dt = dtype_of(c.next());
      bool has_mask = c.flag();
      Rec mask_rec = read_rec(c);
      bool has_pre = c.flag();
      mx::Shape pre = c.shp();
      bool has_perm = c.flag();
      std::vector<int> perm = c.vec();
      bool has_post = c.flag();
      mx::Shape post = c.shp();

      mx::array q = apply_rec(q_rec, in(0));
      mx::array k = apply_rec(k_rec, in(1));
      mx::array v = apply_rec(v_rec, in(2));
      if (q.dtype() != dt) q = mx::astype(q, dt);
      if (k.dtype() != dt) k = mx::astype(k, dt);
      if (v.dtype() != dt) v = mx::astype(v, dt);
      std::optional<mx::array> mask;
      if (has_mask) mask = apply_rec(mask_rec, in(3));
      mx::array out = mx::fast::scaled_dot_product_attention(
          q, k, v, static_cast<float>(e.fattrs[0]), /*mask_mode=*/"",
          mask);
      if (has_pre) out = mx::reshape(out, pre);
      if (has_perm) out = mx::transpose(out, perm);
      if (has_post) out = mx::reshape(out, post);
      if (out.dtype() != out_dt) out = mx::astype(out, out_dt);
      env[e.outs[0]] = out;
      break;
    }

    case kRmsNorm: {
      // metal_norm.cc: jax's spelled-out RMS norm as MLX's fused kernel.
      // ins [x, w]; attrs [out_dtype]; fattrs [eps].  The kernel
      // accumulates in f32 exactly as the literal chain does; the one
      // difference (1-ULP class) is that the chain rounds `x * rsqrt` to
      // bf16 before the weight multiply and the kernel rounds once.
      mx::Dtype out_dt = dtype_of(at[0]);
      mx::array out = mx::fast::rms_norm(
          in(0), in(1), static_cast<float>(e.fattrs[0]));
      if (out.dtype() != out_dt) out = mx::astype(out, out_dt);
      env[e.outs[0]] = out;
      break;
    }

    case kMlaSdpa: {
      // metal_mla.cc: maxtext's multi-span MLA decode attention.  The
      // per-span masked softmax partials joined by the flash renormalization
      // ARE the softmax over the concatenated scores, so: concat the spans,
      // build the additive mask from the segment ids (exactly the numbers
      // the matched `_where` chain selects), and run ONE fused sdpa — whose
      // vector kernel supports the MLA 192/128 head geometry.  The graph
      // pre-scaled q, so scale = 1.
      // ins [q(B,1,H,D), then per span k(B,T,H,D), v(B,T,H,Dv), seg(B,T)];
      // attrs [B, H, D, Dv, nspans, T..., seg_val, dtype, out_dtype];
      // fattrs [mask_true, mask_false].
      Cursor c(at);
      const auto B = static_cast<mx::ShapeElem>(c.next());
      const auto H = static_cast<mx::ShapeElem>(c.next());
      c.next();  // D: carried for the record, read off the arrays
      const auto Dv = static_cast<mx::ShapeElem>(c.next());
      const int64_t nspans = c.next();
      std::vector<int64_t> T(static_cast<size_t>(nspans));
      for (int64_t i = 0; i < nspans; i++) T[static_cast<size_t>(i)] = c.next();
      const int64_t seg_val = c.next();
      mx::Dtype dt = dtype_of(c.next());
      mx::Dtype out_dt = dtype_of(c.next());
      (void)Dv;

      mx::array q = in(0);  // [B, 1, H, D] -> [B, H, 1, D]
      q = mx::transpose(q, {0, 2, 1, 3});
      if (q.dtype() != dt) q = mx::astype(q, dt);
      std::vector<mx::array> ks, vs, masks;
      for (int64_t i = 0; i < nspans; i++) {
        mx::array k = in(static_cast<size_t>(1 + 3 * i));
        mx::array v = in(static_cast<size_t>(2 + 3 * i));
        mx::array seg = in(static_cast<size_t>(3 + 3 * i));
        if (k.dtype() != dt) k = mx::astype(k, dt);
        if (v.dtype() != dt) v = mx::astype(v, dt);
        ks.push_back(k);
        vs.push_back(v);
        masks.push_back(mx::where(
            mx::equal(seg, weak_int(seg_val, seg)),
            mx::array(e.fattrs[0], dt), mx::array(e.fattrs[1], dt)));
      }
      // [B, T, H, D] concat on T, then to [B, H, T, D].
      mx::array k = mx::transpose(mx::concatenate(ks, 1), {0, 2, 1, 3});
      mx::array v = mx::transpose(mx::concatenate(vs, 1), {0, 2, 1, 3});
      mx::array mask =
          mx::reshape(mx::concatenate(masks, 1),
                      mx::Shape{B, 1, 1, static_cast<mx::ShapeElem>(
                                             k.shape(2))});
      mx::array out = mx::fast::scaled_dot_product_attention(
          q, k, v, /*scale=*/1.0f, /*mask_mode=*/"", mask);
      out = mx::transpose(out, {0, 2, 1, 3});  // [B, H, 1, Dv] -> [B, 1, H, Dv]
      if (out.dtype() != out_dt) out = mx::astype(out, out_dt);
      (void)H;
      env[e.outs[0]] = out;
      break;
    }

    // --- the MoE expert gather (metaljax.moe.emit) -------------------
    //
    // `emit` is a small interpreter over a plan of pair-space nodes, so
    // the lowering spreads that plan over the tape: one entry per node,
    // in the plan's own dependency order, with the root's result written
    // by the tail. Liveness then prunes the pair-space intermediates
    // exactly as the tape prunes anything else, and only the tail is
    // charged the root's bytes — so the eager flush cadence sees the one
    // sync-point-moving number the Python engine sees.

    case kMoeEIdx:
      // ctx.eidx = reshape(indices, (P,)).astype(uint32)
      env[e.outs[0]] = mx::astype(
          mx::reshape(in(0), mx::Shape{static_cast<mx::ShapeElem>(at[0])}),
          mx::uint32);
      break;

    case kMoeTIdx:
      // ctx.tidx = repeat(arange(T, uint32), K)
      env[e.outs[0]] = mx::repeat(
          mx::arange(static_cast<double>(at[0]), mx::uint32),
          static_cast<int>(at[1]));
      break;

    case kMoeGather: {
      // moe._emit_ext, the `env` arm. attrs [ea?, ea, ta?, ta, P];
      // ins [value, eidx, tidx].
      bool has_ea = at[0] != 0;
      int ea = static_cast<int>(at[1]);
      bool has_ta = at[2] != 0;
      int ta = static_cast<int>(at[3]);
      const mx::array& x0 = in(0);
      if (!has_ea && !has_ta) {
        env[e.outs[0]] = x0;
        break;
      }
      if (has_ea && has_ta) {
        mx::array x = mx::moveaxis(x0, ea, 0);
        x = mx::moveaxis(x, ta < ea ? ta + 1 : ta, 1);
        // x[eidx, tidx]: one element of the (expert, token) grid per
        // pair, the whole trailing slice.
        mx::Shape sizes = x.shape();
        sizes[0] = 1;
        sizes[1] = 1;
        mx::array g = mx::gather(x, {in(1), in(2)}, {0, 1}, sizes);
        mx::Shape out{static_cast<mx::ShapeElem>(at[4])};
        for (size_t i = 2; i < x.shape().size(); i++)
          out.push_back(x.shape()[i]);
        env[e.outs[0]] = mx::reshape(g, out);
        break;
      }
      int axis = has_ea ? ea : ta;
      const mx::array& idx = has_ea ? in(1) : in(2);
      env[e.outs[0]] = mx::moveaxis(mx::take(x0, idx, axis), axis, 0);
      break;
    }

    case kMoeConcat: {
      // moe._emit_elem's concatenate arm: pair leads may be P or 1.
      int axis = static_cast<int>(at[0]);
      int lead = 0;
      for (size_t i = 0; i < e.ins.size(); i++)
        lead = std::max(lead, in(i).shape()[0]);
      std::vector<mx::array> parts;
      parts.reserve(e.ins.size());
      for (size_t i = 0; i < e.ins.size(); i++) {
        mx::array x = in(i);
        if (x.shape()[0] != lead) {
          mx::Shape s = x.shape();
          s[0] = lead;
          x = mx::broadcast_to(x, s);
        }
        parts.push_back(x);
      }
      env[e.outs[0]] = mx::concatenate(std::move(parts), axis);
      break;
    }

    case kMoeView: {
      // moe._emit_view. attrs [src paired?, kind, ...]:
      //   0 bcast    [nkeep, (kept?, dest)..., trailing]
      //   1 perm     [order]
      //   2 reshape  [trailing]
      //   3 slice    [n, (start, stop, stride)...]
      //   4 dotperm  [transpose?, perm, trailing]
      Cursor c(at);
      mx::array x = pair_lead(in(0), c.flag());
      int64_t kind = c.next();
      if (kind == 0) {
        int64_t nkeep = c.next();
        std::vector<int64_t> kept(static_cast<size_t>(nkeep));
        std::vector<int64_t> dest(static_cast<size_t>(nkeep));
        for (int64_t j = 0; j < nkeep; j++) {
          kept[j] = c.next();
          dest[j] = c.next();
        }
        mx::Shape trailing = c.shp();
        mx::Shape view(trailing.size(), 1);
        mx::Shape lo(x.shape().size(), 0), hi = x.shape();
        std::vector<int> squeeze_axes;
        for (int64_t j = 0; j < nkeep; j++) {
          if (!kept[j]) {
            hi[static_cast<size_t>(1 + j)] = 1;
            squeeze_axes.push_back(static_cast<int>(1 + j));
          } else {
            view[static_cast<size_t>(dest[j])] = x.shape()[1 + j];
          }
        }
        if (!squeeze_axes.empty())
          x = mx::squeeze(mx::slice(x, lo, hi), squeeze_axes);
        mx::Shape mid{x.shape()[0]};
        for (auto d : view) mid.push_back(d);
        mx::Shape out{x.shape()[0]};
        for (auto d : trailing) out.push_back(d);
        env[e.outs[0]] = mx::broadcast_to(mx::reshape(x, mid), out);
      } else if (kind == 1) {
        std::vector<int> order = c.vec();
        std::vector<int> perm{0};
        for (int i : order) perm.push_back(1 + i);
        env[e.outs[0]] = mx::transpose(x, perm);
      } else if (kind == 2) {
        mx::Shape trailing = c.shp();
        mx::Shape out{x.shape()[0]};
        for (auto d : trailing) out.push_back(d);
        env[e.outs[0]] = mx::reshape(x, out);
      } else if (kind == 3) {
        int64_t n = c.next();
        mx::Shape lo{0}, hi{x.shape()[0]}, st{1};
        for (int64_t j = 0; j < n; j++) {
          lo.push_back(static_cast<mx::ShapeElem>(c.next()));
          hi.push_back(static_cast<mx::ShapeElem>(c.next()));
          st.push_back(static_cast<mx::ShapeElem>(c.next()));
        }
        env[e.outs[0]] = mx::slice(x, lo, hi, st);
      } else if (kind == 4) {
        bool do_transpose = c.flag();
        std::vector<int> order = c.vec();
        mx::Shape trailing = c.shp();
        if (do_transpose) {
          std::vector<int> perm{0};
          for (int i : order) perm.push_back(1 + i);
          x = mx::transpose(x, perm);
        }
        mx::Shape out{x.shape()[0]};
        for (auto d : trailing) out.push_back(d);
        env[e.outs[0]] = mx::reshape(x, out);
      } else {
        throw std::invalid_argument("tape: unknown moe view");
      }
      break;
    }

    case kMoeDot: {
      // moe._emit_dot. attrs [shortcut?, src paired?, ta, packed?, E,
      //   n_first?, group_size, bits, mode, perm?, P, M, K, N, out_dtype,
      //   mshape, nshape]; ins [data, eidx, (tidx if shortcut),
      //   weight | w, scales, (biases if affine), (perm if any)].
      Cursor c(at);
      bool shortcut = c.flag();
      bool src_paired = c.flag();
      int ta = static_cast<int>(c.next());
      bool packed = c.flag();
      int64_t E = c.next();
      bool n_first = c.flag();
      int64_t gs = c.next(), bits = c.next(), mode = c.next();
      bool has_perm = c.flag();
      int64_t P = c.next(), M = c.next(), K = c.next(), N = c.next();
      mx::Dtype out_dt = dtype_of(c.next());
      mx::Shape mshape = c.shp(), nshape = c.shp();

      mx::array x = in(0);
      std::optional<mx::array> lhs_idx;
      if (shortcut) {
        // Token-indexed and nothing else: hand gather_* the ungathered
        // rows and let it do the indexing, instead of materializing K
        // copies.
        std::vector<int> perm{ta};
        for (size_t i = 0; i < x.shape().size(); i++)
          if (static_cast<int>(i) != ta) perm.push_back(static_cast<int>(i));
        int rows = x.shape()[ta];
        x = mx::reshape(mx::transpose(x, perm),
                        mx::Shape{rows, static_cast<mx::ShapeElem>(M),
                                  static_cast<mx::ShapeElem>(K)});
        lhs_idx = in(2);
      } else {
        x = pair_lead(x, src_paired);
        if (x.shape()[0] != static_cast<int>(P)) {
          mx::Shape s = x.shape();
          s[0] = static_cast<mx::ShapeElem>(P);
          x = mx::broadcast_to(x, s);
        }
        x = mx::reshape(x, mx::Shape{static_cast<mx::ShapeElem>(P),
                                     static_cast<mx::ShapeElem>(M),
                                     static_cast<mx::ShapeElem>(K)});
        lhs_idx = mx::arange(static_cast<double>(P), mx::uint32);
      }
      std::optional<mx::array> rhs_idx = in(1);
      const size_t wbase = shortcut ? 3 : 2;

      mx::array y = x;
      if (packed) {
        mx::array w = split_experts(in(wbase), E);
        mx::array scales = split_experts(in(wbase + 1), E);
        std::optional<mx::array> biases;
        if (mode == 0) biases = split_experts(in(wbase + 2), E);
        if (has_perm)
          x = mx::take(x, in(e.ins.size() - 1),
                       static_cast<int>(x.ndim()) - 1);
        if (!is_float(x.dtype())) x = mx::astype(x, out_dt);
        y = mx::gather_qmm(x, w, scales, biases, lhs_idx, rhs_idx,
                           /*transpose=*/true, static_cast<int>(gs),
                           static_cast<int>(bits), qmode(mode));
      } else {
        const mx::array& wv = in(wbase);
        mx::array w3 =
            n_first
                // gather_mm computes `a @ b`, so an [E, N, K] weight has
                // to be transposed — as a LAZY view: materializing it
                // would copy the whole expert set.
                ? mx::swapaxes(
                      mx::reshape(wv,
                                  mx::Shape{static_cast<mx::ShapeElem>(E),
                                            static_cast<mx::ShapeElem>(N),
                                            static_cast<mx::ShapeElem>(K)}),
                      -1, -2)
                : mx::reshape(wv,
                              mx::Shape{static_cast<mx::ShapeElem>(E),
                                        static_cast<mx::ShapeElem>(K),
                                        static_cast<mx::ShapeElem>(N)});
        y = mx::gather_mm(x, w3, lhs_idx, rhs_idx);
      }
      y = mx::reshape(y, mx::Shape{static_cast<mx::ShapeElem>(P),
                                   static_cast<mx::ShapeElem>(M),
                                   static_cast<mx::ShapeElem>(N)});
      if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
      mx::Shape out{static_cast<mx::ShapeElem>(P)};
      for (auto d : mshape) out.push_back(d);
      for (auto d : nshape) out.push_back(d);
      env[e.outs[0]] = mx::reshape(y, out);
      break;
    }

    case kMoeTail: {
      // moe.emit's tail: weight the pairs, sum over K, put the token axis
      // back. attrs [out paired?, P, T, K, sum_dtype, out_dtype,
      // out_axis, out_shape]; ins [plan output, router weights].
      Cursor c(at);
      mx::array y = pair_lead(in(0), c.flag());
      int64_t P = c.next(), T = c.next(), Kk = c.next();
      mx::Dtype sum_dt = dtype_of(c.next());
      mx::Dtype out_dt = dtype_of(c.next());
      int out_axis = static_cast<int>(c.next());
      mx::Shape want = c.shp();

      mx::Shape trailing(y.shape().begin() + 1, y.shape().end());
      if (y.shape()[0] != static_cast<int>(P)) {
        mx::Shape s{static_cast<mx::ShapeElem>(P)};
        for (auto d : trailing) s.push_back(d);
        y = mx::broadcast_to(y, s);
      }
      mx::array w = mx::astype(
          mx::reshape(in(1), mx::Shape{static_cast<mx::ShapeElem>(P)}),
          y.dtype());
      mx::Shape wshape{static_cast<mx::ShapeElem>(P)};
      for (size_t i = 0; i < trailing.size(); i++) wshape.push_back(1);
      y = mx::multiply(y, mx::reshape(w, wshape));
      // The dense graph converts the products before summing them.
      if (y.dtype() != sum_dt) y = mx::astype(y, sum_dt);
      mx::Shape split{static_cast<mx::ShapeElem>(T),
                      static_cast<mx::ShapeElem>(Kk)};
      for (auto d : trailing) split.push_back(d);
      y = mx::sum(mx::reshape(y, split), std::vector<int>{1});
      if (out_axis) y = mx::moveaxis(y, 0, out_axis);
      if (y.shape() != want)
        throw std::runtime_error("tape: moe produced the wrong shape");
      if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
      env[e.outs[0]] = y;
      break;
    }

    case kRaggedDot: {
      // metal_ragged.cc: jax's `lax.ragged_dot` dense fallback, rewritten.
      // The dense form runs every row against every group's [K, N] and masks
      // the non-members to zero; this runs each row against ITS group only.
      // attrs [g, m, M, k, n, out_dtype, stacked?, L]; ins [x [m, k],
      // w ([g, k, n], or the carried [L, g, k, n] stack when `stacked`),
      // ends [g] = cumsum(group_sizes), (layer index when `stacked`)].
      // m is the REAL row count (the absorbed pad's input); M >= m is the
      // row count the root declared, and the dense semantics for the M - m
      // padded rows — masked to zero everywhere — is an exact zero row, so
      // the result is zero-padded.
      //
      // Row i belongs to group e iff ends[e-1] <= i < ends[e]; the group
      // index is #(ends <= i), which is nondecreasing in i whatever `ends`
      // holds — so `sorted_indices` below is a structural fact, not an
      // assumption about the data.  A row at or past ends[g-1] is in no
      // group (the dense mask zeroes it everywhere), so its gather — clamped
      // to the last group — is discarded by `valid`.
      Cursor c(at);
      int64_t g = c.next(), m = c.next(), M = c.next();
      int64_t k = c.next(), n = c.next();
      mx::Dtype out_dt = dtype_of(c.next());
      const bool stacked = c.flag();
      int64_t L = c.next();
      mx::Shape out_shape{static_cast<mx::ShapeElem>(M),
                          static_cast<mx::ShapeElem>(n)};
      if (m == 0 || k == 0 || n == 0) {
        env[e.outs[0]] = mx::zeros(out_shape, out_dt);
        break;
      }
      const mx::array& x = in(0);
      const mx::array& w = in(1);
      mx::array ends =
          mx::reshape(mx::astype(in(2), mx::int32),
                      mx::Shape{1, static_cast<mx::ShapeElem>(g)});
      mx::array row = mx::reshape(
          mx::arange(static_cast<double>(m), mx::int32),
          mx::Shape{static_cast<mx::ShapeElem>(m), 1});
      mx::array eid =
          mx::sum(mx::astype(mx::greater_equal(row, ends), mx::int32),
                  std::vector<int>{1});
      mx::array valid = mx::less(
          eid, mx::array(static_cast<int32_t>(g), mx::int32));
      mx::array eidc = mx::astype(
          mx::minimum(eid, mx::array(static_cast<int32_t>(g - 1), mx::int32)),
          mx::uint32);
      mx::array rhs_idx = eidc;
      mx::array wg = w;
      if (stacked) {
        // The analysis proved `w` is the [1, 0, 2, 3]-transposed view of a
        // [g, L, k, n] buffer riding a pass-through carry: transposing it
        // BACK and flattening [g, L] is a zero-copy view of the original,
        // and matrix (e, l) sits at linear index e * L + l.  The dense
        // chain's dynamic_slice clamped the layer index; so does this.
        mx::array l = mx::minimum(
            mx::maximum(mx::reshape(mx::astype(in(3), mx::int32), mx::Shape{}),
                        mx::array(0, mx::int32)),
            mx::array(static_cast<int32_t>(L - 1), mx::int32));
        rhs_idx = mx::astype(
            mx::add(mx::multiply(mx::astype(eidc, mx::int32),
                                 mx::array(static_cast<int32_t>(L),
                                           mx::int32)),
                    l),
            mx::uint32);
        wg = mx::reshape(mx::transpose(w, {1, 0, 2, 3}),
                         mx::Shape{static_cast<mx::ShapeElem>(g * L),
                                   static_cast<mx::ShapeElem>(k),
                                   static_cast<mx::ShapeElem>(n)});
      }
      mx::array lhs_idx = mx::arange(static_cast<double>(m), mx::uint32);
      mx::array y = mx::gather_mm(
          mx::reshape(x, mx::Shape{static_cast<mx::ShapeElem>(m), 1,
                                   static_cast<mx::ShapeElem>(k)}),
          wg, lhs_idx, rhs_idx, /*sorted_indices=*/true);
      y = mx::reshape(y, mx::Shape{static_cast<mx::ShapeElem>(m),
                                   static_cast<mx::ShapeElem>(n)});
      y = mx::where(mx::reshape(valid,
                                mx::Shape{static_cast<mx::ShapeElem>(m), 1}),
                    y, mx::array(0.0f, y.dtype()));
      if (M > m)
        y = mx::pad(y, {0}, mx::Shape{0},
                    mx::Shape{static_cast<mx::ShapeElem>(M - m)},
                    mx::array(0.0f, y.dtype()));
      if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
      env[e.outs[0]] = y;
      break;
    }

    case kStackedDot: {
      // metal_stacked.cc: `dot_general(x, dynamic_index_in_dim(stack,
      // layer))` — jax's scanned-layer weight read — as one gather_mm over
      // an [L, K, N] strided view of the ORIGINAL contiguous stack.  attrs
      // [nperm, back_perm..., L, K, N, sl, sk, sn, M, out_dtype, out rank,
      // out shape...]; ins [x, stack (possibly a carried transposed view),
      // layer].  The analysis proved transpose(stack, back_perm) is the
      // row-contiguous original, so `as_strided`'s flatten is a view, and
      // MLX's gather kernels take the batch/ld strides from the array — no
      // copy on any path.  The dense chain's dynamic_slice clamped the
      // index; so does this.
      Cursor c(at);
      std::vector<int> back_perm = c.vec();
      const int64_t L = c.next(), K = c.next(), N = c.next();
      const int64_t sl = c.next(), sk = c.next(), sn = c.next();
      const int64_t M = c.next();
      mx::Dtype out_dt = dtype_of(c.next());
      mx::Shape out_shape = c.shp();
      const mx::array& x = in(0);
      const mx::array& w = in(1);
      mx::array wb = is_identity_perm(back_perm)
                         ? w
                         : mx::transpose(w, back_perm);
      mx::array wv = mx::as_strided(
          wb,
          mx::Shape{static_cast<mx::ShapeElem>(L),
                    static_cast<mx::ShapeElem>(K),
                    static_cast<mx::ShapeElem>(N)},
          mx::Strides{sl, sk, sn}, /*offset=*/0);
      mx::array a2 = mx::reshape(
          x, mx::Shape{1, static_cast<mx::ShapeElem>(M),
                       static_cast<mx::ShapeElem>(K)});
      mx::array idx = mx::astype(
          mx::minimum(mx::maximum(mx::reshape(mx::astype(in(2), mx::int32),
                                              mx::Shape{1}),
                                  mx::array(0, mx::int32)),
                      mx::array(static_cast<int32_t>(L - 1), mx::int32)),
          mx::uint32);
      mx::array lhs_idx = mx::zeros(mx::Shape{1}, mx::uint32);
      mx::array y =
          mx::gather_mm(a2, wv, lhs_idx, idx, /*sorted_indices=*/true);
      y = mx::reshape(y, out_shape);
      if (y.dtype() != out_dt) y = mx::astype(y, out_dt);
      env[e.outs[0]] = y;
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
