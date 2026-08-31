// metaljax native engine — contractions (ported from Stage 1's
// src/metaljax/ops/linalg.py, deleted 0.11.6, ef5774d).
//
// dot_general, in its three arms. Which one runs is a static property of the
// operand dtypes that tape.py resolved; all three are here because MLX's
// matmul is float-only, so an integer dot is either exact f32 K-chunks or a
// materialized outer product summed in integer arithmetic.
//
// (The rest of ops/linalg.py -- the LAPACK targets -- computed on the host
// and reached the tape as a host call; still the case here, see host.cc.)

#include "program.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

bool Program::step_linalg(const Entry& e,
                          std::vector<std::optional<mx::array>>& env,
                          bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    case kDotGeneral: {
      // ops/linalg.py _dot_general. Which of the three arms runs is a
      // static property of the operand dtypes, so tape.py resolved it;
      // all three are here because MLX has no integer matmul.
      size_t p = 0;
      int64_t lrank = at[p++];
      std::vector<int> lperm = axes(at, p, lrank);
      p += static_cast<size_t>(lrank);
      int64_t rrank = at[p++];
      std::vector<int> rperm = axes(at, p, rrank);
      p += static_cast<size_t>(rrank);
      int64_t b = at[p++], m = at[p++], k = at[p++], n = at[p++];
      mx::Dtype out_dt = dtype_of(at[p++]);
      int64_t out_rank = at[p++];
      mx::Shape out_shape = shape(at, p, out_rank);
      p += static_cast<size_t>(out_rank);
      int64_t kind = at[p++];
      int64_t chunk = at[p++];
      int64_t bside = at[p++];
      int64_t bgroups = at[p++];
      int64_t btail = at[p++];

      mx::array l = mx::transpose(in(0), lperm);
      mx::array r = mx::transpose(in(1), rperm);
      if (b * m * n == 0) {
        // mx.matmul with an empty M/N output yields an array whose host
        // conversion segfaults (null data pointer, MLX 0.32).
        env[e.outs[0]] = mx::zeros(out_shape, out_dt);
        break;
      }
      if (bside != 0) {
        // The MIDDLE-contracted operand, contracted WHERE IT LIES.  The
        // lowering has already checked that the plain arm would copy it,
        // that this arm would not, and that its own permutation
        // (`lperm`/`rperm` on that side) is the identity -- so every
        // reshape below is a view, and the transposes above are no-ops.
        // mx::matmul broadcasts the batch dims, so the other operand only
        // needs a unit axis to ride.
        const int gi = static_cast<int>(bgroups), ti = static_cast<int>(btail);
        const int bi = static_cast<int>(b), ki = static_cast<int>(k);
        mx::array a = l, c = r;
        if (bside == 1) {
          // The weight is the RHS: [B, G, K, Ntail] against [B, 1, M, K].
          a = mx::reshape(l, mx::Shape{bi, 1, static_cast<int>(m), ki});
          c = mx::reshape(r, mx::Shape{bi, gi, ki, ti});
        } else {
          // The weight is the LHS: [B, G, Mtail, K] (last two axes swapped,
          // which `check_transpose` reads off `stx == 1` with no copy)
          // against [B, 1, K, N].
          a = mx::transpose(mx::reshape(l, mx::Shape{bi, gi, ki, ti}),
                            std::vector<int>{0, 1, 3, 2});
          c = mx::reshape(r, mx::Shape{bi, 1, ki, static_cast<int>(n)});
        }
        if (a.dtype() != out_dt) a = mx::astype(a, out_dt);
        if (c.dtype() != out_dt) c = mx::astype(c, out_dt);
        mx::array o4 = mx::matmul(a, c);
        // The lhs arm lands on [B, G, Mtail, N], which IS the result layout
        // (batch ++ lhs free ++ rhs free).  The rhs arm lands on
        // [B, G, M, Ntail] and the layout wants [B, M, G, Ntail] -- the same
        // bytes in the same order when M == 1, so the permute is skipped
        // there rather than left to MLX's unit-axis contiguity rule.
        if (bside == 1 && m != 1)
          o4 = mx::transpose(o4, std::vector<int>{0, 2, 1, 3});
        env[e.outs[0]] = mx::reshape(o4, out_shape);
        break;
      }
      mx::array l3 = mx::reshape(
          l, mx::Shape{static_cast<int>(b), static_cast<int>(m),
                       static_cast<int>(k)});
      mx::array r3 = mx::reshape(
          r, mx::Shape{static_cast<int>(b), static_cast<int>(k),
                       static_cast<int>(n)});
      mx::array o3 = l3;
      if (kind == 1) {
        // _int_dot_via_f32: f32 holds every integer up to 2**24 exactly,
        // so an 8-bit integer dot over short enough K-slices is exact in
        // a real matmul; the per-chunk results accumulate in integer
        // arithmetic, which wraps exactly like XLA's integer dot.
        if (chunk <= 0)
          throw std::invalid_argument("tape: bad dot chunk");
        mx::Dtype acc_dt = out_dt.size() == 8 ? mx::int64 : mx::int32;
        std::optional<mx::array> acc;
        for (int64_t s = 0; s < k; s += chunk) {
          mx::array lp = l3, rp = r3;
          if (k > chunk) {
            int64_t hi = std::min(s + chunk, k);
            lp = mx::slice(l3, mx::Shape{0, 0, static_cast<int>(s)},
                           mx::Shape{static_cast<int>(b),
                                     static_cast<int>(m),
                                     static_cast<int>(hi)});
            rp = mx::slice(r3, mx::Shape{0, static_cast<int>(s), 0},
                           mx::Shape{static_cast<int>(b),
                                     static_cast<int>(hi),
                                     static_cast<int>(n)});
          }
          mx::array part = mx::astype(
              mx::matmul(mx::astype(lp, mx::float32),
                         mx::astype(rp, mx::float32)),
              acc_dt);
          acc = acc ? mx::add(*acc, part) : part;
        }
        o3 = mx::astype(*acc, out_dt);
      } else if (kind == 2 || kind == 3) {
        // MLX matmul is float-only: an explicit multiply-accumulate over
        // a materialized [B, M, K, N] product. kind 3 is the bool arm,
        // whose accumulator stays bool until mx.sum promotes it.
        mx::array acc = kind == 3 ? l3 : mx::astype(l3, mx::int64);
        mx::array prod = mx::multiply(mx::expand_dims(acc, 3),
                                      mx::expand_dims(
                                          mx::astype(r3, acc.dtype()), 1));
        o3 = mx::astype(mx::sum(prod, std::vector<int>{2}), out_dt);
      } else {
        if (l3.dtype() != out_dt) l3 = mx::astype(l3, out_dt);
        if (r3.dtype() != out_dt) r3 = mx::astype(r3, out_dt);
        if (k == 1 && !is_complex(out_dt)) {
          // A K=1 contraction has no sum: the batched matmul is ONE product
          // per output element, i.e. a broadcast multiply.  Bit-identical —
          // the matmul path computes the exact product in its f32
          // accumulator and rounds once on the way out, which for bf16/f16
          // (8- and 11-bit mantissas: the product is exact in f32) and for
          // f32 itself is the elementwise multiply's own RNE rounding.
          // What it buys: jax spells `x * vec` einsums as batching-only
          // dots (maxtext's norm scales and rope: 4 per layer, B=hidden,
          // M=N=K=1), and each was a 1024-batch 1x1x1 GEMM launch; a
          // multiply is one fusable elementwise node instead.
          o3 = mx::multiply(l3, r3);  // [b,m,1] * [b,1,n] -> [b,m,n]
        } else {
          o3 = mx::matmul(l3, r3);
        }
      }
      env[e.outs[0]] = mx::reshape(o3, out_shape);
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
