// metaljax native engine — rng_bit_generator (src/metaljax/ops/rng.py).
//
// Nothing here is "an implementation of philox": it is XLA's, transliterated
// from the Python handler that was reverse-engineered from
// xla/hlo/builder/lib/prng.cc and verified word for word against the CPU
// backend, because the whole value of the family is that its bits match.

#include "program.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace metaljax {

namespace {

// XLA's Philox4x32 and ThreeFry2x32, transliterated from the Python handler
// that was reverse-engineered from xla/hlo/builder/lib/prng.cc and verified
// word for word against the CPU backend. Nothing here is "an implementation
// of philox": it is THAT implementation, in the same order, with the same
// widths, because the whole value of the family is that its bits match.

constexpr int64_t kM0 = 0xD2511F53;
constexpr int64_t kM1 = 0xCD9E8D57;
constexpr int64_t kW0 = 0x9E3779B9;
constexpr int64_t kW1 = 0xBB67AE85;
constexpr int64_t kLo32 = 0xFFFFFFFF;

mx::array u64c(int64_t v) { return mx::array(v, mx::uint64); }
mx::array u32c(int64_t v) { return mx::array(v, mx::uint32); }

// _philox_blocks: ten rounds over vectors of u32 counter words.
void philox_blocks(mx::array& x0, mx::array& x1, mx::array& x2, mx::array& x3,
                   mx::array k0, mx::array k1) {
  for (int i = 0; i < 10; i++) {
    mx::array p0 = mx::multiply(mx::astype(x0, mx::uint64), u64c(kM0));
    mx::array p1 = mx::multiply(mx::astype(x2, mx::uint64), u64c(kM1));
    mx::array hi0 = mx::astype(mx::right_shift(p0, u64c(32)), mx::uint32);
    mx::array lo0 = mx::astype(mx::bitwise_and(p0, u64c(kLo32)), mx::uint32);
    mx::array hi1 = mx::astype(mx::right_shift(p1, u64c(32)), mx::uint32);
    mx::array lo1 = mx::astype(mx::bitwise_and(p1, u64c(kLo32)), mx::uint32);
    mx::array n0 = mx::bitwise_xor(mx::bitwise_xor(hi1, x1), k0);
    mx::array n2 = mx::bitwise_xor(mx::bitwise_xor(hi0, x3), k1);
    x0 = n0;
    x1 = lo1;
    x2 = n2;
    x3 = lo0;
    k0 = mx::add(k0, u32c(kW0));
    k1 = mx::add(k1, u32c(kW1));
  }
}

const int kTfRot[8] = {13, 15, 26, 6, 17, 29, 16, 24};

// _threefry2x32: 20 rounds, key injection every 4.
void threefry2x32(mx::array& x0, mx::array& x1, const mx::array& k0,
                  const mx::array& k1) {
  mx::array ks2 = mx::bitwise_xor(mx::bitwise_xor(u32c(0x1BD11BDA), k0), k1);
  mx::array ks[3] = {k0, k1, ks2};
  x0 = mx::add(x0, ks[0]);
  x1 = mx::add(x1, ks[1]);
  for (int g = 0; g < 5; g++) {
    for (int r = 0; r < 4; r++) {
      int rot = kTfRot[(g % 2) * 4 + r];
      x0 = mx::add(x0, x1);
      x1 = mx::bitwise_or(mx::left_shift(x1, u32c(rot)),
                          mx::right_shift(x1, u32c(32 - rot)));
      x1 = mx::bitwise_xor(x0, x1);
    }
    int j = g + 1;
    x0 = mx::add(x0, ks[j % 3]);
    x1 = mx::add(mx::add(x1, ks[(j + 1) % 3]), u32c(j));
  }
}

}  // namespace

bool Program::step_rng(const Entry& e,
                       std::vector<std::optional<mx::array>>& env,
                       bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  const std::vector<int64_t>& at = e.attrs;

  switch (e.op) {
    case kRng: {
      // ops/rng.py _rng_bit_generator. tape.py resolved the algorithm,
      // the block/half counts and every shape; what is left is the
      // arithmetic and the two ways of laying the words out.
      Cursor c(at);
      bool threefry = c.flag();
      bool state_u32 = c.flag();
      mx::Dtype out_dt = dtype_of(c.next());
      mx::Dtype unsigned_dt = dtype_of(c.next());
      mx::Shape out_shape = c.shp();

      mx::array state = in(0);
      if (state_u32) state = mx::view(state, mx::uint64);
      mx::array key = mx::squeeze(
          mx::slice(state, mx::Shape{0}, mx::Shape{1}), 0);
      mx::array ctr = mx::squeeze(
          mx::slice(state, mx::Shape{1}, mx::Shape{2}), 0);
      mx::array k0 = mx::astype(mx::bitwise_and(key, u64c(kLo32)),
                                mx::uint32);
      mx::array k1 = mx::astype(mx::right_shift(key, u64c(32)), mx::uint32);

      mx::array bits = state;   // replaced below on every path
      int64_t consumed = 0;
      if (threefry) {
        int64_t n_half = c.next();
        int split = static_cast<int>(c.next());
        bool scalar = c.flag();
        bool needs_slice = c.flag();
        mx::Shape h = c.shp(), rounded = c.shp(), dims = c.shp();
        // One threefry block per half-element pair; the counter is the
        // state's plus the row-major linear index.
        mx::array u64s = mx::add(
            ctr, mx::arange(static_cast<double>(n_half), mx::uint64));
        mx::array x0 = mx::astype(mx::bitwise_and(u64s, u64c(kLo32)),
                                  mx::uint32);
        mx::array x1 = mx::astype(mx::right_shift(u64s, u64c(32)),
                                  mx::uint32);
        threefry2x32(x0, x1, k0, k1);
        mx::array both = mx::concatenate(
            {mx::reshape(x0, h), mx::reshape(x1, h)}, split + 1);
        both = mx::reshape(both, rounded);
        if (needs_slice) {
          mx::Shape start(dims.size(), 0), stop = rounded;
          stop[static_cast<size_t>(split)] = dims[split];
          both = mx::slice(both, start, stop);
        }
        if (scalar) both = mx::reshape(both, mx::Shape{});
        // The narrow arm casts (truncating each element); the handler
        // does NOT view a signed output back, so neither does this.
        if (unsigned_dt != mx::uint32) both = mx::astype(both, unsigned_dt);
        bits = both;
        consumed = n_half;
      } else {
        int64_t n = c.next(), width = c.next(), num_u32 = c.next(),
                nv4 = c.next();
        if (nv4 == 0) {
          // Empty output: XLA consumes no blocks, so the state comes
          // back unchanged.
          mx::array st = state_u32 ? mx::view(state, mx::uint32) : state;
          env[e.outs[0]] = st;
          env[e.outs[1]] = mx::zeros(out_shape, out_dt);
          break;
        }
        mx::array low = mx::add(
            ctr, mx::arange(static_cast<double>(nv4), mx::uint64));
        mx::array carry = mx::astype(mx::less(low, ctr), mx::uint64);
        mx::array high = mx::add(key, carry);
        mx::array x0 = mx::astype(mx::bitwise_and(low, u64c(kLo32)),
                                  mx::uint32);
        mx::array x1 = mx::astype(mx::right_shift(low, u64c(32)),
                                  mx::uint32);
        mx::array x2 = mx::astype(mx::bitwise_and(high, u64c(kLo32)),
                                  mx::uint32);
        mx::array x3 = mx::astype(mx::right_shift(high, u64c(32)),
                                  mx::uint32);
        philox_blocks(x0, x1, x2, x3, k0, k1);
        if (width == 64) {
          mx::array b0 = mx::bitwise_or(
              mx::astype(x0, mx::uint64),
              mx::left_shift(mx::astype(x1, mx::uint64), u64c(32)));
          mx::array b1 = mx::bitwise_or(
              mx::astype(x2, mx::uint64),
              mx::left_shift(mx::astype(x3, mx::uint64), u64c(32)));
          bits = mx::slice(
              mx::reshape(mx::stack({b0, b1}, 1),
                          mx::Shape{static_cast<mx::ShapeElem>(2 * nv4)}),
              mx::Shape{0}, mx::Shape{static_cast<mx::ShapeElem>(n)});
        } else {
          bits = mx::slice(
              mx::reshape(mx::stack({x0, x1, x2, x3}, 1),
                          mx::Shape{static_cast<mx::ShapeElem>(4 * nv4)}),
              mx::Shape{0},
              mx::Shape{static_cast<mx::ShapeElem>(num_u32)});
          // Narrow types truncate one u32 per element.
          if (width < 32) bits = mx::astype(bits, unsigned_dt);
        }
        // Signed variants are the same bits, reinterpreted.
        if (out_dt != unsigned_dt) bits = mx::view(bits, out_dt);
        bits = mx::reshape(bits, out_shape);
        consumed = nv4;
      }
      mx::array new_state =
          mx::stack({key, mx::add(ctr, u64c(consumed))}, 0);
      if (state_u32) new_state = mx::view(new_state, mx::uint32);
      env[e.outs[0]] = new_state;
      env[e.outs[1]] = bits;
      break;
    }

    default:
      return false;
  }
  return true;
}

}  // namespace metaljax
