// metaljax native engine -- the gated-delta-net decode step (see gdn.h).
//
// THE KERNEL.  The state is [B, Hv, Dk, Dv] with Dv the FAST axis, and it
// has to stay that way: the layer writes it straight back into the stacked
// recurrent cache.  So the thread mapping is
//
//     thread.x = dv        (tx = Dv threads)   consecutive threads read
//                                              consecutive state addresses
//     thread.y = dk chunk  (ty chunks)         each thread holds ceil(Dk/ty)
//                                              state values in REGISTERS
//     grid.z   = b * Hv + hv                   one threadgroup per head
//
// which reads the state once and writes it once, fully coalesced.  The two
// sums over dk cross thread.y, so they reduce through a threadgroup scratch
// array with barriers rather than `simd_sum` (which would only cover
// thread.x).  mlx-lm's comparator kernel makes the opposite choice because
// its state layout has Dk fast; ours cannot.
//
// The arithmetic is keras' own, in keras' order:
//
//     state *= exp(g);  kv += state * k;   [reduce]
//     delta  = (v - kv) * beta;
//     state += k * delta;  out += state * q;   [reduce]
//
// so the only numeric difference from the literal chain is the ORDER of
// those two reductions.  Every dtype narrowing the chain does -- the
// `_l2norm`'s bf16 rsqrt and product, the state's bf16 round trip -- is
// replayed, which is what keeps a 128-token decode from drifting.
//
// Licensed under the Apache License, Version 2.0.

#include "gdn.h"

#include "program.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace metaljax {
namespace {

std::mutex& Mu() {
  static std::mutex mu;
  return mu;
}

struct Built {
  bool ok = false;
  mx::fast::CustomKernelFunction fn;
};

// Keyed by the generated SOURCE: two matches whose geometry and numerics
// agree share one Metal library, and a source is never built twice.
std::map<std::string, Built>& Cache() {
  static std::map<std::string, Built>* c = new std::map<std::string, Built>();
  return *c;
}

GdnStats& Counters() {
  static GdnStats s;
  return s;
}

bool Debugging() {
  static const bool on = [] {
    const char* v = std::getenv("METALJAX_DEBUG");
    return v != nullptr && std::string(v) == "1";
  }();
  return on;
}

void Debug(const std::string& line) {
  if (!Debugging()) return;
  std::fprintf(stderr, "[metaljax-native] gdn: %s\n", line.c_str());
  std::fflush(stderr);
}

// MLX's kernel preamble (utils.h -> bf16.h) supplies `bfloat16_t`; `half` is
// MSL's own.  Anything else declines at the caller.
const char* MslType(mx::Dtype d) {
  if (d == mx::float32) return "float";
  if (d == mx::float16) return "half";
  if (d == mx::bfloat16) return "bfloat16_t";
  return nullptr;
}

// Round `expr` through a narrower float, the way a convert in the chain
// does.  A no-op when the chain stayed in f32.
std::string Round(mx::Dtype narrow, const std::string& expr) {
  if (narrow == mx::float32) return expr;
  const char* t = MslType(narrow);
  if (t == nullptr) return expr;
  return std::string("static_cast<float>(static_cast<") + t + ">(" + expr +
         "))";
}

// Shortest literal that round-trips, with a forced '.' -- `1f` is not an MSL
// float literal.
std::string Lit(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  std::string s(buf);
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
      s.find("inf") == std::string::npos && s.find("nan") == std::string::npos)
    s += ".0";
  return s + "f";
}

std::string U(int64_t v) { return std::to_string(v) + "u"; }

std::atomic<int64_t>& Seq() {
  static std::atomic<int64_t> seq{0};
  return seq;
}

}  // namespace

bool GdnSpec::operator==(const GdnSpec& o) const {
  return B == o.B && Hv == o.Hv && Hk == o.Hk && Dk == o.Dk && Dv == o.Dv &&
         q == o.q && k == o.k && v == o.v && g == o.g && beta == o.beta &&
         l2q == o.l2q && l2k == o.l2k && eps_q == o.eps_q &&
         eps_k == o.eps_k && narrow_q == o.narrow_q && narrow_k == o.narrow_k &&
         round_state == o.round_state && narrow_state == o.narrow_state &&
         scale == o.scale;
}

GdnGeometry GdnPickGeometry(const GdnSpec& s) {
  GdnGeometry geo;
  if (s.Dv < 1 || s.Dv > 512 || s.Dk < 1) return geo;
  for (int64_t ty : {1, 2, 4, 8, 16, 32}) {
    const int64_t ch = (s.Dk + ty - 1) / ty;
    if (ch > 32) continue;
    // The threadgroup is tiled along dv: the largest divisor of Dv that is
    // at most 32 (one simdgroup's worth), so the GPU gets Dv/tgx times as
    // many threadgroups as the untiled shape did.
    int64_t tgx = s.Dv;
    for (int64_t t = std::min<int64_t>(32, s.Dv); t >= 1; t--)
      if (s.Dv % t == 0) { tgx = t; break; }
    if (tgx * ty > 1024) continue;
    geo.ok = true;
    geo.tx = s.Dv;
    geo.ty = ty;
    geo.ch = ch;
    geo.tgx = tgx;
    return geo;
  }
  return geo;
}

std::string GdnSource(const GdnSpec& s) {
  const GdnGeometry geo = GdnPickGeometry(s);
  if (!geo.ok) return std::string();
  const int64_t TY = geo.ty, CH = geo.ch, TG = geo.tgx;
  std::string L;
  auto add = [&](const std::string& line) { L += line; L += "\n"; };

  add("  uint dv = thread_position_in_grid.x;");
  add("  uint lx = thread_position_in_threadgroup.x;");
  add("  uint yy = thread_position_in_threadgroup.y;");
  add("  uint n  = thread_position_in_grid.z;");
  add("  uint hv = n % " + U(s.Hv) + ";");
  add("  uint b  = n / " + U(s.Hv) + ";");
  add("  uint hk = hv / " + U(s.Hv / s.Hk) + ";");
  add("  threadgroup float red0[" + std::to_string(TG * TY) + "];");
  if (s.l2q || s.l2k)
    add("  threadgroup float red1[" + std::to_string(TG * TY) + "];");
  add("  float qs[" + std::to_string(CH) + "];");
  add("  float ks[" + std::to_string(CH) + "];");
  add("  float sv[" + std::to_string(CH) + "];");
  add("");
  // ---- load q and k, and their sums of squares.  Every dv lane loads the
  // same head vector (it does not depend on dv); the loads all hit cache and
  // the redundancy buys a reduction that needs no extra barrier phase.
  add("  float pq = 0.0f;");
  add("  float pk = 0.0f;");
  add("  for (uint i = 0; i < " + U(CH) + "; i++) {");
  add("    uint dk = yy * " + U(CH) + " + i;");
  add("    float qq = 0.0f;");
  add("    float kk = 0.0f;");
  add("    if (dk < " + U(s.Dk) + ") {");
  add("      uint o = (b * " + U(s.Hk) + " + hk) * " + U(s.Dk) + " + dk;");
  add("      qq = static_cast<float>(q[o]);");
  add("      kk = static_cast<float>(k[o]);");
  add("    }");
  add("    qs[i] = qq;");
  add("    ks[i] = kk;");
  add("    pq += qq * qq;");
  add("    pk += kk * kk;");
  add("  }");
  if (s.l2q || s.l2k) {
    add("  red0[yy * " + U(TG) + " + lx] = pq;");
    add("  red1[yy * " + U(TG) + " + lx] = pk;");
    add("  threadgroup_barrier(metal::mem_flags::mem_threadgroup);");
    add("  float sq = 0.0f;");
    add("  float sk = 0.0f;");
    add("  for (uint j = 0; j < " + U(TY) + "; j++) {");
    add("    sq += red0[j * " + U(TG) + " + lx];");
    add("    sk += red1[j * " + U(TG) + " + lx];");
    add("  }");
    add("  threadgroup_barrier(metal::mem_flags::mem_threadgroup);");
  }
  // ---- normalize, then apply the pre-scale exactly where the chain does:
  // the norm rounds to the model dtype, the scale runs in f32 afterwards.
  std::string qexpr = "qs[i]";
  if (s.l2q) {
    add("  float rq = " +
        Round(s.narrow_q,
              "metal::precise::rsqrt(" +
                  Round(s.narrow_q,
                        Round(s.narrow_q, "sq") + " + " + Lit(s.eps_q)) +
                  ")") +
        ";");
    qexpr = Round(s.narrow_q, "qs[i] * rq");
  }
  if (s.scale != 1.0) qexpr = "(" + qexpr + ") * " + Lit(s.scale);
  std::string kexpr = "ks[i]";
  if (s.l2k) {
    add("  float rk = " +
        Round(s.narrow_k,
              "metal::precise::rsqrt(" +
                  Round(s.narrow_k,
                        Round(s.narrow_k, "sk") + " + " + Lit(s.eps_k)) +
                  ")") +
        ";");
    kexpr = Round(s.narrow_k, "ks[i] * rk");
  }
  if (qexpr != "qs[i]" || kexpr != "ks[i]") {
    add("  for (uint i = 0; i < " + U(CH) + "; i++) {");
    if (qexpr != "qs[i]") add("    qs[i] = " + qexpr + ";");
    if (kexpr != "ks[i]") add("    ks[i] = " + kexpr + ";");
    add("  }");
  }
  add("");
  // ---- the gates and the value, one scalar each per head / lane.
  add("  float ge = metal::precise::exp(static_cast<float>(g[b * " +
      U(s.Hv) + " + hv]));");
  add("  float bt = static_cast<float>(beta[b * " + U(s.Hv) + " + hv]);");
  add("  float vv = static_cast<float>(v[(b * " + U(s.Hv) + " + hv) * " +
      U(s.Dv) + " + dv]);");
  add("");
  // ---- read the state ONCE into registers, decay it, and accumulate the
  // memory read.
  add("  float pkv = 0.0f;");
  add("  for (uint i = 0; i < " + U(CH) + "; i++) {");
  add("    uint dk = yy * " + U(CH) + " + i;");
  add("    float sc = 0.0f;");
  add("    if (dk < " + U(s.Dk) + ") {");
  add("      sc = state[(n * " + U(s.Dk) + " + dk) * " + U(s.Dv) + " + dv];");
  if (s.round_state) add("      sc = " + Round(s.narrow_state, "sc") + ";");
  add("      sc = sc * ge;");
  add("    }");
  add("    sv[i] = sc;");
  add("    pkv += sc * ks[i];");
  add("  }");
  add("  red0[yy * " + U(TG) + " + lx] = pkv;");
  add("  threadgroup_barrier(metal::mem_flags::mem_threadgroup);");
  add("  float kv = 0.0f;");
  add("  for (uint j = 0; j < " + U(TY) + "; j++) kv += red0[j * " + U(TG) +
      " + lx];");
  add("  threadgroup_barrier(metal::mem_flags::mem_threadgroup);");
  add("");
  add("  float delta = (vv - kv) * bt;");
  // ---- write the state ONCE, and accumulate the output from the registers
  // rather than reading it back.
  add("  float pout = 0.0f;");
  add("  for (uint i = 0; i < " + U(CH) + "; i++) {");
  add("    uint dk = yy * " + U(CH) + " + i;");
  add("    float sn = sv[i] + ks[i] * delta;");
  add("    if (dk < " + U(s.Dk) + ") {");
  add("      new_state[(n * " + U(s.Dk) + " + dk) * " + U(s.Dv) + " + dv] = sn;");
  add("      pout += sn * qs[i];");
  add("    }");
  add("  }");
  add("  red0[yy * " + U(TG) + " + lx] = pout;");
  add("  threadgroup_barrier(metal::mem_flags::mem_threadgroup);");
  add("  if (yy == 0u) {");
  add("    float o = 0.0f;");
  add("    for (uint j = 0; j < " + U(TY) + "; j++) o += red0[j * " + U(TG) +
      " + lx];");
  add("    out[(b * " + U(s.Hv) + " + hv) * " + U(s.Dv) + " + dv] = o;");
  add("  }");
  return L;
}

bool GdnProve(const GdnSpec& s) {
  // METALJAX_GDN_KERNEL=0 keeps the recognizer but forces the MLX fallback,
  // which is what splits the win into "fewer tape entries" and "the kernel
  // itself" without rebuilding.  It is also the bisect handle if a generated
  // source ever misbehaves on an OS update.
  if (const char* v = std::getenv("METALJAX_GDN_KERNEL"))
    if (std::string(v) == "0") return false;
  const GdnGeometry geo = GdnPickGeometry(s);
  if (!geo.ok) return false;
  if (MslType(s.q) == nullptr || MslType(s.k) == nullptr ||
      MslType(s.v) == nullptr || MslType(s.g) == nullptr ||
      MslType(s.beta) == nullptr)
    return false;
  const std::string src = GdnSource(s);
  if (src.empty()) return false;

  std::lock_guard<std::mutex> lock(Mu());
  auto it = Cache().find(src);
  if (it != Cache().end()) return it->second.ok;

  Built built;
  const std::string name =
      "mjn_gdn_" + std::to_string(++Seq());
  if (const char* dump = std::getenv("MJDBG_GDN_SOURCE"))
    if (std::string(dump) == "1")
      std::fprintf(stderr, "[metaljax-native] gdn: %s\n%s\n", name.c_str(),
                   src.c_str());
  try {
    built.fn = mx::fast::metal_kernel(
        name, {"q", "k", "v", "g", "beta", "state"}, {"out", "new_state"}, src);
    // Prove it on dummy operands of the real geometry, synchronously: the
    // Metal library is generated at eval, and this is the last moment there
    // is a thread that may safely take the exception.
    std::vector<mx::array> probe{
        mx::zeros({static_cast<int>(s.B), static_cast<int>(s.Hk),
                   static_cast<int>(s.Dk)}, s.q),
        mx::zeros({static_cast<int>(s.B), static_cast<int>(s.Hk),
                   static_cast<int>(s.Dk)}, s.k),
        mx::zeros({static_cast<int>(s.B), static_cast<int>(s.Hv),
                   static_cast<int>(s.Dv)}, s.v),
        mx::zeros({static_cast<int>(s.B), static_cast<int>(s.Hv)}, s.g),
        mx::zeros({static_cast<int>(s.B), static_cast<int>(s.Hv)}, s.beta),
        mx::zeros({static_cast<int>(s.B), static_cast<int>(s.Hv),
                   static_cast<int>(s.Dk), static_cast<int>(s.Dv)},
                  mx::float32)};
    std::vector<mx::Shape> shapes{
        {static_cast<int>(s.B), static_cast<int>(s.Hv), static_cast<int>(s.Dv)},
        {static_cast<int>(s.B), static_cast<int>(s.Hv), static_cast<int>(s.Dk),
         static_cast<int>(s.Dv)}};
    std::vector<mx::Dtype> dts{mx::float32, mx::float32};
    std::vector<mx::array> outs = built.fn(
        probe, shapes, dts,
        std::tuple<int, int, int>{static_cast<int>(geo.tx),
                                  static_cast<int>(geo.ty),
                                  static_cast<int>(s.B * s.Hv)},
        std::tuple<int, int, int>{static_cast<int>(geo.tgx),
                                  static_cast<int>(geo.ty), 1},
        {}, std::nullopt, false, {});
    mx::eval(outs);
    built.ok = true;
    Counters().builds++;
    Debug("built " + name + " (dv=" + std::to_string(geo.tx) +
          " tg=" + std::to_string(geo.tgx) + "x" + std::to_string(geo.ty) +
          " regs=" + std::to_string(geo.ch) + " groups=" +
          std::to_string(s.B * s.Hv * (geo.tx / geo.tgx)) + ")");
  } catch (const std::exception& e) {
    // A governor refusal says the MACHINE is out of memory, not that this
    // source is unbuildable.  Caching that verdict would cost the process
    // its kernel for good over a transient -- the same distinction
    // `Program::settle_msl` draws -- so leave the source unproven and let a
    // later lowering try again.  This call still answers false, and the step
    // runs on the MLX fallback until it is proven.
    if (is_oom(e)) {
      Debug(std::string("kernel probe hit the memory governor (") + e.what() +
            "); leaving the source unproven");
      return false;
    }
    built.ok = false;
    Counters().build_failures++;
    Debug(std::string("kernel did not build (") + e.what() +
          "); the step runs on MLX ops");
  }
  const bool ok = built.ok;
  Cache().emplace(src, std::move(built));
  return ok;
}

namespace {

// The op-by-op fallback: the same arithmetic, and the same narrowings, run
// as ordinary MLX ops.  It is what a source Metal will not build falls back
// to, and it is the reference the differential test compares the kernel
// against.
mx::array Narrow(const mx::array& x, mx::Dtype d) {
  if (d == mx::float32) return x;
  return mx::astype(mx::astype(x, d), mx::float32);
}

mx::array L2(const mx::array& x, double eps, mx::Dtype narrow) {
  mx::array f = mx::astype(x, mx::float32);
  mx::array ss = mx::sum(mx::multiply(f, f), -1, true);
  mx::array r = Narrow(
      mx::rsqrt(Narrow(mx::add(Narrow(ss, narrow),
                               mx::array(static_cast<float>(eps), mx::float32)),
                       narrow)),
      narrow);
  return Narrow(mx::multiply(f, r), narrow);
}

}  // namespace

std::vector<mx::array> GdnRun(const GdnSpec& s, const mx::array& q,
                              const mx::array& k, const mx::array& v,
                              const mx::array& g, const mx::array& beta,
                              const mx::array& state) {
  const GdnGeometry geo = GdnPickGeometry(s);
  const mx::Shape qk{static_cast<int>(s.B), static_cast<int>(s.Hk),
                     static_cast<int>(s.Dk)};
  const mx::Shape vs{static_cast<int>(s.B), static_cast<int>(s.Hv),
                     static_cast<int>(s.Dv)};
  const mx::Shape gs{static_cast<int>(s.B), static_cast<int>(s.Hv)};
  const mx::Shape ss{static_cast<int>(s.B), static_cast<int>(s.Hv),
                     static_cast<int>(s.Dk), static_cast<int>(s.Dv)};
  // The operands arrive in whatever rank the graph gave them; every peeled
  // step preserved row-major order, so a reshape is the whole conversion.
  mx::array qr = mx::reshape(q, qk);
  mx::array kr = mx::reshape(k, qk);
  mx::array vr = mx::reshape(v, vs);
  mx::array gr = mx::reshape(g, gs);
  mx::array br = mx::reshape(beta, gs);
  mx::array sr = mx::reshape(state, ss);

  static const bool kernel_on = [] {
    const char* v = std::getenv("METALJAX_GDN_KERNEL");
    return v == nullptr || std::string(v) != "0";
  }();
  if (geo.ok && kernel_on) {
    const std::string src = GdnSource(s);
    mx::fast::CustomKernelFunction fn;
    bool have = false;
    {
      std::lock_guard<std::mutex> lock(Mu());
      auto it = Cache().find(src);
      if (it != Cache().end() && it->second.ok) {
        fn = it->second.fn;
        have = true;
        Counters().kernel_runs++;
      }
    }
    if (have) {
      std::vector<mx::array> outs = fn(
          {qr, kr, vr, gr, br, sr}, {vs, ss}, {mx::float32, mx::float32},
          std::tuple<int, int, int>{static_cast<int>(geo.tx),
                                    static_cast<int>(geo.ty),
                                    static_cast<int>(s.B * s.Hv)},
          std::tuple<int, int, int>{static_cast<int>(geo.tgx),
                                    static_cast<int>(geo.ty), 1},
          {}, std::nullopt, false, {});
      return outs;
    }
  }

  // ---- fallback, in the chain's own order.
  {
    std::lock_guard<std::mutex> lock(Mu());
    Counters().fallback_runs++;
  }
  mx::array qn = s.l2q ? L2(qr, s.eps_q, s.narrow_q) : mx::astype(qr, mx::float32);
  mx::array kn = s.l2k ? L2(kr, s.eps_k, s.narrow_k) : mx::astype(kr, mx::float32);
  if (s.scale != 1.0)
    qn = mx::multiply(qn, mx::array(static_cast<float>(s.scale), mx::float32));
  // Repeat the Hk key heads up to Hv value heads: head hv reads hk = hv / R,
  // which is what the absorbed `ops.repeat` did.
  const int rep = static_cast<int>(s.Hv / s.Hk);
  if (rep > 1) {
    qn = mx::reshape(mx::broadcast_to(
                         mx::reshape(qn, {static_cast<int>(s.B),
                                          static_cast<int>(s.Hk), 1,
                                          static_cast<int>(s.Dk)}),
                         {static_cast<int>(s.B), static_cast<int>(s.Hk), rep,
                          static_cast<int>(s.Dk)}),
                     {static_cast<int>(s.B), static_cast<int>(s.Hv),
                      static_cast<int>(s.Dk)});
    kn = mx::reshape(mx::broadcast_to(
                         mx::reshape(kn, {static_cast<int>(s.B),
                                          static_cast<int>(s.Hk), 1,
                                          static_cast<int>(s.Dk)}),
                         {static_cast<int>(s.B), static_cast<int>(s.Hk), rep,
                          static_cast<int>(s.Dk)}),
                     {static_cast<int>(s.B), static_cast<int>(s.Hv),
                      static_cast<int>(s.Dk)});
  }
  mx::array sf = s.round_state ? Narrow(sr, s.narrow_state) : sr;
  mx::array ge = mx::exp(mx::astype(gr, mx::float32));
  mx::array decayed =
      mx::multiply(sf, mx::reshape(ge, {static_cast<int>(s.B),
                                        static_cast<int>(s.Hv), 1, 1}));
  mx::array k4 = mx::reshape(kn, {static_cast<int>(s.B), static_cast<int>(s.Hv),
                                  static_cast<int>(s.Dk), 1});
  mx::array kvmem = mx::sum(mx::multiply(decayed, k4), 2, false);
  mx::array delta = mx::multiply(
      mx::subtract(mx::astype(vr, mx::float32), kvmem),
      mx::reshape(mx::astype(br, mx::float32),
                  {static_cast<int>(s.B), static_cast<int>(s.Hv), 1}));
  mx::array snew = mx::add(
      decayed,
      mx::multiply(k4, mx::reshape(delta, {static_cast<int>(s.B),
                                           static_cast<int>(s.Hv), 1,
                                           static_cast<int>(s.Dv)})));
  mx::array q4 = mx::reshape(qn, {static_cast<int>(s.B), static_cast<int>(s.Hv),
                                  static_cast<int>(s.Dk), 1});
  mx::array out = mx::sum(mx::multiply(snew, q4), 2, false);
  return {out, snew};
}

GdnStats GdnStatsRead() {
  std::lock_guard<std::mutex> lock(Mu());
  return Counters();
}

}  // namespace metaljax
