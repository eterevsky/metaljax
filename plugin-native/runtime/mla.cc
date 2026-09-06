// metaljax native engine -- the two-span decode attention kernel (mla.h).
//
// The generated source is MLX's `sdpa_vector` (mlx/backend/metal/kernels/
// sdpa_vector.h, v0.32.0) rewritten for two key spans and an in-kernel
// segment mask.  The float-mask arm of that kernel is what the concat path
// reaches on these operands, and it is replayed here term for term:
//
//   q[j]  = scale * queries[j]                      (scale == 1: the graph
//                                                    pre-scaled q)
//   use   = fmask >= Limits<T>::finite_min          (decided per mask value)
//   score = simd_sum(sum_j q[j] * k[j]) + fmask
//   new_max = max(max_score, score)
//   factor = fast::exp(max_score - new_max); e = fast::exp(score - new_max)
//   sum = sum * factor + e;  o[j] = o[j] * factor + e * v[j]
//   ...then the cross-simdgroup combine, verbatim.
//
// Key i of the joined sequence is simdgroup i % 32's, in order, exactly as
// the concat path hands MLX's kernel key i of the concatenated array; only
// WHERE the key is read from changes (span 0 for i < T0, span 1 after), and
// the mask value the concat path read from a bf16 array is here the same
// value rounded through the same dtype on the host and written into the
// source as a hex literal.
//
// Licensed under the Apache License, Version 2.0.

#include "mla.h"

#include "program.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
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

// Keyed by the Spec's key: two entries of one geometry share one Metal
// library, and a source is never built twice.
std::map<std::string, Built>& Cache() {
  static std::map<std::string, Built>* c = new std::map<std::string, Built>();
  return *c;
}

MlaStats& Counters() {
  static MlaStats s;
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
  std::fprintf(stderr, "[metaljax-native] mla: %s\n", line.c_str());
  std::fflush(stderr);
}

bool KernelEnabled() {
  static const bool on = [] {
    const char* v = std::getenv("METALJAX_MLA_KERNEL");
    return v == nullptr || std::string(v) != "0";
  }();
  return on;
}

// MLX's kernel preamble (utils.h -> bf16.h) supplies `bfloat16_t`; `half` is
// MSL's own.  Anything else is ineligible.
const char* MslType(mx::Dtype d) {
  if (d == mx::float32) return "float";
  if (d == mx::float16) return "half";
  if (d == mx::bfloat16) return "bfloat16_t";
  return nullptr;
}

// The largest finite value of the dtype, negated: `Limits<T>::finite_min`
// in MLX's kernel utils, which decides whether a masked key is visited at
// all (a float mask at or above it is ADDED; below it the key is skipped).
float FiniteMin(mx::Dtype d) {
  if (d == mx::float16) return -65504.0f;
  if (d == mx::bfloat16) return -3.3895313892515355e+38f;
  return -3.4028234663852886e+38f;
}

// A float as a hex literal: exact, whatever the value.
std::string Hex(float v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%a", static_cast<double>(v));
  return std::string(buf) + "f";
}

std::atomic<int64_t>& Seq() {
  static std::atomic<int64_t> seq{0};
  return seq;
}

uint64_t Bits(double v) {
  uint64_t b = 0;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}

// The mask value as the concat path holds it: `mx::array(v, dt)`, read back
// as f32.  Evaluates (a scalar); the callers are outside any trace.
float RoundThrough(double v, mx::Dtype dt) {
  mx::array a = mx::astype(mx::array(v, dt), mx::float32);
  return a.item<float>();
}

}  // namespace

bool MlaSpec::operator==(const MlaSpec& o) const {
  return B == o.B && H == o.H && Hkv == o.Hkv && D == o.D && Dv == o.Dv &&
         T0 == o.T0 && T1 == o.T1 && dt == o.dt && seg_val == o.seg_val &&
         Bits(mask_true) == Bits(o.mask_true) &&
         Bits(mask_false) == Bits(o.mask_false);
}

std::string MlaSpec::key() const {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "B%lldH%lldHkv%lldD%lldDv%lldT%lld+%lld_%s_seg%lld_%llx_%llx",
                static_cast<long long>(B), static_cast<long long>(H),
                static_cast<long long>(Hkv), static_cast<long long>(D),
                static_cast<long long>(Dv), static_cast<long long>(T0),
                static_cast<long long>(T1),
                MslType(dt) == nullptr ? "?" : MslType(dt),
                static_cast<long long>(seg_val),
                static_cast<unsigned long long>(Bits(mask_true)),
                static_cast<unsigned long long>(Bits(mask_false)));
  return buf;
}

bool MlaKernelEligible(const MlaSpec& s) {
  if (MslType(s.dt) == nullptr) return false;
  if (s.B < 1 || s.H < 1 || s.Hkv < 1 || s.H % s.Hkv != 0) return false;
  // Exactly the geometries MLX's own `sdpa_vector` takes
  // (scaled_dot_product_attention.cpp `use_fallback`): the head dims of its
  // instantiations, and a query-group count its threadgroup covers.  The
  // lane split would take any multiple of 32, but outside this set the
  // concat path runs MLX's UNFUSED fallback (matmul + softmax), and the
  // kernel's bit-for-bit contract is against the fused kernel it copies.
  const bool head_dims =
      (s.D == s.Dv && (s.D == 64 || s.D == 96 || s.D == 128 || s.D == 256)) ||
      (s.D == 192 && s.Dv == 128);
  if (!head_dims) return false;
  if (s.H / s.Hkv > 32) return false;
  if (s.T0 < 1 || s.T1 < 1) return false;
  // int offsets: the largest span index the kernel forms.
  const int64_t span_elems =
      s.B * std::max(s.T0, s.T1) * s.Hkv * std::max(s.D, s.Dv);
  if (span_elems > (1LL << 30)) return false;
  if (s.B * s.H * 1024 > (1LL << 30)) return false;
  return true;
}

std::string MlaSource(const MlaSpec& s) {
  const char* T = MslType(s.dt);
  if (T == nullptr) return std::string();
  const float mt = RoundThrough(s.mask_true, s.dt);
  const float mf = RoundThrough(s.mask_false, s.dt);
  const float fmin = FiniteMin(s.dt);
  const bool use_true = mt >= fmin;
  const bool use_false = mf >= fmin;
  std::string src;
  auto add = [&](const std::string& line) {
    src += line;
    src += '\n';
  };
  auto I = [](int64_t v) { return std::to_string(v); };
  add("  // metaljax two-span decode attention: MLX sdpa_vector over spans");
  add("  // 0 (T0 keys) and 1 (T1 keys), the additive segment mask computed");
  add("  // at the key.  Key i is simdgroup (i % 32)'s, as in the concat path.");
  add("  constexpr int BN = 32;");
  add("  constexpr int BD = 32;");
  add("  constexpr int D = " + I(s.D) + ";");
  add("  constexpr int V = " + I(s.Dv) + ";");
  add("  constexpr int qk_per_thread = D / BD;");
  add("  constexpr int v_per_thread = V / BD;");
  add("  constexpr int T0 = " + I(s.T0) + ";");
  add("  constexpr int T1 = " + I(s.T1) + ";");
  add("  constexpr int N = T0 + T1;");
  add("  constexpr int H = " + I(s.H) + ";");
  add("  constexpr int G = " + I(s.H / s.Hkv) + ";");
  add("  constexpr int SEG_VAL = " + I(s.seg_val) + ";");
  add("  constexpr float SCALE = 1.0f;");
  add("  constexpr float MASK_TRUE = " + Hex(mt) + ";");
  add("  constexpr float MASK_FALSE = " + Hex(mf) + ";");
  add(std::string("  constexpr bool USE_TRUE = ") +
      (use_true ? "true" : "false") + ";");
  add(std::string("  constexpr bool USE_FALSE = ") +
      (use_false ? "true" : "false") + ";");
  add("  typedef float U;");
  add("  thread U q[qk_per_thread];");
  add("  thread U k[qk_per_thread];");
  add("  thread U o[v_per_thread];");
  add("  threadgroup U outputs[BN * BD];");
  add("  threadgroup U max_scores[BN];");
  add("  threadgroup U sum_exp_scores[BN];");
  add("  const int qbh = int(threadgroup_position_in_grid.x);");
  add("  const int b = qbh / H;");
  add("  const int h = qbh - b * H;");
  add("  const int hk = h / G;");
  add("  const int simd_gid = int(simdgroup_index_in_threadgroup);");
  add("  const int simd_lid = int(thread_index_in_simdgroup);");
  // q [B,1,H,D]: strides from the array (a transposed or sliced view reads
  // in place).
  add("  const int q_off = b * int(qin_strides[0]) + h * int(qin_strides[2])"
      " + (simd_lid * qk_per_thread) * int(qin_strides[3]);");
  add("  const int qs = int(qin_strides[3]);");
  add("  for (int j = 0; j < qk_per_thread; j++) {");
  add("    q[j] = SCALE * static_cast<U>(qin[q_off + j * qs]);");
  add("  }");
  add("  for (int j = 0; j < v_per_thread; j++) {");
  add("    o[j] = 0;");
  add("  }");
  add("  U max_score = Limits<U>::finite_min;");
  add("  U sum_exp_score = 0;");
  add("  for (int i = simd_gid; i < N; i += BN) {");
  add("    int sv;");
  add("    if (i < T0) {");
  add("      sv = int(s0[b * int(s0_strides[0]) + i * int(s0_strides[1])]);");
  add("    } else {");
  add("      sv = int(s1[b * int(s1_strides[0]) + (i - T0) * "
      "int(s1_strides[1])]);");
  add("    }");
  add("    const bool hit = (sv == SEG_VAL);");
  add("    const U mval = hit ? MASK_TRUE : MASK_FALSE;");
  add("    const bool use_key = hit ? USE_TRUE : USE_FALSE;");
  add("    if (use_key) {");
  add("      if (i < T0) {");
  add("        const int kb = b * int(k0_strides[0]) + i * int(k0_strides[1])"
      " + hk * int(k0_strides[2]) + (simd_lid * qk_per_thread) * "
      "int(k0_strides[3]);");
  add("        const int ks = int(k0_strides[3]);");
  add("        for (int j = 0; j < qk_per_thread; j++) {");
  add("          k[j] = static_cast<U>(k0[kb + j * ks]);");
  add("        }");
  add("      } else {");
  add("        const int kb = b * int(k1_strides[0]) + (i - T0) * "
      "int(k1_strides[1]) + hk * int(k1_strides[2]) + "
      "(simd_lid * qk_per_thread) * int(k1_strides[3]);");
  add("        const int ks = int(k1_strides[3]);");
  add("        for (int j = 0; j < qk_per_thread; j++) {");
  add("          k[j] = static_cast<U>(k1[kb + j * ks]);");
  add("        }");
  add("      }");
  add("      U score = 0;");
  add("      for (int j = 0; j < qk_per_thread; j++) {");
  add("        score += q[j] * k[j];");
  add("      }");
  add("      score = simd_sum(score);");
  add("      score += mval;");
  add("      U new_max = max(max_score, score);");
  add("      U factor = fast::exp(max_score - new_max);");
  add("      U exp_score = fast::exp(score - new_max);");
  add("      max_score = new_max;");
  add("      sum_exp_score = sum_exp_score * factor + exp_score;");
  add("      if (i < T0) {");
  add("        const int vb = b * int(v0_strides[0]) + i * int(v0_strides[1])"
      " + hk * int(v0_strides[2]) + (simd_lid * v_per_thread) * "
      "int(v0_strides[3]);");
  add("        const int vs = int(v0_strides[3]);");
  add("        for (int j = 0; j < v_per_thread; j++) {");
  add("          o[j] = o[j] * factor + exp_score * "
      "static_cast<U>(v0[vb + j * vs]);");
  add("        }");
  add("      } else {");
  add("        const int vb = b * int(v1_strides[0]) + (i - T0) * "
      "int(v1_strides[1]) + hk * int(v1_strides[2]) + "
      "(simd_lid * v_per_thread) * int(v1_strides[3]);");
  add("        const int vs = int(v1_strides[3]);");
  add("        for (int j = 0; j < v_per_thread; j++) {");
  add("          o[j] = o[j] * factor + exp_score * "
      "static_cast<U>(v1[vb + j * vs]);");
  add("        }");
  add("      }");
  add("    }");
  add("  }");
  // The cross-simdgroup combine, verbatim.
  add("  if (simd_lid == 0) {");
  add("    max_scores[simd_gid] = max_score;");
  add("    sum_exp_scores[simd_gid] = sum_exp_score;");
  add("  }");
  add("  threadgroup_barrier(mem_flags::mem_threadgroup);");
  add("  max_score = max_scores[simd_lid];");
  add("  U new_max = simd_max(max_score);");
  add("  U factor = fast::exp(max_score - new_max);");
  add("  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);");
  add("  for (int i = 0; i < v_per_thread; i++) {");
  add("    outputs[simd_lid * BD + simd_gid] = o[i];");
  add("    threadgroup_barrier(mem_flags::mem_threadgroup);");
  add("    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);");
  add("    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);");
  add("    threadgroup_barrier(mem_flags::mem_threadgroup);");
  add("  }");
  add("  if (simd_lid == 0) {");
  add("    for (int i = 0; i < v_per_thread; i++) {");
  add(std::string("      out[qbh * V + simd_gid * v_per_thread + i] = "
                  "static_cast<") +
      T + ">(o[i]);");
  add("    }");
  add("  }");
  return src;
}

namespace {

const std::vector<std::string>& InputNames() {
  static const std::vector<std::string> n{"qin", "k0", "v0", "s0",
                                          "k1",  "v1", "s1"};
  return n;
}

// The header goes after MLX's own preamble (utils.h): the simdgroup
// intrinsics and the unqualified names the copied kernel uses.
const char* kHeader = "#include <metal_simdgroup>\nusing namespace metal;\n";

std::vector<mx::array> Launch(const Built& built, const MlaSpec& s,
                              const std::vector<mx::array>& ins) {
  const auto B = static_cast<int>(s.B), H = static_cast<int>(s.H);
  std::vector<mx::Shape> shapes{{B, 1, H, static_cast<int>(s.Dv)}};
  std::vector<mx::Dtype> dts{s.dt};
  return built.fn(ins, shapes, dts,
                  std::tuple<int, int, int>{1024 * B * H, 1, 1},
                  std::tuple<int, int, int>{1024, 1, 1}, {}, std::nullopt,
                  false, {});
}

}  // namespace

bool MlaProve(const MlaSpec& s) {
  if (!KernelEnabled()) {
    static bool said = false;
    if (!said) {
      said = true;
      Debug("kernel disabled (METALJAX_MLA_KERNEL=0); the concat path runs");
    }
    return false;
  }
  if (!MlaKernelEligible(s)) {
    Debug("kernel ineligible for " + s.key() + "; the concat path runs");
    return false;
  }
  const std::string key = s.key();
  std::lock_guard<std::mutex> lock(Mu());
  auto it = Cache().find(key);
  if (it != Cache().end()) return it->second.ok;

  Built built;
  const std::string name = "mjn_mla2_" + std::to_string(++Seq());
  try {
    const std::string src = MlaSource(s);
    if (src.empty()) return false;
    if (const char* dump = std::getenv("MJDBG_MLA_SOURCE"))
      if (std::string(dump) == "1")
        std::fprintf(stderr, "[metaljax-native] mla: %s\n%s\n", name.c_str(),
                     src.c_str());
    built.fn = mx::fast::metal_kernel(name, InputNames(), {"out"}, src,
                                      kHeader,
                                      /*ensure_row_contiguous=*/false);
    // Prove it on dummy operands of the real geometry, synchronously: the
    // Metal library is generated at eval, and this is the last moment there
    // is a thread that may safely take the exception.
    const auto B = static_cast<int>(s.B), H = static_cast<int>(s.H);
    const auto Hkv = static_cast<int>(s.Hkv), D = static_cast<int>(s.D);
    const auto Dv = static_cast<int>(s.Dv);
    const auto T0 = static_cast<int>(s.T0), T1 = static_cast<int>(s.T1);
    std::vector<mx::array> probe{
        mx::zeros({B, 1, H, D}, s.dt),      mx::zeros({B, T0, Hkv, D}, s.dt),
        mx::zeros({B, T0, Hkv, Dv}, s.dt),  mx::zeros({B, T0}, mx::int32),
        mx::zeros({B, T1, Hkv, D}, s.dt),   mx::zeros({B, T1, Hkv, Dv}, s.dt),
        mx::zeros({B, T1}, mx::int32)};
    std::vector<mx::array> outs = Launch(built, s, probe);
    mx::eval(outs);
    built.ok = true;
    Counters().builds++;
    Debug("kernel built " + name + " for " + key + " (" +
          std::to_string(s.B * s.H) + " threadgroups x 1024)");
  } catch (const std::exception& e) {
    // A governor refusal says the MACHINE is out of memory, not that this
    // source is unbuildable (gdn.cc draws the same line).
    if (is_oom(e)) {
      Debug(std::string("kernel probe hit the memory governor (") + e.what() +
            "); leaving the source unproven");
      return false;
    }
    built.ok = false;
    Counters().build_failures++;
    Debug(std::string("kernel did not build (") + e.what() +
          "); the concat path runs");
  }
  const bool ok = built.ok;
  Cache().emplace(key, std::move(built));
  return ok;
}

namespace {

// The concat path: what the emit ran before B3, and what a kernel that did
// not build (or a geometry the kernel does not take) still runs.
mx::array Fallback(const MlaSpec& s, const mx::array& q,
                   const std::vector<mx::array>& ks,
                   const std::vector<mx::array>& vs,
                   const std::vector<mx::array>& segs) {
  const auto B = static_cast<mx::ShapeElem>(s.B);
  mx::array qt = mx::transpose(q, {0, 2, 1, 3});  // [B, H, 1, D]
  std::vector<mx::array> masks;
  for (const mx::array& seg : segs)
    masks.push_back(mx::where(mx::equal(seg, weak_int(s.seg_val, seg)),
                              mx::array(s.mask_true, s.dt),
                              mx::array(s.mask_false, s.dt)));
  // [B, T, Hkv, D] concat on T, then to [B, Hkv, T, D].
  mx::array k = mx::transpose(mx::concatenate(ks, 1), {0, 2, 1, 3});
  mx::array v = mx::transpose(mx::concatenate(vs, 1), {0, 2, 1, 3});
  mx::array mask = mx::reshape(
      mx::concatenate(masks, 1),
      mx::Shape{B, 1, 1, static_cast<mx::ShapeElem>(k.shape(2))});
  mx::array out = mx::fast::scaled_dot_product_attention(
      qt, k, v, /*scale=*/1.0f, /*mask_mode=*/"", mask);
  return mx::transpose(out, {0, 2, 1, 3});  // [B, H, 1, Dv] -> [B, 1, H, Dv]
}

bool ShapeIs(const mx::array& a, std::initializer_list<int64_t> dims) {
  if (static_cast<size_t>(a.ndim()) != dims.size()) return false;
  size_t i = 0;
  for (int64_t d : dims)
    if (a.shape(static_cast<int>(i++)) != d) return false;
  return true;
}

}  // namespace

mx::array MlaRun(const MlaSpec& s, const mx::array& q, const mx::array& k0,
                 const mx::array& v0, const mx::array& s0, const mx::array& k1,
                 const mx::array& v1, const mx::array& s1) {
  bool kernel = false;
  {
    std::lock_guard<std::mutex> lock(Mu());
    auto it = Cache().find(s.key());
    kernel = it != Cache().end() && it->second.ok;
  }
  // The kernel was proven on exactly this geometry; anything else -- a
  // segment dtype it was not built for, a shape the tape disagrees on --
  // takes the concat path, which handles all of them.
  if (kernel) {
    kernel = ShapeIs(q, {s.B, 1, s.H, s.D}) &&
             ShapeIs(k0, {s.B, s.T0, s.Hkv, s.D}) &&
             ShapeIs(v0, {s.B, s.T0, s.Hkv, s.Dv}) &&
             ShapeIs(s0, {s.B, s.T0}) &&
             ShapeIs(k1, {s.B, s.T1, s.Hkv, s.D}) &&
             ShapeIs(v1, {s.B, s.T1, s.Hkv, s.Dv}) &&
             ShapeIs(s1, {s.B, s.T1}) && q.dtype() == s.dt &&
             k0.dtype() == s.dt && v0.dtype() == s.dt &&
             k1.dtype() == s.dt && v1.dtype() == s.dt &&
             s0.dtype() == mx::int32 && s1.dtype() == mx::int32;
  }
  if (kernel) {
    Built* built = nullptr;
    {
      std::lock_guard<std::mutex> lock(Mu());
      built = &Cache().at(s.key());
    }
    Counters().kernel_runs++;
    return Launch(*built, s, {q, k0, v0, s0, k1, v1, s1})[0];
  }
  Counters().fallback_runs++;
  return Fallback(s, q, {k0, k1}, {v0, v1}, {s0, s1});
}

MlaStats MlaStatsRead() { return Counters(); }

}  // namespace metaljax
