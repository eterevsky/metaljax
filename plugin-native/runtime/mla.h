// metaljax native engine -- maxtext's two-span decode attention, as one
// generated Metal kernel over BOTH cache spans.
//
// metal_mla.cc recognizes the per-span masked softmax partials and their
// flash-style combine and hands the executor q, the two spans' keys, values
// and segment ids, and this Spec.  Until B3 the emit joined the spans for
// MLX's fused sdpa -- concat(K), concat(V), where(seg) per span, concat(mask)
// -- which is ~8 copy kernels per layer around the one attention kernel, all
// of them data movement the attention could have read in place: 26 layers x
// 8 dispatches at the ~2.4 us/dispatch device floor is ~0.5 ms of a 24 ms
// DeepSeek-V2-Lite token (`~/.cache/metaljax-bench/logs/gap-rows/row10/
// report.md` item 7).
//
// The kernel here is MLX's own `sdpa_vector` (kernels/sdpa_vector.h) with
// two key pointers: key i < T0 reads span 0, the rest read span 1, and the
// additive mask is computed from the segment id at the key instead of read
// from an array.  Everything numeric is kept verbatim -- one simdgroup per
// key at a time with the lanes splitting the head dim, `fast::exp`, the
// running max/sum, the cross-simdgroup combine -- so the fused answer is the
// concat path's answer to the bit when the Metal compiler contracts both
// sources alike (execute_test.py `_p37_mla_kernel` pins it; the disclosure,
// if an OS update ever moves it, is the fused-attention ULP class the
// recognizer's header already carries).
//
// WHY THE PROBE IS EAGER: gdn.h's reason, verbatim -- MLX generates the
// Metal library at EVAL, a build error on an async worker aborts the
// process, and this entry is traced into an enclosing `mx::compile` graph
// where nothing is evaluated.  So the build is forced at LOWERING time, on
// dummy operands of the real geometry, and a source Metal refuses leaves the
// emit on the concat path (the correct slow program).
//
// Licensed under the Apache License, Version 2.0.

#ifndef METALJAX_MLA_H_
#define METALJAX_MLA_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace metaljax {

namespace mx = mlx::core;

// The geometry and the numerics of one two-span decode attention.  Two
// entries with equal Specs generate the same kernel source and share one
// build.
struct MlaSpec {
  int64_t B = 0, H = 0, Hkv = 0, D = 0, Dv = 0;
  int64_t T0 = 0, T1 = 0;      // the two spans' key counts
  mx::Dtype dt = mx::float32;  // q/k/v/out compute dtype
  int64_t seg_val = 1;         // keep key i when seg[i] == seg_val
  double mask_true = 0.0;      // the additive mask's two values, as the
  double mask_false = 0.0;     // matched `_where` chain selects them

  bool operator==(const MlaSpec& o) const;
  // The cache key: geometry, dtype and the mask bits -- no evaluation.
  std::string key() const;
};

// Whether the kernel can take this geometry at all: the lanes split the
// head dims 32 ways, so D and Dv are multiples of 32 (up to 512), H is a
// multiple of Hkv, and the dtype is one the kernel names.
bool MlaKernelEligible(const MlaSpec& s);

// Generate the kernel source for a Spec (exposed for MJDBG_MLA_SOURCE).
// Evaluates two scalars through MLX to round the mask values the way the
// emit's `mx::array(v, dt)` does -- call it outside any trace.
std::string MlaSource(const MlaSpec& s);

// Build and probe the kernel for `s`, synchronously, and remember the
// verdict.  Safe to call repeatedly and from the lowering: the first call
// for a given Spec does the work, the rest read the cache.  Returns true
// when the tape may use the kernel.  METALJAX_MLA_KERNEL=0 answers false
// without building, which keeps the recognizer (and the entry count) but
// runs the attention on the concat path -- the arm that prices the kernel
// alone.
bool MlaProve(const MlaSpec& s);

// Run one attention.  The kernel when it is proven and the operands are
// what it was proven on; the concat path otherwise.  q [B,1,H,D]; per span
// k [B,T,Hkv,D], v [B,T,Hkv,Dv], seg [B,T] integral.  Returns [B,1,H,Dv] in
// `s.dt`.
mx::array MlaRun(const MlaSpec& s, const mx::array& q, const mx::array& k0,
                 const mx::array& v0, const mx::array& s0, const mx::array& k1,
                 const mx::array& v1, const mx::array& s1);

// Counters, for the tests: how many attentions ran on the kernel and how
// many on the concat path.
struct MlaStats {
  int64_t builds = 0;
  int64_t build_failures = 0;
  int64_t kernel_runs = 0;
  int64_t fallback_runs = 0;
};
MlaStats MlaStatsRead();

}  // namespace metaljax

#endif  // METALJAX_MLA_H_
