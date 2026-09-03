// metaljax native engine -- the gated-delta-net decode step, as one
// generated Metal kernel.
//
// metal_gdn.cc recognizes the Qwen3.5 linear-attention layer's decode step
// and hands the executor six operands and this Spec; everything about HOW it
// runs lives here.  The kernel is generated from the Spec, built and probed
// ONCE per distinct source (process-global, keyed by the source itself), and
// a source Metal refuses to build leaves the step running on the MLX
// fallback below -- the same arithmetic, op by op, which is the correct slow
// program the recognizer file rule asks for.
//
// WHY THE PROBE IS EAGER.  MLX generates the Metal library at EVAL, and a
// build error raised on an async worker aborts the process (CLAUDE.md item
// 7).  Worse, this entry is meant to be TRACED into an enclosing
// `mx::compile` graph, where there is nothing to evaluate at all.  So the
// build is forced at LOWERING time -- which for a fused program is the first
// execute, on concrete buffers and outside any trace -- against dummy
// operands of the real geometry.  By the time any tape can reach the kernel
// it is already known to build.
//
// Licensed under the Apache License, Version 2.0.

#ifndef METALJAX_GDN_H_
#define METALJAX_GDN_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace metaljax {

namespace mx = mlx::core;

// The geometry and the numerics of one decode step.  Two matches with equal
// Specs generate the same kernel source and share one build.
struct GdnSpec {
  int64_t B = 0, Hv = 0, Hk = 0, Dk = 0, Dv = 0;
  // Operand dtypes, as the tape's arrays carry them.
  mx::Dtype q = mx::float32, k = mx::float32, v = mx::float32;
  mx::Dtype g = mx::float32, beta = mx::float32;

  // The `_l2norm` the match absorbed, per operand, and the dtype its rsqrt
  // and product were rounded to (`float32` = no rounding).  The kernel
  // replays the rounding so the fused answer sits on the literal chain's own
  // values.
  bool l2q = false, l2k = false;
  double eps_q = 0.0, eps_k = 0.0;
  mx::Dtype narrow_q = mx::float32, narrow_k = mx::float32;
  // The state's dead `convert(f32 -> T) -> convert(T -> f32)` round trip.
  bool round_state = false;
  mx::Dtype narrow_state = mx::float32;
  // The pre-scale folded onto q (1 / sqrt(Dk)), applied after the norm.
  double scale = 1.0;

  bool operator==(const GdnSpec& o) const;
};

// The launch geometry the Spec resolves to.  The grid is (Dv, ty, B*Hv)
// THREADS: `dv` is the fast axis so consecutive threads read consecutive
// state addresses, and `ty` chunks split Dk into `ch` registers a thread.
//
// `tgx` is the threadgroup's width along dv, and it is NOT Dv.  Every dv
// column is independent -- only the dk reduction is shared, and that lives
// along ty -- so making the threadgroup the full Dv wide put one
// threadgroup on each (b, hv) and left the GPU with 32 of them on row 8 and
// 48 on row 21: far too few to hide the memory latency of a 32-element
// register loop, and MEASURED at ~10 % of bandwidth (row 21 lost 7 ms to it,
// `~/.cache/metaljax-bench/logs/gdn-fuse/findings.txt`).  Tiling dv at 32
// gives Dv/tgx times as many threadgroups for the same total threads.
struct GdnGeometry {
  bool ok = false;
  int64_t tx = 0, ty = 0, ch = 0, tgx = 0;
};
GdnGeometry GdnPickGeometry(const GdnSpec& s);

// Generate the kernel source for a Spec (exposed for the tests and for
// MJDBG_GDN_SOURCE).
std::string GdnSource(const GdnSpec& s);

// Build and probe the kernel for `s`, synchronously, and remember the
// verdict.  Safe to call repeatedly and from the lowering: the first call
// for a given source does the work, the rest read the cache.  Returns true
// when the tape may use the kernel.  METALJAX_GDN_KERNEL=0 answers false
// without building, which keeps the recognizer (and so the entry count) but
// runs the step on the MLX fallback -- the arm that splits the win between
// "fewer tape entries" and "the kernel itself".
bool GdnProve(const GdnSpec& s);

// Run one decode step.  Uses the generated kernel when it is proven, and the
// MLX op-by-op fallback otherwise.  Returns {out [B,Hv,Dv] f32,
// new_state [B,Hv,Dk,Dv] f32}.
std::vector<mx::array> GdnRun(const GdnSpec& s, const mx::array& q,
                              const mx::array& k, const mx::array& v,
                              const mx::array& g, const mx::array& beta,
                              const mx::array& state);

// Counters, for the tests: how many steps ran on the kernel and how many on
// the fallback.
struct GdnStats {
  int64_t builds = 0;
  int64_t build_failures = 0;
  int64_t kernel_runs = 0;
  int64_t fallback_runs = 0;
};
GdnStats GdnStatsRead();

}  // namespace metaljax

#endif  // METALJAX_GDN_H_
