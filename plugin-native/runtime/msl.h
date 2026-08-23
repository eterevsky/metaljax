// metaljax native engine — generated persistent kernels (M5b).
//
// A counted loop whose body msl_scan can express becomes ONE Metal kernel:
// a thread per lane, the carries in registers, the whole time loop inside
// the shader. The planning — pattern match, mode choice, MSL generation —
// is compile-time work and stays in Python. What crosses is the LAUNCH: the
// generated source, the binding order, the geometry, and the little recipe
// `Plan.run` applies afterwards to turn the kernel's outputs back into the
// loop's carries. Everything below is that recipe, transliterated; the
// layouts are documented at each parse, and metaljax.tape._lower_msl writes
// them in this order.
//
// A plan that fails at run time is DEAD for the rest of the process
// (Metal's shader compiler rejecting a generated source is not a transient
// condition), and the entry falls back to the interpreted loop it carries
// alongside — the C++ half of engine.MetalExecutable.disable_msl.
//
// The class is separate from program.h because a tape carries a plan only
// as an opaque handle (Entry::msl): msl.cc runs it, config.cc binds its
// constructor for metaljax.tape._lower_msl, and nothing else needs to know
// what a launch recipe is made of.

#ifndef METALJAX_MSL_H_
#define METALJAX_MSL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "program.h"

namespace metaljax {

// One node of an accumulator's stacking recipe (`Plan.run`'s `stacked`).
struct AccNode {
  int kind = 0;             // 0 hidden, 1 buffer, 2 red, 3 dot
  int idx = 0;              // hidden: output index; buffer: source id
  int64_t a = 0, b = 0;     // buffer: the read's affine index a*t + b
  mx::Shape shape;          // buffer/red: the per-step shape
  std::vector<int> dims, perm;
  std::vector<int> lb, rb, lc, rc;   // dot: batch/contracting dims
  mx::Shape lshape, rshape;
  std::vector<AccNode> kids;
};

class MslPlan {
 public:
  MslPlan(std::string name, std::string source, std::string header,
          std::vector<std::string> input_names,
          std::vector<std::string> output_names,
          std::vector<std::vector<int>> out_shapes,
          std::vector<int> out_dtypes, std::vector<int64_t> layout);

  // How many arrays the entry hands us after the loop's own inputs.
  size_t num_sources() const { return norms_.size(); }
  bool dead() const { return dead_; }
  void kill() { dead_ = true; }
  bool validated() const { return validated_; }
  void validate() { validated_ = true; }
  const std::string& name() const { return name_; }

  // One launch: the whole loop. `carries` are the loop's initial carries,
  // `srcs` the arrays its sources resolved to, in plan order.
  std::vector<mx::array> run(const std::vector<mx::array>& carries,
                             const std::vector<mx::array>& srcs,
                             bool in_trace,
                             std::vector<MslPlan*>& pending);

 private:
  struct Norm {
    bool on = false;
    mx::Shape shape;
    mx::Strides strides;
    size_t offset = 0;
    std::vector<int> perm;
  };
  struct Pack {
    mx::Dtype dtype = mx::float32;
    std::vector<int> sids;
  };

  // The launch recipe metaljax.tape._lower_msl writes; its layout is
  // documented where it is read (msl.cc).
  void parse(const std::vector<int64_t>& layout);
  static AccNode parse_node(Cursor& c);
  void build();

  // The (L, *per-step) array an accumulator term contributes, and the
  // rank restore a unit-squeezed hidden stack needs (msl.cc).
  mx::array stacked_of(const AccNode& s, const std::vector<mx::array>& outs,
                       size_t ns, const std::vector<mx::array>& srcs) const;
  static mx::array with_lead(const mx::array& x, const mx::Shape& want);

  std::string name_, source_, header_;
  std::vector<std::string> in_names_, out_names_;
  std::vector<mx::Shape> out_shapes_;
  std::vector<mx::Dtype> out_dtypes_;
  int64_t N_ = 0, tg_ = 1, trip_ = 0, start_ = 0;
  int nhidden_ = 0, ncarry_ = 0;
  std::vector<Norm> norms_;
  std::vector<int> unpacked_, state_pos_, stacked_pos_, passthrough_;
  // Sources fed to the kernel as f32 whatever their buffer's dtype (bf16
  // dot weights, converted once per call): the layout's optional tail.
  std::vector<int> conv_f32_;
  std::vector<Pack> packs_;
  std::vector<std::pair<int, int64_t>> counters_;
  std::vector<std::pair<int, std::vector<AccNode>>> acc_;
  mx::fast::CustomKernelFunction kernel_;
  bool built_ = false;
  bool validated_ = false;
  bool dead_ = false;
  bool narrated_ = false;
};

}  // namespace metaljax

#endif  // METALJAX_MSL_H_
