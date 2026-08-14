// metaljax native engine — msl_scan launches (src/metaljax/msl_scan.py).
//
// The planning -- pattern match, mode choice, MSL generation -- is compile
// time work and stays in Python; what lives here is the LAUNCH and the
// recipe that turns a kernel's outputs back into the loop's carries
// (MslPlan), the tape entry that dispatches one (Program::run_msl, which
// falls back to the interpreted loop the entry still carries), and the
// synchronous settling a traced kernel needs before its Metal library is
// known to build (Program::settle_msl).

#include "msl.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace metaljax {

// The pending list program.h declares.
thread_local std::vector<MslPlan*> t_msl_pending;

namespace {

int64_t shape_prod(const mx::Shape& s, const std::vector<int>& idx) {
  int64_t p = 1;
  for (int i : idx) p *= s[static_cast<size_t>(i)];
  return p;
}

}  // namespace

MslPlan::MslPlan(std::string name, std::string source, std::string header,
                 std::vector<std::string> input_names,
                 std::vector<std::string> output_names,
                 std::vector<std::vector<int>> out_shapes,
                 std::vector<int> out_dtypes, std::vector<int64_t> layout)
    : name_(std::move(name)), source_(std::move(source)),
      header_(std::move(header)), in_names_(std::move(input_names)),
      out_names_(std::move(output_names)) {
  for (const auto& s : out_shapes)
    out_shapes_.push_back(mx::Shape(s.begin(), s.end()));
  for (int c : out_dtypes) out_dtypes_.push_back(dtype_of(c));
  if (out_shapes_.size() != out_dtypes_.size() ||
      out_shapes_.size() != out_names_.size())
    throw std::invalid_argument("msl: output spec mismatch");
  parse(layout);
}

std::vector<mx::array> MslPlan::run(const std::vector<mx::array>& carries,
                                    const std::vector<mx::array>& srcs,
                                    bool in_trace,
                                    std::vector<MslPlan*>& pending) {
  if (carries.size() != static_cast<size_t>(ncarry_))
    throw std::runtime_error("msl: wrong carry count");
  if (srcs.size() != norms_.size())
    throw std::runtime_error("msl: wrong source count");
  build();

  // `Plan.run`'s `bufs`: a weight whose layout was canonicalized at plan
  // time is materialized here, per call, exactly as it is there (the
  // backward pass reads W^T, which is uncoalesced without this).
  std::vector<mx::array> bufs;
  bufs.reserve(srcs.size());
  for (size_t i = 0; i < srcs.size(); i++) {
    const Norm& n = norms_[i];
    if (!n.on) {
      bufs.push_back(srcs[i]);
      continue;
    }
    mx::array v = mx::as_strided(srcs[i], n.shape, n.strides, n.offset);
    bufs.push_back(mx::contiguous(mx::transpose(v, n.perm)));
  }

  // ...and its `feed`: the unpacked sources, then one pooled buffer per
  // dtype (the 0.4.3 input packing, whose element offsets are already
  // baked into the generated source), then the state initializers.
  std::vector<mx::array> feed;
  feed.reserve(unpacked_.size() + packs_.size() + state_pos_.size());
  for (int sid : unpacked_) feed.push_back(bufs[static_cast<size_t>(sid)]);
  for (const Pack& p : packs_) {
    std::vector<mx::array> parts;
    parts.reserve(p.sids.size());
    for (int sid : p.sids)
      parts.push_back(mx::astype(
          mx::reshape(bufs[static_cast<size_t>(sid)], mx::Shape{-1}),
          p.dtype));
    feed.push_back(mx::concatenate(parts, 0));
  }
  for (int pos : state_pos_)
    feed.push_back(carries[static_cast<size_t>(pos)]);
  if (feed.size() != in_names_.size())
    throw std::runtime_error("msl: input count does not match the kernel");

  // Once per plan (the first launch), like msl_scan's own narration: the
  // binding order is what a mis-encoded recipe gets wrong, and it does
  // not change from call to call.
  if (g_cfg.debug && !narrated_) {
    narrated_ = true;
    std::string msg = "[metaljax] msl " + name_ +
                      ": grid=" + std::to_string(N_) +
                      " tg=" + std::to_string(tg_) + " in";
    for (size_t i = 0; i < feed.size(); i++) {
      msg += " " + in_names_[i] + "(";
      for (auto d : feed[i].shape()) msg += std::to_string(d) + ",";
      msg += std::to_string(feed[i].itemsize()) + "B)";
    }
    debug_line(msg);
  }
  std::vector<mx::array> outs = kernel_(
      feed, out_shapes_, out_dtypes_,
      std::tuple<int, int, int>{static_cast<int>(N_), 1, 1},
      std::tuple<int, int, int>{static_cast<int>(tg_), 1, 1}, {},
      std::nullopt, false, {});
  g_stats.msl_launches++;

  if (!validated_) {
    if (!in_trace) {
      // The first call proves the kernel: MLX generates the Metal library
      // at EVAL, and a build error raised on an async worker aborts the
      // process. Synchronous here, so the caller's catch can retire the
      // plan to the interpreted loop instead.
      mx::eval(outs);
      validated_ = true;
    } else {
      // Traced into an enclosing mx::compile graph: there is nothing to
      // evaluate here (these are tracers). The plan is unproven until the
      // whole call is settled, which Program::run does synchronously
      // while this list is non-empty -- msl_scan's own contract, and
      // engine.execute's `_msl_pending`.
      bool known = false;
      for (MslPlan* p : pending) known = known || p == this;
      if (!known) pending.push_back(this);
    }
  }

  if (outs.size() != out_shapes_.size())
    throw std::runtime_error("msl: kernel returned the wrong output count");
  std::vector<mx::array> vals(carries);
  const size_t ns = stacked_pos_.size();
  for (size_t q = 0; q < ns; q++)
    vals[static_cast<size_t>(stacked_pos_[q])] = outs[q];
  for (size_t j = 0; j < state_pos_.size(); j++)
    vals[static_cast<size_t>(state_pos_[j])] =
        outs[ns + static_cast<size_t>(nhidden_) + j];
  for (const auto& c : counters_) {
    const mx::array& x = carries[static_cast<size_t>(c.first)];
    vals[static_cast<size_t>(c.first)] =
        mx::add(x, weak_int(c.second * trip_, x));
  }
  for (const auto& a : acc_) {
    const mx::array& x = carries[static_cast<size_t>(a.first)];
    std::optional<mx::array> total;
    for (const AccNode& spec : a.second) {
      mx::array t = mx::sum(stacked_of(spec, outs, ns, srcs),
                            std::vector<int>{0});
      t = mx::reshape(t, x.shape());
      total = total ? mx::add(*total, t) : t;
    }
    if (!total) throw std::runtime_error("msl: accumulator with no terms");
    vals[static_cast<size_t>(a.first)] = mx::add(x, *total);
  }
  return vals;
}

// Layout, in the order metaljax.tape._lower_msl writes it:
//   N, threadgroup, trip, start, nsources, nhidden, ncarry
//   per source: on?, [shape], [strides], offset, [perm]   (weight norm)
//   [unpacked sids]
//   npacks, then per pack: dtype, [sids]
//   [state carry positions], [stacked carry positions], [pass-throughs]
//   ncounters, then per counter: position, per-iteration delta
//   naccumulators, then per one: position, nterms, term trees
void MslPlan::parse(const std::vector<int64_t>& layout) {
  Cursor c(layout);
  N_ = c.next();
  tg_ = c.next();
  trip_ = c.next();
  start_ = c.next();
  const int64_t nsrc = c.next();
  nhidden_ = static_cast<int>(c.next());
  ncarry_ = static_cast<int>(c.next());
  for (int64_t i = 0; i < nsrc; i++) {
    Norm n;
    n.on = c.flag();
    if (n.on) {
      n.shape = c.shp();
      std::vector<int64_t> st = c.vec64();
      n.strides = mx::Strides(st.begin(), st.end());
      n.offset = static_cast<size_t>(c.next());
      n.perm = c.vec();
    }
    norms_.push_back(std::move(n));
  }
  unpacked_ = c.vec();
  int64_t npacks = c.next();
  for (int64_t i = 0; i < npacks; i++) {
    Pack p;
    p.dtype = dtype_of(c.next());
    p.sids = c.vec();
    packs_.push_back(std::move(p));
  }
  state_pos_ = c.vec();
  stacked_pos_ = c.vec();
  passthrough_ = c.vec();
  int64_t ncnt = c.next();
  for (int64_t i = 0; i < ncnt; i++) {
    int pos = static_cast<int>(c.next());
    counters_.emplace_back(pos, c.next());
  }
  int64_t nacc = c.next();
  for (int64_t i = 0; i < nacc; i++) {
    int pos = static_cast<int>(c.next());
    int64_t nterms = c.next();
    std::vector<AccNode> terms;
    for (int64_t t = 0; t < nterms; t++) terms.push_back(parse_node(c));
    acc_.emplace_back(pos, std::move(terms));
  }
  if (!c.done()) throw std::invalid_argument("msl: layout has trailing data");
  if (out_shapes_.size() !=
      stacked_pos_.size() + static_cast<size_t>(nhidden_) +
          state_pos_.size())
    throw std::invalid_argument("msl: output count does not match the plan");
  // Every index the recipe will subscript with, checked once, here.
  auto carry = [&](int p) {
    if (p < 0 || p >= ncarry_)
      throw std::invalid_argument("msl: carry index out of range");
  };
  auto source = [&](int s) {
    if (s < 0 || static_cast<size_t>(s) >= norms_.size())
      throw std::invalid_argument("msl: source index out of range");
  };
  for (int p : state_pos_) carry(p);
  for (int p : stacked_pos_) carry(p);
  for (int p : passthrough_) carry(p);
  for (const auto& c2 : counters_) carry(c2.first);
  for (const auto& a : acc_) carry(a.first);
  for (int s : unpacked_) source(s);
  for (const Pack& p : packs_)
    for (int s : p.sids) source(s);
}

AccNode MslPlan::parse_node(Cursor& c) {
  AccNode n;
  n.kind = static_cast<int>(c.next());
  switch (n.kind) {
    case 0:
      n.idx = static_cast<int>(c.next());
      break;
    case 1:
      n.idx = static_cast<int>(c.next());
      n.a = c.next();
      n.b = c.next();
      n.shape = c.shp();
      break;
    case 2:
      n.kids.push_back(parse_node(c));
      n.dims = c.vec();
      n.perm = c.vec();
      n.shape = c.shp();
      break;
    case 3:
      n.kids.push_back(parse_node(c));
      n.kids.push_back(parse_node(c));
      n.lb = c.vec();
      n.rb = c.vec();
      n.lc = c.vec();
      n.rc = c.vec();
      n.perm = c.vec();
      n.lshape = c.shp();
      n.rshape = c.shp();
      break;
    default:
      throw std::invalid_argument("msl: bad accumulator node");
  }
  return n;
}

void MslPlan::build() {
  if (built_) return;
  kernel_ = mx::fast::metal_kernel(name_, in_names_, out_names_, source_,
                                   header_);
  built_ = true;
}

// `Plan.run`'s `stacked`: the (L, *per-step) array an accumulator term
// contributes, either straight out of the kernel, straight out of a
// device buffer, or reduced/contracted after it (loop fission -- a
// cross-lane dot cannot run per lane, so the kernel stacks its operands
// and one batched matmul finishes the job here).
mx::array MslPlan::stacked_of(const AccNode& s,
                             const std::vector<mx::array>& outs, size_t ns,
                             const std::vector<mx::array>& srcs) const {
  // Bounds are checked rather than trusted: an index the lowering got
  // wrong reads whatever is next in memory, which is a NaN or a crash
  // depending on the day (it was both, before the source count and the
  // accumulator recipes were made to agree).
  switch (s.kind) {
    case 0: {
      const size_t i = ns + static_cast<size_t>(s.idx);
      if (s.idx < 0 || i >= outs.size())
        throw std::runtime_error("msl: hidden stack out of range");
      return outs[i];
    }
    case 1: {
      if (s.idx < 0 || static_cast<size_t>(s.idx) >= srcs.size())
        throw std::runtime_error("msl: accumulator source out of range");
      const mx::array& src = srcs[static_cast<size_t>(s.idx)];
      const int64_t b2 = s.a * start_ + s.b;
      mx::Shape start(src.ndim(), 0), stop(src.shape());
      mx::array sl = src;
      if (s.a == 1) {
        start[0] = static_cast<mx::ShapeElem>(b2);
        stop[0] = static_cast<mx::ShapeElem>(b2 + trip_);
        sl = mx::slice(src, start, stop);
      } else {
        start[0] = static_cast<mx::ShapeElem>(b2 - trip_ + 1);
        stop[0] = static_cast<mx::ShapeElem>(b2 + 1);
        sl = mx::flip(mx::slice(src, start, stop), 0);
      }
      mx::Shape want{static_cast<mx::ShapeElem>(trip_)};
      for (auto d : s.shape) want.push_back(d);
      return mx::reshape(sl, want);
    }
    case 2: {
      mx::array a = stacked_of(s.kids[0], outs, ns, srcs);
      std::vector<int> axes;
      for (int d : s.dims) axes.push_back(d + 1);
      a = mx::sum(a, axes);
      if (!is_identity_perm(s.perm)) {
        std::vector<int> p{0};
        for (int d : s.perm) p.push_back(d + 1);
        a = mx::transpose(a, p);
      }
      mx::Shape want{static_cast<mx::ShapeElem>(trip_)};
      for (auto d : s.shape) want.push_back(d);
      return mx::reshape(a, want);
    }
    case 3: {
      mx::array A = stacked_of(s.kids[0], outs, ns, srcs);
      mx::array B = stacked_of(s.kids[1], outs, ns, srcs);
      // Hidden stacks may have been unit-squeezed at plan time; restore
      // the operand's exact rank (free -- all the dropped dims are 1).
      A = with_lead(A, s.lshape);
      B = with_lead(B, s.rshape);
      const int lrank = static_cast<int>(s.lshape.size());
      const int rrank = static_cast<int>(s.rshape.size());
      std::vector<int> zb_l{0}, zb_r{0}, c_l, c_r, f_l, f_r;
      for (int d : s.lb) zb_l.push_back(d + 1);
      for (int d : s.rb) zb_r.push_back(d + 1);
      for (int d : s.lc) c_l.push_back(d + 1);
      for (int d : s.rc) c_r.push_back(d + 1);
      auto has = [](const std::vector<int>& v, int x) {
        return std::find(v.begin(), v.end(), x) != v.end();
      };
      for (int i = 0; i <= lrank; i++)
        if (!has(zb_l, i) && !has(c_l, i)) f_l.push_back(i);
      for (int i = 0; i <= rrank; i++)
        if (!has(zb_r, i) && !has(c_r, i)) f_r.push_back(i);
      std::vector<int> pl(zb_l), pr(zb_r);
      pl.insert(pl.end(), f_l.begin(), f_l.end());
      pl.insert(pl.end(), c_l.begin(), c_l.end());
      pr.insert(pr.end(), c_r.begin(), c_r.end());
      pr.insert(pr.end(), f_r.begin(), f_r.end());
      mx::array At = mx::transpose(A, pl);
      mx::array Bt = mx::transpose(B, pr);
      const int64_t bsz = shape_prod(A.shape(), zb_l);
      const int64_t m = shape_prod(A.shape(), f_l);
      const int64_t kk = shape_prod(A.shape(), c_l);
      const int64_t n2 = shape_prod(B.shape(), f_r);
      mx::array out = mx::matmul(
          mx::reshape(At, mx::Shape{static_cast<mx::ShapeElem>(bsz),
                                    static_cast<mx::ShapeElem>(m),
                                    static_cast<mx::ShapeElem>(kk)}),
          mx::reshape(Bt, mx::Shape{static_cast<mx::ShapeElem>(bsz),
                                    static_cast<mx::ShapeElem>(kk),
                                    static_cast<mx::ShapeElem>(n2)}));
      mx::Shape want;
      for (int i : zb_l) want.push_back(A.shape()[static_cast<size_t>(i)]);
      for (int i : f_l) want.push_back(A.shape()[static_cast<size_t>(i)]);
      for (int i : f_r) want.push_back(B.shape()[static_cast<size_t>(i)]);
      mx::array arr = mx::reshape(out, want);
      if (!is_identity_perm(s.perm)) {
        std::vector<int> p{0};
        for (int d : s.perm) p.push_back(d + 1);
        arr = mx::transpose(arr, p);
      }
      return arr;
    }
    default:
      throw std::runtime_error("msl: bad accumulator node");
  }
}

// `x` reshaped to (L, *want) when its trailing dims are not already that.
mx::array MslPlan::with_lead(const mx::array& x, const mx::Shape& want) {
  if (x.ndim() == want.size() + 1) {
    bool same = true;
    for (size_t i = 0; i < want.size(); i++)
      same = same && x.shape()[i + 1] == want[i];
    if (same) return x;
  }
  mx::Shape s{x.shape()[0]};
  for (auto d : want) s.push_back(d);
  return mx::reshape(x, s);
}

// A counted loop msl_scan planned into one generated kernel. The entry is
// a kWhile in every other respect -- same attrs, same carries, the same
// cond/body sub-programs when they lowered -- with the kernel's launch
// recipe and the arrays it reads appended. So a plan that dies falls back
// to the loop it was standing in for, in this very call: the carries are
// untouched (the kernel writes only its own outputs), and the Python
// engine does exactly this when `try_run` raises.
void Program::run_msl(const Entry& e,
                      std::vector<std::optional<mx::array>>& env,
                      bool in_trace) const {
  auto in = [&](size_t i) -> const mx::array& {
    const auto& v = env[e.ins[i]];
    if (!v) throw std::runtime_error("tape: read of a dropped slot");
    return *v;
  };
  MslPlan* plan = e.msl.get();
  if (plan != nullptr && !plan->dead()) {
    const int64_t ncarry = e.attrs[0];
    const size_t base = e.ins.size() - plan->num_sources();
    std::vector<mx::array> carries, srcs;
    for (int64_t i = 0; i < ncarry; i++) carries.push_back(in(i));
    for (size_t i = base; i < e.ins.size(); i++) srcs.push_back(in(i));
    std::exception_ptr err;
    try {
      std::vector<mx::array> vals =
          plan->run(carries, srcs, in_trace, t_msl_pending);
      write_results(e, env, vals);
      return;
    } catch (const std::exception&) {
      err = std::current_exception();
    }
    // Outside the handler, holding none of the failed launch's arrays.
    // msl_scan.try_run's policy: whatever went wrong, this kernel is not
    // going to work, so the loop stops asking for it.
    plan->kill();
    g_stats.msl_failures++;
    if (e.regions.size() < 2) {
      // Nothing to fall back to (the body is outside the native op set):
      // the caller retires the tape and the Python engine runs the
      // program, kernel recovery and all.
      std::rethrow_exception(err);
    }
    debug_print("msl_scan kernel failed; running the loop instead");
  }
  run_while(e, env, in_trace);
}

// engine.execute's msl recovery, on the native side of the boundary: a
// generated kernel that was TRACED into a compiled graph has never been
// built, so the call is settled here, synchronously, while any such plan
// is unproven -- a Metal build error surfacing later, on an async worker,
// aborts the process. On a failure the plans that could be at fault are
// retired (they fall back to their interpreted loops), the compiled
// graphs that embedded their kernels go with them, and the program is run
// again. Once, and only for programs with no host call in them: a rerun
// repeats whatever the first attempt already did to the world.
void Program::settle_msl(const std::vector<mx::array>& inputs,
                         std::vector<mx::array>& outs) {
  if (t_msl_pending.empty()) return;
  // Bounded, and it has to be: the rerun can REACH a plan the first attempt
  // never launched. A loop whose kernel is retired runs its body instead, and
  // the body may hold a loop of its own -- so a nested scan hands back a
  // second unproven plan on the very run that was recovering from the first
  // (P21: found by METALJAX_MSL_FORCE_BUILD_FAIL over execute_test's "msl
  // nested unrolled loop"). Stage 1 answered this by disabling the whole
  // persistent-kernel path for the program and letting the interpreter run it
  // (engine.execute's `_no_msl`); with no Python engine underneath, the same
  // answer is: retire EVERY plan this program holds and run it once more,
  // after which no kernel can fail because none is left.
  for (int round = 0; !t_msl_pending.empty(); round++) {
    std::exception_ptr err;
    try {
      mx::eval(outs);
      for (MslPlan* p : t_msl_pending) p->validate();
      t_msl_pending.clear();
      return;
    } catch (const std::exception&) {
      err = std::current_exception();
    }
    for (MslPlan* p : t_msl_pending) p->kill();
    g_stats.msl_failures++;
    t_msl_pending.clear();
    // A rerun repeats whatever the first attempt already did to the world, so
    // a program holding a host call keeps the failure.
    if (has_host() || round >= 1) {
      if (has_host() || round >= 2) std::rethrow_exception(err);
      // Second failure: no more one-plan-at-a-time recovery.
      debug_print("a second msl_scan kernel failed to build; retiring every "
                  "kernel in this program");
      disable_msl_deep();
    } else {
      debug_print("a traced msl_scan kernel failed to build; dropping it and "
                  "rerunning the program");
    }
    drop_compiled_deep();
    gc_collect();
    mx::clear_cache();
    outs = run_recovering(inputs);
  }
}

// Every generated kernel this program (or any region of it) holds, retired.
// The entries fall back to the interpreted loops they carry alongside, which
// is what makes the recovery above terminate.
void Program::disable_msl_deep() {
  for (Entry& e : ops_) {
    if (e.msl) e.msl->kill();
    for (const auto& r : e.regions) r->disable_msl_deep();
  }
}

}  // namespace metaljax
