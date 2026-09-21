// metaljax native engine — the compiled path (M3).
//
// A Program whose tape Python judged compilable is traced through
// mx::compile once per repeat count and replayed as a fused graph
// afterwards. Three things make that safe, and all three are here: cache
// ids that cannot collide with MLX's own (which are function addresses),
// the anchoring that keeps two equal-valued constant outputs from colliding
// inside MLX's compiler, and the one-way retirement of a compiled variant
// that failed -- the eager path is always correct, so every failure moves
// toward it and never back.

#include "program.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

#include <mlx/compile_impl.h>

namespace metaljax {

namespace {

// A distinctive high tag: mx::detail::compile keys its cache by an integer
// the CALLER owns, and MLX's own Python bindings use the address of a Python
// function object. Ids from this counter can collide with neither those nor
// any real pointer (user-space addresses on arm64 macOS live far below).
std::atomic<std::uintptr_t> g_next_compile_id{0x6D6A5F0000000001ULL};

std::uintptr_t new_compile_id() { return g_next_compile_id.fetch_add(1); }

// P49. The start table one specialized body application is traced with, set
// for exactly the span of that application's `interpret` and restored after
// it -- so the second half of a K=2 variant reads position pos0+1, and a
// nested region (an inner while's body, a reduce's fold) walked inside the
// same trace sees a frame that is not its program's and keeps the dynamic
// spelling.
class SpecScope {
 public:
  SpecScope(const Program* prog, const LoopSpec* spec) : prev_(t_spec_frame) {
    frame_.prog = prog;
    frame_.spec = spec;
    t_spec_frame = &frame_;
  }
  ~SpecScope() { t_spec_frame = prev_; }
  SpecScope(const SpecScope&) = delete;
  SpecScope& operator=(const SpecScope&) = delete;

 private:
  SpecFrame frame_;
  const SpecFrame* prev_;
};

// ops.control._anchor_outputs: give constant outputs a bitwise-exact data
// dependency on an input. where(x == x, out, out) == out for every bit
// pattern (both branches are `out`), but the result is a computed node, not
// a constant MLX's compiler can bake into a table KEYED BY VALUE -- where
// two equal-valued constant outputs collide and the compiled call dies with
// unordered_map::at.
void anchor_outputs(std::vector<mx::array>& outs,
                    const std::vector<mx::array>& args,
                    const std::vector<int>& underived) {
  if (underived.empty() || args.empty()) return;
  std::optional<mx::array> anchor;
  for (const mx::array& a : args) {
    if (a.size() == 0) continue;
    mx::array head = mx::slice(mx::reshape(a, mx::Shape{-1}), mx::Shape{0},
                               mx::Shape{1});
    anchor = mx::reshape(mx::equal(head, head), mx::Shape{});
    break;
  }
  if (!anchor) return;
  for (int i : underived) {
    if (i >= 0 && static_cast<size_t>(i) < outs.size())
      outs[i] = mx::where(*anchor, outs[i], outs[i]);
  }
}

}  // namespace

thread_local const SpecFrame* t_spec_frame = nullptr;

const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
Program::compiled(int repeat) {
  auto it = compiled_.find(repeat);
  if (it != compiled_.end()) return it->second.fn;
  Compiled c;
  c.id = new_compile_id();
  Program* self = this;
  std::vector<int> anchors = anchors_;
  auto traced = [self, repeat, anchors](const std::vector<mx::array>& flat)
      -> std::vector<mx::array> {
    std::vector<mx::array> vals = self->interpret(flat, true);
    if (repeat > 1) {
      // A body's outputs are its next carries, and its captures ride
      // along unchanged: feed the outputs back in, keep the tail.
      for (int r = 1; r < repeat; r++) {
        std::vector<mx::array> next(vals);
        next.insert(next.end(), flat.begin() + vals.size(), flat.end());
        vals = self->interpret(next, true);
      }
    }
    anchor_outputs(vals, flat, anchors);
    return vals;
  };
  g_stats.compiles++;
  c.fn = mx::detail::compile(traced, c.id, false, {});
  auto ins = compiled_.emplace(repeat, std::move(c));
  return ins.first->second.fn;
}

// P49. `repeat` applications of this body as one graph, each traced with the
// start table of the position it will run at. Identical to `compiled` above
// except for the scopes: the folding reaches the dynamic-slice handler and
// nothing else, so the graph MLX builds differs from the generic one only in
// which spelling those slices took.
const std::function<std::vector<mx::array>(const std::vector<mx::array>&)>&
Program::compiled_spec(int repeat, int64_t pos0,
                       const std::vector<const LoopSpec*>& specs) {
  const std::pair<int, int64_t> key(repeat, pos0);
  auto it = spec_compiled_.find(key);
  if (it != spec_compiled_.end()) return it->second.fn;
  Compiled c;
  c.id = new_compile_id();
  Program* self = this;
  std::vector<int> anchors = anchors_;
  // By pointer: the tables live in `spec_cache_` for the life of the program
  // (`drop_spec` is what clears both, and it clears the variants first).
  std::vector<const LoopSpec*> tables = specs;
  auto traced = [self, repeat, anchors, tables](
                    const std::vector<mx::array>& flat)
      -> std::vector<mx::array> {
    std::vector<mx::array> vals;
    {
      SpecScope scope(self, tables[0]);
      vals = self->interpret(flat, true);
    }
    for (int r = 1; r < repeat; r++) {
      std::vector<mx::array> next(vals);
      next.insert(next.end(), flat.begin() + vals.size(), flat.end());
      SpecScope scope(self, tables[r]);
      vals = self->interpret(next, true);
    }
    anchor_outputs(vals, flat, anchors);
    return vals;
  };
  g_stats.compiles++;
  g_stats.spec_variants++;
  for (const LoopSpec* s : specs) g_stats.spec_folds += s->folded();
  c.fn = mx::detail::compile(traced, c.id, false, {});
  auto ins = spec_compiled_.emplace(key, std::move(c));
  return ins.first->second.fn;
}

void Program::prove_spec(int repeat, int64_t pos0,
                         const std::vector<mx::array>& outs) {
  auto it = spec_compiled_.find(std::pair<int, int64_t>(repeat, pos0));
  if (it == spec_compiled_.end() || it->second.proven) return;
  mx::eval(outs);
  it->second.proven = true;
}

// A specialized variant failed. Like every other compiled-path failure this
// moves one way only -- toward the generic variant, which is always correct.
void Program::drop_spec() {
  spec_state_ = 2;
  for (const auto& kv : spec_compiled_) mx::detail::compile_erase(kv.second.id);
  spec_compiled_.clear();
  spec_cache_.clear();
  g_stats.spec_declines++;
}

bool Program::may_compile(int repeat) const {
  return compile_ && !compile_disabled_ && repeat <= max_repeat_;
}

void Program::drop_compiled() {
  compile_disabled_ = true;
  for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
  compiled_.clear();
  // The specialized variants are the same tape through the same compiler:
  // whatever it rejected, it will reject again.
  for (const auto& kv : spec_compiled_) mx::detail::compile_erase(kv.second.id);
  spec_compiled_.clear();
  spec_cache_.clear();
  spec_state_ = 2;
  g_stats.compile_drops++;
}

// The first call of a compiled variant, settled. Declared in program.h,
// where the rule it enforces is written out: nothing that can still fail to
// BUILD may be handed to `mx::async_eval`, because a build failure raised
// inside an async walk abandons MLX's per-stream events unsignaled and the
// next blocking eval of an array that walk had visited never returns.
//
// Deliberately not wrapped in a try: the caller's ladder is what decides
// what a failure means (a chunk that cannot be built stops being chunked, a
// main tape falls back to the eager path), and every one of them is
// positioned to catch this. `proven` is set only on success, so a variant
// that threw is probed again if it is ever called again -- it will not be:
// every caller retires it.
void Program::prove_compiled(int repeat,
                             const std::vector<mx::array>& outs) {
  auto it = compiled_.find(repeat);
  if (it == compiled_.end() || it->second.proven) return;
  mx::eval(outs);
  it->second.proven = true;
}

// Drop every compiled graph in this program and its regions: a compiled
// trace that embedded a now-dead kernel would keep calling it. The
// compile DECISION stands (disable_msl clears `_compiled` and no more) --
// the next trace simply builds the loop where the kernel was.
void Program::drop_compiled_deep() {
  for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
  compiled_.clear();
  for (const auto& kv : spec_compiled_) mx::detail::compile_erase(kv.second.id);
  spec_compiled_.clear();
  for (const Entry& e : ops_)
    for (const auto& r : e.regions) r->drop_compiled_deep();
}

}  // namespace metaljax
