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

bool Program::may_compile(int repeat) const {
  return compile_ && !compile_disabled_ && repeat <= max_repeat_;
}

void Program::drop_compiled() {
  compile_disabled_ = true;
  for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
  compiled_.clear();
  g_stats.compile_drops++;
}

// Drop every compiled graph in this program and its regions: a compiled
// trace that embedded a now-dead kernel would keep calling it. The
// compile DECISION stands (disable_msl clears `_compiled` and no more) --
// the next trace simply builds the loop where the kernel was.
void Program::drop_compiled_deep() {
  for (const auto& kv : compiled_) mx::detail::compile_erase(kv.second.id);
  compiled_.clear();
  for (const Entry& e : ops_)
    for (const auto& r : e.regions) r->drop_compiled_deep();
}

}  // namespace metaljax
