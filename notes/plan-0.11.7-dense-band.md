# 0.11.7 campaign plan — the dense-band decode optimizations (Oleg, 2026-08-30)

**The release goal**: all four framework-gap items landed, lifting the dense
bf16 decode band (rows 1/2/5/6/9/12, currently 1.6–1.9× behind the
frontier) plus whatever they give the rest of the table; then the standard
release process (full gate on one frozen binary → tables → wheel → Oleg's
review → TestPyPI → his publish).

## State at campaign start

- HEAD `ce42938`; tree clean. 0.11.6 released (tag v0.11.6, PyPI).
- Row-10 campaign complete: 1948.2 → **25.4 ms/tok** (76×), via the
  ragged_dot recognizer (6590c05), stacked-dot/K=1 fixes (ff569eb), and the
  P30 op-count campaign (37b0cee: CSE + region constant folding, MLA-sdpa,
  rms_norm recognizer). Streams identical throughout.
- HEAD cells (models.md ʰ column): row 4 = 24.7, row 10 = 25.4,
  row 11 = 12.03, row 14 = 29.84.
- Frontier verified like-for-like bf16 (notes/comparator-survey-2026-08-29.md,
  notes/data/llamacpp-bf16-2026-08-29.jsonl): mlx-lm and llama.cpp within
  ~3% on most rows — dense rows ride 551–570 GB/s; row 8 mlx-lm wins
  (13.7 vs 15.34), row 9 llama.cpp wins (114.9 vs 131.8). Row 12 has NO
  published 16-bit gguf (proven). Like-for-like rule: STATUS.md fn 20.

## The four items (prices from notes/framework-gap-gemma31b.md, measured on
## the 31B two engine-generations ago — RE-MEASURE ON HEAD FIRST)

1. **`dot_general` middle-axis handling** (~42 ms/tok on 31B then) — the
   biggest; needs fresh diagnosis before building (the engine has changed
   twice since it was priced).
2. **KV-cache in-place / donation forwarding** (~25 ms) — XLA emits
   functional `dynamic_update_slice` per layer per token; make it forward
   in place through the donation machinery. Endgame = custom-kernel ladder
   rung 1 (fuse KV-append INTO the sdpa kernel).
3. **Attention-vector path** (~23 ms) — decode-shaped attention epilogue.
4. **`rms_norm` recognizer coverage** (~12 ms) — the recognizer EXISTS
   (metal_norm.cc, built in P30 for DeepSeek's spelling); verify/extend it
   to gemma/llama/qwen lowerings (they spell norms differently). Cheapest;
   do first.

Complementary, larger, deferred unless Oleg green-lights separately: the
**prepared-replay executor** (build the mx graph once per executable,
rebind buffers per token — attacks the ~10–15 ms/tok host graph
construction every row pays; projected row 10 → ~13–15 vs its ~10–12
floor).

## Campaign rules (the same discipline that produced the 76×)

- Diagnose before building; re-measure stale prices on HEAD; follow the
  measured nodes, not the plan (P30's rope-table lesson).
- Engine-side only; benchmark-code tweaks (scripts/model_bench only, never
  ~/texmo) require genuine blockage + bit-exact results + both cells.
- Per fix: differential tests; row 1 (or 2/5) full protocol before/after;
  rows 10/11/14 + a keras row as no-regression sentinels vs their HEAD
  cells; execute_test/ingest/bazel/texmo_gate-20.
- Machine lock; mem_guard budgets; no-panic contract absolute; rerun-first;
  release rules 1 & 2 verbatim at gate time.
- Stream changes only of the documented tie-flip class, disclosed.
- One subagent at a time (Oleg's process); main agent reviews every diff
  before commit; subagents never commit.
- Vehicle rows: row 2 (gemma4-12B, cheap ~49 s load) for iteration;
  row 1 (31B) and row 5 (qwen3-8B) for confirmation. Row protocols in
  ~/.cache/metaljax-bench/logs/gate-0.11.6/models/ (gate invocations) and
  regate-0.11.5/models/rg_rows.sh (templates; pin the CURRENT binary).

## Also queued for 0.11.7 (from the ledger, not this campaign)

- Row 20 pack-wave optimization (68-min load verification pass).
- The coop-lrnn emitter design (tc112–121 + db07/08/10/12 pockets;
  design in ~/.cache/metaljax-bench/logs/lrnn-split/).
- Row 11's token-stream change at gate time (tie-flip class, phase-2) —
  the gate's token record moves; TOKEN_KNOWN handling.
- Row 10 protocol pin METALJAX_MEM_SYS_MB=107520 (documented envelope).
- MLX fork/upstream: Event::wait patch (notes/patches/mlx-0001-…), 16-bit
  scatter atomics, fusion #8 (needs MLX-level repro), #4099 release request.
