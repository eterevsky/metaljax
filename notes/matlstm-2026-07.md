# matlstm on metal: 300x slow, needs register blocks (2026-07)

Report (Oleg, post-0.4.3 search): worker "seriously stuck" on
`bits.1+bp|dense.1.tanh-suffix.4-dense.2.tanh-matlstm.2` fp32 4x2048
S32768 — ~723 ms/step observed. Reproduced exactly: 735-742 ms/step;
jax-CPU does it in 2.46 ms/step (300x). At S32768 that's ~6.7 hours of
GPU for a 39-weight config.

## Why

matlstm (xLSTM mLSTM, texmo/layers/matlstm.py) carries a MATRIX state
per sample: C (D,D), n (D,), m scalar; per step
C' = f·C + i·outer(v,k), h = o·(C'@q)/max(|n'·q|, 1). Neither msl_scan
mode can host it, so both time loops (fwd + AD bwd, trip 2048) fall to
compiled-body replay (~0.18 ms/iteration x 2048 x 2 loops) and their
pessimistic cost makes the whole chunk eager. The failure cascade:

1. (fixed) `broadcast of SymAccDot (4,2)->(1,4,2)`: _broadcasted had no
   accumulator case for pure unit-dim insertion (vmapped `carry +
   broadcast(dot)`). Now delegates to _reshaped; benign — run()
   reshapes summed accs to the carry shape and dot specs only use
   operand shapes.
2. (fixed) both cell dots have NO invariant operand (outer(v,k), C'@q
   are all loop-computed) → classified SymAccDot → "acc-dot outside
   accumulator update" (the C update is DECAYED, f_t·C — fission
   requires plain sums, and h reads the running state anyway). New
   `_inlane_dot` rewrite: outer products → broadcast-multiply,
   trailing-dim contractions → multiply + SymRedReg. Gated: both
   operands f32, contraction width <= _REG_LIMIT, and RANK >= 3
   somewhere (matrix-state signature — ungated it stole db02-class
   rank-2 dW dots from fission, 1.4x regression, measured). Env:
   METALJAX_MSL_INLANE=0. build_plan retries with the rewrite off if
   the rewritten build fails (same protocol as the coop flip; the two
   retry flags combine).
3. (fixed) with all dots rewritten away, mode selection fell to
   "scalar", whose emitter has no registers for SymRedReg ("emit
   type"). New _needs_registers() DAG probe → vector mode when
   register-resident constructs survive without dots.
4. (BLOCKED) the real wall, hit after 1-3: lane-space unification.
   C (4,2,2) wants lanes (batch,row) with cols in registers; n (4,2)
   and h want lanes (batch,) with D in registers → `broadcast (4,2) vs
   (4,)`. And the bwd has ACCRED[dims=[1,2]] — a per-batch reduce over
   the WHOLE matrix — mid-expression: cross-lane under (batch,row)
   lanes, illegal outside accumulator position.

## The feature that would fix it: 2-D register blocks (vector mode)

Lane = batch element only; each lane holds C as a DxD register BLOCK
(D<=4-5: 16-25 regs), n as a D-tail, m as a scalar. Everything becomes
in-lane: outer = register outer product, C@q = register matvec,
dims=[1,2] reduces = full block reduce. Touches: lane derivation,
per-leaf R -> block shapes, emitter loops over block coords, RedReg
over chosen block dims, broadcast strides per block dim, stacked
writes. Est. 1-2 days + full validation. Covers matlstm.2/.4 fwd+bwd;
matlstm.8+ (64+ regs/lane) would need a coop-style threadgroup variant.

## Meanwhile

The three recognizer fixes stand on their own (validated: 200 pytest
incl. new matrix-state cell test that PLANS in vector mode; gate
104/104; standalone A/B spots db02/db08/db11/mid08 neutral, db17
6.2 vs 8.4 ms/step — a kept win from a rank-3 in-lane rewrite).
matlstm itself stays on the replay path (~740 ms/step) until register
blocks land — if search keeps drawing matlstm configs at long L,
routing them to CPU workers is 300x better until then.
