"""Training-step benchmark: metaljax vs JAX-CPU vs PyTorch (MPS/CPU).

Models:
  transformer -- GPT-style byte LM: tok+pos embeddings, N pre-LN blocks
                 (causal MHA + GELU MLP), CE loss. Matmul-dominated; the
                 GPU showcase.
  gru         -- the same custom GRU cell as texmo's scripts/bench_jax.py.
                 Sequential scan; the GPU-unfriendly case.

Each run times full training steps (fwd + bwd + AdamW update) on synthetic
byte data, after warmup. Examples:

  .venv/bin/python scripts/bench_compare.py --backend jax --platform metal
  .venv/bin/python scripts/bench_compare.py --backend jax --platform cpu
  .venv/bin/python scripts/bench_compare.py --backend torch --device mps
"""

import argparse
import math
import time

import numpy as np

V = 256  # byte vocabulary


def synthetic_batches(batch, seq, steps, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, V, (batch, seq)).astype(np.int32)
            for _ in range(steps)]


# --------------------------------------------------------------------- jax

def run_jax(args):
    import jax
    if args.platform:
        jax.config.update("jax_platforms", args.platform)
    import jax.numpy as jnp
    import optax

    print(f"backend: jax  platform: {jax.default_backend()}  "
          f"devices: {jax.devices()}")

    D, L, H, T = args.d_model, args.layers, args.heads, args.seq
    key = jax.random.PRNGKey(0)

    def norm_shape(*shape):
        nonlocal key
        key, k = jax.random.split(key)
        return jax.random.normal(k, shape, jnp.float32) * 0.02

    if args.model == "transformer":
        params = {
            "tok": norm_shape(V, D),
            "pos": norm_shape(T, D),
            "blocks": [
                {
                    "ln1": (jnp.ones(D), jnp.zeros(D)),
                    "qkv": (norm_shape(D, 3 * D), jnp.zeros(3 * D)),
                    "proj": (norm_shape(D, D), jnp.zeros(D)),
                    "ln2": (jnp.ones(D), jnp.zeros(D)),
                    "mlp1": (norm_shape(D, 4 * D), jnp.zeros(4 * D)),
                    "mlp2": (norm_shape(4 * D, D), jnp.zeros(D)),
                }
                for _ in range(L)
            ],
            "lnf": (jnp.ones(D), jnp.zeros(D)),
            "head": norm_shape(D, V),
        }

        def layernorm(x, gb):
            g, b = gb
            m = x.mean(-1, keepdims=True)
            v = ((x - m) ** 2).mean(-1, keepdims=True)
            return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b

        causal = jnp.tril(jnp.ones((T, T), bool))

        def forward(params, tokens):
            x = params["tok"][tokens] + params["pos"][None, :, :]
            for blk in params["blocks"]:
                h = layernorm(x, blk["ln1"])
                qkv = h @ blk["qkv"][0] + blk["qkv"][1]
                q, k, v = jnp.split(qkv, 3, axis=-1)
                split = lambda z: z.reshape(z.shape[0], T, H, D // H).transpose(0, 2, 1, 3)
                q, k, v = split(q), split(k), split(v)
                att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D // H)
                att = jnp.where(causal, att, -jnp.inf)
                att = jax.nn.softmax(att, axis=-1)
                o = (att @ v).transpose(0, 2, 1, 3).reshape(x.shape)
                x = x + o @ blk["proj"][0] + blk["proj"][1]
                h = layernorm(x, blk["ln2"])
                x = x + jax.nn.gelu(h @ blk["mlp1"][0] + blk["mlp1"][1]) @ blk["mlp2"][0] + blk["mlp2"][1]
            return layernorm(x, params["lnf"]) @ params["head"]

        def loss_fn(params, tokens):
            logits = forward(params, tokens)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1], tokens[:, 1:]).mean() / math.log(2.0)

    else:  # gru — mirrors texmo scripts/bench_jax.py
        Hs = args.d_model

        def rnd(*shape):
            nonlocal key
            key, k = jax.random.split(key)
            return jax.random.normal(k, shape, jnp.float32) * 0.01

        params = {
            "wi": rnd(Hs, V), "wh": rnd(Hs, Hs), "bi": jnp.zeros(Hs),
            "wr": rnd(Hs, V), "wrh": rnd(Hs, Hs), "br": jnp.zeros(Hs),
            "wz": rnd(Hs, V), "wzh": rnd(Hs, Hs), "bz": jnp.zeros(Hs),
            "wout": rnd(V, Hs), "bout": jnp.zeros(V),
        }

        def loss_fn(w, tokens):
            oh = jax.nn.one_hot(tokens, V, dtype=jnp.float32)
            oh_in = jnp.pad(oh[:, :-1], ((0, 0), (1, 0), (0, 0)))

            def step(h, x):
                r = jax.nn.sigmoid(x @ w["wr"].T + h @ w["wrh"].T + w["br"])
                z = jax.nn.sigmoid(x @ w["wz"].T + h @ w["wzh"].T + w["bz"])
                c = jnp.tanh(x @ w["wi"].T + (r * h) @ w["wh"].T + w["bi"])
                h = (1 - z) * h + z * c
                return h, h

            h0 = jnp.zeros((tokens.shape[0], Hs), jnp.float32)
            _, hs = jax.lax.scan(step, h0, oh_in.transpose(1, 0, 2))
            logits = hs.transpose(1, 0, 2) @ w["wout"].T + w["bout"]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
            return loss.mean() / math.log(2.0)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=0.01,
                    mask=lambda tree: jax.tree.map(lambda g: g.ndim > 1, tree)),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    batches = synthetic_batches(args.batch, args.seq, args.steps + args.warmup)
    for i in range(args.warmup):
        params, opt_state, loss = train_step(params, opt_state, batches[i])
    jax.block_until_ready(loss)
    t0 = time.perf_counter()
    for i in range(args.warmup, args.warmup + args.steps):
        params, opt_state, loss = train_step(params, opt_state, batches[i])
    jax.block_until_ready(loss)
    dt = time.perf_counter() - t0
    report(args, dt, float(loss))


# ------------------------------------------------------------------- torch

def run_torch(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device(args.device)
    print(f"backend: torch  device: {dev}")
    torch.manual_seed(0)

    D, L, H, T = args.d_model, args.layers, args.heads, args.seq

    if args.model == "transformer":
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(D)
                self.qkv = nn.Linear(D, 3 * D)
                self.proj = nn.Linear(D, D)
                self.ln2 = nn.LayerNorm(D)
                self.mlp1 = nn.Linear(D, 4 * D)
                self.mlp2 = nn.Linear(4 * D, D)

            def forward(self, x):
                B = x.shape[0]
                h = self.ln1(x)
                q, k, v = self.qkv(h).chunk(3, dim=-1)
                shp = lambda z: z.view(B, T, H, D // H).transpose(1, 2)
                o = F.scaled_dot_product_attention(
                    shp(q), shp(k), shp(v), is_causal=True)
                x = x + self.proj(o.transpose(1, 2).reshape(B, T, D))
                x = x + self.mlp2(F.gelu(self.mlp1(self.ln2(x))))
                return x

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = nn.Embedding(V, D)
                self.pos = nn.Parameter(torch.randn(T, D) * 0.02)
                self.blocks = nn.ModuleList(Block() for _ in range(L))
                self.lnf = nn.LayerNorm(D)
                self.head = nn.Linear(D, V, bias=False)

            def forward(self, tokens):
                x = self.tok(tokens) + self.pos
                for b in self.blocks:
                    x = b(x)
                return self.head(self.lnf(x))

        model = Model().to(dev)

        def loss_of(tokens):
            logits = model(tokens)
            return F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                tokens[:, 1:].reshape(-1).long()) / math.log(2.0)

    else:  # gru: torch's fused nn.GRU as the "best case" target
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(V, D, batch_first=True)
                self.out = nn.Linear(D, V)

            def forward(self, tokens):
                oh = F.one_hot(tokens.long(), V).float()
                oh = F.pad(oh[:, :-1], (0, 0, 1, 0))
                hs, _ = self.gru(oh)
                return self.out(hs)

        model = Model().to(dev)

        def loss_of(tokens):
            logits = model(tokens)
            return F.cross_entropy(
                logits.reshape(-1, V), tokens.reshape(-1).long()) / math.log(2.0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    batches = [torch.from_numpy(b).to(dev) for b in
               synthetic_batches(args.batch, args.seq, args.steps + args.warmup)]
    for i in range(args.warmup):
        opt.zero_grad()
        loss = loss_of(batches[i])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sync()
    t0 = time.perf_counter()
    for i in range(args.warmup, args.warmup + args.steps):
        opt.zero_grad()
        loss = loss_of(batches[i])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sync()
    dt = time.perf_counter() - t0
    report(args, dt, float(loss.item()))


def report(args, dt, loss):
    print(f"{args.model}  d={args.d_model} L={args.layers} T={args.seq} "
          f"batch={args.batch}  {args.steps} steps: "
          f"{dt:.2f}s  ({1000 * dt / args.steps:.1f} ms/step)  loss={loss:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("jax", "torch"), default="jax")
    p.add_argument("--platform", default=None,
                   help="jax platform: metal / cpu / metal,cpu")
    p.add_argument("--device", default="mps", help="torch device: mps / cpu")
    p.add_argument("--model", choices=("transformer", "gru"),
                   default="transformer")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    (run_jax if args.backend == "jax" else run_torch)(args)


if __name__ == "__main__":
    main()
