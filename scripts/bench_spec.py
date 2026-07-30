"""Time one texmo train chunk (ms/step) for a spec on a given platform.

Usage: bench_spec.py <platform> <spec> <batch> <length> [nsteps] [reps]
"""

import os
import sys
import time
from pathlib import Path

TEXMO = Path(os.environ.get("TEXMO_DIR", Path.home() / "texmo"))
sys.path.insert(0, str(TEXMO))

import jax

jax.config.update("jax_enable_x64", False)
platform = sys.argv[1]
jax.config.update("jax_platforms", platform)

import jax.numpy as jnp
import numpy as np
import logging

logging.disable(logging.INFO)

from texmo.tokens import set_tokens_dir
from texmo.dataset import DataSet
from texmo.configuration import Configuration
from texmo.spec_parser import parse_model2
from texmo.precision import Precision
from texmo.manager_jax import ManagerJax

spec = sys.argv[2]
batch = int(sys.argv[3])
length = int(sys.argv[4])
nsteps = int(sys.argv[5]) if len(sys.argv) > 5 else 8
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 5

set_tokens_dir(str(TEXMO / "tokens"))
train_set = DataSet(path=str(TEXMO / "data" / "pride.txt"))
conf = Configuration(
    parse_model2(spec, precision=Precision("fp32")),
    lr=0.01, length=length, batch=batch, steps=nsteps,
    decay=1.0, cosine=False,
)
manager = ManagerJax(conf=conf, system="bench", dataset=train_set,
                     test_sample_len=64, test_batch=8)
tokens_name = manager.model_def.input.tokens_name
data = manager.dataset.sample_tokens(
    ntokens=length, batch=nsteps * batch, tokenset_name=tokens_name)
batches = jnp.asarray(data).reshape(nsteps, batch, length)
args = (manager.weights, manager._opt_state, batches)


def force(tree):
    # jax.block_until_ready is a no-op on the metal backend; pull one
    # element of every leaf through numpy to force execution.
    for leaf in jax.tree_util.tree_leaves(tree):
        np.asarray(leaf)


t0 = time.perf_counter()
out = manager._train_chunk(*args)
force(out)
t_compile = time.perf_counter() - t0

times = []
for _ in range(reps):
    t0 = time.perf_counter()
    out = manager._train_chunk(*args)
    force(out)
    times.append(time.perf_counter() - t0)

best = min(times)
print(f"RESULT platform={jax.default_backend()} spec={spec} "
      f"batch={batch} len={length} nsteps={nsteps} "
      f"first={t_compile*1000:.1f}ms best_chunk={best*1000:.2f}ms "
      f"ms_per_step={best*1000/nsteps:.3f}")
