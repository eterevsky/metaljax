"""Run a texmo training job on the metaljax backend, without modifying texmo.

texmo pins its JAX platform in its own config.py at import time; this driver
imports ManagerJax directly and selects the platform itself.

Usage (from the metaljax repo root):

    .venv/bin/python scripts/texmo_train.py \
        [platform] [spec] [steps] [batch] [length] [precision]

Defaults: platform=metal,cpu  spec=bits.1+bp|rnn.1.tanh  steps=64
          batch=16  length=64  precision=fp32

The texmo checkout location comes from $TEXMO_DIR (default: ~/texmo).
Requires texmo's import-time deps in this venv (optax, scipy, scikit-learn,
safetensors, regex, torch — torch is only imported, never executed, by the
JAX path).
"""

import os
import sys
import time
from pathlib import Path

TEXMO = Path(os.environ.get("TEXMO_DIR", Path.home() / "texmo"))
if not TEXMO.is_dir():
    sys.exit(f"texmo checkout not found at {TEXMO}; set TEXMO_DIR")
sys.path.insert(0, str(TEXMO))

import jax

jax.config.update("jax_enable_x64", False)  # matches texmo.py
platform = sys.argv[1] if len(sys.argv) > 1 else "metal,cpu"
jax.config.update("jax_platforms", platform)

spec = sys.argv[2] if len(sys.argv) > 2 else "bits.1+bp|rnn.1.tanh"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 64
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16
length = int(sys.argv[5]) if len(sys.argv) > 5 else 64
precision = sys.argv[6] if len(sys.argv) > 6 else "fp32"

import logging

logging.basicConfig(level=logging.INFO)

from texmo.tokens import set_tokens_dir
from texmo.dataset import DataSet
from texmo.configuration import Configuration
from texmo.spec_parser import parse_model2
from texmo.precision import Precision
from texmo.manager_jax import ManagerJax

set_tokens_dir(str(TEXMO / "tokens"))
train_set = DataSet(path=str(TEXMO / "data" / "pride.txt"))
conf = Configuration(
    parse_model2(spec, precision=Precision(precision)),
    lr=0.01,
    length=length,
    batch=batch,
    steps=steps,
    decay=1.0,
    cosine=False,
)
manager = ManagerJax(
    conf=conf,
    system="metaljax",
    dataset=train_set,
    test_sample_len=256,
    test_batch=64,
)
print("default backend:", jax.default_backend())
t0 = time.perf_counter()
run, final_conf = manager.train_and_eval(steps, None)
dt = time.perf_counter() - t0
print(f"DONE platform={platform} spec={spec} steps={steps} "
      f"loss={run.loss:.4f} wall={dt:.1f}s")
