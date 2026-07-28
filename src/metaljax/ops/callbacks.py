"""Host callbacks (jax.debug.print / debug_callback / pure_callback).

The plugin registers metal lowerings that stash the Python callable in
_CALLBACKS and emit a `metaljax_callback` custom call carrying its
index; the interpreter invokes it with numpy inputs. Because the whole
backend runs in-process, no PJRT host-callback machinery is needed.
Blocks containing the target are impure (lapack.TARGETS hook), so
callbacks never land inside mx.compile traces and run in program order
on the eager path.
"""

import numpy as np

from metaljax import _ir, dtypes
from metaljax.interpreter import register
from metaljax.ops.lapack import TARGETS, _result_specs

_CALLBACKS = {}


def register_callback(fn) -> int:
    idx = len(_CALLBACKS)
    _CALLBACKS[idx] = fn
    return idx


def _run_callback(op, ins):
    idx = int(_ir.str_attr(op, "backend_config"))
    fn = _CALLBACKS[idx]
    np_ins = [np.asarray(dtypes.to_np(x)) for x in ins]
    outs = fn(*np_ins)
    if outs is None:
        outs = []
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    specs = _result_specs(op)
    return [np.asarray(o).reshape(spec[0]).astype(spec[1], copy=False)
            for o, spec in zip(outs, specs)]


TARGETS["metaljax_callback"] = _run_callback


@register("stablehlo.create_token", "stablehlo.after_all")
def _token(interp, op, ins, env):
    # Tokens only sequence effects; a zero-size sentinel suffices.
    import mlx.core as mx
    return mx.zeros((0,))
