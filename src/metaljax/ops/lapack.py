"""LAPACK-family custom_call targets, computed on the host.

These factorizations are CPU-bound in every backend (XLA's CPU backend
calls LAPACK; MLX's mx.linalg runs on its CPU stream); unified memory
makes the host round-trip cheap. Any block containing one of these
targets is marked impure (interpreter.custom_call_host_hook), so it
never lands inside an mx.compile trace.

Result shapes/dtypes come from the op's result types; inputs may carry
arbitrary leading batch dimensions.
"""

import numpy as np

import mlx.core as mx

from jaxlib.mlir import ir

from metaljax import _ir, dtypes
from metaljax.interpreter import Interpreter, UnsupportedOpError

TARGETS = {}


def _target(*names):
    def deco(fn):
        for n in names:
            TARGETS[n] = fn
        return fn
    return deco


def _np_in(x):
    return np.asarray(dtypes.to_np(x))


def _result_specs(op):
    out = []
    for r in op.results:
        t = ir.RankedTensorType(r.type)
        out.append((tuple(t.shape), dtypes.np_dtype_for_mlir(t.element_type)))
    return out


def _batch_apply(fn, args, batch_shape, out_specs):
    """Apply fn over leading batch dims; fn maps per-item ndarrays to a
    tuple of per-item ndarrays matching out_specs' trailing shapes."""
    nb = len(batch_shape)
    flat = 1
    for b in batch_shape:
        flat *= b
    outs = [np.zeros(spec[0], dtype=spec[1]) for spec in out_specs]
    flat_args = [a.reshape((flat,) + a.shape[nb:]) for a in args]
    flat_outs = [o.reshape((flat,) + o.shape[nb:]) for o in outs]
    for i in range(flat):
        res = fn(*[a[i] for a in flat_args])
        for dst, r in zip(flat_outs, res):
            dst[i] = r
    return [o.reshape(spec[0]).astype(spec[1], copy=False)
            for o, spec in zip(flat_outs, out_specs)]


@_target("Qr")
def _qr(op, ins):
    from scipy.linalg import lapack
    (a,) = [_np_in(x) for x in ins]
    specs = _result_specs(op)
    batch = a.shape[:-2]

    def one(x):
        qr, tau, _, info = lapack.sgeqrf(x)
        return qr, tau
    return _batch_apply(one, [a], batch, specs)


@_target("ProductOfElementaryHouseholderReflectors")
def _householder_product(op, ins):
    from scipy.linalg import lapack
    a, taus = (_np_in(x) for x in ins)
    specs = _result_specs(op)
    batch = a.shape[:-2]
    m = a.shape[-2]
    k = taus.shape[-1]
    ncols = specs[0][0][-1]

    def one(x, t):
        q, _, info = lapack.sorgqr(np.ascontiguousarray(x[:, :max(k, 1)]), t)
        if q.shape[1] < ncols:
            q = np.pad(q, [(0, 0), (0, ncols - q.shape[1])])
        return (q[:, :ncols],)
    return _batch_apply(one, [a, taus], batch, specs)


def _register_ffi(name_map):
    for names, fn in name_map:
        for n in names:
            TARGETS[n] = fn


@_target("lapack_ssyevd_ffi", "lapack_dsyevd_ffi", "Eigh")
def _eigh(op, ins):
    # FFI result convention: (eigenvectors, eigenvalues, info)
    a = _np_in(ins[0])
    specs = _result_specs(op)
    lower = True
    if "mhlo.backend_config" in op.attributes:
        cfg = str(op.attributes["mhlo.backend_config"])
        lower = "uplo = 76" in cfg or "lower" in cfg.lower()
    batch = a.shape[:-2]

    def one(x):
        x = np.tril(x) + np.tril(x, -1).T if lower else \
            np.triu(x) + np.triu(x, 1).T
        w, v = np.linalg.eigh(x)
        return (v, w) + tuple(
            np.zeros(shape[len(batch):], dtype=dt)
            for shape, dt in specs[2:])
    return _batch_apply(one, [a], batch, specs)


@_target("lapack_sgesdd_ffi", "lapack_dgesdd_ffi", "lapack_sgesvd_ffi")
def _svd(op, ins):
    # FFI result convention: (a_workspace_copy, s, u, vt, info)
    a = _np_in(ins[0])
    specs = _result_specs(op)
    batch = a.shape[:-2]
    nb = len(batch)
    u_cols = specs[2][0][nb:][1]
    vt_rows = specs[3][0][nb:][0]

    def one(x):
        m, n = x.shape
        u, s, vt = np.linalg.svd(
            x, full_matrices=(u_cols == m and vt_rows == n))
        return (x, s, u[:, :u_cols], vt[:vt_rows, :]) + tuple(
            np.zeros(shape[nb:], dtype=dt) for shape, dt in specs[4:])
    return _batch_apply(one, [a], batch, specs)


def custom_call_host_hook(op):
    try:
        return _ir.str_attr(op, "call_target_name") in TARGETS
    except Exception:
        return False


Interpreter.custom_call_host_hook = staticmethod(custom_call_host_hook)


def run_target(interp, op, ins, env):
    target = _ir.str_attr(op, "call_target_name")
    fn = TARGETS.get(target)
    if fn is None:
        return None
    outs = fn(op, ins)
    return [dtypes.to_mx(np.ascontiguousarray(o)) for o in outs]
