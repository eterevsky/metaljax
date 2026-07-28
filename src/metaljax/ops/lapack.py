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
    """Device array -> numpy, upcasting half precisions to f32: LAPACK
    (and scipy/numpy linalg) has no bf16/f16 routines, but per policy we
    support them by computing in f32 — _batch_apply casts results back
    to the op's declared (half) result dtypes."""
    import ml_dtypes
    a = np.asarray(dtypes.to_np(x))
    if a.dtype in (np.dtype(ml_dtypes.bfloat16), np.dtype(np.float16)):
        a = a.astype(np.float32)
    return a


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


@_target("Qr", "lapack_sgeqrf_ffi", "lapack_dgeqrf_ffi", "lapack_cgeqrf_ffi")
def _qr(op, ins):
    from scipy.linalg import lapack
    (a,) = [_np_in(x) for x in ins]
    specs = _result_specs(op)
    batch = a.shape[:-2]

    geqrf = lapack.cgeqrf if np.iscomplexobj(a) else lapack.sgeqrf

    def one(x):
        qr, tau, _, info = geqrf(x)
        return qr, tau
    return _batch_apply(one, [a], batch, specs)


@_target("ProductOfElementaryHouseholderReflectors",
         "lapack_sorgqr_ffi", "lapack_dorgqr_ffi", "lapack_cungqr_ffi")
def _householder_product(op, ins):
    from scipy.linalg import lapack
    a, taus = (_np_in(x) for x in ins)
    specs = _result_specs(op)
    batch = a.shape[:-2]
    m = a.shape[-2]
    k = taus.shape[-1]
    ncols = specs[0][0][-1]

    orgqr = lapack.cungqr if np.iscomplexobj(a) else lapack.sorgqr

    def one(x, t):
        cols = max(k, 1)
        if ncols > cols:
            # full_matrices: LAPACK orgqr completes an orthonormal basis
            # when given room; zero taus are identity reflectors.
            x = np.pad(np.ascontiguousarray(x[:, :cols]),
                       [(0, 0), (0, ncols - cols)])
            t = np.pad(t, (0, ncols - len(t)))
            q, _, info = orgqr(x, t)
            return (q,)
        q, _, info = orgqr(np.ascontiguousarray(x[:, :cols]), t)
        return (q[:, :ncols],)
    return _batch_apply(one, [a, taus], batch, specs)


def _register_ffi(name_map):
    for names, fn in name_map:
        for n in names:
            TARGETS[n] = fn


@_target("lapack_ssyevd_ffi", "lapack_dsyevd_ffi", "lapack_cheevd_ffi",
         "Eigh")
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
        x = (np.tril(x) + np.tril(x, -1).conj().T if lower
             else np.triu(x) + np.triu(x, 1).conj().T)
        w, v = np.linalg.eigh(x)
        return (v, w) + tuple(
            np.zeros(shape[len(batch):], dtype=dt)
            for shape, dt in specs[2:])
    return _batch_apply(one, [a], batch, specs)


@_target("lapack_sgesdd_ffi", "lapack_dgesdd_ffi", "lapack_sgesvd_ffi",
         "lapack_cgesdd_ffi")
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


@_target("lapack_sgeev_ffi", "lapack_dgeev_ffi", "lapack_cgeev_ffi")
def _eig(op, ins):
    # sgeev results: (wr, wi, vl, vr, info); cgeev: (w, vl, vr, info).
    # jax's jnp.linalg.eig only consumes the right eigenvectors
    # (compute_left = 'N'), so vl is zero-filled.
    a = _np_in(ins[0])
    specs = _result_specs(op)
    batch = a.shape[:-2]
    nb = len(batch)
    split_real = len(specs) == 5

    def one(x):
        w, vr = np.linalg.eig(x)
        if split_real:
            vl = np.zeros(specs[2][0][nb:], dtype=specs[2][1])
            return (w.real, w.imag, vl, vr,
                    np.zeros(specs[4][0][nb:], dtype=specs[4][1]))
        vl = np.zeros(specs[1][0][nb:], dtype=specs[1][1])
        return (w, vl, vr,
                np.zeros(specs[3][0][nb:], dtype=specs[3][1]))
    return _batch_apply(one, [a], batch, specs)


@_target("metaljax_eigh")
def _metal_eigh(op, ins):
    # our own lowering: results (v, w); backend_config "L"/"U"
    a = _np_in(ins[0])
    try:
        lower = "L" in _ir.str_attr(op, "backend_config")
    except Exception:
        lower = True
    specs = _result_specs(op)
    batch = a.shape[:-2]

    def one(x):
        if not np.isfinite(x).all():
            return tuple(np.full(spec[0][len(batch):], np.nan,
                                 dtype=spec[1]) for spec in specs)
        x = (np.tril(x) + np.tril(x, -1).conj().T if lower
             else np.triu(x) + np.triu(x, 1).conj().T)
        w, v = np.linalg.eigh(x)
        return (v, w)
    return _batch_apply(one, [a], batch, specs)


@_target("metaljax_svd")
def _metal_svd(op, ins):
    # our own lowering: results (s,) or (s, u, vt) per jax svd_p order
    a = _np_in(ins[0])
    specs = _result_specs(op)
    batch = a.shape[:-2]
    nb = len(batch)
    compute_uv = len(specs) > 1

    def one(x):
        m, n = x.shape
        if not np.isfinite(x).all():
            return tuple(np.full(spec[0][nb:], np.nan, dtype=spec[1])
                         if np.issubdtype(spec[1], np.inexact)
                         else np.zeros(spec[0][nb:], dtype=spec[1])
                         for spec in specs)
        if not compute_uv:
            return (np.linalg.svd(x, compute_uv=False),)
        u_cols = specs[1][0][nb:][1]
        vt_rows = specs[2][0][nb:][0]
        u, s, vt = np.linalg.svd(
            x, full_matrices=(u_cols == m and vt_rows == n))
        return (s, u[:, :u_cols], vt[:vt_rows, :])
    return _batch_apply(one, [a], batch, specs)


@_target("metaljax_eig")
def _metal_eig(op, ins):
    # our own lowering: results (w[, vl][, vr]) per backend_config "LR"
    a = _np_in(ins[0])
    try:
        cfg = _ir.str_attr(op, "backend_config")
    except Exception:
        cfg = "R"
    specs = _result_specs(op)
    batch = a.shape[:-2]
    nb = len(batch)

    def one(x):
        if x.shape[0] == 0:
            return tuple(np.zeros(spec[0][nb:], dtype=spec[1])
                         for spec in specs)
        w, vr = np.linalg.eig(x)
        res = [w]
        if "L" in cfg:
            # left eigenvectors: eig of the adjoint, matched by order of
            # conj eigenvalues (numpy returns them consistently ordered
            # for the adjoint in conj pairs; recompute directly instead)
            wl, vl = np.linalg.eig(x.conj().T)
            # reorder columns of vl to match conj(w)
            order = np.argmin(
                np.abs(wl[None, :] - np.conj(w)[:, None]), axis=1)
            res.append(vl[:, order])
        if "R" in cfg:
            res.append(vr)
        return tuple(res)
    return _batch_apply(one, [a], batch, specs)


from metaljax.interpreter import register  # noqa: E402


@register("stablehlo.cholesky")
def _cholesky(interp, op, ins, env):
    a = _np_in(ins[0])
    lower = True
    if "lower" in op.attributes:
        lower = "true" in str(op.attributes["lower"])
    specs = _result_specs(op)
    batch = a.shape[:-2]

    def one(x):
        try:
            c = np.linalg.cholesky(
                x if lower else x.conj().swapaxes(-1, -2))
            c = c if lower else c.conj().swapaxes(-1, -2)
        except np.linalg.LinAlgError:
            c = np.full_like(x, np.nan)
        return (c,)
    return dtypes.to_mx(_batch_apply(one, [a], batch, specs)[0])


@register("stablehlo.triangular_solve")
def _triangular_solve(interp, op, ins, env):
    from scipy.linalg import solve_triangular
    a, b = (_np_in(x) for x in ins)
    attrs = {n: str(op.attributes[n]) for n in op.attributes}
    left = "true" in attrs.get("left_side", "true")
    lower = "true" in attrs.get("lower", "false")
    unit = "true" in attrs.get("unit_diagonal", "false")
    ta = attrs.get("transpose_a", "NO_TRANSPOSE")
    specs = _result_specs(op)
    batch = b.shape[:-2]
    a = np.broadcast_to(a, batch + a.shape[-2:])

    def one(aa, bb):
        m, lo = aa, lower
        if "ADJOINT" in ta:
            m, lo = m.conj().swapaxes(-1, -2), not lower
        elif "TRANSPOSE" in ta and "NO_" not in ta:
            m, lo = m.swapaxes(-1, -2), not lower
        if left:
            x = solve_triangular(m, bb, lower=lo, unit_diagonal=unit,
                                 check_finite=False)
        else:
            # X op(A) = B  <=>  op(A)^T X^T = B^T
            x = solve_triangular(m.swapaxes(-1, -2), bb.swapaxes(-1, -2),
                                 lower=not lo, unit_diagonal=unit,
                                 check_finite=False)
            x = x.swapaxes(-1, -2)
        return (x,)
    return dtypes.to_mx(_batch_apply(one, [a, b], batch, specs)[0])


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


@_target("metaljax_schur")
def _metal_schur(op, ins):
    from scipy.linalg import schur
    a = _np_in(ins[0])
    specs = _result_specs(op)
    batch = a.shape[:-2]

    def one(x):
        kind = "complex" if np.iscomplexobj(x) else "real"
        t, z = schur(x, output=kind, check_finite=False)
        return (t, z)[: len(specs)]
    return _batch_apply(one, [a], batch, specs)


@_target("metaljax_hessenberg")
def _metal_hessenberg(op, ins):
    from scipy.linalg import lapack
    a = _np_in(ins[0])
    specs = _result_specs(op)
    batch = a.shape[:-2]
    gehrd = lapack.cgehrd if np.iscomplexobj(a) else lapack.sgehrd

    def one(x):
        ht, tau, info = gehrd(x)
        return ht, tau
    return _batch_apply(one, [a], batch, specs)


@_target("metaljax_tridiagonal")
def _metal_tridiagonal(op, ins):
    from scipy.linalg import lapack
    a = _np_in(ins[0])
    try:
        lower = "L" in _ir.str_attr(op, "backend_config")
    except Exception:
        lower = True
    specs = _result_specs(op)
    batch = a.shape[:-2]
    sytrd = lapack.chetrd if np.iscomplexobj(a) else lapack.ssytrd

    def one(x):
        c, d, e, tau, info = sytrd(x, lower=int(lower))
        return c, d, e, tau
    return _batch_apply(one, [a], batch, specs)


@_target("metaljax_triangular_solve")
def _metal_tri_solve(op, ins):
    # cfg: side L/R, lower l/u, transpose t/n, conj c/-, unit 1/0,
    # perturb p/-  (see the plugin's tri_solve_rule)
    from scipy.linalg import solve_triangular
    a, b = (_np_in(x) for x in ins)
    cfg = _ir.str_attr(op, "backend_config")
    left, lower = cfg[0] == "L", cfg[1] == "l"
    trans, conj = cfg[2] == "t", cfg[3] == "c"
    unit, perturb = cfg[4] == "1", cfg[5] == "p"
    specs = _result_specs(op)
    batch = b.shape[:-2]
    a = np.broadcast_to(a, batch + a.shape[-2:])

    def one(aa, bb):
        m, lo = aa, lower
        if conj:
            m = m.conj()
        if trans:
            m, lo = m.swapaxes(-1, -2), not lo
        if perturb and not unit:
            # XLA's perturb_singular: nudge tiny diagonal entries so the
            # solve produces finite values instead of inf/nan.
            d = np.diagonal(m).copy()
            tiny = np.finfo(np.float64).tiny if m.dtype.kind == "f" else 0
            eps = max(np.abs(d).max(), 1.0) * 1e-30
            bad = np.abs(d) < eps
            if bad.any():
                m = m.copy()
                np.fill_diagonal(m, np.where(bad, eps, d))
        if left:
            x = solve_triangular(m, bb, lower=lo, unit_diagonal=unit,
                                 check_finite=False)
        else:
            x = solve_triangular(m.swapaxes(-1, -2), bb.swapaxes(-1, -2),
                                 lower=not lo, unit_diagonal=unit,
                                 check_finite=False)
            x = x.swapaxes(-1, -2)
        return (x,)
    # custom-call target convention: list of numpy arrays
    return _batch_apply(one, [a, b], batch, specs)


@_target("metaljax_tridiagonal_solve")
def _metal_tridiag_solve(op, ins):
    dl, d, du, b = (_np_in(x) for x in ins)
    perturb = "p" in _ir.str_attr(op, "backend_config")
    specs = _result_specs(op)
    batch = b.shape[:-2]

    def one(dl_, d_, du_, b_):
        n = d_.shape[0]
        m = np.zeros((n, n), dtype=d_.dtype)
        np.fill_diagonal(m, d_)
        if n > 1:
            m[np.arange(1, n), np.arange(n - 1)] = dl_[1:]
            m[np.arange(n - 1), np.arange(1, n)] = du_[:-1]
        if perturb:
            dd = np.diagonal(m).copy()
            eps = max(np.abs(dd).max(), 1.0) * 1e-30
            bad = np.abs(dd) < eps
            if bad.any():
                np.fill_diagonal(m, np.where(bad, eps, dd))
        return (np.linalg.solve(m, b_),)
    return _batch_apply(one, [dl, d, du, b], batch, specs)
