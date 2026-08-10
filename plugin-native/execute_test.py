#!/usr/bin/env python
"""Differential test for the native PJRT plugin's executor (phase 2, P2).

Every expression below is evaluated twice: once through the bazel-built
native plugin (`JAX_PLATFORMS=metal`, `METALJAX_PLUGIN_PATH` pointing at the
dylib) and once through jax on the CPU backend, in a subprocess of its own.
The CPU answer is the bar -- this is the same doctrine the Stage 1 suite runs
under -- so a case that disagrees is a failure here whatever the two engines
have in common.

Run it from the repo venv:

    plugin-native/../.venv/bin/python plugin-native/execute_test.py

Exit status is the number of failing cases (0 = all good).  A `--reference
<path>` invocation is how the child process computes the CPU side; nothing
else needs it.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_DYLIB = _HERE / "bazel-bin" / "metal" / "libmetal_pjrt_native.dylib"


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------
#
# (name, fn, args, rtol, atol).  Integer and boolean results are compared
# EXACTLY (the tolerances are ignored for them): a gather of bits that is one
# ULP out is a bug, not rounding.


def _rand(shape, seed, dtype=np.float32):
    return np.asarray(np.random.RandomState(seed).standard_normal(shape),
                      dtype=dtype)


def _cases():
    import jax
    import jax.numpy as jnp

    f = np.float32
    # A tolerance band per class of arithmetic, not one global number.
    EXACT = (0.0, 0.0)
    F32 = (1e-6, 1e-6)
    # Contractions accumulate in a different order from the CPU's, so they get
    # a band of their own.  Not a WIDE one: the M5's low-precision matmul path
    # (CLAUDE.md's "M5 GPU MLX f32 matmul is low-precision", ~4e-3) is off
    # here, because src/jax_plugins/metal/__init__.py pins
    # MLX_METAL_GPU_ARCH before dlopening the plugin -- on the native branch
    # too.  Measured on 512x512: 7.6e-7 relative against an f64 reference,
    # where jax-CPU itself is 1.3e-6.
    DOT = (1e-5, 1e-5)
    HALF = (5e-3, 5e-3)

    cases = [
        # milestone zero, exactly as CLAUDE.md states it
        ("2*x int32", lambda x: 2 * x, [np.array([1, 2, 3], np.int32)], *EXACT),
        ("elementwise chain f32", lambda x, y: jnp.tanh(x * y + x) - y / 3.0,
         [_rand((4, 5), 0), _rand((4, 5), 1)], *F32),
        ("unary mix f32",
         lambda x: (jnp.exp(x) + jnp.tanh(x) * jax.lax.rsqrt(jnp.abs(x) + 1.0)
                    + jnp.log1p(jnp.abs(x)) + jnp.floor(x) + jnp.sign(x)),
         [np.linspace(-3, 3, 32, dtype=f)], *F32),
        ("logistic/erf/sqrt f32",
         lambda x: jax.nn.sigmoid(x) + jax.scipy.special.erf(x) + jnp.sqrt(
             jnp.abs(x)),
         [np.linspace(-4, 4, 17, dtype=f)], *F32),
        ("elementwise chain f16", lambda x: jnp.tanh(x) * 2 - 1,
         [np.arange(12, dtype=np.float16).reshape(3, 4) / 8],
         *HALF),
        ("elementwise chain bf16",
         lambda x: (x * x + x).astype(jnp.float32),
         [np.arange(12, dtype=np.float32).reshape(3, 4).astype(jnp.bfloat16)],
         *HALF),
        ("integer arithmetic",
         lambda x, y: (x * y - x // 3 + y % 5, x & y, x | y, x ^ y, -x),
         [np.arange(-6, 6, dtype=np.int32),
          np.arange(1, 13, dtype=np.int32)], *EXACT),
        ("unsigned arithmetic", lambda x: x * 3 + 1,
         [np.arange(8, dtype=np.uint8)], *EXACT),
        ("bool logic", lambda a, b: (a & b, a | b, ~a, a ^ b),
         [np.array([True, False, True, False]),
          np.array([True, True, False, False])], *EXACT),

        # reductions
        ("sum over one axis", lambda x: x.sum(0),
         [np.arange(12, dtype=f).reshape(3, 4)], *F32),
        ("sum over two axes", lambda x: x.sum((0, 2)),
         [_rand((2, 3, 4), 2)], *F32),
        ("max/min/prod", lambda x: (x.max(1), x.min(0), jnp.prod(x, 1)),
         [_rand((3, 4), 3)], *F32),
        ("any/all", lambda x: ((x > 0).any(1), (x > 0).all(0), (x > 0).any()),
         [_rand((3, 4), 4)], *EXACT),
        ("sum of everything", lambda x: x.sum(),
         [_rand((5, 6), 5)], *F32),
        ("argmax / argmin", lambda x: (jnp.argmax(x, 1), jnp.argmin(x, 0)),
         [_rand((3, 4), 6)], *EXACT),
        ("argmax with a NaN", lambda x: jnp.argmax(x),
         [np.array([1.0, np.nan, 3.0], f)], *EXACT),
        ("integer sum", lambda x: x.sum(1),
         [np.arange(12, dtype=np.int32).reshape(3, 4)], *EXACT),

        # shape ops
        ("transpose + reshape", lambda x: x.T.reshape(4, 3),
         [np.arange(12, dtype=f).reshape(3, 4)], *EXACT),
        ("transpose rank 3", lambda x: jnp.transpose(x, (2, 0, 1)),
         [_rand((2, 3, 4), 7)], *EXACT),
        ("broadcast", lambda x, y: x + y[:, None],
         [_rand((3, 4), 8), _rand((3,), 9)], *F32),
        ("broadcast unsorted dims",
         lambda x: jnp.broadcast_to(x.T[None], (2, 4, 3)),
         [np.arange(12, dtype=f).reshape(3, 4)], *EXACT),
        ("strided slice", lambda x: x[1:4, ::2],
         [np.arange(24, dtype=f).reshape(4, 6)], *EXACT),
        ("concatenate", lambda x, y: jnp.concatenate([x, y], 0),
         [np.ones((2, 3), f), np.zeros((1, 3), f)], *EXACT),
        ("concatenate axis 1", lambda x, y: jnp.concatenate([x, y], 1),
         [np.ones((2, 3), f), np.zeros((2, 2), f)], *EXACT),
        ("iota", lambda x: x + jnp.arange(4, dtype=jnp.float32),
         [np.zeros((3, 4), f)], *EXACT),
        ("pad", lambda x: jnp.pad(x, ((1, 2), (0, 1))),
         [np.arange(6, dtype=f).reshape(2, 3)], *EXACT),
        ("pad with interior dilation",
         lambda x: jax.lax.pad(x, np.float32(0), ((0, 0, 1), (1, 1, 2))),
         [np.arange(6, dtype=f).reshape(2, 3)], *EXACT),
        ("pad with negative edges (a crop)",
         lambda x: jax.lax.pad(x, np.float32(9), ((-1, 0, 0), (0, -1, 0))),
         [np.arange(6, dtype=f).reshape(2, 3)], *EXACT),
        ("rank-0 scalar", lambda x: x * 2 + 1,
         [np.float32(2.5)], *F32),
        ("empty array", lambda x: x * 2,
         [np.zeros((0, 3), f)], *EXACT),

        # contractions
        ("matmul 2D", lambda a, b: a @ b,
         [_rand((4, 5), 10), _rand((5, 6), 11)], *DOT),
        ("batched matmul", lambda a, b: jnp.einsum("bij,bjk->bik", a, b),
         [_rand((2, 3, 4), 12), _rand((2, 4, 5), 13)], *DOT),
        ("dot with batch and free dims",
         lambda a, b: jnp.einsum("bik,bjk->bij", a, b),
         [_rand((2, 3, 7), 14), _rand((2, 5, 7), 15)], *DOT),
        ("vector dot", lambda a, b: a @ b,
         [_rand((16,), 16), _rand((16,), 17)], *DOT),
        ("matmul 128x128", lambda a, b: a @ b,
         [_rand((128, 128), 31), _rand((128, 128), 32)], *DOT),
        ("int32 matmul", lambda a, b: a @ b,
         [np.arange(12, dtype=np.int32).reshape(3, 4),
          np.arange(20, dtype=np.int32).reshape(4, 5)], *EXACT),
        ("int8 matmul", lambda a, b: (a.astype(jnp.int32)
                                      @ b.astype(jnp.int32)),
         [np.arange(-6, 6, dtype=np.int8).reshape(3, 4),
          np.arange(-10, 10, dtype=np.int8).reshape(4, 5)], *EXACT),

        # selection
        ("select / compare / clamp",
         lambda x: jnp.where(x > 0, jnp.clip(x, -1.0, 1.0), -x),
         [np.linspace(-3, 3, 21, dtype=f)], *F32),
        ("comparison ladder",
         lambda a, b: (a == b, a != b, a < b, a <= b, a > b, a >= b),
         [np.array([1.0, 2.0, 3.0, np.nan], f),
          np.array([1.0, 3.0, 2.0, 1.0], f)], *EXACT),
        ("maximum / minimum", lambda a, b: (jnp.maximum(a, b),
                                            jnp.minimum(a, b)),
         [_rand((7,), 18), _rand((7,), 19)], *F32),

        # constants and converts
        ("f32 constant", lambda x: x * jnp.array([1.5, 2.5, 3.5], jnp.float32),
         [np.ones(3, f)], *F32),
        ("bf16 constant",
         lambda x: (x * jnp.array([0.5, 1.25, -2.75], jnp.bfloat16)
                    ).astype(jnp.float32),
         [np.ones(3, np.float32).astype(jnp.bfloat16)], *HALF),
        ("splat constant", lambda x: x + 7.25,
         [np.zeros((3, 4), f)], *F32),
        ("bf16 splat constant",
         lambda x: (x + jnp.bfloat16(3.5)).astype(jnp.float32),
         [np.zeros((2, 3), np.float32).astype(jnp.bfloat16)], *HALF),
        ("bool constant",
         lambda x: x & jnp.array([True, False, True]),
         [np.array([True, True, True])], *EXACT),
        ("int constant", lambda x: x * jnp.array([2, 3, 4], jnp.int32),
         [np.ones(3, np.int32)], *EXACT),
        ("convert chain",
         lambda x: (x.astype(jnp.int32).astype(jnp.float32)
                    .astype(jnp.bfloat16).astype(jnp.float32)),
         [np.linspace(-5, 5, 9, dtype=f)], *EXACT),
        ("bool convert", lambda x: (x > 0).astype(jnp.float32),
         [np.linspace(-1, 1, 9, dtype=f)], *EXACT),

        # complex64: in the dtype table because its storage IS its bits
        ("complex arithmetic",
         lambda z: (z * z + z, jnp.abs(z), jnp.real(z), jnp.imag(z)),
         [np.array([1 + 1j, -2 - 3j, 0 + 0j, 4 - 0.5j], np.complex64)],
         1e-6, 1e-6),
        ("complex from parts", lambda a, b: jnp.abs(a + 1j * b),
         [_rand((5,), 20), _rand((5,), 21)], 1e-6, 1e-6),

        # program shapes
        ("several outputs", lambda x, y: (x + y, x - y, x * y),
         [_rand((3,), 22), _rand((3,), 23)], *F32),
        ("identity", lambda x: x, [_rand((4,), 24)], *EXACT),
        ("argument returned twice", lambda x: (x, x),
         [_rand((4,), 25)], *EXACT),
        ("constant output", lambda x: (x, jnp.arange(3, dtype=jnp.float32)),
         [_rand((4,), 26)], *EXACT),
        ("no argument", lambda: jnp.arange(5, dtype=jnp.float32) * 2,
         [], *EXACT),

        # a realistic little block: the shapes a model's forward pass has
        ("dense + norm + gelu",
         lambda x, w, b: jax.nn.gelu(
             (x @ w + b) / jnp.sqrt((x @ w + b).var(-1, keepdims=True) + 1e-5)),
         [_rand((8, 16), 27), _rand((16, 32), 28), _rand((32,), 29)], *DOT),
        ("softmax", lambda x: jax.nn.softmax(x, axis=-1),
         [_rand((4, 9), 30)], *F32),
    ]
    return cases


# Programs that must DECLINE, with the op the message has to name.  A decline
# is a feature here: the plugin refuses whole programs it cannot lower, and it
# says which op stopped it.
def _declines():
    import jax
    import jax.numpy as jnp

    return [
        ("sort", lambda x: jnp.sort(x), [np.array([3.0, 1.0, 2.0], np.float32)],
         "stablehlo.sort"),
        ("gather", lambda x, i: x[i],
         [np.arange(6, dtype=np.float32), np.array([0, 2], np.int32)],
         "stablehlo.gather"),
        ("while loop", lambda x: jax.lax.fori_loop(0, 4, lambda i, c: c + i, x),
         [np.float32(0.0)], "stablehlo.while"),
    ]


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _canonical(value):
    """An output as (dtype-class, widened array), comparable across backends.

    Widening to f64/c128 is exact for every dtype this backend has, so the
    comparison never loses a bit that the tolerance then has to hide; the
    class is what decides whether a tolerance applies at all.
    """
    arr = np.asarray(value)
    if arr.dtype.kind == "c":
        return "c", arr.astype(np.complex128)
    if arr.dtype.kind in "fV":       # 'V' is how ml_dtypes' bfloat16 presents
        return "f", arr.astype(np.float64)
    if arr.dtype.kind == "b":
        return "b", arr.astype(np.int64)
    return "i", arr.astype(np.int64)


def _flatten(out):
    if isinstance(out, (tuple, list)):
        return [np.asarray(v) for v in out]
    return [np.asarray(out)]


def _compare(name, got, want, rtol, atol):
    """Largest error, or a message.  Returns (ok, detail)."""
    if len(got) != len(want):
        return False, f"{len(got)} outputs, the CPU backend gave {len(want)}"
    worst = 0.0
    for j, (g, (wk, wa)) in enumerate(zip(got, want)):
        gk, ga = _canonical(g)
        if gk != wk:
            return False, f"output {j} is {gk!r}, the CPU backend gave {wk!r}"
        if ga.shape != wa.shape:
            return False, f"output {j} is {ga.shape}, CPU gave {wa.shape}"
        if gk in "ib":
            if not np.array_equal(ga, wa):
                bad = int(np.count_nonzero(ga != wa))
                return False, f"output {j}: {bad} of {ga.size} elements differ"
            continue
        # NaNs must land in the same places, then the finite parts compare.
        if not np.array_equal(np.isnan(ga), np.isnan(wa)):
            return False, f"output {j}: NaNs are in different places"
        finite = ~np.isnan(ga)
        if not np.array_equal(np.isinf(ga[finite]), np.isinf(wa[finite])):
            return False, f"output {j}: infinities are in different places"
        both = finite & ~np.isinf(ga)
        if not np.any(both):
            continue
        err = np.abs(ga[both] - wa[both])
        scale = atol + rtol * np.abs(wa[both])
        rel = np.max(err / np.maximum(scale, 1e-300))
        worst = max(worst, float(np.max(err)))
        if rel > 1.0:
            return False, (f"output {j}: max |error| {float(np.max(err)):.3e} "
                           f"exceeds atol {atol} + rtol {rtol}")
    return True, worst


# --------------------------------------------------------------------------
# the two halves
# --------------------------------------------------------------------------


def write_reference(path):
    """Run every case on the CPU backend and save the answers."""
    import jax

    saved = {}
    for i, (name, fn, args, _rtol, _atol) in enumerate(_cases()):
        outs = _flatten(jax.jit(fn)(*args))
        saved[f"n{i}"] = np.asarray(len(outs))
        for j, out in enumerate(outs):
            kind, arr = _canonical(out)
            saved[f"k{i}_{j}"] = np.asarray(kind)
            saved[f"v{i}_{j}"] = arr
    np.savez(path, **saved)


def read_reference(path, index):
    """The CPU answers for one case, as (kind, widened array) pairs.

    The kind is stored rather than re-derived: canonicalization widens a bool
    to an integer array, so reading it back would say "integer" for a case
    whose result really is boolean, and a dtype disagreement between the
    backends would then go unnoticed.
    """
    data = np.load(path)
    n = int(data[f"n{index}"])
    return [(str(data[f"k{index}_{j}"]), data[f"v{index}_{j}"])
            for j in range(n)]


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--reference":
        write_reference(sys.argv[2])
        return 0

    os.environ.setdefault("METALJAX_PLUGIN_PATH", str(_DEFAULT_DYLIB))
    dylib = pathlib.Path(os.environ["METALJAX_PLUGIN_PATH"])
    if not dylib.exists():
        sys.exit(f"plugin dylib not found: {dylib}")
    os.environ["JAX_PLATFORMS"] = "metal"
    print(f"plugin: {dylib} ({dylib.stat().st_size / 1e6:.1f} MB)")

    # The CPU answers come from a subprocess: in this one, jax sees only the
    # metal platform, and pointing it at both would let a case silently
    # compare the metal backend against itself.
    ref_path = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"metaljax-native-reference-{os.getpid()}.npz")
    child = dict(os.environ)
    child["JAX_PLATFORMS"] = "cpu"
    child.pop("METALJAX_PLUGIN_PATH", None)
    print("computing CPU references ...", flush=True)
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--reference",
         str(ref_path)],
        env=child, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit("the CPU reference run failed")

    import jax  # noqa: E402  (after JAX_PLATFORMS is fixed)
    import jax.numpy as jnp  # noqa: E402

    failures = []
    print(f"\n{'case':<32} {'max error':>12}  result")
    print("-" * 62)
    for i, (name, fn, args, rtol, atol) in enumerate(_cases()):
        try:
            got = _flatten(jax.jit(fn)(*args))
            want = read_reference(ref_path, i)
            ok, detail = _compare(name, got, want, rtol, atol)
        except BaseException as exc:  # noqa: BLE001 - report and continue
            ok, detail = False, f"{type(exc).__name__}: " \
                                f"{str(exc).splitlines()[0][:110]}"
        if ok:
            print(f"{name:<32} {detail:>12.3e}  ok")
        else:
            print(f"{name:<32} {'-':>12}  FAIL: {detail}")
            failures.append(name)

    print("\ndeclines (the message must name the op)")
    print("-" * 62)
    for name, fn, args, op in _declines():
        try:
            jax.jit(fn)(*args)
        except BaseException as exc:  # noqa: BLE001
            msg = str(exc)
            if "UNIMPLEMENTED" in msg and op in msg:
                print(f"{name:<32} {'':>12}  ok ({op})")
                continue
            print(f"{name:<32} {'':>12}  FAIL: {msg.splitlines()[0][:90]}")
            failures.append(f"decline {name}")
            continue
        print(f"{name:<32} {'':>12}  FAIL: it did not decline")
        failures.append(f"decline {name}")

    # XLA's no-alias contract: an output may not share a buffer with an input.
    # The tape works this out statically (metal_lowering.cc's copy rule), and
    # a jitted identity is the case that exercises it.
    print("\ncontracts")
    print("-" * 62)
    try:
        src = jax.device_put(np.arange(4, dtype=np.float32))
        out = jax.jit(lambda a: a)(src)
        same = src.unsafe_buffer_pointer() == out.unsafe_buffer_pointer()
        print(f"{'identity returns a fresh buffer':<32} {'':>12}  "
              f"{'FAIL: aliased' if same else 'ok'}")
        if same:
            failures.append("no-alias contract")
    except BaseException as exc:  # noqa: BLE001
        print(f"{'identity returns a fresh buffer':<32} {'':>12}  "
              f"FAIL: {exc}")
        failures.append("no-alias contract")

    # Every dtype the transfer path claims, host -> device -> host, bit exact,
    # plus a negative-stride view (whose logical first element is not its
    # lowest address).
    dtypes = [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8,
              np.uint16, np.uint32, np.uint64, np.float16, np.float32,
              np.complex64]
    for dt in dtypes:
        src = (np.arange(6) % 2).astype(dt) if dt is np.bool_ else \
            np.arange(6).astype(dt)
        # With x64 off jax canonicalizes 64-bit integers down to 32 on the way
        # in, so the dtype to expect back is jax's answer, not numpy's.
        want_dtype = np.dtype(jax.dtypes.canonicalize_dtype(src.dtype))
        try:
            back = np.asarray(jax.device_put(src))
            ok = back.dtype == want_dtype and np.array_equal(
                back, src.astype(want_dtype))
        except BaseException as exc:  # noqa: BLE001
            ok, back = False, exc
        label = f"round-trip {np.dtype(dt).name}"
        print(f"{label:<32} {'':>12}  {'ok' if ok else f'FAIL: {back}'}")
        if not ok:
            failures.append(label)
    try:
        import ml_dtypes
        src = np.arange(6, dtype=np.float32).astype(ml_dtypes.bfloat16)
        back = np.asarray(jax.device_put(src))
        ok = back.dtype == src.dtype and np.array_equal(
            back.astype(np.float32), src.astype(np.float32))
    except BaseException as exc:  # noqa: BLE001
        ok, back = False, exc
    print(f"{'round-trip bfloat16':<32} {'':>12}  "
          f"{'ok' if ok else f'FAIL: {back}'}")
    if not ok:
        failures.append("round-trip bfloat16")
    try:
        src = np.arange(12, dtype=np.float32).reshape(3, 4)[::-1, ::-2]
        back = np.asarray(jax.device_put(src))
        ok = np.array_equal(back, src)
    except BaseException as exc:  # noqa: BLE001
        ok, back = False, exc
    print(f"{'round-trip negative strides':<32} {'':>12}  "
          f"{'ok' if ok else f'FAIL: {back}'}")
    if not ok:
        failures.append("round-trip negative strides")

    # One executable, called from many threads at once.  MLX streams are
    # thread-bound; the plugin's answer is a cross-thread-evaluable stream per
    # entering thread (metal_stream.cc), and this is what exercises it.
    import concurrent.futures

    try:
        # An all-positive sum on purpose: a cancelling one would measure f32
        # summation order, not the threading.
        fn = jax.jit(lambda x: jnp.abs(jnp.tanh(x * 2)).sum())
        xs = [_rand((64, 64), 100 + i) for i in range(32)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda x: float(np.asarray(fn(x))), xs))
        want = [float(np.abs(np.tanh(x.astype(np.float64) * 2)).sum())
                for x in xs]
        worst = max(abs(a - b) / max(abs(b), 1.0) for a, b in zip(got, want))
        ok = worst < 1e-5
        detail = "ok" if ok else f"FAIL: max relative error {worst:.2e}"
    except BaseException as exc:  # noqa: BLE001
        ok, detail = False, f"FAIL: {str(exc).splitlines()[0][:90]}"
    print(f"{'32 executes on 8 threads':<32} {'':>12}  {detail}")
    if not ok:
        failures.append("threaded execute")

    # The f64 policy, STRICT exactly as Stage 1 defines it: an f64 buffer
    # passes through (stored f32) and an f64 COMPUTATION declines naming the
    # element type.  In a subprocess because enabling x64 is a global switch.
    probe = (
        "import os, numpy as np, jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "x = np.arange(4, dtype=np.float64)\n"
        "back = np.asarray(jax.device_put(x))\n"
        "assert back.dtype == np.float64 and np.array_equal(back, x), back\n"
        "try:\n"
        "    jax.jit(lambda a: a * 2)(x)\n"
        "except Exception as e:\n"
        "    assert 'element type f64' in str(e), str(e)\n"
        "    print('ok')\n"
        "else:\n"
        "    raise SystemExit('an f64 computation did not decline')\n")
    proc = subprocess.run([sys.executable, "-c", probe], env=dict(os.environ),
                          capture_output=True, text=True)
    ok = proc.returncode == 0 and "ok" in proc.stdout
    print(f"{'f64 passes through, f64 math not':<32} {'':>12}  "
          f"{'ok' if ok else 'FAIL: ' + (proc.stderr or proc.stdout).strip()[-90:]}")
    if not ok:
        failures.append("f64 policy")

    try:
        ref_path.unlink()
    except OSError:
        pass

    if failures:
        print(f"\n{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("\nall cases match the CPU backend")
    return 0


if __name__ == "__main__":
    # jax is imported only inside main(), after JAX_PLATFORMS is fixed: an
    # import up here would pin the platform before the child process could
    # choose the CPU one.
    sys.exit(main())
