#!/usr/bin/env bash
# Build THE release wheel: native plugin + vendored Metal runtime, and NO
# Stage 1 Python engine.
#
#   scripts/build_native_wheel.sh [--dylib <path>] [--out <dir>]
#
# Oleg, 2026-08-18: "the release wheel is native-only -- the old Python
# implementation is fully out of release scope and should not ship in the
# wheel."  Two things make that awkward to express in pyproject.toml, and
# this script is the answer to both:
#
#  1. `packages = ["src/metaljax", ...]` sweeps in every module under it,
#     and hatchling's file selection is static config, not something a
#     build hook can subtract from.  So the wheel is built from a STAGED
#     source tree that simply does not contain the Stage 1 modules.  The
#     alternative -- moving files out of src/ during the build, the way the
#     hook already parks the trampoline -- would mutate a tree other agents
#     are running against.  Staging touches nothing.
#
#  2. `src/metaljax/__init__.py` imports the interpreter and the whole ops
#     tree at package import, so shipping it verbatim over an excluded
#     engine would make `import metaljax` an ImportError (and, in a venv
#     with no mlx, an ImportError about mlx).  The staged tree therefore
#     carries a GENERATED minimal __init__ that keeps the one thing the
#     package still owes its users -- __version__, parsed out of the real
#     file so it cannot drift -- and nothing else.
#
# What ships:
#   jax_plugins/metal/__init__.py   the loader (its Stage 1 branches are
#                                   dead code here: the native dylib is
#                                   found first, by find_spec, and the
#                                   module is never imported)
#   metaljax/__init__.py            generated, __version__ only
#   metaljax/lib/libmetal_pjrt_native.dylib
#   metaljax/lib/mlx/lib/{libmlx_metaljax,libjaccl_metaljax}.dylib,
#                       mlx.metallib, ../VENDOR_STAMP
#
# The dylib is COPIED FROM THE GATED BINARY by default (release rule 1:
# every number in a release table comes from the release binary), not
# rebuilt by bazel.
#
# `mlx` is dropped from the staged dependency list: no packaged module
# imports it any more, and the Metal runtime is inside the wheel.  A user
# who installs mlx anyway gets two independent libraries, which is what the
# private install name is for (coexist tests).
set -euo pipefail

ROOT=/Users/oleg/metaljax
L=$HOME/.cache/metaljax-bench/logs/mlx-vendoring
DYLIB=""
OUT=$HOME/.cache/metaljax-bench/wheels-vendored/native
STAGE=$HOME/.cache/metaljax-bench/wheel-stage-native

while [ $# -gt 0 ]; do
  case "$1" in
    --dylib) DYLIB=$2; shift ;;
    --out) OUT=$2; shift ;;
    *) echo "usage: $0 [--dylib <path>] [--out <dir>]" >&2; exit 2 ;;
  esac
  shift
done
# Default to the gated frozen binary only when --dylib was not given, so an
# explicit --dylib works even without a frozen-path.txt on disk.
[ -n "$DYLIB" ] || DYLIB=$(cat "$L/frozen-path.txt")

[ -e "$DYLIB" ] || { echo "no plugin dylib at $DYLIB" >&2; exit 2; }
MLXDIR=$ROOT/src/metaljax/lib/mlx
[ -e "$MLXDIR/VENDOR_STAMP" ] || { echo "no vendored MLX at $MLXDIR (scripts/vendor_mlx.sh)" >&2; exit 2; }

VERSION=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$ROOT/src/metaljax/__init__.py" | head -1)
[ -n "$VERSION" ] || { echo "could not parse __version__" >&2; exit 2; }
# The wheel's dist version comes from pyproject; __version__ is what the
# installed package reports.  They drifted once (the 0.11.3-in-0.11.4-wheel
# cosmetic bug) -- assert they agree before building anything.
PYVER=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$ROOT/pyproject.toml" | head -1)
[ "$VERSION" = "$PYVER" ] || {
  echo "__version__ ($VERSION) != pyproject version ($PYVER)" >&2; exit 2; }
echo "staging native-only wheel: version $VERSION, plugin $(basename "$DYLIB")"

rm -rf "$STAGE"; mkdir -p "$STAGE/src/metaljax/lib" "$STAGE/src"
cp "$ROOT/pyproject.toml" "$ROOT/hatch_build.py" "$ROOT/README.md" \
   "$ROOT/LICENSE" "$STAGE/"
cp -R "$ROOT/src/jax_plugins" "$STAGE/src/jax_plugins"
find "$STAGE/src" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# `mlx` out of the dependency list -- nothing packaged imports it.
python3 - "$STAGE/pyproject.toml" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
new = re.sub(r'"mlx>=[^"]*",\s*', "", s, count=1)
if new == s:
    raise SystemExit("expected an mlx dependency to drop")
open(p, "w").write(new)
EOF

cat > "$STAGE/src/metaljax/__init__.py" <<EOF
"""metaljax: a Metal backend for JAX (native PJRT plugin).

This wheel carries the Stage 2 NATIVE plugin together with its own private
Metal runtime under \`metaljax/lib/\`.  The Stage 1 Python engine -- the
StableHLO interpreter, tape, msl_scan codegen and op handlers -- is
deliberately NOT packaged (Oleg, 2026-08-18: the release wheel is
native-only).  \`jax_plugins.metal\` locates the plugin with find_spec and
never imports this package on the native path, so nothing here is on any
hot path; \`__version__\` is kept because it is a published surface.

Generated by scripts/build_native_wheel.sh -- do not edit.
"""

__version__ = "$VERSION"
__all__ = ["__version__"]
EOF

echo "--- staged tree"
find "$STAGE/src" -type f | sed "s|$STAGE/||" | sort

mkdir -p "$OUT"; rm -f "$OUT"/*.whl
( cd "$STAGE" && METALJAX_WHEEL_PLUGIN=native \
    METALJAX_NATIVE_DYLIB="$DYLIB" \
    METALJAX_VENDORED_MLX="$MLXDIR" \
    uv build --wheel --out-dir "$OUT" )

W=$(ls "$OUT"/*.whl)
echo "--- built $W"
echo "size: $(stat -f %z "$W") B  ($(du -h "$W" | cut -f1))"
shasum -a 256 "$W"
echo "--- wheel contents"
unzip -l "$W" | sed -n '1,40p'
echo "--- Stage 1 modules in the wheel (want: none)"
if unzip -l "$W" | grep -E "metaljax/(engine|interpreter|tape|msl_scan|qmm|moe|sdpa|dtypes|_ir|compile_options|diagnostics)\.py|metaljax/ops/"; then
  echo "FAIL: Stage 1 modules present"
  exit 1
fi
echo "none -- clean"
echo "--- the dylib in the wheel vs the gated binary"
rm -rf "$STAGE/verify"; mkdir -p "$STAGE/verify"
( cd "$STAGE/verify" && unzip -q "$W" metaljax/lib/libmetal_pjrt_native.dylib )
A=$(shasum -a 256 "$STAGE/verify/metaljax/lib/libmetal_pjrt_native.dylib" | cut -d' ' -f1)
B=$(shasum -a 256 "$DYLIB" | cut -d' ' -f1)
echo "wheel  $A"
echo "gated  $B"
[ "$A" = "$B" ] || { echo "FAIL: the wheel does not carry the gated binary"; exit 1; }
echo "identical -- the wheel carries the gated binary"
