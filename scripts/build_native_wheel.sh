#!/usr/bin/env bash
# Build THE release wheel and verify it.
#
#   scripts/build_native_wheel.sh [--dylib <path>] [--out <dir>]
#
# Since the Stage-1 retirement (0.11.6) the repository IS native-only, so
# this is a thin wrapper around `uv build`: the three things it used to work
# around are gone.  For the record, they were --
#
#   1. a STAGED source tree, because `packages = ["src/metaljax", ...]`
#      swept in the whole Stage 1 engine and hatchling's file selection is
#      static config a build hook cannot subtract from;
#   2. a GENERATED minimal `src/metaljax/__init__.py`, because the real one
#      imported the interpreter and the entire ops tree at package import;
#   3. stripping `mlx` from the staged dependency list.
#
# All three are now simply true of the tree: `src/metaljax/` is
# `__init__.py` (version only) plus `lib/`, and pyproject declares no mlx.
#
# What ships:
#   jax_plugins/metal/__init__.py   the loader
#   metaljax/__init__.py            __version__ only
#   metaljax/lib/libmetal_pjrt_native.dylib
#   metaljax/lib/mlx/lib/{libmlx_metaljax,libjaccl_metaljax}.dylib,
#                       mlx.metallib, ../VENDOR_STAMP
#
# The dylib is COPIED FROM THE GATED BINARY by default (release rule 1:
# every number in a release table comes from the release binary), not
# rebuilt by bazel.  hatch_build.py asserts __version__ == the pyproject
# version before building anything; the checks after the build are here:
# no Stage 1 module may reappear, and the wheel must carry the gated binary
# bit for bit.
set -euo pipefail

ROOT=/Users/oleg/metaljax
L=$HOME/.cache/metaljax-bench/logs/mlx-vendoring
DYLIB=""
OUT=$HOME/.cache/metaljax-bench/wheels-vendored/native

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
echo "building wheel: version $VERSION, plugin $(basename "$DYLIB")"

mkdir -p "$OUT"; rm -f "$OUT"/*.whl
( cd "$ROOT" && METALJAX_NATIVE_DYLIB="$DYLIB" \
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
VERIFY=$(mktemp -d)
trap 'rm -rf "$VERIFY"' EXIT
( cd "$VERIFY" && unzip -q "$W" metaljax/lib/libmetal_pjrt_native.dylib )
A=$(shasum -a 256 "$VERIFY/metaljax/lib/libmetal_pjrt_native.dylib" | cut -d' ' -f1)
B=$(shasum -a 256 "$DYLIB" | cut -d' ' -f1)
echo "wheel  $A"
echo "gated  $B"
[ "$A" = "$B" ] || { echo "FAIL: the wheel does not carry the gated binary"; exit 1; }
echo "identical -- the wheel carries the gated binary"
