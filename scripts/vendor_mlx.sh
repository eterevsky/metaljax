#!/usr/bin/env bash
# Stage our patched MLX as metaljax's PRIVATE, vendored Metal runtime.
#
#   scripts/vendor_mlx.sh [--build] [--python <interpreter>] [--from <sdk dir>]
#
# Input : an MLX SDK directory holding `lib/` + `include/` -- either
#         mlx-src/python/mlx (what the in-tree cmake build writes, and the
#         default) or the site-packages/mlx of a venv the fork was INSTALLED
#         into, which is the stronger choice: it makes the library the plugin
#         links and the library the Python side imports one file, compiled
#         once.
# Output: src/metaljax/lib/mlx/{lib,include} -- the tree
#         plugin-native/third_party/mlx/workspace.bzl consumes and the tree
#         hatch_build.py copies into the native wheel.
#
# THE RENAME IS THE POINT.  MLX's own install name is `@rpath/libmlx.dylib`,
# and a consumer process may well also import the public `mlx` wheel (Stage 1
# does, and so does anything the user writes).  Two images with the same
# install name in one process is a symbol-interposition lottery -- two
# Devices, two allocators, two residency sets, two command-encoder pools over
# one GPU.  So the vendored copy is renamed:
#
#     libmlx.dylib    -> libmlx_metaljax.dylib     (@rpath/libmlx_metaljax.dylib)
#     libjaccl.dylib  -> libjaccl_metaljax.dylib   (@rpath/libjaccl_metaljax.dylib)
#
# after which "the user also has mlx installed" is merely two independent
# libraries -- which is a supported state and is what coexist_test.py checks.
#
# mlx.metallib keeps its name: MLX finds it by dladdr'ing itself and looking
# for `mlx.metallib` NEXT TO the dylib (mlx/backend/metal/device.cpp,
# load_colocated_library), so it is located by directory, never by install
# name, and it must stay beside libmlx_metaljax.dylib in every layout.
set -euo pipefail

ROOT=/Users/oleg/metaljax
SRC=$ROOT/mlx-src
DEST=$ROOT/src/metaljax/lib/mlx
BUILD=0
PY=${MLX_BUILD_PYTHON:-$HOME/.cache/metaljax-bench/venvs/mlxsrc/bin/python}
FROM=$SRC/python/mlx

while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=1 ;;
    --python) PY=$2; shift ;;
    --from) FROM=$2; shift ;;
    *) echo "usage: $0 [--build] [--python <interpreter>] [--from <sdk dir>]" >&2
       exit 2 ;;
  esac
  shift
done

[ -d "$SRC" ] || { echo "no fork checkout at $SRC" >&2; exit 2; }

BRANCH=$(git -C "$SRC" rev-parse --abbrev-ref HEAD)
SHA=$(git -C "$SRC" rev-parse --short HEAD)
DIRTY=$(git -C "$SRC" status --porcelain | wc -l | tr -d ' ')
echo "fork: $BRANCH @ $SHA (dirty files: $DIRTY)"
# The stamp is THE provenance record (the version string can no longer carry
# it -- see below), so refuse to write one that could lie: a dirty checkout
# means the staged bits may not match the recorded sha.
if [ "$DIRTY" != "0" ]; then
  echo "refusing to stage from a dirty mlx-src ($DIRTY files) -- commit or stash first" >&2
  exit 2
fi
case "$BRANCH" in
  vendor/*) ;;
  *) echo "WARNING: staging from branch '$BRANCH', not a vendor/* branch" >&2 ;;
esac

if [ "$BUILD" = "1" ]; then
  # Pin metal-cpp: MLX's CMake otherwise fetches
  # https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip at configure
  # time, and Apple's server has returned 502 during this work.
  MCPP=$(ls -d "$SRC"/build/temp.*/mlx.core/_deps/metal_cpp-src 2>/dev/null | head -1)
  [ -n "$MCPP" ] || { echo "no local metal-cpp; run one build by hand first" >&2; exit 2; }
  echo "building mlx-src with $PY (metal-cpp: $MCPP)"
  # PYPI_RELEASE=1 is LOAD-BEARING: without it setup.py appends
  # .dev<date>+<sha> to the version, the dylib reports that string at run
  # time, and the plugin's ABI handshake (compiled == linked, from numeric
  # macros that can never carry a suffix) refuses the library.  The stamp
  # below records this policy; the assert after it enforces it.
  ( cd "$SRC" && PYPI_RELEASE=1 \
      CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_METAL_CPP=$MCPP" \
      CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-12} \
      uv pip install --python "$PY" --no-build-isolation -e . )
fi

for f in lib/libmlx.dylib lib/libjaccl.dylib lib/mlx.metallib include/mlx; do
  [ -e "$FROM/$f" ] || { echo "missing SDK file: $FROM/$f" >&2; exit 2; }
done
echo "source SDK: $FROM"
SRCSHA=$(shasum -a 256 "$FROM/lib/libmlx.dylib" | cut -d' ' -f1)

rm -rf "$DEST"
mkdir -p "$DEST/lib"
cp -R "$FROM/include" "$DEST/include"
cp "$FROM/lib/mlx.metallib" "$DEST/lib/mlx.metallib"
cp "$FROM/lib/libmlx.dylib" "$DEST/lib/libmlx_metaljax.dylib"
cp "$FROM/lib/libjaccl.dylib" "$DEST/lib/libjaccl_metaljax.dylib"
chmod u+w "$DEST/lib"/*.dylib

install_name_tool -id @rpath/libjaccl_metaljax.dylib "$DEST/lib/libjaccl_metaljax.dylib"
install_name_tool -id @rpath/libmlx_metaljax.dylib "$DEST/lib/libmlx_metaljax.dylib"
install_name_tool -change @rpath/libjaccl.dylib @rpath/libjaccl_metaljax.dylib \
  "$DEST/lib/libmlx_metaljax.dylib"
# libmlx carries no LC_RPATH of its own upstream -- it relies on whoever loads
# it to have one that finds libjaccl.  Give the vendored copy its own, so the
# pair is self-locating in every layout (wheel, bazel-bin, staging tree).
install_name_tool -add_rpath @loader_path "$DEST/lib/libmlx_metaljax.dylib" 2>/dev/null || true
# install_name_tool invalidates the adhoc signature on arm64.
codesign --force --sign - "$DEST/lib/libjaccl_metaljax.dylib"
codesign --force --sign - "$DEST/lib/libmlx_metaljax.dylib"

MLXVER=$("$PY" - <<'EOF' 2>/dev/null || echo unknown
import mlx.core as mx
print(mx.__version__)
EOF
)
# Enforce what the stamp is about to claim.  A dev-suffixed version here
# means the build ran without PYPI_RELEASE=1 -- the native handshake will
# refuse that library at load time, so fail NOW, not in the first venv that
# installs the wheel.  ("unknown" = $PY has no mlx: only legitimate when
# staging a prebuilt SDK with --from; the version then rides in the dylib
# itself, which the smoke run exercises.)
if [ "$MLXVER" != "0.32.0" ] && [ "$MLXVER" != "unknown" ]; then
  echo "staged mlx reports version '$MLXVER', want exactly 0.32.0" >&2
  echo "(rebuild with PYPI_RELEASE=1 -- the --build path now sets it)" >&2
  exit 2
fi

{
  echo "vendored-from: mlx-src ($BRANCH @ $SHA, dirty=$DIRTY)"
  echo "mlx-version:   $MLXVER"
  echo "sdk-dir:       $FROM"
  echo "source-libmlx: sha256 $SRCSHA  (before the install-name rename)"
  echo "staged:        $(date '+%F %T')"
  echo "install-names: libmlx_metaljax.dylib libjaccl_metaljax.dylib (private)"
  # WHY THE VERSION STRING LOOKS UNPATCHED.  The extension's
  # compiled_mlx_version() is built from MLX_VERSION_{MAJOR,MINOR,PATCH} --
  # numbers, no suffix -- and engine.py refuses the native engine unless
  # compiled == linked == mlx.__version__.  So the vendored build is made
  # with PYPI_RELEASE=1, whose version is exactly "0.32.0", and THIS STAMP
  # (fork branch@sha + sha256s) is the provenance record that the version
  # string can no longer carry.  Read it, not `pip list`, to answer "which
  # MLX is this?".
  echo "version-policy: PYPI_RELEASE=1 (exact 0.32.0) -- provenance is this stamp"
  for f in "$DEST"/lib/*; do
    echo "sha256 $(shasum -a 256 "$f" | cut -d' ' -f1)  $(basename "$f")  $(stat -f %z "$f") B"
  done
} > "$DEST/VENDOR_STAMP"

echo "--- staged $DEST"
cat "$DEST/VENDOR_STAMP"
echo "--- linkage"
otool -D "$DEST/lib/libmlx_metaljax.dylib" | tail -1
otool -L "$DEST/lib/libmlx_metaljax.dylib" | grep -E "jaccl|mlx" || true
