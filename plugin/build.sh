#!/bin/zsh
# Build the metaljax PJRT plugin dylib.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(dirname "$PWD")
PY="$ROOT/.venv/bin/python"
PY_INCLUDE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
mkdir -p build
clang++ -std=c++20 -O2 -fPIC -shared \
  -Wall -Wno-unused-function \
  -undefined dynamic_lookup \
  -I vendor -I "$PY_INCLUDE" \
  metal_pjrt.cc \
  -o build/libmetal_pjrt.dylib
# Also place it inside the python package so non-repo installs find it.
mkdir -p "$ROOT/src/metaljax/lib"
cp build/libmetal_pjrt.dylib "$ROOT/src/metaljax/lib/"
echo "built plugin/build/libmetal_pjrt.dylib (+ copy in src/metaljax/lib/)"
