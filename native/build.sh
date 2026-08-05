#!/usr/bin/env bash
# Build the metaljax native engine extension (Stage 2, M0+).
#
# Same artisanal-clang pattern as plugin/build.sh: no CMake until the
# build outgrows it. Links against the INSTALLED mlx wheel's libmlx —
# the supported extension path (the wheel ships include/ + lib/) — with
# an rpath into site-packages, so the extension is only valid for the
# venv it was built against. engine.py's version handshake enforces
# that at import.
#
# NB: unlike plugin/ (limited API), nanobind needs the full CPython
# API — this artifact is per-Python-version. Wheel packaging for the
# native engine is an M6 question; for now it builds into native/build/
# and METALJAX_ENGINE=native finds it there.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-../.venv/bin/python}
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
NBINC=$($PY -c "import nanobind; print(nanobind.include_dir())")
NBSRC=$(dirname "$NBINC")/src
MLX=$SP/mlx
EXT_SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

mkdir -p build
clang++ -std=c++17 -O2 -fPIC -shared \
  -Wall -Wextra -Wno-unused-parameter \
  -I "$NBINC" -I "$PYINC" -I "$MLX/include" \
  -I "$(dirname "$NBINC")/ext/robin_map/include" \
  -DNDEBUG -DNB_COMPACT_ASSERTIONS \
  -DNB_DOMAIN=mlx \
  metaljax_native.cc "$NBSRC/nb_combined.cpp" \
  -L "$MLX/lib" -lmlx -Wl,-rpath,"$MLX/lib" \
  -undefined dynamic_lookup \
  -o "build/metaljax_native$EXT_SUFFIX"

echo "built build/metaljax_native$EXT_SUFFIX"
