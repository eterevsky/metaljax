#!/usr/bin/env bash
# Convert the HF checkpoints the MaxText rows need into MaxText/Orbax format.
#
#   scripts/model_bench/convert_checkpoints.sh [VENV_PATH] [model ...]
#
# models: qwen3-0.6b (default), deepseek2-16b, all
#
# Reads the EXISTING ~/.cache/huggingface snapshots (--hf_model_path), so
# nothing is re-downloaded. Output:
#
#   $MAXTEXT_CKPT_ROOT/qwen3-0.6b/0/items      1.1 GB   (~10 s)
#   $MAXTEXT_CKPT_ROOT/deepseek2-16b/0/items   23  GB   (~2 min, ~62 GB peak RAM)
#
# The conversion itself forces jax_platforms=cpu (to_maxtext.py does this
# internally); metaljax is not involved.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="${METALJAX_BENCH_HOME:-$HOME/.cache/metaljax-bench}"
MAXTEXT_REPO="${MAXTEXT_REPO:-$BENCH_HOME/maxtext/repo}"
CKPT_ROOT="${MAXTEXT_CKPT_ROOT:-$BENCH_HOME/maxtext/ckpt}"
VENV="${1:-$BENCH_HOME/maxtext/venv}"
shift || true
MODELS=("${@:-qwen3-0.6b}")
[ "${MODELS[0]}" = "all" ] && MODELS=(qwen3-0.6b deepseek2-16b)

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no python at $PY -- run setup_maxtext.sh first"; exit 1; }

# newest snapshot dir for a HF repo id, or empty
snapshot() {
  local repo="${1//\//--}"
  local d="$HOME/.cache/huggingface/hub/models--$repo/snapshots"
  [ -d "$d" ] && find "$d" -mindepth 1 -maxdepth 1 -type d | head -1 || true
}

convert() {
  local name="$1" hf_id="$2"
  local snap; snap="$(snapshot "$hf_id")"
  if [ -z "$snap" ]; then
    echo "!! no local HF snapshot for $hf_id -- skipping $name"
    echo "   (download it first; this script never fetches weights)"
    return
  fi
  if [ -d "$CKPT_ROOT/$name/0/items" ]; then
    echo "==> $name already converted ($CKPT_ROOT/$name)"
    return
  fi
  echo "==> converting $name from $snap"
  ( cd "$MAXTEXT_REPO" && \
    DECOUPLE_GCLOUD=TRUE KMP_DUPLICATE_LIB_OK=TRUE JAX_PLATFORMS=cpu \
    "$PY" -m maxtext.checkpoint_conversion.to_maxtext \
      src/maxtext/configs/base.yml \
      model_name="$name" \
      base_output_directory="$CKPT_ROOT/$name" \
      hardware=cpu \
      skip_jax_distributed_system=True \
      scan_layers=true \
      --hf_model_path="$snap" \
      --simulated_cpu_devices_count=1 \
      --save_dtype=bfloat16 )
  echo "==> $name -> $CKPT_ROOT/$name/0/items"
}

for m in "${MODELS[@]}"; do
  case "$m" in
    # NB: HF_IDS maps deepseek2-16b to DeepSeek-V2-Lite (base). The manifest
    # row is the -Chat variant, which is the same architecture; --hf_model_path
    # points the converter at the local Chat snapshot.
    qwen3-0.6b)     convert qwen3-0.6b     "Qwen/Qwen3-0.6B" ;;
    qwen3-8b)       convert qwen3-8b       "Qwen/Qwen3-8B" ;;
    deepseek2-16b)  convert deepseek2-16b  "deepseek-ai/DeepSeek-V2-Lite-Chat" ;;
    *) echo "unknown model $m"; exit 1 ;;
  esac
done

echo
echo "Checkpoints under $CKPT_ROOT:"
du -sh "$CKPT_ROOT"/* 2>/dev/null || true
