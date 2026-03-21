#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/mnt/bigdisk/cache_GTRS}"
MODEL_DIR="${MODEL_DIR:-$HOME/navsim_workspace/dataset/models}"

mkdir -p "$CACHE_DIR" "$MODEL_DIR"

# Official URLs from README
DP_URL="https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/gtrs_dp.ckpt"
DENSE_URL="https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/gtrs_dense_vov.ckpt"
VOV_URL="https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/dd3d_det_final.pth"

DP_CKPT="$CACHE_DIR/gtrs_dp.ckpt"
DENSE_CKPT="$CACHE_DIR/gtrs_dense_vov.ckpt"
VOV_CKPT="$MODEL_DIR/dd3d_det_final.pth"

fetch() {
  local url="$1" out="$2" name="$3"
  if [ -f "$out" ]; then
    echo "[OK] $name exists: $out"
    return 0
  fi
  echo "[DL] $name -> $out"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$out" "$url"
  else
    curl -L --fail -o "$out" "$url"
  fi
}

fetch "$DP_URL" "$DP_CKPT" "gtrs_dp.ckpt (Diffusion Policy)"
fetch "$DENSE_URL" "$DENSE_CKPT" "gtrs_dense_vov.ckpt (GTRS-Dense)"
fetch "$VOV_URL" "$VOV_CKPT" "dd3d_det_final.pth (VOV backbone)"

echo ""
echo "[DONE]"
echo "DP_CKPT=$DP_CKPT"
echo "DENSE_CKPT=$DENSE_CKPT"
echo "VOV_CKPT=$VOV_CKPT"
