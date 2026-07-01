#!/bin/bash
# Decode the VAE latents in data/vae back to mp4 -> data/recon (visual sanity check).
# Run extract_vae_example.sh first to produce data/vae.
#
# Requires the official `wan` package installed so `import wan` works:
#   pip install git+https://github.com/Wan-Video/Wan2.2.git   (or clone + pip install -e .)
# VAE_CKPT = the Wan2.1 VAE weight file (Wan2.1_VAE.pth). Default is this machine's local path;
# override it for your setup, e.g. VAE_CKPT=/path/to/Wan2.1_VAE.pth bash <this script>.
set -e
cd "$(dirname "$0")/../.."   # -> repo root (release_my/)

VAE_CKPT="${VAE_CKPT:-/path/to/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/decode_wan_vae_feature.py \
    --vae_ckpt_path "$VAE_CKPT" \
    --pt_dir data/vae \
    --out_dir data/recon
