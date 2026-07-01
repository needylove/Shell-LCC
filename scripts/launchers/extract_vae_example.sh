#!/bin/bash
# Extract Wan2.1 VAE latents from the example videos in data/videos -> data/vae.
# Each clip becomes a .pt ([T,16,H//8,W//8]) plus a side-car .json.
#
# Requires the official `wan` package installed so `import wan` works:
#   pip install git+https://github.com/Wan-Video/Wan2.2.git   (or clone + pip install -e .)
# VAE_CKPT = the Wan2.1 VAE weight file (Wan2.1_VAE.pth). Default is this machine's local path;
# override it for your setup, e.g. VAE_CKPT=/path/to/Wan2.1_VAE.pth bash <this script>.
set -e
cd "$(dirname "$0")/../.."   # -> repo root (release_my/)

VAE_CKPT="${VAE_CKPT:-/path/to/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/extract_wan_vae_feature.py \
    --vae_ckpt_path "$VAE_CKPT" \
    --video_dir data/videos \
    --save_path data/vae
