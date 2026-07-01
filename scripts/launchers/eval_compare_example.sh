#!/bin/bash
# Evaluate PRE-GENERATED videos (run generate_videos_example.sh first) — CPU only, seconds, no GPU.
#   - eval_detail      : detail metrics (lap / hf / change) -> eval/results.jsonl
#   - generate_compare : side-by-side comparison videos     -> <VIDEOS_DIR>/*_compare.mp4
#   - make_montage     : base|ckpt detail montage PNGs       -> <VIDEOS_DIR>/*_montage.png
# VIDEOS_DIR must match the --out_dir used by generate_videos_example.sh; CKPT_LABEL its --ckpt_label.
set -e
cd "$(dirname "$0")/../.."   # -> repo root (release_my/)

VIDEOS_DIR="${VIDEOS_DIR:-eval/videos}"     # output dir of generate_videos_example.sh (has manifest.json)
CKPT_LABEL="${CKPT_LABEL:-step10}"          # the finetuned column to compare against "base"
ZOOM="${ZOOM:-0.4}"                         # montage: 0 = full frame; 0<z<1 = zoom into central detail

# 1. detail metrics vs base (lap/hf/change). NOTE: lap/hf are proxies, only meaningful when
#    `change` is small; large content change -> compare the videos by eye instead.
python scripts/eval_detail.py --videos_dir "$VIDEOS_DIR" --ckpt_label "$CKPT_LABEL" \
    --tag "$CKPT_LABEL" --results eval/results.jsonl

# 2. side-by-side comparison videos (base | ckpt)
python scripts/generate_compare.py --videos_dir "$VIDEOS_DIR"

# 3. detail montage PNGs (base | ckpt, center-zoomed) — the eyeball companion to the proxy metrics
python scripts/make_montage.py --videos_dir "$VIDEOS_DIR" --zoom "$ZOOM"

echo "metrics -> eval/results.jsonl ; *_compare.mp4 + *_montage.png -> $VIDEOS_DIR/"
