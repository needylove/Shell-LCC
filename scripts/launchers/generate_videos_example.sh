#!/bin/bash
# Generate base + finetuned videos for a set of prompts, sharded across GPUs (data-parallel).
# This is the ONLY GPU / denoise-heavy step. Output (--out_dir) then feeds the two CPU loaders:
#   python scripts/eval_detail.py      --videos_dir <out_dir> --ckpt_label <label>   # detail metrics
#   python scripts/generate_compare.py --videos_dir <out_dir>                        # side-by-side mp4
set -e
cd "$(dirname "$0")/../.."   # -> repo root (release_my/)

WAN_DIR="${WAN_DIR:-/path/to/Wan2.1-T2V-1.3B}"      # must match the finetuned model
CKPT="${CKPT:-model/t2v_run/step10/wan_full.pth}"          # finetuned checkpoint
CKPT_LABEL="${CKPT_LABEL:-step10}"                          # label/filename for the ckpt column
PROMPTS="${PROMPTS:-data/captions.txt}"
OUT_DIR="${OUT_DIR:-eval/videos}"
NPROC="${NPROC:-4}"                # number of GPUs to shard across
MAX_PROMPTS="${MAX_PROMPTS:-4}"    # 0 = all prompts
STEPS="${STEPS:-40}"

for i in $(seq 0 $((NPROC - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_videos.py \
      --wan_dir "$WAN_DIR" --full_ckpt "$CKPT" --ckpt_label "$CKPT_LABEL" \
      --prompt_file "$PROMPTS" --max_prompts "$MAX_PROMPTS" --steps "$STEPS" \
      --shard "$i/$NPROC" --out_dir "$OUT_DIR" &
done
wait
echo "all $NPROC shards done -> $OUT_DIR (manifest.json ready for eval_detail.py / generate_compare.py)"
