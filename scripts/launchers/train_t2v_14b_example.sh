#!/bin/bash
# Finetune Wan2.1-T2V-14B with the manifold reward (full-param + FSDP).
#
# 14B notes (measured, see README "Training tips"):
#   - full-param + FSDP only: LoRA (rank 16) barely moves a 14B model — no usable window.
#   - the useful window is VERY short (~10 steps at effective batch 8): save every 5 steps
#     and stop early; by ~step 30 texture-hacking creeps in and quality collapses later.
#   - lower lr (e.g. 5e-6) does not widen the window on 14B — it only smooths.
# GPU memory: ~98GB/GPU with 4x FSDP shards + batch 2 (effective batch 8).
set -e
cd "$(dirname "$0")/../.."   # -> repo root (release_my/)

WAN_DIR="${WAN_DIR:-/path/to/Wan2.1-T2V-14B}"
NPROC="${NPROC:-4}"
BATCH="${BATCH:-2}"          # per-GPU; effective batch = BATCH x NPROC

torchrun --nproc_per_node="$NPROC" scripts/train_T2V_model.py \
    --mode full --full_parallel fsdp --no_drift \
    --wan_dir "$WAN_DIR" \
    --captions data/captions.txt \
    --manifold_ckpt model/shell_lcc.pth \
    --batch "$BATCH" --seed 0 \
    --lr 1e-5 --iters 20 --save_interval 5 \
    --out_dir outputs/t2v_14b_run
