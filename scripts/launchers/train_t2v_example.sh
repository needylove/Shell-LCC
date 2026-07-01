#!/bin/bash
# Finetune a Wan2.1-T2V model with the (frozen) Shell-LCC manifold reward.
# Prereqs:
#   - a trained manifold : model/shell_lcc.pth   (from train_manifold_example.sh)
#   - a prompt file       : data/captions.txt    (generate with scripts/make_captions.py)
#   - a local Wan2.1-T2V checkpoint dir (WAN_DIR); default below is this machine's local path.
#
# --batch is a TRUE batch (B noises in one forward), decoupled from #GPUs. Experiments show a
# larger effective batch (lr x batch x #GPUs) usually gives better detail — but it also peaks and
# collapses earlier, so stop at the peak step. Effective batch = BATCH x NPROC.
set -e
cd "$(dirname "$0")/../.."   # -> repo root (release_my/)

WAN_DIR="${WAN_DIR:-/path/to/Wan2.1-T2V-1.3B}"   # local Wan2.1-T2V checkpoint dir
MANIFOLD_CKPT="${MANIFOLD_CKPT:-model/shell_lcc.pth}"
CAPTIONS="${CAPTIONS:-data/captions.txt}"
NPROC="${NPROC:-4}"     # number of GPUs
BATCH="${BATCH:-8}"     # per-GPU true batch

torchrun --nproc_per_node="$NPROC" scripts/train_T2V_model.py \
    --mode full --full_parallel ddp --no_drift \
    --wan_dir "$WAN_DIR" \
    --manifold_ckpt "$MANIFOLD_CKPT" \
    --captions "$CAPTIONS" \
    --batch "$BATCH" \
    --lr 1e-5 --iters 200 --save_interval 10 \
    --out_dir model/t2v_run
