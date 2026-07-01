#!/bin/bash
# Train a Shell-LCC manifold (two stages) on the example VAE latents in data/vae -> data/manifold.
# Run extract_vae_example.sh first to produce data/vae.
#
# Stage 1 fits the LCC skeleton, Stage 2 freezes it and trains only the shell head.
# Single-GPU by default. Epoch counts are small here just to demo on the 4 example clips;
# for a real run use ~100 + ~100 (and more data). Override with EPOCHS_LCC / EPOCHS_SHELL.
set -e
cd "$(dirname "$0")/../.."   # -> repo root (release_my/)

EPOCHS_LCC="${EPOCHS_LCC:-11}"
EPOCHS_SHELL="${EPOCHS_SHELL:-11}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python manifold/train_manifold_2stage.py \
    --data_dir data/vae \
    --save_dir model/test \
    --epochs_lcc "$EPOCHS_LCC" \
    --epochs_shell "$EPOCHS_SHELL" \
    --save_interval 10 

# Multi-GPU (e.g. 4 GPUs), for a real training run:
#   torchrun --nproc_per_node=4 manifold/train_manifold_2stage.py \
#       --data_dir data/vae --save_dir model/manifold --epochs_lcc 500 --epochs_shell 100
