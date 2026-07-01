"""Two-stage training for the Shell-LCC manifold (multi-GPU DDP, single-GPU fallback).

Stage 1 (LCC)   Train basis / predictor / global_scale / global_bias with
                loss = l1 + l2 + [usage_kl] + [basis_to_patch]; surface_head untouched. 100 epochs.
Stage 2 (Shell) Load the Stage-1 checkpoint, freeze the LCC params, train only surface_head with
                loss = LCC(no grad) + shell + reg. 100 epochs. The result is the final Shell-LCC manifold.

Why two stages (the paper's main text uses one-stage / joint training): training LCC and the shell
head jointly lets global_bias (and the rest of the LCC skeleton) keep drifting while surface_head is
fitting the shell, so the shell is learned on a moving skeleton, ends up mis-fit, and the final reward
is mis-calibrated. Freezing the converged LCC skeleton first pins the shell to a stable skeleton and
keeps the reward well-calibrated.

Data: VAE latents saved as .pt (see scripts/extract_wan_vae_feature.py), each shaped [T, 16, H, W].
Model: ShellCoordinateManifold from shell_lcc.py (same directory).

Usage:
  # multi-GPU (4 GPUs, DDP); per-GPU batch = --batch_size, effective batch = batch_size * n_gpus
  torchrun --nproc_per_node=4 train_manifold_2stage.py \
      --data_dir /path/to/vae_latents --save_dir ./runs/run1 --epochs_lcc 100 --epochs_shell 100
  # single GPU (auto fallback)
  python train_manifold_2stage.py --data_dir /path/to/vae_latents --save_dir ./runs/run1
  # Stage 2 only (already have an LCC checkpoint)
  torchrun --nproc_per_node=4 train_manifold_2stage.py --save_dir ./runs/run1 \
      --resume_lcc ./runs/run1/lcc_ep100.pth --epochs_lcc 0 --epochs_shell 100
"""
import os
import sys
import glob
import argparse
import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shell_lcc import ShellCoordinateManifold, init_basis_with_random_samples


def setup_distributed():
    """torchrun sets RANK/WORLD_SIZE -> use DDP; otherwise fall back to single GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), True
    return 0, 0, 1, False


class LatentPTDataset(Dataset):
    """Reads VAE-latent .pt files ([T,16,H,W]). Takes the first `num_frames` frames and returns
    (16,T,H,W) so same-resolution clips batch together. Clips with too few frames are skipped;
    if a .pt has 32 channels (mean+logvar concatenated) the first 16 (mean) are used."""

    def __init__(self, data_dir, num_frames=16, verbose=True):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not paths:
            raise SystemExit(f"no .pt found in {data_dir}")
        self.num_frames = num_frames
        self.paths = [p for p in paths if torch.load(p, map_location="cpu").shape[0] >= num_frames]
        if verbose:
            dropped = len(paths) - len(self.paths)
            print(f"[data] {len(self.paths)} clips (>= {num_frames} frames) from {data_dir}"
                  + (f"; dropped {dropped} too-short" if dropped else ""))
        if not self.paths:
            raise SystemExit(f"no clip has >= {num_frames} frames; lower --num_frames")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        z = torch.load(self.paths[idx], map_location="cpu").float()  # [T,C,H,W]
        if z.shape[1] == 32:
            z = z[:, :16]
        z = z[: self.num_frames]                 # [nf,C,H,W]
        return z.permute(1, 0, 2, 3).contiguous()  # [C,nf,H,W]


def log(path, msg):
    with open(path, "a") as f:
        f.write(msg + "\n")
        f.flush()


def save_ckpt(raw_model, optimizer, epoch, config, path):
    torch.save({"epoch": epoch, "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": config}, path)


def run_stage(ddp_model, raw_model, loader, sampler, optimizer, device, stage, n_epochs,
              save_dir, prefix, save_interval, log_path, config, fwd_kwargs, is_main):
    """Run n_epochs. ddp_model does forward/backward; raw_model is used for saving.
    Only rank 0 (is_main) prints / logs / saves checkpoints."""
    step = 0
    last = None
    for ep in range(1, n_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)  # different shuffle each epoch
        it = tqdm(loader, desc=f"[{prefix}] ep{ep}/{n_epochs}") if is_main else loader
        for batch in it:
            batch = batch.to(device)
            _, loss, logs, _ = ddp_model(batch, stage=stage, **fwd_kwargs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if is_main:
                it.set_postfix(loss=f"{loss.item():.3f}", l1=f"{logs.get('l1',0):.3f}",
                               l2=f"{logs.get('l2',0):.3f}", kl=f"{logs.get('kl',0):.3f}",
                               b2p=f"{logs.get('l_basis_to_patch',0):.3f}",
                               shell=f"{logs.get('l_NLL',0):.3f}")
                if step % config["log_interval"] == 0:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log(log_path, f"{ts} | {prefix} | ep{ep} step{step} | total {loss.item():.6f} | "
                                  f"l1 {logs.get('l1',0):.6f} | l2 {logs.get('l2',0):.6f} | "
                                  f"kl {logs.get('kl',0):.6f} | b2p {logs.get('l_basis_to_patch',0):.6f} | "
                                  f"shell {logs.get('l_NLL',0):.6f}")
            step += 1
        if is_main and (ep % save_interval == 0 or ep == n_epochs):
            last = os.path.join(save_dir, f"{prefix}_ep{ep}.pth")
            save_ckpt(raw_model, optimizer, ep, config, last)
            log(log_path, f"--- saved {last} ---")
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="dir of VAE-latent .pt files ([T,16,H,W])")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--embedding_dim", type=int, default=16)
    p.add_argument("--num_bases", type=int, default=4096)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=16, help="use the first N frames of each clip (so clips batch)")
    p.add_argument("--batch_size", type=int, default=2, help="per-GPU batch; effective = batch_size * n_gpus")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs_lcc", type=int, default=500, help="Stage-1 LCC epochs (0 = skip, go straight to Stage 2)")
    p.add_argument("--epochs_shell", type=int, default=100, help="Stage-2 Shell epochs")
    p.add_argument("--lambda_usage", type=float, default=0.1,
                   help="Stage-1 usage-KL weight (0 disables). Usually no effect; optional.")
    p.add_argument("--lambda_basis_to_patch", type=float, default=0.1,
                   help="Stage-1 basis->patch weight (0 disables). Usually no effect; optional.")
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--resume_lcc", default="", help="existing LCC ckpt; if given, skip Stage 1 and train Shell only")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    local_rank, global_rank, world_size, is_dist = setup_distributed()
    is_main = global_rank == 0
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + global_rank)  # per-rank sampling differs; DDP broadcasts params to match
    np.random.seed(args.seed + global_rank)

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
    if is_dist:
        dist.barrier()
    log_path = os.path.join(args.save_dir, "training_log.txt")
    config = vars(args)
    if is_main:
        mode = f"DDP x{world_size}" if is_dist else "single-GPU"
        log(log_path, f"\n{'='*20} 2-stage manifold train [{mode}] {datetime.datetime.now()} {'='*20}")
        print(f"[dist] mode={mode}")

    dataset = LatentPTDataset(args.data_dir, num_frames=args.num_frames, verbose=is_main)
    sampler = DistributedSampler(dataset, shuffle=True) if is_dist else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)

    raw_model = ShellCoordinateManifold(embedding_dim=args.embedding_dim,
                                        num_bases=args.num_bases,
                                        hidden_dim=args.hidden_dim).to(device)

    def wrap(m):
        # DDP constructor broadcasts rank-0 params to all ranks -> identical init across GPUs.
        return DDP(m, device_ids=[local_rank], output_device=local_rank,
                   find_unused_parameters=True) if is_dist else m

    # ---------- Stage 1: LCC ----------
    if args.resume_lcc:
        if is_main:
            print(f"[stage1] skip; load LCC ckpt {args.resume_lcc}")
        sd = torch.load(args.resume_lcc, map_location=device)["model"]
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        miss, unexp = raw_model.load_state_dict(sd, strict=False)
        if is_main:
            print("  missing:", miss, "| unexpected:", unexp)
    elif args.epochs_lcc > 0:
        init_basis_with_random_samples(raw_model, loader, device)
        raw_model.train()
        ddp1 = wrap(raw_model)
        opt1 = torch.optim.Adam(ddp1.parameters(), lr=args.lr)
        if is_main:
            print(f"[stage1] train LCC {args.epochs_lcc} ep "
                  f"(usage={args.lambda_usage}, basis_to_patch={args.lambda_basis_to_patch})")
        lcc_ckpt = run_stage(ddp1, raw_model, loader, sampler, opt1, device, "lcc", args.epochs_lcc,
                             args.save_dir, "lcc", args.save_interval, log_path, config,
                             fwd_kwargs=dict(lambda_usage=args.lambda_usage,
                                             lambda_basis_to_patch=args.lambda_basis_to_patch),
                             is_main=is_main)
        if is_main:
            log(log_path, f"[stage1] LCC done -> {lcc_ckpt}")
        del ddp1

    # ---------- Stage 2: Shell (freeze LCC, train surface_head only) ----------
    raw_model.predictor.requires_grad_(False)
    raw_model.basis.requires_grad_(False)
    raw_model.global_scale.requires_grad_(False)
    raw_model.global_bias.requires_grad_(False)
    raw_model.surface_head.requires_grad_(True)
    raw_model.train()
    raw_model.predictor.eval()  # freeze BatchNorm running stats (buffers already broadcast under DDP)

    ddp2 = wrap(raw_model)
    raw_model.predictor.eval()  # DDP wrap resets training flags; pin predictor to eval again
    trainable = [q for q in raw_model.parameters() if q.requires_grad]
    if is_main:
        n_tr, n_all = sum(q.numel() for q in trainable), sum(q.numel() for q in raw_model.parameters())
        print(f"[stage2] trainable {n_tr/1e6:.3f}M / {n_all/1e6:.3f}M (surface_head only)")
        log(log_path, f"[stage2] trainable {n_tr} / {n_all}")
    opt2 = torch.optim.Adam(trainable, lr=args.lr)
    shell_ckpt = run_stage(ddp2, raw_model, loader, sampler, opt2, device, "joint", args.epochs_shell,
                           args.save_dir, "shell", args.save_interval, log_path, config,
                           fwd_kwargs={}, is_main=is_main)
    if is_main:
        log(log_path, f"[stage2] Shell-LCC done -> {shell_ckpt}")
        print(f"DONE. final Shell-LCC manifold: {shell_ckpt}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
