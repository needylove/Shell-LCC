"""
Manifold-reward finetuning of a Wan2.1-T2V model (multi-GPU): LoRA (DDP) or full-param (FSDP/DDP).

Mechanism (each rank samples independently; gradients are all-reduced):
    ctx   = T5(caption);  noise ~ N(0,I);  sigma ~ U[lo,hi];  x_t := noise;  t := sigma*1000
    z_gen = WanModel(x_t, t, ctx)  (train)               -> z_gen     = x_t - sigma*v
    z_ref = WanModel(x_t, t, ctx)  (frozen ref, no_grad) -> z_gen_ref = x_t - sigma*v_ref
    loss  = manifold.distance_reward(z_gen) + relu(MSE(z_gen, z_gen_ref) - margin)

Use --no_drift for the pure manifold reward (paper setting, no anti-drift regularizer).
Launch with train.sh (torchrun + LD_PRELOAD); single-process (no torchrun) also works.
"""
import os
import sys


def _ensure_libstdcpp():
    """Preload the conda env's (usually newer) libstdc++ so C++ extensions like torch/pyarrow don't
    fail with a `CXXABI_x.x.x not found` error against an older system libstdc++. This is NOT a file
    you ship: it points at libstdc++.so.6 already present in the active env. No-op when that file is
    absent (os.path.exists guard), so it is harmless on machines that don't need it; re-execs once
    with LD_PRELOAD set."""
    pre = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
    if os.path.exists(pre) and pre not in os.environ.get("LD_PRELOAD", ""):
        old = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = pre + (":" + old if old else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_libstdcpp()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import types
import math
import json
import time
import random
import argparse
import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist


def _import_manifold_cls():
    # The manifold package lives next to this script (../manifold); override with MANIFOLD_DIR.
    manifold_dir = os.environ.get(
        "MANIFOLD_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manifold"),
    )
    sys.path.insert(0, manifold_dir)
    from shell_lcc import ShellCoordinateManifold
    return ShellCoordinateManifold


def enable_grad_ckpt(wan):
    import torch.utils.checkpoint as cp

    def wrap(orig):
        def fwd(x, **kw):
            return cp.checkpoint(orig, x, use_reentrant=False, **kw)
        return fwd
    for blk in wan.blocks:
        blk.forward = wrap(blk.forward)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["lora", "full"], default="lora")
    p.add_argument("--full_parallel", choices=["fsdp", "ddp"], default="fsdp",
                   help="full-param parallelism: fsdp (shard, for 14B) or ddp (replica, 4x batch, for 1.3B)")
    p.add_argument("--no_drift", action="store_true",
                   help="pure manifold reward (drop the anti-drift regularizer and ref forward); paper "
                        "setting. The anti-drift term usually has no effect; recommended.")
    p.add_argument("--wan_dir", default="/path/to/Wan2.1-T2V-14B")
    p.add_argument("--init_ckpt", default="",
                   help="full state_dict .pth to replace the wan_dir weights as the training start "
                        "and drift reference (e.g. an UltraWan-merged checkpoint)")
    p.add_argument("--manifold_ckpt", default="manifolds/ep10.pth")
    p.add_argument("--captions", default="data/captions.txt")
    p.add_argument("--out_dir", default="outputs/runs/exp")
    p.add_argument("--lat_t", type=int, default=5)
    p.add_argument("--lat_h", type=int, default=90)
    p.add_argument("--lat_w", type=int, default=160)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--batch", type=int, default=1,
                   help="true batch: B samples in one forward (decoupled from #GPUs); larger usually helps")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--sigma_lo", type=float, default=0.7)
    p.add_argument("--sigma_hi", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    # distributed setup
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    if world > 1:
        dist.barrier()
    logf = open(os.path.join(args.out_dir, "train_log.txt"), "a") if is_main else None

    def log(msg):
        if is_main:
            line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            logf.write(line + "\n"); logf.flush()

    log(f"mode={args.mode} world={world} args={vars(args)}")
    ShellCoordinateManifold = _import_manifold_cls()

    # ---- 1. T5 encode all captions, cache, free ----
    from wan.modules.t5 import T5EncoderModel
    caps_txt = [l.strip() for l in open(args.captions, encoding="utf-8") if l.strip()]
    text_encoder = T5EncoderModel(
        text_len=512, dtype=torch.bfloat16, device=device,
        checkpoint_path=os.path.join(args.wan_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.wan_dir, "google/umt5-xxl"),
    )
    contexts = []
    with torch.no_grad():
        for i in range(0, len(caps_txt), 64):   # chunked: large caption sets would OOM in one T5 pass
            contexts += [c.detach() for c in text_encoder(caps_txt[i:i + 64], device)]
    del text_encoder
    torch.cuda.empty_cache()
    log(f"cached {len(contexts)} caption embeddings")

    # ---- 2. model ----
    from wan.modules.model import WanModel
    base = WanModel.from_pretrained(args.wan_dir, torch_dtype=torch.bfloat16).to(device).eval()
    if args.init_ckpt:
        log(f"Loading init_ckpt from {args.init_ckpt} ...")
        sd = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        missing, unexpected = base.load_state_dict(sd, strict=False)
        log(f"init_ckpt loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    enable_grad_ckpt(base)

    ref_model = None
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                         target_modules=["q", "k", "v", "o"], bias="none")
        peft_model = get_peft_model(base, cfg)
        for _, pm in peft_model.named_parameters():
            if pm.requires_grad:
                pm.data = pm.data.float()
        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        if world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(peft_model, device_ids=[local_rank], find_unused_parameters=False)
            core = model.module
        else:
            model, core = peft_model, peft_model
        fwd_model = model            # gradient forward goes through this (DDP triggers sync)
        ref_fwd = core               # ref forward uses disable_adapter
    else:  # full finetune
        for p in base.parameters():
            p.requires_grad_(True)
        # frozen reference (only loaded when anti-drift is enabled)
        ref_model = None
        if not args.no_drift:
            ref_model = WanModel.from_pretrained(args.wan_dir, torch_dtype=torch.bfloat16).to(device).eval()
            if args.init_ckpt:
                sd = torch.load(args.init_ckpt, map_location=device, weights_only=True)
                ref_model.load_state_dict(sd, strict=False)
            for p in ref_model.parameters():
                p.requires_grad_(False)
        if world > 1 and args.full_parallel == "fsdp":
            from wan.distributed.fsdp import shard_model
            model = shard_model(base, device_id=local_rank, sync_module_states=True)
            if ref_model is not None:
                ref_model = shard_model(ref_model, device_id=local_rank, sync_module_states=True)
        elif world > 1:  # ddp: full model replica (used for 1.3B)
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(base, device_ids=[local_rank], find_unused_parameters=False)
        else:
            model = base
        core = model
        trainable = [p for p in model.parameters() if p.requires_grad]
        fwd_model = model
        ref_fwd = ref_model
    n_train = sum(p.numel() for p in trainable)
    log(f"trainable params: {n_train/1e6:.2f}M")

    # ---- 3. manifold (frozen fp32) ----
    ck = torch.load(args.manifold_ckpt, map_location="cpu")
    mc = ck["config"]
    manifold = ShellCoordinateManifold(embedding_dim=mc["embedding_dim"],
                                       num_bases=mc["num_bases"], hidden_dim=mc["hidden_dim"])
    manifold.load_state_dict({k[7:] if k.startswith("module.") else k: v
                              for k, v in ck["model"].items()})
    manifold.to(device).eval()
    for p in manifold.parameters():
        p.requires_grad_(False)

    # ---- 4. train ----
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    T, H, W = args.lat_t, args.lat_h, args.lat_w
    seq_len = T * (H // 2) * (W // 2)
    log(f"latent [1,16,{T},{H},{W}] seq_len={seq_len}")

    def save(step):
        tag = os.path.join(args.out_dir, f"step{step}")
        if args.mode == "lora":
            if is_main:
                core.save_pretrained(tag)
        else:
            if world > 1 and args.full_parallel == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                          FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    sd = model.state_dict()
            else:  # ddp (model.module is WanModel) or single GPU
                sd = (model.module if world > 1 else model).state_dict()
            if is_main:
                os.makedirs(tag, exist_ok=True)
                torch.save(sd, os.path.join(tag, "wan_full.pth"))
        if is_main:
            log(f"saved -> {tag}")

    model.train() if args.mode == "full" else core.train()
    t0 = time.time()
    for step in range(1, args.iters + 1):
        B = args.batch
        noises, sigmas, ctxs = [], [], []
        for b in range(B):
            gi = ((step - 1) * B + b) * world + rank   # distinct context/noise per sample & rank
            ctxs.append(contexts[gi % len(contexts)])
            noises.append(torch.randn(16, T, H, W, device=device, dtype=torch.bfloat16))
            sigmas.append(random.uniform(args.sigma_lo, args.sigma_hi))
        tt = torch.tensor([s * 1000.0 for s in sigmas], device=device).unsqueeze(1).expand(B, seq_len).contiguous()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vs = fwd_model(noises, t=tt, context=ctxs, seq_len=seq_len)   # one forward, list of B
            z_gens = [noises[b] - sigmas[b] * vs[b] for b in range(B)]
            if not args.no_drift:
                with torch.no_grad():
                    if args.mode == "lora":
                        with ref_fwd.disable_adapter():
                            vr = ref_fwd(noises, t=tt, context=ctxs, seq_len=seq_len)
                    else:
                        vr = ref_fwd(noises, t=tt, context=ctxs, seq_len=seq_len)
                    z_refs = [noises[b] - sigmas[b] * vr[b] for b in range(B)]

        zf = torch.stack([z.float() for z in z_gens], 0)   # [B,16,T,H,W]
        reward = manifold.distance_reward(zf)
        if args.no_drift:                       # pure manifold reward (paper setting)
            drift = torch.zeros((), device=device)
            loss = reward
        else:
            zrf = torch.stack([z.float() for z in z_refs], 0)
            drift = torch.relu(F.mse_loss(zf, zrf) - args.margin)
            loss = reward + drift
        sigma = sigmas[0]                       # for logging only

        opt.zero_grad(set_to_none=True)
        bad = torch.isnan(loss) | torch.isinf(loss)
        if not bad:
            loss.backward()
            if args.mode == "full" and world > 1 and args.full_parallel == "fsdp":
                gnorm = model.clip_grad_norm_(args.grad_clip)
            else:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if not (torch.isnan(gnorm) | torch.isinf(gnorm)):
                opt.step()
            else:
                log(f"step {step}: bad grad, skip")
        else:
            log(f"step {step}: bad loss, skip")

        if step % args.log_interval == 0 or step == 1:
            sps = (time.time() - t0) / step
            log(f"step {step:4d} | loss {loss.item():.5f} | reward {reward.item():.5f} "
                f"| drift {drift.item():.5f} | sigma {sigma:.3f} | {sps:.1f}s/it")
        if step % args.save_interval == 0 or step == args.iters:
            save(step)
            if world > 1:
                dist.barrier()

    log("done.")
    if logf:
        logf.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
