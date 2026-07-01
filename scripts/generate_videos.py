"""
Generate videos for a set of prompts with the base model and one or more finetuned models
(full-param via --full_ckpt, LoRA via --adapters), using the SAME seed/noise per prompt so the
outputs are directly comparable. Saves one mp4 per (prompt, model) plus a manifest.json.

This is the ONLY GPU / denoise-heavy step. Point --out_dir at the two loaders below (no GPU needed):
  - eval_detail.py      : load videos -> detail metrics (lap / hf / change)
  - generate_compare.py : load videos -> side-by-side comparison mp4

manifest.json layout:
  {"labels": ["base", "full"], "fps": 16,
   "items": [{"idx": 0, "prompt": "...", "videos": {"base": "00_base.mp4", "full": "00_full.mp4"}}, ...]}
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

import math
import json
import argparse

import torch
import numpy as np
import imageio


def to_uint8(video):
    # video: [C,T,H,W] in ~[-1,1] -> [T,H,W,C] uint8
    v = ((video.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return v.permute(1, 2, 3, 0).cpu().numpy()


def write_mp4(path, frames_uint8, fps):
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames_uint8:
        w.append_data(f)
    w.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wan_dir", default="/path/to/Wan2.1-T2V-1.3B")
    p.add_argument("--base_ckpt", default="",
                   help="optional: override the base column weights with this full state_dict "
                        "(e.g. an UltraWan-merged ckpt) so base and the finetuned column share a start")
    p.add_argument("--full_ckpt", default="",
                   help="full-param checkpoint (wan_full.pth) — the finetuned model column")
    p.add_argument("--ckpt_label", default="full", help="label/filename for the --full_ckpt column")
    p.add_argument("--adapters", default="",
                   help="comma-separated LoRA adapter dirs; each becomes an extra column")
    p.add_argument("--prompt_file", default="data/captions.txt")
    p.add_argument("--prompt_idx", default="", help="comma-separated subset of prompt indices; empty = all")
    p.add_argument("--max_prompts", type=int, default=0, help="0=all")
    p.add_argument("--shard", default="0/1",
                   help="i/n : this process handles prompt k where k%%n==i (data-parallel across GPUs)")
    p.add_argument("--out_dir", required=True, help="dir for the mp4s + manifest.json")
    p.add_argument("--width", type=int, default=1280)   # 720p, matches the manifold training domain
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--frame_num", type=int, default=49)  # T_lat=(F-1)//4+1
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)   # Wan 720p default shift=5
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    all_prompts = [l.strip() for l in open(args.prompt_file, encoding="utf-8") if l.strip()]
    if args.prompt_idx.strip():
        full_idxs = [int(x) for x in args.prompt_idx.split(",")]
    else:
        full_idxs = list(range(len(all_prompts)))
        if args.max_prompts > 0:
            full_idxs = full_idxs[: args.max_prompts]
    si, sn = [int(x) for x in args.shard.split("/")]
    idxs = [full_idxs[k] for k in range(len(full_idxs)) if k % sn == si]   # this shard's slice
    prompts = [all_prompts[i] for i in idxs]
    adapters = [a for a in args.adapters.split(",") if a.strip()]
    print(f"[shard {si}/{sn}] {len(prompts)}/{len(full_idxs)} prompts, {len(adapters)} adapters", flush=True)

    # Wan negative prompt (kept in Chinese on purpose: Wan2.1 was trained with Chinese negatives,
    # which work better than an English translation — this is model input, not a comment).
    neg = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
           "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
           "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
           "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

    # ---- 1. T5-encode all prompts + negative, cache, then free ----
    from wan.modules.t5 import T5EncoderModel
    text_encoder = T5EncoderModel(
        text_len=512, dtype=torch.bfloat16, device=device,
        checkpoint_path=os.path.join(args.wan_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.wan_dir, "google/umt5-xxl"),
    )
    with torch.no_grad():
        ctx_list = [c.detach() for c in text_encoder(prompts, device)]
        ctx_null = text_encoder([neg], device)[0].detach()
    del text_encoder
    torch.cuda.empty_cache()
    print("text cached", flush=True)

    # ---- 2. base model (+ optional base_ckpt) + LoRA adapters + full ckpt ----
    from wan.modules.model import WanModel
    from peft import PeftModel
    base = WanModel.from_pretrained(args.wan_dir, torch_dtype=torch.bfloat16).to(device).eval()
    if args.base_ckpt:
        miss, unexp = base.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"), strict=False)
        print(f"base_ckpt loaded: {args.base_ckpt} ({len(miss)} missing, {len(unexp)} unexpected)", flush=True)
    adapter_names = [f"a{i}" for i in range(len(adapters))]
    model = None
    if adapters:
        model = PeftModel.from_pretrained(base, adapters[0], adapter_name=adapter_names[0])
        for nm, ap in zip(adapter_names[1:], adapters[1:]):
            model.load_adapter(ap, adapter_name=nm)
        model.eval()
    full_model = None
    if args.full_ckpt:
        full_model = WanModel.from_pretrained(args.wan_dir, torch_dtype=torch.bfloat16)
        full_model.load_state_dict(torch.load(args.full_ckpt, map_location="cpu"), strict=False)
        full_model = full_model.to(device).eval()
        print(f"full ckpt loaded: {args.full_ckpt}", flush=True)

    # ---- 3. VAE + scheduler ----
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    vae = Wan2_1_VAE(vae_pth=os.path.join(args.wan_dir, "Wan2.1_VAE.pth"), device=device)

    vae_stride, patch = (4, 8, 8), (1, 2, 2)
    F = args.frame_num
    T = (F - 1) // vae_stride[0] + 1
    Hl, Wl = args.height // vae_stride[1], args.width // vae_stride[2]
    target_shape = (16, T, Hl, Wl)
    seq_len = math.ceil(Hl * Wl / (patch[1] * patch[2]) * T)
    print(f"latent {target_shape}, seq_len {seq_len}", flush=True)

    from contextlib import nullcontext

    @torch.no_grad()
    def sample(noise, ctx, mode):
        # mode: "base" -> clean base (disable LoRA if present); "full" -> full ckpt; else an adapter_name
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sched.set_timesteps(args.steps, device=device, shift=args.shift)
        lat = noise.clone()
        ctxs = {"context": [ctx], "seq_len": seq_len}
        ctxn = {"context": [ctx_null], "seq_len": seq_len}
        if mode == "full":
            m, adapter_ctx = full_model, nullcontext()
        elif mode == "base":
            if model is not None:
                m, adapter_ctx = model, model.disable_adapter()
            else:
                m, adapter_ctx = base, nullcontext()
        else:
            m = model
            model.set_adapter(mode)
            adapter_ctx = nullcontext()
        with adapter_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            for t in sched.timesteps:
                ts = torch.stack([t])
                nc = m([lat], t=ts, **ctxs)[0]
                nu = m([lat], t=ts, **ctxn)[0]
                npred = nu + args.guide_scale * (nc - nu)
                lat = sched.step(npred.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False)[0].squeeze(0)
        return to_uint8(vae.decode([lat])[0])  # [T,H,W,C] uint8

    # columns: base | adapter labels... | ckpt_label (full)
    col_modes = ["base"] + adapter_names + (["full"] if full_model is not None else [])
    col_labels = ["base"] + [os.path.basename(a) for a in adapters] + ([args.ckpt_label] if full_model is not None else [])
    print("columns:", col_labels, flush=True)

    # ---- 4. per prompt: run all columns from the SAME noise, save one mp4 per column ----
    for orig_i, ctx in zip(idxs, ctx_list):
        g = torch.Generator(device=device).manual_seed(args.seed + orig_i)
        noise = torch.randn(*target_shape, generator=g, device=device, dtype=torch.float32)
        print(f"[shard {si}/{sn}][{orig_i}] generating {len(col_modes)} columns ...", flush=True)
        for mode, label in zip(col_modes, col_labels):
            write_mp4(os.path.join(args.out_dir, f"{orig_i:02d}_{label}.mp4"), sample(noise, ctx, mode), args.fps)

    # shard 0 writes the FULL manifest for all prompts (video filenames are deterministic and
    # independent of which shard produced them). Other shards only write their videos.
    if si == 0:
        items = [{"idx": i, "prompt": all_prompts[i],
                  "videos": {lbl: f"{i:02d}_{lbl}.mp4" for lbl in col_labels}} for i in full_idxs]
        manifest = {"labels": col_labels, "fps": args.fps, "items": items}
        with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"done. manifest: {len(items)} prompts x {len(col_labels)} columns -> {args.out_dir}/manifest.json", flush=True)
    else:
        print(f"[shard {si}/{sn}] done {len(idxs)} prompts", flush=True)


if __name__ == "__main__":
    main()
