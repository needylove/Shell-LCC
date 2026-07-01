"""Decode Wan2.1 VAE latents back to MP4 (visual sanity check).

Inverse of extract_wan_vae_feature.py. Takes a VAE latent tensor and runs the
Wan2.1 VAE decoder to reconstruct an RGB video, then writes an .mp4.

Accepted inputs (either a single .pt file, or a dir / glob of .pt files):
  * an extracted feature .pt      -> shape [T, 16, H, W]   (this repo's format)
  * a raw decoder latent          -> shape [16, T, H, W]   (C-first)
  * a manifold vae representation -> shape [B, 16, T, H, W] (batched, each decoded)
The channel dim is always 16 (Wan2.1 latent channels); layout is auto-detected,
override with --layout {tchw,cthw} if a tensor is ambiguous (e.g. T==16).

Usage:
  # one feature .pt -> one .mp4
  python decode_wan_vae_feature.py --pt output/clips_short_1920/xxx.pt --out recon/xxx.mp4

  # a whole dir of .pt -> mp4s in --out_dir (same basenames)
  python decode_wan_vae_feature.py --pt_dir output/clips_short_1920 --out_dir recon --limit 10
"""
import os
import sys


def _pre():
    """Preload libstdc++ from the conda env before importing the wan VAE
    (else CXXABI / GLIBCXX symbol error). Re-exec once with LD_PRELOAD set.
    (Same as extract_wan_vae_feature.py / explore_gen.py.)"""
    pre = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
    if os.path.exists(pre) and pre not in os.environ.get("LD_PRELOAD", ""):
        os.environ["LD_PRELOAD"] = pre + (":" + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


_pre()

import glob
import time
import argparse

import torch
import imageio
from einops import rearrange


def load_vae(vae_ckpt_path, device, dtype):
    # Requires the official `wan` package installed (pip install the Wan2.1/2.2 repo).
    from wan.modules.vae2_1 import Wan2_1_VAE
    return Wan2_1_VAE(vae_pth=vae_ckpt_path, device=device, dtype=dtype)


def to_cthw_list(z, layout="auto"):
    """Normalize an arbitrary latent tensor to a list of [C=16, T, H, W] tensors."""
    if z.dim() == 5:  # [B, C, T, H, W] -> list of [C,T,H,W]
        return [z[b] for b in range(z.shape[0])]
    if z.dim() != 4:
        raise ValueError(f"expected 4D or 5D latent, got shape {tuple(z.shape)}")

    if layout == "cthw":
        return [z]
    if layout == "tchw":
        return [rearrange(z, "t c h w -> c t h w")]

    # auto: channel dim must be 16.
    d0, d1 = z.shape[0], z.shape[1]
    if d1 == 16 and d0 != 16:
        return [rearrange(z, "t c h w -> c t h w")]     # [T,C,H,W]  (this repo)
    if d0 == 16 and d1 != 16:
        return [z]                                      # [C,T,H,W]
    if d0 == 16 and d1 == 16:
        # ambiguous (T==16). This repo saves [T,C,H,W]; assume that.
        print("WARN ambiguous layout (dim0==dim1==16); assuming [T,C,H,W]. Use --layout to force.")
        return [rearrange(z, "t c h w -> c t h w")]
    raise ValueError(f"no dim==16 (latent channels) in shape {tuple(z.shape)}")


@torch.no_grad()
def decode_to_frames(vae, z_cthw, device, dtype):
    """[C,T,H,W] latent -> uint8 numpy [T,H,W,3]."""
    z = z_cthw.to(device).to(dtype)
    v = vae.decode([z])[0]                       # [3, T, H, W] in [-1, 1]
    v = ((v.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return v.permute(1, 2, 3, 0).cpu().numpy()   # [T, H, W, 3]


def save_mp4(frames, path, fps):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames:
        w.append_data(f)
    w.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt_path", default="Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                   help="Wan2.1 VAE weight file (Wan2.1_VAE.pth)")
    # input: exactly one of --pt / --pt_dir / --pt_glob
    p.add_argument("--pt", default="", help="a single latent .pt file")
    p.add_argument("--pt_dir", default="", help="a dir of .pt files")
    p.add_argument("--pt_glob", default="", help="a glob like 'output/**/*.pt'")
    # output
    p.add_argument("--out", default="", help="output .mp4 (only for single --pt)")
    p.add_argument("--out_dir", default="recon", help="output dir for batch mode")
    p.add_argument("--fps", type=int, default=24, help="playback fps (extract used 24)")
    p.add_argument("--layout", default="auto", choices=["auto", "tchw", "cthw"])
    p.add_argument("--limit", type=int, default=0, help="cap number of files (0=all)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "requires at least one GPU"
    device, dtype = "cuda", torch.bfloat16

    # collect input .pt paths
    if args.pt:
        pts = [args.pt]
    elif args.pt_dir:
        pts = sorted(glob.glob(os.path.join(args.pt_dir, "*.pt")))
    elif args.pt_glob:
        pts = sorted(glob.glob(args.pt_glob, recursive=True))
    else:
        raise SystemExit("provide one of --pt / --pt_dir / --pt_glob")
    if args.limit:
        pts = pts[: args.limit]
    if not pts:
        raise SystemExit("no .pt inputs found")

    vae = load_vae(args.vae_ckpt_path, device, dtype)
    print(f"decoding {len(pts)} latent file(s) @ fps={args.fps}")

    done, t0 = 0, time.time()
    for pt in pts:
        base = os.path.splitext(os.path.basename(pt))[0]
        if args.pt and args.out:
            out_path = args.out
        else:
            out_path = os.path.join(args.out_dir, base + ".mp4")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip (exists): {out_path}")
            continue

        z = torch.load(pt, map_location="cpu")
        if not torch.is_tensor(z):
            print(f"ERROR {pt}: not a tensor ({type(z)})")
            continue
        try:
            clips = to_cthw_list(z, args.layout)
            for j, z_cthw in enumerate(clips):
                frames = decode_to_frames(vae, z_cthw, device, dtype)
                op = out_path if len(clips) == 1 else out_path.replace(".mp4", f"_{j}.mp4")
                save_mp4(frames, op, args.fps)
                print(f"[{done+1}] {base}{'' if len(clips)==1 else f' clip{j}'} "
                      f"latent{tuple(z_cthw.shape)} -> {op} ({frames.shape[0]}f {frames.shape[2]}x{frames.shape[1]})")
        except Exception as e:
            print(f"ERROR {pt}: {e}")
            continue
        done += 1

    dt = time.time() - t0
    print(f"done {done} file(s) in {dt:.1f}s" + (f", {dt/done:.2f}s/file" if done else ""))


if __name__ == "__main__":
    main()
