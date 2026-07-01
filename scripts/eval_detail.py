"""
Load pre-generated videos (from generate_videos.py) and compute detail metrics — NO GPU, NO denoise.
For each prompt, compare the base video against a finetuned-checkpoint video and report, averaged
over frames and prompts:

  lap    : Laplacian variance — sharpness / edge energy (variance of a 3x3 Laplacian per frame).
           Higher = sharper (more edges/texture). NOTE: also responds to NOISE, so a high lap can
           be noise/artifacts, not "good" detail.
  hf     : high-frequency FFT energy ratio — fraction of spectral energy above 1/4 Nyquist.
           Higher = more fine texture. NOTE: high frequency = detail OR noise (grain, compression
           artifacts, ringing), so a high hf is NOT proof of better visual quality.
  change : content drift — mean absolute pixel difference (0~1) between base and ckpt frames.
           Small = same composition, only detail tweaked; large = the content itself changed.

Reported as ratios: lap_ratio = ckpt_lap/base_lap, hf_ratio = ckpt_hf/base_hf  ( >1 = more than base ).

IMPORTANT — lap/hf are PROXIES, only meaningful when `change` is small:
  When `change` is large, base and ckpt are effectively two DIFFERENT videos, so comparing their
  lap/hf is meaningless (it just reflects which content happens to be higher-frequency, not whether
  detail improved). A "genuine" improvement is roughly  hf_ratio>1 AND lap_ratio>1 AND change modest
  — and even then, WATCH THE VIDEOS to confirm it is real detail and not noise/artifacts.

Reads <videos_dir>/manifest.json, appends the aggregate to results.jsonl. Pure numpy/torch on CPU.
"""
import os
import json
import argparse

import numpy as np
import imageio.v2 as imageio
import torch
import torch.nn.functional as F


def read_video(path):
    """mp4 -> [T,H,W,3] uint8."""
    rd = imageio.get_reader(path)
    frames = np.stack([f[..., :3] for f in rd])
    rd.close()
    return frames


def detail_metrics(vid_uint8):
    """vid_uint8: [T,H,W,C] uint8 -> (lap_var, hf_ratio) averaged over frames."""
    x = torch.from_numpy(vid_uint8).float() / 255.0          # [T,H,W,C]
    gray = x.mean(dim=3)                                       # [T,H,W]
    t = gray.unsqueeze(1)                                      # [T,1,H,W]
    # Laplacian variance (sharpness)
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    lap_var = F.conv2d(t, k, padding=1).var(dim=(1, 2, 3)).mean().item()
    # high-frequency FFT energy ratio (energy at radius > 1/4 Nyquist / total energy)
    F2 = torch.fft.fftshift(torch.fft.fft2(gray), dim=(1, 2)).abs() ** 2  # [T,H,W]
    _, H, W = gray.shape
    yy, xx = torch.meshgrid(torch.arange(H) - H / 2, torch.arange(W) - W / 2, indexing="ij")
    rad = torch.sqrt((yy / (H / 2)) ** 2 + (xx / (W / 2)) ** 2)
    hf_mask = (rad > 0.25).float()
    hf_ratio = ((F2 * hf_mask).sum(dim=(1, 2)) / (F2.sum(dim=(1, 2)) + 1e-8)).mean().item()
    return lap_var, hf_ratio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", required=True, help="dir produced by generate_videos.py (has manifest.json)")
    p.add_argument("--base_label", default="base", help="manifest column used as the reference")
    p.add_argument("--ckpt_label", default="full", help="manifest column to evaluate against base")
    p.add_argument("--tag", default="", help="result label, e.g. t2v_run/step25")
    p.add_argument("--results", default="results.jsonl", help="jsonl file to append the aggregate to")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.load(open(os.path.join(args.videos_dir, "manifest.json"), encoding="utf-8"))
    labels = manifest["labels"]
    for lbl in (args.base_label, args.ckpt_label):
        if lbl not in labels:
            raise SystemExit(f"label '{lbl}' not in manifest labels {labels}")

    base_lap, base_hf, ck_lap, ck_hf, changes = [], [], [], [], []
    for it in manifest["items"]:
        vb = read_video(os.path.join(args.videos_dir, it["videos"][args.base_label]))
        va = read_video(os.path.join(args.videos_dir, it["videos"][args.ckpt_label]))
        lb, hb = detail_metrics(vb)
        la, ha = detail_metrics(va)
        base_lap.append(lb); base_hf.append(hb); ck_lap.append(la); ck_hf.append(ha)
        n = min(len(vb), len(va))
        changes.append(float(np.abs(vb[:n].astype(np.float32) - va[:n].astype(np.float32)).mean() / 255.0))
        print(f"[{it['idx']}] base lap={lb:.1f} hf={hb:.4f} | {args.ckpt_label} lap={la:.1f} hf={ha:.4f}", flush=True)

    res = {
        "tag": args.tag, "base_label": args.base_label, "ckpt_label": args.ckpt_label,
        "n_prompts": len(manifest["items"]),
        "base_lap": float(np.mean(base_lap)), "ckpt_lap": float(np.mean(ck_lap)),
        "base_hf": float(np.mean(base_hf)), "ckpt_hf": float(np.mean(ck_hf)),
        "lap_ratio": float(np.mean(ck_lap) / (np.mean(base_lap) + 1e-8)),
        "hf_ratio": float(np.mean(ck_hf) / (np.mean(base_hf) + 1e-8)),
        "change": float(np.mean(changes)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(res) + "\n")
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
