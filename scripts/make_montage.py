"""
Read pre-generated videos (from generate_videos.py) and build side-by-side detail montages (PNG):
for each prompt, take one frame from every column (base, ckpt, ...), optionally center-crop and
zoom in to inspect fine detail, label each cell, and stitch them horizontally into one image.
Pure numpy / PIL / imageio — NO GPU.

This is the "look at the actual pixels" companion to eval_detail.py: lap/hf are only proxies, so
eyeball the montages (especially when `change` is large) to judge whether detail truly improved.
Reads <videos_dir>/manifest.json; writes {idx:02d}_montage.png per prompt.
"""
import os
import json
import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def read_frame(path, frame):
    """Return one RGB frame (H,W,3 uint8) from an mp4. frame<0 -> the middle frame."""
    rd = imageio.get_reader(path)
    frames = [f[..., :3] for f in rd]
    rd.close()
    i = len(frames) // 2 if frame < 0 else max(0, min(frame, len(frames) - 1))
    return frames[i]


def center_crop(img, frac):
    """Keep the central `frac` (0<frac<=1) of H and W (to zoom into fine detail)."""
    H, W = img.shape[:2]
    h, w = int(round(H * frac)), int(round(W * frac))
    y0, x0 = (H - h) // 2, (W - w) // 2
    return img[y0:y0 + h, x0:x0 + w]


def label_cell(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 9 + len(text) * 7, 22], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(60, 230, 150))
    return im


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", required=True, help="dir produced by generate_videos.py (has manifest.json)")
    p.add_argument("--out_dir", default="", help="where to write *_montage.png (default: videos_dir)")
    p.add_argument("--labels", default="", help="comma-separated subset of columns (default: all, in manifest order)")
    p.add_argument("--frame", type=int, default=-1, help="frame index to grab; <0 = middle frame")
    p.add_argument("--zoom", type=float, default=0.0,
                   help="0 = full frame; 0<z<1 = keep central z fraction and enlarge (inspect detail)")
    p.add_argument("--cell_width", type=int, default=600, help="width each column is resized to")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.load(open(os.path.join(args.videos_dir, "manifest.json"), encoding="utf-8"))
    labels = [l for l in args.labels.split(",") if l.strip()] or manifest["labels"]
    for lbl in labels:
        if lbl not in manifest["labels"]:
            raise SystemExit(f"label '{lbl}' not in manifest labels {manifest['labels']}")
    out_dir = args.out_dir or args.videos_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"columns: {labels} | frame={args.frame} zoom={args.zoom}", flush=True)

    for it in manifest["items"]:
        cells = []
        for lbl in labels:
            fr = read_frame(os.path.join(args.videos_dir, it["videos"][lbl]), args.frame)
            if 0 < args.zoom < 1:
                fr = center_crop(fr, args.zoom)
            im = Image.fromarray(fr)
            h = int(round(im.height * args.cell_width / im.width))
            im = im.resize((args.cell_width, h))
            cells.append(label_cell(im, lbl))
        W = sum(c.width for c in cells)
        H = max(c.height for c in cells)
        montage = Image.new("RGB", (W, H), (0, 0, 0))
        x = 0
        for c in cells:
            montage.paste(c, (x, 0))
            x += c.width
        out = os.path.join(out_dir, f"{it['idx']:02d}_montage.png")
        montage.save(out)
        print(f"[{it['idx']}] -> {out}  ({W}x{H})", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
