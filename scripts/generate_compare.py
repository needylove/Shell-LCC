"""
Load pre-generated videos (from generate_videos.py) and write side-by-side comparison mp4s
— NO GPU, NO denoise. Reads <videos_dir>/manifest.json and, for each prompt, horizontally
concatenates the columns (default: all, in manifest order) into {idx}_compare.mp4.
Pure numpy/imageio — run it anywhere.
"""
import os
import json
import argparse

import numpy as np
import imageio.v2 as imageio


def read_video(path):
    """mp4 -> [T,H,W,3] uint8."""
    rd = imageio.get_reader(path)
    frames = np.stack([f[..., :3] for f in rd])
    rd.close()
    return frames


def write_mp4(path, frames_uint8, fps):
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames_uint8:
        w.append_data(f)
    w.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_dir", required=True, help="dir produced by generate_videos.py (has manifest.json)")
    p.add_argument("--out_dir", default="", help="where to write *_compare.mp4 (default: videos_dir)")
    p.add_argument("--labels", default="", help="comma-separated subset of columns to show (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.load(open(os.path.join(args.videos_dir, "manifest.json"), encoding="utf-8"))
    fps = manifest.get("fps", 16)
    labels = [l for l in args.labels.split(",") if l.strip()] or manifest["labels"]
    for lbl in labels:
        if lbl not in manifest["labels"]:
            raise SystemExit(f"label '{lbl}' not in manifest labels {manifest['labels']}")
    out_dir = args.out_dir or args.videos_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"columns left->right: {labels}", flush=True)

    for it in manifest["items"]:
        cols = [read_video(os.path.join(args.videos_dir, it["videos"][lbl])) for lbl in labels]
        n = min(len(c) for c in cols)
        strip = np.concatenate([c[:n] for c in cols], axis=2)  # horizontal concat: [T,H,W*ncol,C]
        out = os.path.join(out_dir, f"{it['idx']:02d}_compare.mp4")
        write_mp4(out, strip, fps)
        print(f"[{it['idx']}] -> {out}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
