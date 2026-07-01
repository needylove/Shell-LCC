"""Local Wan2.1 VAE feature extractor for manifold training.

Extracts Wan2.1 VAE latents from videos and saves them as `.pt` tensors
(shape [T, 16, H//8, W//8], bfloat16) plus a side-car `.json` with metadata.
Numerically matches the NPU/cloud reference pipeline (same preprocessing,
same Wan2.1 VAE, same latent normalization).

All cloud/NPU dependencies (moxing, torch_npu, deepspeed, Feature_Codec,
OBS/S3 upload) have been removed. Frames are extracted with the bundled
ffmpeg-7.0.2 (via imageio_ffmpeg) using the exact same filter chain as the
reference so numbers line up.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python extract_wan_vae_feature.py \
        --video_dir /path/to/videos --save_path ./output

Usage (4-GPU sharding, one process per GPU):
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python extract_wan_vae_feature.py \
        --video_dir /path/to/videos --save_path ./output --shard $i/4 &
    done; wait
"""
import os
import sys


def _pre():
    """Preload libstdc++ from the conda env, else the wan VAE import
    fails with a CXXABI / GLIBCXX symbol error. Re-exec once with LD_PRELOAD set.
    (Copied from scripts/explore_gen.py.)"""
    pre = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
    if os.path.exists(pre) and pre not in os.environ.get("LD_PRELOAD", ""):
        os.environ["LD_PRELOAD"] = pre + (":" + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


_pre()

import glob
import json
import time
import argparse
import subprocess

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision import transforms as T


# --------------------------------------------------------------------------
# Preprocessing constants / helpers  (verbatim from the reference pipeline)
# --------------------------------------------------------------------------
TARGET_SIZE_272P = {1.7778: [480, 272], 1.0000: [336, 336], 0.5625: [272, 480]}
TARGET_SIZE_480P = {1.0000: [512, 512], 1.5000: [720, 480], 0.6667: [480, 720]}
TARGET_SIZE_720P = {1.0000: [960, 960], 0.5625: [720, 1280], 1.7778: [1280, 720]}
TARGET_SIZE_1080P = {1.0000: [1440, 1440], 0.5625: [1080, 1920], 1.7778: [1920, 1080]}
TARGET_SIZE_DICT = {
    "272p": TARGET_SIZE_272P,
    "480p": TARGET_SIZE_480P,
    "720p": TARGET_SIZE_720P,
    "1080p": TARGET_SIZE_1080P,
}


class NormalizeToTensor(object):
    """image (HWC, uint8) -> tensor (CHW, float32) in [-1, 1]."""

    def __init__(self, reshape=True):
        self.reshape = reshape

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        if self.reshape:
            image = np.reshape(image, (image.shape[0], image.shape[1], -1))
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


def get_resize_crop_size(input_size, target_size):
    """input_size = [H, W]. Pick closest aspect target, resize (cover), crop."""
    input_ratio = input_size[0] / input_size[1]
    closest_ratio = min(target_size.keys(), key=lambda r: abs(float(r) - input_ratio))
    crop_size = target_size[closest_ratio]
    ratio = max(crop_size[0] / input_size[0], crop_size[1] / input_size[1])
    resize_size = (int(input_size[0] * ratio), int(input_size[1] * ratio))
    return resize_size, crop_size


def get_ffmpeg_exe(explicit=None):
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


class RawVideoExtractor:
    """ffmpeg-subprocess frame extractor, matching the reference filter chain."""

    # HDR transfer characteristics that need tone mapping to SDR (BT.709).
    HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG
    # HDR->SDR tone-map filter (done in linear light, then back to bt709).
    TONEMAP_FILTER = ("zscale=t=linear:npl=100,tonemap=tonemap=hable:desat=0,"
                      "zscale=p=bt709:t=bt709:m=bt709:r=tv")

    def __init__(self, target_size, max_frames=121, fps=24, ffmpeg="ffmpeg", ffprobe="ffprobe", tonemap="auto"):
        self.target_size = target_size
        self.max_frames = max_frames
        self.fps = fps
        self.end_time = max_frames // fps + 1  # 121//24+1 = 6
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.tonemap = tonemap  # "auto": tonemap only HDR videos; "off": never
        self._norm = NormalizeToTensor()

    def _probe(self, video_file, attribute):
        cmd = [
            self.ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", f"stream={attribute}",
            "-of", "default=noprint_wrappers=1:nokey=1", video_file, "-loglevel", "quiet",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        if attribute == "duration":
            return float(out)
        if attribute in ("width", "height"):
            return int(out)
        return out  # strings like color_transfer

    def _is_hdr(self, video_file):
        try:
            return self._probe(video_file, "color_transfer") in self.HDR_TRANSFERS
        except Exception:
            return False

    def get_video_data(self, video_file):
        vid_time = self._probe(video_file, "duration")
        vid_width = self._probe(video_file, "width")
        vid_height = self._probe(video_file, "height")

        start = 0
        end = min(vid_time, self.end_time)

        resize_size, crop_size = get_resize_crop_size([vid_height, vid_width], self.target_size)

        # Insert HDR->SDR tone mapping (before scale/crop) only for HDR sources.
        do_tonemap = self.tonemap == "auto" and self._is_hdr(video_file)
        tm = self.TONEMAP_FILTER + "," if do_tonemap else ""
        cmd = (
            f"{self.ffmpeg} -ss {start} -to {end} -i {video_file} "
            f"-vf fps={self.fps},{tm}scale={resize_size[1]}:{resize_size[0]}:flags=bilinear,"
            f"crop=w={crop_size[1]}:h={crop_size[0]} -vsync 0 -vcodec rawvideo "
            f"-pix_fmt rgb24 -f image2pipe -"
        )
        pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = pipe.communicate()

        bytes_per_img = crop_size[1] * crop_size[0] * 3
        if len(stdout) % bytes_per_img != 0:
            print(f"WARN wrong frame bytes for {video_file}")
        img_num = len(stdout) // bytes_per_img

        img_list = []
        idx = 0
        while idx < min(img_num, self.max_frames):
            s = idx * bytes_per_img
            e = (idx + 1) * bytes_per_img
            img = Image.frombytes("RGB", (crop_size[1], crop_size[0]), stdout[s:e])
            img_list.append(self._norm(img))
            idx += 1

        if len(img_list) == 0:
            raise RuntimeError(f"no frames extracted from {video_file}")
        # stack over time -> [T, C, H, W]
        video_data = torch.stack(img_list, dim=0)
        return video_data


def load_vae(vae_ckpt_path, device, dtype):
    # Requires the official `wan` package installed (pip install the Wan2.1/2.2 repo).
    from wan.modules.vae2_1 import Wan2_1_VAE
    return Wan2_1_VAE(vae_pth=vae_ckpt_path, device=device, dtype=dtype)


def list_videos(video_dir, exts=("mp4", "MP4", "avi")):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
    return sorted(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt_path", default="Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                   help="Wan2.1 VAE weight file (Wan2.1_VAE.pth)")
    p.add_argument("--video_dir", required=True, help="dir of input videos (.mp4)")
    p.add_argument("--save_path", required=True, help="output dir for the .pt/.json files")
    p.add_argument("--image_size", default="720p", choices=list(TARGET_SIZE_DICT.keys()))
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--max_frames_limit", type=int, default=121)
    p.add_argument("--shard", default="0/1", help="i/n : this process handles video index%%n==i")
    p.add_argument("--ffmpeg", default="", help="override ffmpeg binary (default: imageio_ffmpeg 7.0.2)")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--tonemap", default="auto", choices=["auto", "off"],
                   help="auto: HDR(PQ/HLG) videos get HDR->SDR tone mapping, SDR untouched (matches reference). off: never tonemap (byte-exact reference).")
    p.add_argument("--limit", type=int, default=0, help="cap number of videos (0=all), useful for tests")
    args = p.parse_args()

    assert torch.cuda.is_available(), "requires at least one GPU"
    device = "cuda"
    dtype = torch.bfloat16
    target_size = TARGET_SIZE_DICT[args.image_size]

    si, sn = [int(x) for x in args.shard.split("/")]

    ffmpeg = get_ffmpeg_exe(args.ffmpeg if args.ffmpeg else None)
    print(f"[shard {si}/{sn}] ffmpeg={ffmpeg}")

    vae = load_vae(args.vae_ckpt_path, device, dtype)
    reader = RawVideoExtractor(target_size, max_frames=args.max_frames_limit,
                               fps=args.fps, ffmpeg=ffmpeg, ffprobe=args.ffprobe, tonemap=args.tonemap)

    os.makedirs(args.save_path, exist_ok=True)

    videos = list_videos(args.video_dir)
    videos = [v for i, v in enumerate(videos) if i % sn == si]
    if args.limit:
        videos = videos[: args.limit]
    print(f"[shard {si}/{sn}] {len(videos)} videos to process")

    done, t0 = 0, time.time()
    for vp in videos:
        vid_name = os.path.splitext(os.path.basename(vp))[0]
        save_pt = os.path.join(args.save_path, vid_name + ".pt")
        save_json = os.path.join(args.save_path, vid_name + ".json")
        if os.path.exists(save_json):
            print(f"skip (exists): {save_json}")
            continue

        try:
            tic = time.time()
            video = reader.get_video_data(vp)  # [T, C, H, W]
            video = rearrange(video, "t c h w -> c t h w").to(device).to(dtype)
            with torch.no_grad():
                z = vae.encode([video])[0]  # [C=16, T, H, W]
            z = rearrange(z, "c t h w -> t c h w").to(torch.bfloat16)
        except Exception as e:
            print(f"ERROR {vp}: {e}")
            continue

        output_dict = {
            "video_fn": vid_name + ".mp4",
            "4_vae_feature_shape": [int(s) for s in z.shape],
            "4_vae_feature_length": int(z.shape[0]),
        }
        torch.save(z.cpu(), save_pt)
        with open(save_json, "w") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False))

        done += 1
        dt = time.time() - tic
        print(f"[{done}] {vid_name} shape={tuple(z.shape)} {dt:.2f}s")

    elapsed = time.time() - t0
    if done:
        print(f"[shard {si}/{sn}] done {done} videos, {elapsed:.1f}s, {elapsed/done:.2f}s/video")
    else:
        print(f"[shard {si}/{sn}] nothing processed")


if __name__ == "__main__":
    main()
