<!-- # Shell-LCC: Your Data Manifold is Secretly a Reward Model -->
<div align="center">

# Your Data Manifold is Secretly a Reward Model
### Shell-LCC for Text-to-Video Generation &nbsp;·&nbsp; ECCV 2026

[Project Page](https://needylove.github.io/Shell-LCC/) · [Paper](https://arxiv.org/pdf/2606.30248) · [Models (HuggingFace)](https://huggingface.co/Needylove/Shell-LCC)

</div>


Turn the **manifold of high-quality video data into a cost-free reward model** and use it to
finetune a text-to-video (Wan2.1-T2V) generator so it produces sharper, more detailed videos —
**no human labels, no external reward model**.

The manifold is modeled as an isotropic *shell* (Shell Local Coordinate Coding). Generated video
latents are pulled onto that shell, giving a dense, differentiable, annotation-free reward:
`R(z) = ‖ Σ^{-1/2}(z − ẑ) ‖`, where `ẑ` is the local linear (LCC) reconstruction of latent patch
`z` and `Σ` is a learned per-dimension scale.

---

## Pipeline

```
videos ──①extract──▶ VAE latents ──②train manifold──▶ Shell-LCC ckpt ──③finetune──▶ T2V model ──④evaluate
        (Wan VAE)      (.pt)          (2 stages)         (reward)        (Wan2.1)        (metrics + compare)
```

| # | Stage | Script | GPU? |
|---|---|---|---|
| ① | Extract VAE latents from videos | `scripts/extract_wan_vae_feature.py` | yes |
| ② | Train the Shell-LCC manifold (2 stages) | `manifold/train_manifold_2stage.py` | yes (DDP) |
| ③ | Finetune T2V with the manifold reward | `scripts/train_T2V_model.py` | yes (DDP) |
| ④ | Generate → metrics + compare videos + detail montages | `scripts/generate_videos.py` + `eval_detail.py` + `generate_compare.py` + `make_montage.py` | ④a GPU, ④b–d CPU |

Every stage has a copy-paste launcher in `scripts/launchers/`.

---

## Install

```bash
# 1. Python deps
pip install -r requirements.txt
# 2. T2V backbone: the official Wan package (WanModel / Wan2.1 VAE / T5 / schedulers), so `import wan` works
pip install git+https://github.com/Wan-Video/Wan2.2.git
# 3. VAE / T5 weights
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

> If you hit a `CXXABI_x.x.x not found` error (old system libstdc++), the scripts auto-preload the
> conda env's libstdc++ (`_ensure_libstdcpp` / `_pre`); it is a no-op when not needed.

---

## Released checkpoints

Finetuned T2V checkpoints on [HuggingFace](https://huggingface.co/Needylove/Shell-LCC/tree/main)
(full state dicts; load with `--full_ckpt` in `scripts/generate_videos.py`, or `--init_ckpt` to
continue training):

| Checkpoint | Backbone | Recipe | Character |
|---|---|---|---|
| `wan2.1-t2v-1.3b_shell-lcc_lr1e-5_step100.pth` | Wan2.1-T2V-1.3B | lr 1e-5, eff. batch 8 | **balanced default** — strongest overall detail gain |
| `wan2.1-t2v-1.3b_shell-lcc_lr5e-6_step700.pth` | Wan2.1-T2V-1.3B | lr 5e-6, eff. batch 8 | **most faithful** — smallest content change; best on texture-heavy prompts |
| `wan2.1-t2v-14b_shell-lcc_lr1e-5_band0.444_step20.pth` | Wan2.1-T2V-14B | lr 1e-5, FSDP full, **band reward** (`--reward_target 0.444`), 20 steps | **recommended 14B** — the band target roughly doubles both the usable window and the detail gain vs. the plain reward |
| `wan2.1-t2v-14b_shell-lcc_lr1e-5_step10.pth` | Wan2.1-T2V-14B | lr 1e-5, FSDP full, 10 steps | plain-reward variant; richer fine detail (the 14B base is already sharp, so the gain is subtler than on 1.3B) |

`model/shell_lcc.pth` in this repo is the frozen Shell-LCC manifold used to train all three.

---

## Layout

```
manifold/
  shell_lcc.py               # ShellCoordinateManifold: LCC skeleton + shell head + distance_reward
  train_manifold_2stage.py   # two-stage manifold training (Stage-1 LCC, Stage-2 shell); DDP + single-GPU
model/
  shell_lcc.pth              # Trained shell-lcc model for wan2.1 VAE
scripts/
  extract_wan_vae_feature.py # encode videos -> Wan2.1 VAE latents (.pt), HDR tone-mapping, --shard
  decode_wan_vae_feature.py  # decode latents back to mp4 (sanity-check features / manifold reps)
  make_captions.py           # build a diverse prompt set (data/captions.txt) for reward finetuning
  train_T2V_model.py         # finetune Wan2.1-T2V with the frozen manifold reward (full/LoRA, DDP; --batch)
  generate_videos.py         # (eval ④a) generate base + finetuned videos, sharded across GPUs
  eval_detail.py             # (eval ④b) load videos -> detail metrics (lap/hf/change); CPU
  generate_compare.py        # (eval ④c) load videos -> side-by-side comparison mp4; CPU
  make_montage.py            # (eval ④d) load videos -> base|ckpt detail montage PNGs (eyeball); CPU
  launchers/                 # one .sh per stage (edit the paths at the top, then run)
requirements.txt             # python deps (Wan backbone installed separately, see Install)
```

---

## Quick start

```bash
# ① Extract VAE latents (shard across GPUs with --shard)
bash scripts/launchers/extract_vae_example.sh          # data/videos -> data/vae
#    (optional) eyeball a latent:  bash scripts/launchers/decode_vae_example.sh

# ② Train the Shell-LCC manifold (Stage-1 LCC, Stage-2 shell)
bash scripts/launchers/train_manifold_example.sh       # data/vae -> model/... (shell_lcc.pth)

# ③ Finetune the T2V model with the manifold reward
python scripts/make_captions.py                        # -> data/captions.txt
bash scripts/launchers/train_t2v_example.sh            # 1.3B -> model/t2v_run/step*/wan_full.pth
#    (14B: bash scripts/launchers/train_t2v_14b_example.sh — full-param + FSDP, stop early)

# ④ Evaluate: generate once (multi-GPU), then metrics + compare + montage (CPU, seconds)
NPROC=4 bash scripts/launchers/generate_videos_example.sh   # -> eval/videos + manifest.json
bash scripts/launchers/eval_compare_example.sh              # results.jsonl + *_compare.mp4 + *_montage.png
```

Launchers read paths from environment variables (e.g. `WAN_DIR`, `CKPT`, `NPROC`, `MAX_PROMPTS`,
`STEPS`); defaults are set at the top of each `.sh`.

---

## Two-stage manifold training

`train_manifold_2stage.py` mirrors the original manual "train LCC, then train the shell on top":

| Stage | Trains | Frozen | Loss | Default |
|---|---|---|---|---|
| **1 — LCC** | `basis` / `predictor` / `global_scale` / `global_bias` | — | `l1` + `l2` + `usage_kl` + `basis_to_patch` | 500 ep |
| **2 — Shell** | `surface_head` | all LCC params | `shell` + `reg` | 100 ep |

> **Why two stages (the paper's main text describes one-stage / joint training).** In practice we
> found the two-stage schedule works better on Wan. Training the LCC skeleton and the shell head
> *jointly* lets `global_bias` (and the rest of the LCC skeleton) drift substantially while
> `surface_head` is trying to fit the shell distribution — so the shell is learned on top of a
> **moving skeleton**, ends up mis-fit, and the final reward is **mis-calibrated**. Freezing the
> converged LCC skeleton first and only then training `surface_head` pins the shell to a stable
> skeleton and keeps the reward well-calibrated.

Multi-GPU: `torchrun --nproc_per_node=4 manifold/train_manifold_2stage.py ...` (single-GPU auto-fallback).
Output: `shell_ep{N}.pth = {"model": state_dict, "config": {embedding_dim, num_bases, hidden_dim}}`,
usable as `--manifold_ckpt` for stage ③.

---

## Training tips (from experiments)

**Finetuning the T2V model (stage ③) — learning rate (`--lr`), measured at effective batch 8 on
Wan2.1-T2V-1.3B (per-prompt metrics + frame inspection):**
- **`1e-5`** — default sweet spot; peaks around step 80–100 (balanced: strong detail on most prompts).
- **`5e-6`** — most faithful (content change stays smallest); improves slowly but keeps improving —
  peak around step 600–700, where texture-heavy prompts even overtake `1e-5`.

**Finetuning the T2V model (stage ③) — batch size:** going from tiny batches (1–2) to a moderate
effective batch (~8 = `batch × #GPUs`) clearly helps. Beyond that, at a fixed lr, larger batches
mostly make the model traverse the reward landscape faster — the peak arrives earlier. Treat effective batch 8 as
the default; if you scale it up, scale the lr schedule too.

**Scaling to Wan2.1-T2V-14B (stage ③):** use **full-param + FSDP** (`--full_parallel fsdp`) with lr `1e-5`
and effective batch 8 (`scripts/launchers/train_t2v_14b_example.sh`). With the plain reward the useful window is
**very short (~10 steps)**: the reward is minimized toward zero, but real data sits on a *shell* at a non-zero
distance, so a 14B model (which descends fast) quickly overshoots through the shell and collapses. The fix is the
**band reward** — `--reward_target τ` optimizes `|dist − τ|` instead, where τ is simply the logged `reward` value
of a known-good checkpoint (we use `0.444`, the value at plain-reward step 10). This roughly **doubles both the
usable window (~20 steps) and the detail gain**. Two caveats: (i) reward hacking still accumulates with steps —
high-frequency noise gradually stamps onto flat backgrounds (see the cat sequence on the project page) — so still
stop early and inspect frames; (ii) the gain on 14B is mainly **richer fine detail**: since the 14B base is
already quite sharp, the improvement is less pronounced than on 1.3B. In general the benefit scales with how
over-smoothed the base model is — our 4.5B model, whose outputs have the strongest over-smoothed "plastic" look,
shows the most striking improvement (see the project page).

**Training the manifold (stage ②) — loss can look flat while quality changes a lot.** The manifold
training loss often barely moves, yet the resulting generation quality differs substantially. **Do not
judge a manifold checkpoint by its loss curve** — generate videos and look. (Flat loss ≠ flat performance.)

---

## Evaluation metrics (`eval_detail.py`)

Generate once (`generate_videos.py`), then metrics and comparison are pure-CPU and reusable.

- **`lap`** — Laplacian variance = sharpness / edge energy. Higher = sharper. ⚠️ also responds to
  **noise**, so a high `lap` can be artifacts, not "good" detail.
- **`hf`** — high-frequency FFT energy ratio (energy above 1/4 Nyquist). Higher = more fine texture.
  ⚠️ high frequency = **detail OR noise** (grain, compression, ringing) — a high `hf` is **not** proof
  of better quality.
- **`change`** — mean pixel difference (0~1) between base and ckpt frames = how much content changed.

Reported as `lap_ratio`, `hf_ratio` (ckpt / base; >1 = more than base).

> **`lap`/`hf` are proxies — no metric here replaces looking at the videos.** Three failure modes we
> hit in practice: (1) when `change` is large, base and ckpt are effectively different videos, so their
> `lap`/`hf` are not comparable; (2) a **small `change` does not prove the video is healthy** — a model
> that collapses to a flat noise canvas whose color matches the scene's mean keeps `change` low while
> `lap`/`hf` skyrocket; (3) `hf_ratio` explodes on low-texture prompts (a clear sky has near-zero base
> high-frequency energy), and noise itself is high-frequency — treat `hf_ratio >> lap_ratio` as a
> **noise / reward-hacking alarm**, not as quality. A *genuine* improvement is roughly `lap_ratio>1 AND
> hf_ratio>1 AND change modest` — and even then, **watch the `*_compare.mp4` videos / montages** to
> confirm it is real detail.

For a static, zoom-in look at the pixels, `make_montage.py` stitches `base | ckpt` frames into a PNG
per prompt (optionally center-cropped and enlarged) — the eyeball companion to the proxy metrics:

```bash
python scripts/make_montage.py --videos_dir eval/videos              # full-frame base|ckpt
python scripts/make_montage.py --videos_dir eval/videos --zoom 0.4   # zoom into central detail
```

---

## Optional tricks 

Optional knobs / choices we explored, documented for reproducibility. Most had **little effect** and
none is part of the core method:

- **Anti-drift regularizer** (`train_T2V_model.py`; enabled by dropping `--no_drift`): adds
  `relu(MSE(z_gen, z_ref) − margin)` to keep the finetuned output near the base. Usually does not
  help. A **high** drift weight can *delay* collapse, but once the model has collapsed that is moot —
  so we default to `--no_drift` (pure manifold reward, the paper setting).
- **Codebook regularizers in manifold Stage 1** (`--lambda_usage`, `--lambda_basis_to_patch`):
  `usage_kl` pushes codebook usage toward uniform (anti-collapse); `basis_to_patch` pulls each basis
  toward its nearest data patch. We did **not** find either to help meaningfully, but they are
  available (default `0.1`, set to `0` to disable) and harmless if you want to try them.
- **Cross-resolution manifold.** The default is 720p, but a manifold trained at a *higher* resolution
  can reward a *lower*-resolution T2V model (e.g. use a 1080p-VAE manifold to finetune a 720p T2V
  model) — this **does work**. Caveat: the higher-resolution patches can be **too fine** for the lower
  target, sometimes making the output look busy / cluttered.

---

## Notes

- `--batch` in `train_T2V_model.py` is a **true batch** (B noises in one forward), decoupled from #GPUs;
  effective batch = `batch × #GPUs`. Moderate batches (effective ≈ 8) work best in our runs; larger
  batches at a fixed lr peak earlier and can underperform — stop at the peak step.
- The `neg` (negative prompt) strings in the generation scripts are kept in Chinese on purpose: Wan2.1
  was trained with Chinese negatives, which work better than an English translation. They are model
  input, not comments.

## Acknowledgements

This project builds on excellent open-source work:

- **[Wan2.1 / Wan2.2](https://github.com/Wan-Video/Wan2.2)** — the text-to-video backbone we finetune
  (WanModel, Wan2.1 VAE, T5 encoder, and schedulers).
- **UltraWan** — high-resolution Wan LoRA, usable as an optional finetuning start (`--init_ckpt`).
- **[UltraVideo](https://huggingface.co/datasets/APRIL-AIGC/UltraVideo)** (APRIL-AIGC) — the
  high-quality SFT video data our manifold is trained on, itself built on **Panda-70M** and **Koala-36M**.

Many thanks to the authors of these projects for releasing their models and data.

## Citation

```bibtex
@inproceedings{zhang2026shelllcc,
  title     = {Your Data Manifold is Secretly a Reward Model:
               Shell-LCC for Text-to-Video Generation},
  author    = {Zhang, Shihao and Li, Yunzhi and Yan, Yuguang and
               Zhang, Junzhe and Zhao, Wei and Wang, Bohan and Zhang, Hanwang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```