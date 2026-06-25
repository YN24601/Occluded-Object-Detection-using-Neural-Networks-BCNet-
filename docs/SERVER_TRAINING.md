# Full-COCOA Training on the WTM GPU Server (BCNet vs. Baseline)

End-to-end runbook for training **BCNet** and a **vanilla Mask R-CNN baseline**
on the **full COCOA** dataset on the WTM group's GPU server (`wtmgws11`), then
collecting convergence curves, random test-case visualizations, and a
side-by-side comparison.

Target GPU: **`[2] NVIDIA GeForce GTX 1080 Ti` (11 GB)** — selected with
`CUDA_VISIBLE_DEVICES=2`.

> **Storage rule (important).** `/export/home` has only ~10 GB free and is
> shared with pip/conda/HF caches. **Everything heavy — the dataset, the conda
> environment, caches, and all run outputs — must live under `/data`** (~158 GB
> free). This runbook puts everything under `/data/5zhang/`.

---

## 0. Conventions

```bash
# Your personal area on the big disk. Change "5zhang" if your username differs.
export WORK=/data/5zhang
```

All commands below assume you have SSH'd into `wtmgws11`.

---

## 1. Create the directory layout

```bash
mkdir -p $WORK/{miniconda3,envs,.cache/pip,.cache/torch}
mkdir -p $WORK/BCNet
```

Redirect caches OFF the small home disk (add these to `~/.bashrc` so they
persist across logins):

```bash
echo "export PIP_CACHE_DIR=$WORK/.cache/pip"   >> ~/.bashrc
echo "export TORCH_HOME=$WORK/.cache/torch"    >> ~/.bashrc   # detectron2 model-zoo + R-50.pkl land here
echo "export TMPDIR=$WORK/.cache/tmp"          >> ~/.bashrc
mkdir -p $WORK/.cache/tmp
source ~/.bashrc
```

Final layout:

```
/data/5zhang/
├── miniconda3/                 # conda install (NOT in $HOME)
├── envs/bcn_server/            # conda env prefix (NOT in $HOME)
├── .cache/{pip,torch,tmp}/     # caches redirected here
└── BCNet/                      # this repo
    ├── data/cocoa-cls/
    │   ├── annotations/        # raw *_with_classes.json + derived JSONs
    │   ├── train2014/          # ~13 GB
    │   └── val2014/            # ~6 GB
    ├── configs/                # server_*.yaml (committed)
    └── output/                 # bcnet_full / baseline_full / smoke
```

---

## 2. Get the code

```bash
cd $WORK
git clone <your-BCNet-remote-url> BCNet     # or rsync your local checkout up (Section 3)
cd $WORK/BCNet
```

If you have local-only commits, push them first or `rsync` the repo (excluding
`data/`, `output/`, `wandb/`):

```bash
# from your Windows machine (Git Bash / WSL):
rsync -avz --exclude data --exclude output --exclude wandb \
    /d/00workshop/pythonProjects/BCNet/ 5zhang@134.100.10.170:/data/5zhang/BCNet/
```

---

## 3. Environment

Install Miniconda **into `/data`** (the installer defaults to `$HOME` — override it):

```bash
cd $WORK
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $WORK/miniconda3
source $WORK/miniconda3/etc/profile.d/conda.sh
```

Create the env with an explicit prefix on `/data`:

```bash
conda create -y -p $WORK/envs/bcn_server python=3.11
conda activate $WORK/envs/bcn_server
```

Install the pinned stack (matches `README.md` / `requirements.txt`):

```bash
cd $WORK/BCNet

# 1. torch + torchvision (CUDA 12.4 wheels; the 1080 Ti / driver 565.x supports this)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. everything else pinned
pip install -r requirements.txt

# 3. detectron2 from the exact commit we developed against
pip install "git+https://github.com/facebookresearch/detectron2.git@b599f139756bd3646a26a909caf86a1a159e53a7"

# 4. setuptools pin (detectron2.model_zoo imports pkg_resources, gone in setuptools 81)
pip install "setuptools<81"
```

> **Linux note:** the Windows-only env vars (`PYTHONUTF8`, `KMP_DUPLICATE_LIB_OK`)
> are **not** needed here. `train.py` sets `KMP_DUPLICATE_LIB_OK` defensively
> anyway; it is harmless on Linux.

Verify CUDA sees the GPU:

```bash
CUDA_VISIBLE_DEVICES=2 python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.6.0+cu124 NVIDIA GeForce GTX 1080 Ti
```

Log in to W&B once (or `export WANDB_API_KEY=...`):

```bash
wandb login
```

---

## 4. Migrate the data

### 4a. COCO 2014 images — download directly on the server (faster than uploading)

```bash
cd $WORK/BCNet/data/cocoa-cls
curl -O https://images.cocodataset.org/zips/train2014.zip   # ~13 GB
curl -O https://images.cocodataset.org/zips/val2014.zip     # ~6.2 GB
unzip -q train2014.zip   # -> ./train2014/
unzip -q val2014.zip     # -> ./val2014/
rm train2014.zip val2014.zip    # reclaim ~19 GB once unzipped
```

### 4b. COCOA annotations — transfer from your machine (not downloadable from COCO)

The two raw `*_with_classes.json` files are the only inputs the pipeline needs.
From your Windows machine:

```bash
# Git Bash / WSL on the local box:
scp /d/00workshop/pythonProjects/BCNet/data/cocoa-cls/annotations/COCO_amodal_train2014_with_classes.json \
    /d/00workshop/pythonProjects/BCNet/data/cocoa-cls/annotations/COCO_amodal_val2014_with_classes.json \
    5zhang@134.100.10.170:/data/5zhang/BCNet/data/cocoa-cls/annotations/
```

---

## 5. Build the derived annotation files

The model trains on **derived** JSONs (occluder masks + a visible-mask eval
split), not the raw `*_with_classes.json`. Build them once. **Run inside `tmux`**
— the train-set occluder derivation is O(n²) per image over RLE masks and can
take a while on the full ~80k-image split.

```bash
cd $WORK/BCNet
tmux new -s prep
conda activate $WORK/envs/bcn_server

A=data/cocoa-cls/annotations

# (1) Derive occluder masks for the full TRAIN split (the long one).
python tools/build_occluder_anns.py \
    --input  $A/COCO_amodal_train2014_with_classes.json \
    --output $A/cocoa_train2014_with_occluder.json

# (2) Same for the VAL split (used for prediction visualization).
python tools/build_occluder_anns.py \
    --input  $A/COCO_amodal_val2014_with_classes.json \
    --output $A/cocoa_val2014_with_occluder.json

# (3) Build the visible-mask eval JSON COCOEvaluator scores against.
python tools/build_eval_anns.py \
    --input  $A/cocoa_val2014_with_occluder.json \
    --output $A/cocoa_val2014_visible_eval.json
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t prep`. Each script prints
how many annotations got a non-empty occluder mask — on the full train split the
coverage is much higher than the ~24 % seen on the mini set.

> These output filenames are exactly what `configs/server_base.yaml` points at.
> If you change them, update `BCNET.*_JSON` in that file too.

---

## 6. Smoke test (validate before committing hours)

Three increasingly heavy checks. Do all three before a real run.

```bash
cd $WORK/BCNet
conda activate $WORK/envs/bcn_server

# (a) static layout / annotation-format checks (no GPU, no torch)
python quick_check.py

# (b) build the model + one forward + loss on a single batch
CUDA_VISIBLE_DEVICES=2 python tools/check_forward.py

# (c) 100-iter run on the FULL data (proves the big JSONs load + register and
#     that the BCNet head fits in 11 GB). Watch VRAM in a second shell:
#       watch -n 1 nvidia-smi
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-file configs/server_smoke.yaml --num-gpus 1
```

`server_smoke.yaml` writes to `output/smoke/`, runs 100 iters, no eval, no W&B.
If it reaches iter 100 and stays under 11 GB, you are clear to train for real.

> If you OOM: drop `MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE` to 64, or set
> `INPUT.MAX_SIZE_TRAIN 1000`, or raise `MODEL.BACKBONE.FREEZE_AT` to 4 — all on
> the CLI as trailing `KEY VALUE` pairs, or edit `configs/server_base.yaml`.

---

## 7. Full training

Both runs share the same recipe (`configs/server_base.yaml`): **20k iterations**,
`IMS_PER_BATCH 2`, `BASE_LR 0.005`, eval every 2.5k iters, AMP on. They differ in
**one line** — the mask head — and write to separate output dirs and W&B run
names (grouped under `full-cocoa-compare` so they overlay automatically).

The single GPU runs them **sequentially**. Use `tmux` so they survive disconnects.

```bash
cd $WORK/BCNet
conda activate $WORK/envs/bcn_server

# --- BCNet (bilayer head: occluder + occludee, GCN + boundary) ---
tmux new -s bcnet
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-file configs/server_bcnet.yaml --num-gpus 1
# Ctrl-b d to detach.

# --- Baseline (stock Mask R-CNN mask head) — start after BCNet finishes ---
tmux new -s baseline
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-file configs/server_baseline.yaml --num-gpus 1
```

Outputs land in `output/bcnet_full/` and `output/baseline_full/`:
`model_*.pth`, `metrics.json`, `log.txt`, `events.out.tfevents.*`,
`eval/<dataset>/`. Live metrics stream to W&B.

To resume an interrupted run, add `--resume` (it picks up from
`last_checkpoint`).

> **Why this is a fair comparison.** `server_baseline.yaml` changes only
> `MODEL.ROI_MASK_HEAD.NAME` to Detectron2's `MaskRCNNConvUpsampleHead`. The
> backbone, FPN, RPN, schedule, inputs, and the dataset mapper are identical,
> and both heads are supervised by the **same** modal `visible_mask` GT
> (`gt_masks`). `CLS_AGNOSTIC_MASK` is kept `True` for both, so the only variable
> is single-layer vs. bilayer. (For a stock per-class Mask R-CNN baseline
> instead, set `MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK False` in that config.)

---

## 8. Collect results

### 8a. Per-run report (curves + random test-case overlays + AP table)

```bash
CUDA_VISIBLE_DEVICES=2 python tools/make_report.py \
    --config-file configs/server_bcnet.yaml --run-dir output/bcnet_full

CUDA_VISIBLE_DEVICES=2 python tools/make_report.py \
    --config-file configs/server_baseline.yaml --run-dir output/baseline_full
```

For each run this re-evaluates `model_final.pth` and writes into the run dir:

- `loss_curves.png` — per-step training losses (from `metrics.json`).
- `ap_curves.png` — eval `segm/AP`, `bbox/AP`, … vs. iteration.
- `viz/pred_*.png` — predictions on random val images (auto-detects the missing
  occluder panel on the baseline, so it works for both models).
- `report/summary.md` — the final visible-mask AP table.

### 8b. BCNet vs. baseline comparison

```bash
python tools/compare_runs.py \
    --runs output/bcnet_full output/baseline_full \
    --out-dir output/comparison
```

Writes `output/comparison/compare.md` (side-by-side final AP table) and
`compare_ap.png` (overlaid `segm/AP`-vs-iteration). The same overlay is also
visible live in the W&B `full-cocoa-compare` group.

### 8c. Pull artifacts back to your machine

```bash
# from the local box:
rsync -avz 5zhang@134.100.10.170:/data/5zhang/BCNet/output/comparison/ ./results/comparison/
rsync -avz --include='*/' --include='*.png' --include='*.md' --exclude='*' \
    5zhang@134.100.10.170:/data/5zhang/BCNet/output/bcnet_full/ ./results/bcnet_full/
rsync -avz --include='*/' --include='*.png' --include='*.md' --exclude='*' \
    5zhang@134.100.10.170:/data/5zhang/BCNet/output/baseline_full/ ./results/baseline_full/
```

(The `.pth` checkpoints are heavy — leave them on the server unless you need
them locally.)

---

## Quick reference

| Task              | Config                          | Output dir              |
|-------------------|---------------------------------|-------------------------|
| Smoke test        | `configs/server_smoke.yaml`     | `output/smoke`          |
| BCNet full train  | `configs/server_bcnet.yaml`     | `output/bcnet_full`     |
| Baseline full     | `configs/server_baseline.yaml`  | `output/baseline_full`  |

| Tunable           | Where                                              |
|-------------------|----------------------------------------------------|
| GPU selection     | `CUDA_VISIBLE_DEVICES=2`                            |
| Schedule length   | `SOLVER.MAX_ITER` / `STEPS` in `server_base.yaml`  |
| VRAM (OOM)        | `BATCH_SIZE_PER_IMAGE`, `INPUT.MAX_SIZE_TRAIN`, `FREEZE_AT` |
| Eval frequency    | `TEST.EVAL_PERIOD` in `server_base.yaml`           |
| W&B on/off        | `WANDB.ENABLED` (CLI: `WANDB.ENABLED False`)       |
