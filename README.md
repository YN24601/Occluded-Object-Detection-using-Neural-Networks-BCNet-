# BCNet on COCOA

Re-implementation of **BCNet** (Ke et al., *Deep Occlusion-Aware Instance
Segmentation with Overlapping BiLayers*, CVPR 2021) on top of
**Detectron2 0.6**, trained on the **COCOA** amodal dataset.

> Status: all four phases (skeleton / data / model / training+eval) are
> complete. The bilayer head matches the paper's structure end-to-end:
> non-local reasoning in *both* layers, additive residual fusion of Layer 1
> features into Layer 2, and an independent boundary deconv per branch.
> Boundary supervision uses the paper's BCE + Gaussian-weighted-BCE
> joint loss. Everything that can be ablated (boundary on/off, GCN on/off,
> class-agnostic vs per-class occludee) is exposed via config flags. An
> amodal head is *not* implemented (the BCNet paper doesn't ship one
> either).

## Idea in one paragraph

BCNet replaces Mask R-CNN's single ROI mask head with **two stacked
branches** inside each ROI:

* **Layer 1 (occluder, class-agnostic):** predicts the mask of whatever
  is *covering* the target inside the ROI.
* **Layer 2 (occludee, per-class or class-agnostic):** predicts the
  target's own visible (modal) mask. Layer 1's post-attention feature
  is fused into Layer 2's input via additive residual (`x_roi + f_L1`),
  giving Layer 2 globally-reasoned occluder context for free.

Both layers have a non-local **GCN** block after their conv stack for
ROI-level graph reasoning, and each layer has a parallel **boundary**
predictor (own deconv + 1x1) supervised by a morphological-gradient
contour map under BCE + Gaussian-weighted-BCE loss — the paper's
contour-aware joint loss. Explicitly modelling "what is occluding what"
gives a strong signal in heavily occluded scenes; exactly what COCOA's
amodal annotations expose.

## Project layout

```
BCNet/
├── bcnet/                          # the package (everything importable)
│   ├── data/
│   │   ├── build.py                # DatasetCatalog registration (keeps extra RLE keys)
│   │   └── mapper.py               # BCNetDatasetMapper: visible + occluder + amodal GT
│   ├── models/
│   │   └── bilayer_head.py         # BCNetBilayerMaskHead: occluder + occludee + GCN + boundary
│   ├── utils/
│   │   ├── config.py               # setup_bcnet_config() + cfg.BCNET / cfg.WANDB defaults
│   │   └── wandb_writer.py         # init_wandb() + WandbWriter (EventStorage -> W&B)
│   └── evaluation.py               # build_bcnet_evaluator -> COCOEvaluator on visible eval
├── configs/
│   ├── bcnet_train.yaml            # default — tuned for 4 GB VRAM (GTX 1650)
│   └── bcnet_train_8gb.yaml        # 8 GB+ VRAM variant; longer schedule, bigger inputs
├── tools/
│   ├── build_mini_split.py         # filter COCOA by occlude_rate -> mini JSON
│   ├── build_occluder_anns.py      # derive occluder masks from COCOA cross-instance overlap
│   ├── build_eval_anns.py          # emit visible-mask eval JSON (seg <- visible_mask)
│   ├── viz_occluder_samples.py     # 4-panel: img / visible / invisible / derived occluder
│   ├── check_mapper.py             # mapper end-to-end smoke test (no model)
│   ├── check_forward.py            # full model forward + loss smoke test
│   ├── plot_losses.py              # render loss curves from metrics.json
│   └── viz_predictions.py          # overlay predicted occludee + occluder on val images
├── data/cocoa-cls/                 # NOT in git — see "Dataset" below
│   ├── annotations/                #   put COCOA *_with_classes.json here
│   ├── train2014/                  #   put unzipped COCO 2014 train images here
│   └── val2014/                    #   put unzipped COCO 2014 val images here
├── train.py                        # DefaultTrainer entry-point + BCNetTrainer subclass
├── quick_check.py                  # static checks: no PyTorch required
├── requirements.txt
├── .env.example                    # PYTHONUTF8 / KMP_DUPLICATE_LIB_OK (Windows)
└── output/                         # per-run artifacts (one sub-dir per run)
    ├── run_paper_align/            # example: paper-aligned run reported below
    │   ├── config.yaml             #   resolved cfg actually used by this run (post-merge)
    │   ├── log.txt                 #   full Detectron2 stdout/stderr capture
    │   ├── metrics.json            #   JSON-lines, one record per logging step (feeds plot_losses.py)
    │   ├── events.out.tfevents.*   #   TensorBoard event stream (same data as metrics.json)
    │   ├── last_checkpoint         #   text file pointing at the most recent .pth
    │   ├── model_*.pth             #   periodic + final checkpoints (heavy — gitignore if pushing)
    │   ├── loss_curves.png         #   rendered by tools/plot_losses.py
    │   ├── eval/<dataset>/         #   COCOEvaluator dump: coco_instances_results.json + per-class AP
    │   └── viz/                    #   rendered by tools/viz_predictions.py — pred mask overlays
    └── viz_occluder/               # output of tools/viz_occluder_samples.py (data viz, no model)
```

## Dataset

This repo does **not** ship the dataset. You need to download two
things and drop them under `data/cocoa-cls/`:

### 1. COCO 2014 images

COCOA is built on top of COCO 2014. We use the train + val splits.

| Split    | Where                                                                            | Size     |
| -------- | -------------------------------------------------------------------------------- | -------- |
| `train2014.zip` | https://images.cocodataset.org/zips/train2014.zip                       | ~13 GB   |
| `val2014.zip`   | https://images.cocodataset.org/zips/val2014.zip                         | ~6.2 GB  |

```bash
cd data/cocoa-cls
curl -O https://images.cocodataset.org/zips/val2014.zip
curl -O https://images.cocodataset.org/zips/train2014.zip
unzip val2014.zip      # -> data/cocoa-cls/val2014/
unzip train2014.zip    # -> data/cocoa-cls/train2014/
```

> The default config (`configs/bcnet_train.yaml`) only points at
> `val2014/` because the mini split is sampled from there. You can skip
> downloading `train2014.zip` until you're ready for a full-scale run.

### 2. COCOA "with_classes" annotations

Original release: Patrick Follmann et al., MVTec 2018 — *Learning to
See the Invisible: End-to-end Trainable Amodal Instance Segmentation*
([paper](https://arxiv.org/abs/1804.08864) /
[archive](https://www.amazonaws.cn/en/s3/) — the canonical mirror moves
over time; check the paper's project page for the current link).

What you need to end up with:

```
data/cocoa-cls/annotations/
├── COCO_amodal_train2014_with_classes.json    # (~280 MB)
├── COCO_amodal_val2014_with_classes.json      # (~140 MB)
└── COCO_amodal_info.txt                       # already in repo: upstream README
```

> The two `*_with_classes.json` files are the only inputs the pipeline
> consumes. Everything else (`cocoa_mini_test.json`,
> `cocoa_mini_test_with_occluder.json`, `cocoa_mini_visible_eval.json`)
> is **derived** by `tools/build_*.py` and is gitignored.

### 3. Build the derived annotation files

Once images + raw `*_with_classes.json` are in place:

```bash
# (1) filter COCOA val to a 100-image high-occlusion subset
python tools/build_mini_split.py
# (2) derive the occluder mask per annotation (cross-instance overlap)
python tools/build_occluder_anns.py \
    --input  data/cocoa-cls/annotations/cocoa_mini_test.json \
    --output data/cocoa-cls/annotations/cocoa_mini_test_with_occluder.json
# (3) build a visible-mask eval JSON for COCOEvaluator
python tools/build_eval_anns.py \
    --input  data/cocoa-cls/annotations/cocoa_mini_test_with_occluder.json \
    --output data/cocoa-cls/annotations/cocoa_mini_visible_eval.json
```

Verify with `python quick_check.py` (no PyTorch required).

## Environment


| package      | version          | notes                                                            |
|--------------|------------------|------------------------------------------------------------------|
| Python       | 3.11.15          |                                                                  |
| torch        | 2.6.0 + cu124    | install from the PyTorch wheel index (see `requirements.txt`)    |
| torchvision  | 0.21.0 + cu124   |                                                                  |
| detectron2   | 0.6              | no public wheel for torch 2.6 — install from git (see below)     |
| numpy        | 2.0.1            | NumPy 2 — match this; older versions break torch 2.6 ABI         |
| pycocotools  | 2.0.11           |                                                                  |
| fvcore       | 0.1.5.post...    | bundled by detectron2                                            |
| setuptools   | **< 81**         | detectron2.model_zoo needs `pkg_resources`, gone in setuptools 81|

Install:

```bash
# 1. torch + torchvision (the CUDA wheel must match your driver)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. everything else pinned
pip install -r requirements.txt

# 3. detectron2 from source (matches the commit we developed against)
pip install "git+https://github.com/facebookresearch/detectron2.git@b599f139756bd3646a26a909caf86a1a159e53a7"

# 4. setuptools pin (detectron2.model_zoo imports pkg_resources)
pip install "setuptools<81"
```

### Windows-only env vars

Copy `.env.example` to `.env` and load it, or set the two variables in
your shell / PyCharm run config:

```powershell
$env:PYTHONUTF8 = "1"            # fvcore reads YAML in default codec; UTF-8 crashes on cp936
$env:KMP_DUPLICATE_LIB_OK = "TRUE"  # torch + numpy + matplotlib all ship OpenMP; Windows aborts on duplicates
```

On Linux/macOS neither is needed.

## Quick start

```bash
python quick_check.py                                 # static layout + ann format
python tools/check_mapper.py                          # data mapper end-to-end (no model)
python tools/check_forward.py                         # build model + 1-batch forward + loss
python train.py --num-gpus 1                          # train with default cfg
python train.py --eval-only MODEL.WEIGHTS output/run/model_final.pth
python tools/plot_losses.py --metrics output/run/metrics.json --out output/run/loss_curves.png
python tools/viz_predictions.py --weights output/run/model_final.pth --out-dir output/run/viz --n 6
```

Anything from the YAML can be overridden inline via `KEY VALUE` pairs at
the end of the command. Some common ones:

```bash
# Shorter overfit probe
python train.py SOLVER.MAX_ITER 200 OUTPUT_DIR ./output/overfit

# Toggle the boundary head
python train.py BCNET.HEAD.USE_BOUNDARY True OUTPUT_DIR ./output/run_boundary

# Use the 8 GB config (longer schedule, larger inputs)
python train.py --config-file configs/bcnet_train_8gb.yaml

```

## Experiment tracking (Weights & Biases)

Tracking is **off by default** so the repo runs offline. To stream metrics
to [wandb.ai](https://wandb.ai), log in once (`wandb login`, or export
`WANDB_API_KEY`) then flip `WANDB.ENABLED`:

```bash
python train.py \
    WANDB.ENABLED True \
    WANDB.PROJECT bcnet-cocoa \
    WANDB.RUN_NAME run2k_gcn \
    WANDB.TAGS '["mini","gcn"]' \
    OUTPUT_DIR ./output/run2k_gcn
```

A `WandbWriter` (added to the trainer's writer list in `train.py`) drains
Detectron2's `EventStorage`, so **every** scalar already logged to
`metrics.json` — per-step `total_loss` / `loss_*` / `lr` / timing, plus the
periodic COCOEvaluator metrics (`bbox/AP`, `segm/AP`, ...) `EvalHook` writes
back — is mirrored to W&B under the same keys. The full resolved cfg is
logged as the run config for reproducibility. `--eval-only` runs log the
final flattened eval results. All knobs live under `cfg.WANDB` (see
`configs/bcnet_train.yaml`); set `WANDB_MODE=offline` to log locally without
a network connection.

## Tuning for your GPU

The default config targets **4 GB VRAM**. If you are running on a different GPU, you may need to retune the parameters. The primary knobs are located in `configs/bcnet_train.yaml` and are commented in line. Start with these:

| Symptom                                  | What to try                                                                |
|------------------------------------------|----------------------------------------------------------------------------|
| OOM after `RPN` step                     | lower `MODEL.RPN.PRE_NMS_TOPK_TRAIN`, drop `INPUT.MAX_SIZE_TRAIN` to 600   |
| OOM in mask head                         | lower `MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE` (64 fits 4 GB; 128 fits ~6 GB)|
| Plenty of VRAM left                      | raise `MODEL.BACKBONE.FREEZE_AT` (5 → 2 → 0), then `IMS_PER_BATCH`         |
| Loss stalls at ~0.7 for `loss_occluder_mask` | bump `BCNET.LOSS.OCCLUDER_MASK_WEIGHT` to 3-10 (sparse occluder GT)    |
| AP keeps climbing at `MAX_ITER`           | raise `SOLVER.MAX_ITER`; keep `STEPS` at ~(0.7, 0.9) of the new max       |

For an 8 GB+ GPU, `configs/bcnet_train_8gb.yaml` is a working starting
point — it inherits from the 4 GB config via Detectron2's `_BASE_`
mechanism and only changes the knobs that should change. For larger
cards, edit that file (no need to retype the whole config).

## Architecture status

| Component                                       | Status          | Config flag                                  |
|-------------------------------------------------|-----------------|----------------------------------------------|
| Bilayer ROI head (occluder + occludee)          | implemented     | hardcoded (the whole point)                  |
| Layer 2 input fusion: `x_roi + f_L1` (residual) | implemented     | hardcoded — matches the paper                |
| Class-agnostic occludee head                    | default ON      | `MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK`      |
| GCN (non-local) on Layer 1 *and* Layer 2        | default ON      | `BCNET.HEAD.USE_GCN`                         |
| Boundary supervision (BCE + Gaussian-weighted BCE) | default ON   | `BCNET.HEAD.USE_BOUNDARY`                    |
| Independent deconv per boundary branch          | implemented     | hardcoded — matches the paper                |
| Visible-mask mAP eval                           | implemented     | runs in `--eval-only`                        |

## Reproduced results

**Paper-aligned model** (1000 iter, mini set, GTX 1650, ~14 min). Loss
weights come straight from `configs/bcnet_train.yaml` — `0.5 / 0.5` for
the two boundary weights, matching the source repo's symmetric setup.

| Task | AP    | AP50  | AP75  | APs   | APm   | APl    |
|------|-------|-------|-------|-------|-------|--------|
| bbox | 16.51 | 42.07 | 10.20 | 2.01  | 15.80 | 25.25  |
| segm | 8.32  | 23.22 | 3.86  | 1.46  | 5.76  | 13.37  |

These numbers are about *data*, not about *code*: 100 images is too
small a regime to validate the boundary / GCN heads. For a real
comparison, train on full `train2014` — the schedule + config is
`configs/bcnet_train_8gb.yaml`.

## Caveats specific to COCOA dataset

* **Occluder-GT coverage is ~24 %.** Only ~24 % of mini-set annotations
  recover a non-empty *derived* occluder mask: COCOA only annotates 80
  COCO classes, so anything occluded by un-annotated stuff (walls,
  picture frames, image edges) cannot be recovered cross-instance.
  Loss `loss_occluder_mask` may collapse toward 0.005 with all-zero
  predictions on small splits — this is the data, not the model. Full
  `train2014` recovers a much higher fraction.

* **GCN must run in fp32.** The non-local softmax overflows under AMP
  fp16 and produces NaN. The block forces fp32 via
  `torch.amp.autocast(..., enabled=False)`; cost is negligible.

* **`pred_masks` vs `pred_occluder_masks` have different shapes.**
  `pred_masks` is full-image after `detector_postprocess`;
  `pred_occluder_masks` stays at 28x28 sigmoid probabilities. The viz
  script handles both — see `tools/viz_predictions.py`.

## References

* Ke, Tai, Tang. *Deep Occlusion-Aware Instance Segmentation with
  Overlapping BiLayers.* CVPR 2021.
* Follmann, König, Härtinger, Klostermann. *Learning to See the
  Invisible: End-to-End Trainable Amodal Instance Segmentation.* 2018.
* Zhu, Tian, Metaxas, Dollar. *Semantic Amodal Segmentation.* CVPR 2017.

## License

Code in this repository is provided under the MIT license. The COCOA
annotations and COCO 2014 images keep their original upstream licenses
— see the official COCO and COCOA distributions for terms.
