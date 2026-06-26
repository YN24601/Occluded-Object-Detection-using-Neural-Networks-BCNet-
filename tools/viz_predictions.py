"""Visualize BCNet predictions on a few images from the val split.

For each sampled image we render a 3-panel figure:
    [image + boxes]  [predicted occludee masks]  [predicted occluder masks]

The occluder overlay uses `inst.pred_occluder_masks`, the BCNet-specific
field attached by BCNetBilayerMaskHead during inference.

Usage:
    PYTHONUTF8=1 python tools/viz_predictions.py \
        --config-file configs/bcnet_train.yaml \
        --weights output/overfit200/model_final.pth \
        --out-dir output/viz_pred \
        --n 6
"""

from __future__ import annotations

import os

# Windows: torch, numpy (MKL), matplotlib and opencv each ship their own copy of
# Intel's OpenMP runtime (libiomp5md.dll). The runtime aborts when a duplicate is
# loaded, so allow it here. Must run before any of those libraries are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.modeling import build_model

import bcnet  # noqa: F401  (registers BCNetBilayerMaskHead)
from bcnet.data import register_cocoa_datasets
from bcnet.utils import setup_bcnet_config


def _overlay(img: np.ndarray, mask: np.ndarray, color, alpha=0.5):
    out = img.copy()
    out[mask] = (alpha * np.array(color) + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def _paste_mask_in_image(mask_28: np.ndarray, box_xyxy, h: int, w: int) -> np.ndarray:
    """Resize a 28x28 (or HxW) predicted mask back into the full image canvas."""
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=bool)
    bw, bh = x1 - x0, y1 - y0
    m = torch.as_tensor(mask_28).float().unsqueeze(0).unsqueeze(0)
    m = F.interpolate(m, size=(bh, bw), mode="bilinear", align_corners=False)
    m = (m[0, 0].numpy() >= 0.5)
    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = m
    return full


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/bcnet_train.yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--score-thresh", type=float, default=0.3)
    args = p.parse_args()

    cfg = setup_bcnet_config(args.config_file)
    # Lower the model's internal NMS/score filter so under-trained checkpoints
    # still surface predictions for visual inspection.
    cfg.defrost()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, args.score_thresh
    )
    cfg.freeze()
    register_cocoa_datasets(cfg)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.eval()
    device = next(model.parameters()).device

    # Match COCOEvaluator/DefaultPredictor test-time preprocessing: resize the
    # short edge to MIN_SIZE_TEST (capped at MAX_SIZE_TEST) so the model runs at
    # its trained scale. Feeding native-resolution images degrades predictions.
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dicts = DatasetCatalog.get(cfg.BCNET.VAL_DATASET_NAME)

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(dataset_dicts), size=min(args.n, len(dataset_dicts)), replace=False)

    for k, idx in enumerate(sample_idx):
        d = dataset_dicts[int(idx)]
        bgr = cv2.imread(d["file_name"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        # Feed the model an image in its configured colour order (cfg.INPUT.FORMAT,
        # "BGR" by default), resized to the test scale. height/width stay at the
        # ORIGINAL size so detector_postprocess rescales masks/boxes back onto the
        # full-resolution `rgb` canvases used for the overlays below.
        model_img = rgb if cfg.INPUT.FORMAT == "RGB" else bgr
        model_img = aug.get_transform(model_img).apply_image(model_img)
        img_t = torch.as_tensor(
            np.ascontiguousarray(model_img.transpose(2, 0, 1)).astype("float32")
        ).to(device)
        inputs = [{"image": img_t, "height": h, "width": w}]

        with torch.no_grad():
            outs = model(inputs)
        inst = outs[0]["instances"].to("cpu")
        keep = inst.scores >= args.score_thresh
        inst = inst[keep]
        if len(inst) == 0:
            print(f"  [{k}] image {d['file_name']}: no detections above {args.score_thresh}")
            continue

        # `inst.pred_masks` was already pasted into full-image resolution by
        # detector_postprocess (shape (N, H_img, W_img), binarized). The
        # BCNet-specific `pred_occluder_masks` is raw (N, 1, 28, 28) since
        # postprocess only touches the standard `pred_masks` field. The baseline
        # (stock Mask R-CNN head) does NOT emit it, so render only the occludee
        # panel in that case.
        has_occ = inst.has("pred_occluder_masks")
        occludee_canvas = rgb.copy()
        occluder_canvas = rgb.copy()
        for i in range(len(inst)):
            occludee_full = inst.pred_masks[i].numpy().astype(bool)
            occludee_canvas = _overlay(occludee_canvas, occludee_full, (60, 220, 60))
            if has_occ:
                box = inst.pred_boxes.tensor[i].numpy()
                occluder_28 = inst.pred_occluder_masks[i, 0].numpy()
                occluder_full = _paste_mask_in_image(occluder_28, box, h, w)
                occluder_canvas = _overlay(occluder_canvas, occluder_full, (220, 60, 220))

        n_panels = 3 if has_occ else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
        axes[0].imshow(rgb)
        for i in range(len(inst)):
            x0, y0, x1, y1 = inst.pred_boxes.tensor[i].numpy()
            axes[0].add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1.5)
            )
        axes[0].set_title(f"image + boxes (n={len(inst)})")
        axes[1].imshow(occludee_canvas); axes[1].set_title("occludee (visible) pred")
        if has_occ:
            axes[2].imshow(occluder_canvas); axes[2].set_title("occluder pred")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out_path = args.out_dir / f"pred_{k:02d}_img{d.get('image_id', 'x')}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}  ({len(inst)} dets)")


if __name__ == "__main__":
    main()
