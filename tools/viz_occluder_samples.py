"""Visualize a few derived occluder masks side-by-side with COCOA's GT.

Produces a 4-panel figure per sample:
    [original]  [visible (occludee modal)]  [invisible]  [derived occluder]

Usage:
    python tools/viz_occluder_samples.py \
        --ann data/cocoa-cls/annotations/cocoa_mini_test_with_occluder.json \
        --img-dir data/cocoa-cls/val2014 \
        --out-dir output/viz_occluder \
        --n 8
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_util


def decode_rle(rle: dict | None, h: int, w: int) -> np.ndarray:
    if rle is None:
        return np.zeros((h, w), dtype=bool)
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def overlay(img: np.ndarray, mask: np.ndarray, color=(255, 50, 50), alpha=0.55):
    out = img.copy()
    out[mask] = (alpha * np.array(color) + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n", type=int, default=8, help="number of annotations to viz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--require-occluder",
        action="store_true",
        help="only sample annotations whose occluder mask is non-empty",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.ann.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images_by_id = {img["id"]: img for img in data["images"]}

    candidates = data["annotations"]
    if args.require_occluder:
        candidates = [a for a in candidates if a.get("occluder_area", 0) > 0]

    sample = rng.sample(candidates, k=min(args.n, len(candidates)))

    for k, ann in enumerate(sample):
        img_meta = images_by_id[ann["image_id"]]
        h, w = img_meta["height"], img_meta["width"]
        img_path = args.img_dir / img_meta["file_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  skip {img_path} (read failed)")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        visible = decode_rle(ann.get("visible_mask"), h, w)
        invisible = decode_rle(ann.get("invisible_mask"), h, w)
        occluder = decode_rle(ann.get("occluder_mask"), h, w)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb); axes[0].set_title("image")
        axes[1].imshow(overlay(rgb, visible, (60, 220, 60)))
        axes[1].set_title(f"visible (occludee modal) | area={int(visible.sum())}")
        axes[2].imshow(overlay(rgb, invisible, (220, 220, 60)))
        axes[2].set_title(f"invisible | area={int(invisible.sum())} | occ_rate={ann.get('occlude_rate',0):.2f}")
        axes[3].imshow(overlay(rgb, occluder, (220, 60, 220)))
        axes[3].set_title(f"derived occluder | area={int(occluder.sum())}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"image_id={ann['image_id']}  ann_id={ann['id']}  cat_id={ann['category_id']}")
        fig.tight_layout()
        out_path = args.out_dir / f"sample_{k:02d}_ann{ann['id']}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
