"""Build a COCO-style annotation JSON for *visible* (modal) evaluation.

`COCOEvaluator` compares predictions against whatever the GT JSON's
`segmentation`/`bbox`/`area` fields say. Our BCNet model predicts the
*occludee modal* mask (= COCOA `visible_mask`), so for AP to be
meaningful we have to feed the evaluator a JSON where:
  - `segmentation` is the visible-mask RLE,
  - `bbox` is the tight bbox of that visible mask,
  - `area` is the visible-mask pixel area.

Amodal evaluation is intentionally NOT produced here: our model has no
amodal head, so comparing visible predictions to amodal GT would
penalise the model for not predicting occluded regions it was never
trained on. Add an amodal eval path once an amodal branch exists.

Usage:
    python tools/build_eval_anns.py \
        --input  data/cocoa-cls/annotations/cocoa_mini_test_with_occluder.json \
        --output data/cocoa-cls/annotations/cocoa_mini_visible_eval.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util


def _decode_rle(rle: dict, h: int, w: int) -> np.ndarray:
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _tight_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) tight bbox of a binary mask, or None if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}

    new_anns = []
    n_total = 0
    n_dropped = 0
    for ann in data["annotations"]:
        if "visible_mask" not in ann or ann["visible_mask"] is None:
            n_dropped += 1
            continue
        img = images_by_id[ann["image_id"]]
        h, w = img["height"], img["width"]
        visible = _decode_rle(ann["visible_mask"], h, w)
        bbox = _tight_bbox_xywh(visible)
        if bbox is None:
            # Empty visible mask: object is fully occluded, no signal for
            # modal evaluation. Drop.
            n_dropped += 1
            continue

        new_ann = copy.deepcopy(ann)
        new_ann["segmentation"] = ann["visible_mask"]  # RLE, ready for COCO eval
        new_ann["bbox"] = list(bbox)
        new_ann["area"] = int(visible.sum())
        # iscrowd, category_id, image_id, id stay as-is. Drop the BCNet
        # extras so the file is a clean COCO modal annotation.
        for k in ("visible_mask", "invisible_mask", "occluder_mask", "occluder_area", "occlude_rate"):
            new_ann.pop(k, None)
        new_anns.append(new_ann)
        n_total += 1

    out_data = dict(data)
    out_data["annotations"] = new_anns

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out_data, f)

    print(f"Wrote {args.output}")
    print(f"  annotations kept   : {n_total}")
    print(f"  annotations dropped: {n_dropped}  (empty/missing visible mask)")


if __name__ == "__main__":
    main()
