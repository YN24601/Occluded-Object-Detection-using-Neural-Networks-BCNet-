"""Derive occluder masks for a COCOA-style annotation JSON.

For every annotation A on a given image, the occluder mask is the union of
the *visible* parts of every OTHER instance B that overlaps A's invisible
region:

    occluder(A) = union_over_B { B.visible_mask & A.invisible_mask } restricted
                  to A's amodal bounding box.

This matches the occluder/occludee decomposition used by BCNet (Ke et al.,
CVPR 2021): Layer 1 predicts what's covering the target, Layer 2 predicts
the target's visible region. We only derive occluder masks when COCOA tells
us the object is actually occluded (`occlude_rate > 0`); fully visible
objects get an all-zero occluder mask.

Usage:
    python tools/build_occluder_anns.py \
        --input  data/cocoa-cls/annotations/cocoa_mini_test.json \
        --output data/cocoa-cls/annotations/cocoa_mini_test_with_occluder.json
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util


def _decode_rle(rle: dict | None, h: int, w: int) -> np.ndarray:
    """Decode a COCO RLE into a HxW boolean array, or an all-zero mask if None."""
    if rle is None:
        return np.zeros((h, w), dtype=bool)
    # pycocotools needs `counts` to be bytes, not str.
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def _encode_rle(mask: np.ndarray) -> dict:
    """Encode a HxW boolean array into a COCO RLE with str `counts`."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def derive_occluder_for_image(anns: list[dict], h: int, w: int) -> list[np.ndarray]:
    """Compute one occluder mask per annotation in `anns` for a single image.

    Args:
        anns: annotations belonging to the same image.
        h, w: image height/width.

    Returns:
        A list of HxW boolean arrays, aligned with `anns`.
    """
    visible = [_decode_rle(a.get("visible_mask"), h, w) for a in anns]
    invisible = [_decode_rle(a.get("invisible_mask"), h, w) for a in anns]

    out = []
    for i, ann in enumerate(anns):
        if invisible[i].sum() == 0:
            # Object is fully visible; nothing occludes it.
            out.append(np.zeros((h, w), dtype=bool))
            continue
        occluder = np.zeros((h, w), dtype=bool)
        for j in range(len(anns)):
            if j == i:
                continue
            # Any visible region of another instance that lies on top of
            # this instance's invisible region is, by definition, occluding it.
            overlap = visible[j] & invisible[i]
            if overlap.any():
                occluder |= visible[j] & invisible[i]
        out.append(occluder)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    new_anns: list[dict] = []
    n_occluded = 0
    n_total = 0
    for img_id, anns in by_image.items():
        img = images[img_id]
        h, w = img["height"], img["width"]
        occluders = derive_occluder_for_image(anns, h, w)
        for ann, occ in zip(anns, occluders):
            new_ann = copy.deepcopy(ann)
            new_ann["occluder_mask"] = _encode_rle(occ)
            new_ann["occluder_area"] = int(occ.sum())
            new_anns.append(new_ann)
            n_total += 1
            if occ.any():
                n_occluded += 1

    out_data = dict(data)
    out_data["annotations"] = new_anns

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out_data, f)

    print(f"Wrote {args.output}")
    print(f"  annotations: {n_total}")
    print(f"  with non-empty occluder: {n_occluded} ({100*n_occluded/max(n_total,1):.1f}%)")


if __name__ == "__main__":
    main()
