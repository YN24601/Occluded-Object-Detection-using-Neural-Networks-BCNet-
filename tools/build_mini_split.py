"""Build a small COCOA subset filtered by per-instance occlusion rate.

COCOA stores an `occlude_rate` per annotation (=
`area(invisible_mask) / area(segmentation)`). For BCNet experiments
the interesting samples are those with non-trivial occlusion. This
script picks the first `--limit` images that contain at least one
annotation with `occlude_rate > --threshold` and writes a COCO-format
JSON that's a strict subset of the input.

Typical usage (defaults match `bcnet/utils/config.py`):

    python tools/build_mini_split.py \\
        --input  data/cocoa-cls/annotations/COCO_amodal_val2014_with_classes.json \\
        --output data/cocoa-cls/annotations/cocoa_mini_test.json \\
        --threshold 0.3 \\
        --limit 100

Two convenience modes:
  * `prepare_mini_cocoa(...)` — first N images above the threshold (default).
  * `filter_cocoa_by_occlusion(...)` — keep every annotation above the
    threshold (no image-count cap), useful for a larger filtered split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def filter_cocoa_by_occlusion(input_json: str | Path, output_json: str | Path, threshold: float = 0.3) -> None:
    """Keep every annotation whose `occlude_rate >= threshold` (no cap)."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_anns = []
    valid_image_ids: set[int] = set()
    for ann in data["annotations"]:
        if ann.get("occlude_rate", 0) >= threshold:
            filtered_anns.append(ann)
            valid_image_ids.add(ann["image_id"])

    filtered_images = [img for img in data["images"] if img["id"] in valid_image_ids]
    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_anns,
        "categories": data.get("categories", []),
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f)

    print(
        f"Wrote {output_json}: kept {len(filtered_anns)} annotations "
        f"across {len(filtered_images)} images (threshold={threshold})"
    )


def prepare_mini_cocoa(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.3,
    limit: int = 100,
) -> None:
    """Pick the first `limit` images that contain an annotation above threshold.

    BCNet occlusion rate: R_occ = area(invisible_mask) / area(amodal_mask).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Source annotation file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_anns = []
    selected_img_ids: set[int] = set()
    for ann in data["annotations"]:
        if ann.get("occlude_rate", 0) > threshold:
            filtered_anns.append(ann)
            selected_img_ids.add(ann["image_id"])
            if len(selected_img_ids) >= limit:
                break

    filtered_images = [img for img in data["images"] if img["id"] in selected_img_ids]
    mini_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_anns,
        "categories": data.get("categories", []),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mini_data, f)
    print(
        f"Wrote {output_path}: {len(filtered_images)} images, "
        f"{len(filtered_anns)} annotations (threshold>{threshold}, limit={limit})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/cocoa-cls/annotations/COCO_amodal_val2014_with_classes.json"),
        help="Source COCOA annotation file (default: COCOA val split).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/cocoa-cls/annotations/cocoa_mini_test.json"),
        help="Destination JSON.",
    )
    p.add_argument("--threshold", type=float, default=0.3, help="Minimum per-instance occlude_rate.")
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of images to keep. Use 0 to disable the cap (keep every matching annotation).",
    )
    args = p.parse_args()

    if args.limit and args.limit > 0:
        prepare_mini_cocoa(args.input, args.output, threshold=args.threshold, limit=args.limit)
    else:
        filter_cocoa_by_occlusion(args.input, args.output, threshold=args.threshold)


if __name__ == "__main__":
    main()
