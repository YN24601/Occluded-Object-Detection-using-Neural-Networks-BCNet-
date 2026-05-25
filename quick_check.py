"""Lightweight smoke check — no PyTorch / Detectron2 required.

Run this first on any new machine to confirm the repo is laid out
correctly and the annotation JSON parses. It is a fast "did I clone +
download data correctly?" gate before installing the heavy deps.

Checks:
  1. Project directory + file layout
  2. Annotation file parses as JSON with the expected schema
  3. A handful of referenced image files actually exist on disk
  4. Per-annotation occlusion-rate distribution
"""

import json
import os
import sys


def check_project_structure() -> bool:
    print("\n" + "=" * 60)
    print("CHECK 1: Project structure")
    print("=" * 60)

    required_dirs = [
        "bcnet/data",
        "bcnet/models",
        "bcnet/utils",
        "configs",
        "data/cocoa-cls/annotations",
        "data/cocoa-cls/train2014",
        "data/cocoa-cls/val2014",
    ]

    required_files = [
        "bcnet/__init__.py",
        "bcnet/data/__init__.py",
        "bcnet/data/mapper.py",
        "bcnet/data/build.py",
        "bcnet/models/__init__.py",
        "bcnet/models/bilayer_head.py",
        "bcnet/utils/__init__.py",
        "bcnet/utils/config.py",
        "bcnet/evaluation.py",
        "configs/bcnet_train.yaml",
        "train.py",
        "tools/build_mini_split.py",
        "tools/build_occluder_anns.py",
        "tools/build_eval_anns.py",
        "tools/check_mapper.py",
        "tools/check_forward.py",
        "tools/plot_losses.py",
        "tools/viz_predictions.py",
    ]

    print("Directories:")
    all_good = True
    for d in required_dirs:
        ok = os.path.isdir(d)
        print(f"  {'OK ' if ok else 'MISS'}  {d}")
        all_good &= ok

    print("Files:")
    for f in required_files:
        ok = os.path.isfile(f)
        print(f"  {'OK ' if ok else 'MISS'}  {f}")
        all_good &= ok

    print("PASSED" if all_good else "FAILED")
    return all_good


def check_annotation_format() -> bool:
    print("\n" + "=" * 60)
    print("CHECK 2: Annotation format")
    print("=" * 60)

    ann_file = "data/cocoa-cls/annotations/cocoa_mini_test.json"
    if not os.path.exists(ann_file):
        print(f"MISS  {ann_file}")
        print("      Run `python tools/build_mini_split.py` first.")
        return False

    try:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"JSON decode error: {exc}")
        return False

    print("OK    JSON parsed")
    for key in ("images", "annotations", "categories"):
        if key in data:
            print(f"  OK   '{key}': {len(data[key])} items")
        else:
            print(f"  MISS '{key}'")
            return False

    if data["annotations"]:
        ann = data["annotations"][0]
        print("Sample annotation keys:", sorted(ann.keys()))
        for field in ("visible_mask", "invisible_mask", "segmentation"):
            present = field in ann
            print(f"  {'OK ' if present else 'WARN'}  field {field!r}")

    print("PASSED")
    return True


def check_image_files() -> bool:
    print("\n" + "=" * 60)
    print("CHECK 3: Image files")
    print("=" * 60)

    ann_file = "data/cocoa-cls/annotations/cocoa_mini_test.json"
    img_dir = "data/cocoa-cls/val2014"

    with open(ann_file, "r", encoding="utf-8") as f:
        images = json.load(f).get("images", [])
    print(f"Images in annotation: {len(images)}")

    missing = 0
    for img_meta in images[:5]:
        path = os.path.join(img_dir, img_meta["file_name"])
        if os.path.exists(path):
            print(f"  OK    {img_meta['file_name']} ({os.path.getsize(path)/1024:.1f} KB)")
        else:
            print(f"  MISS  {img_meta['file_name']}")
            missing += 1

    if missing:
        print(f"WARN: {missing}/5 sample images missing — did you unzip val2014.zip?")
        return False
    print("PASSED")
    return True


def check_occlusion_statistics() -> bool:
    print("\n" + "=" * 60)
    print("CHECK 4: Occlusion-rate distribution")
    print("=" * 60)

    ann_file = "data/cocoa-cls/annotations/cocoa_mini_test.json"
    with open(ann_file, "r", encoding="utf-8") as f:
        annotations = json.load(f).get("annotations", [])

    if not annotations:
        print("No annotations.")
        return False

    rates = [ann.get("occlude_rate", 0) for ann in annotations]
    print(f"  count={len(rates)}  min={min(rates):.3f}  max={max(rates):.3f}  mean={sum(rates)/len(rates):.3f}")
    for low, high in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0001)]:
        n = sum(1 for r in rates if low <= r < high)
        print(f"  [{low:.1f}, {high:.1f}): {n:>4d}  ({100 * n / len(rates):.1f}%)")
    print("PASSED")
    return True


def main() -> bool:
    print("\nBCNet lightweight check (no PyTorch / Detectron2 required)")
    results = [
        ("Project structure", check_project_structure()),
        ("Annotation format", check_annotation_format()),
        ("Image files", check_image_files()),
        ("Occlusion statistics", check_occlusion_statistics()),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASSED' if ok else 'FAILED'}  {name}")

    if all(ok for _, ok in results):
        print(
            "\nAll checks passed. Typical workflow:\n"
            "  1. python tools/build_mini_split.py     # filter mini set by occlude_rate\n"
            "  2. python tools/build_occluder_anns.py  # derive occluder masks\n"
            "  3. python tools/build_eval_anns.py      # build visible-mask eval JSON\n"
            "  4. python tools/check_forward.py        # smoke-test model + losses\n"
            "  5. python train.py --num-gpus 1         # train (override cfg via CLI: KEY VALUE)\n"
            "  6. python train.py --eval-only MODEL.WEIGHTS output/<run>/model_final.pth\n"
        )
        return True
    print("\nSome checks failed — fix issues above before proceeding.")
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
