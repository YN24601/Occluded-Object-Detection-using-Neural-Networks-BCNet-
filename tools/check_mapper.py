"""End-to-end sanity check for `BCNetDatasetMapper`.

Runs the mapper on a few samples from the registered COCOA mini set and
prints the shapes of every GT tensor we care about. Also asserts that the
extra BCNet fields (occluder + amodal) align with the standard gt_masks.
"""

from __future__ import annotations

import sys
from pathlib import Path

from detectron2.data import DatasetCatalog

# Ensure we can import the local `bcnet` package when run from anywhere.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcnet.data import BCNetDatasetMapper, register_cocoa_datasets
from bcnet.utils import setup_bcnet_config


def main() -> None:
    cfg = setup_bcnet_config()
    register_cocoa_datasets(cfg)

    name = cfg.BCNET.VAL_DATASET_NAME
    dataset_dicts = DatasetCatalog.get(name)
    print(f"Dataset {name!r}: {len(dataset_dicts)} images")

    mapper = BCNetDatasetMapper(cfg, is_train=True)

    n_checked = 0
    for d in dataset_dicts:
        out = mapper(d)
        if "instances" not in out:
            continue
        inst = out["instances"]
        if len(inst) == 0:
            continue

        print(
            f"  image_id={d.get('image_id', '?'):>8} "
            f"img={tuple(out['image'].shape)} "
            f"n={len(inst):>2} "
            f"gt_masks={tuple(inst.gt_masks.tensor.shape)} "
            f"gt_occ={tuple(inst.gt_occluder_masks.tensor.shape)} "
            f"gt_amod={tuple(inst.gt_amodal_masks.tensor.shape)}"
        )

        # Shape alignment: occluder/amodal must match gt_masks per-instance.
        assert inst.gt_occluder_masks.tensor.shape == inst.gt_masks.tensor.shape
        assert inst.gt_amodal_masks.tensor.shape == inst.gt_masks.tensor.shape
        # Sanity: amodal area >= visible area for each instance (modulo
        # augmentation cropping turning some into all-zero, which the
        # filter_empty_instances pass would have removed).
        v = inst.gt_masks.tensor.sum(dim=(1, 2))
        a = inst.gt_amodal_masks.tensor.sum(dim=(1, 2))
        if not (a >= v).all():
            n_bad = int((a < v).sum())
            print(f"    WARN: amodal < visible on {n_bad}/{len(inst)} instances")

        n_checked += 1
        if n_checked >= 5:
            break

    print(f"\nOK: mapper produced valid samples for {n_checked} images")


if __name__ == "__main__":
    main()
