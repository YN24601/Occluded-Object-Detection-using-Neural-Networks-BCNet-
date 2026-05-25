"""Dataset registration helpers.

`register_cocoa_datasets` registers BCNet's train/val splits with the
Detectron2 catalog. It is idempotent: calling it twice is a no-op.

Detectron2's stock `register_coco_instances` calls `load_coco_json` without
`extra_annotation_keys`, which strips every field outside the COCO core
schema. We need `visible_mask`, `invisible_mask`, `occluder_mask`, and
`occlude_rate`, so we register the loader ourselves.
"""

from __future__ import annotations

from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


_BCNET_EXTRA_KEYS = ["visible_mask", "invisible_mask", "occluder_mask", "occlude_rate"]


def _make_loader(json_path: str, img_root: str, name: str):
    """Return a thunk that lazily loads the dataset on first access."""

    def _load():
        return load_coco_json(
            json_path,
            image_root=img_root,
            dataset_name=name,
            extra_annotation_keys=_BCNET_EXTRA_KEYS,
        )

    return _load


def register_cocoa_datasets(cfg) -> None:
    """Register train, val, and visible-eval COCOA splits from `cfg.BCNET`.

    Visible eval is registered only if its JSON exists on disk so that the
    training-only code path doesn't fail when the user hasn't run
    `tools/build_eval_anns.py` yet.
    """
    splits = {
        cfg.BCNET.TRAIN_DATASET_NAME: (cfg.BCNET.TRAIN_JSON, cfg.BCNET.TRAIN_IMG_DIR),
        cfg.BCNET.VAL_DATASET_NAME: (cfg.BCNET.VAL_JSON, cfg.BCNET.VAL_IMG_DIR),
    }
    if Path(cfg.BCNET.VISIBLE_EVAL_JSON).exists():
        splits[cfg.BCNET.VISIBLE_EVAL_DATASET_NAME] = (
            cfg.BCNET.VISIBLE_EVAL_JSON,
            cfg.BCNET.VISIBLE_EVAL_IMG_DIR,
        )
    for name, (json_path, img_root) in splits.items():
        if name in DatasetCatalog.list():
            continue
        json_path = str(Path(json_path).resolve())
        img_root = str(Path(img_root).resolve())

        DatasetCatalog.register(name, _make_loader(json_path, img_root, name))
        meta = MetadataCatalog.get(name)
        meta.json_file = json_path
        meta.image_root = img_root
        meta.evaluator_type = "coco"
