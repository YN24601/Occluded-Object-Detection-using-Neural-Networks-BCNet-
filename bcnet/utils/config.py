"""Detectron2 config helpers for BCNet.

Adds a CfgNode subtree `cfg.BCNET` for project-specific knobs and loads a base
YAML tuned for a 4 GB GTX 1650.
"""

from pathlib import Path

from detectron2.config import CfgNode, get_cfg
from detectron2.model_zoo import get_config_file


_DEFAULT_YAML = Path(__file__).resolve().parents[2] / "configs" / "bcnet_train.yaml"


def _add_bcnet_defaults(cfg: CfgNode) -> None:
    """Attach BCNet-specific defaults to a Detectron2 cfg."""
    cfg.BCNET = CfgNode()

    # Dataset paths (overridable per experiment).
    cfg.BCNET.DATA_ROOT = "data/cocoa-cls"
    cfg.BCNET.TRAIN_JSON = "data/cocoa-cls/annotations/cocoa_mini_test_with_occluder.json"
    cfg.BCNET.TRAIN_IMG_DIR = "data/cocoa-cls/val2014"
    cfg.BCNET.VAL_JSON = "data/cocoa-cls/annotations/cocoa_mini_test_with_occluder.json"
    cfg.BCNET.VAL_IMG_DIR = "data/cocoa-cls/val2014"
    cfg.BCNET.TRAIN_DATASET_NAME = "cocoa_mini_train"
    cfg.BCNET.VAL_DATASET_NAME = "cocoa_mini_val"

    # Modal (visible) evaluation split. COCOEvaluator compares predictions
    # against this JSON, whose `segmentation` field has been swapped to the
    # COCOA `visible_mask` by tools/build_eval_anns.py.
    cfg.BCNET.VISIBLE_EVAL_JSON = "data/cocoa-cls/annotations/cocoa_mini_visible_eval.json"
    cfg.BCNET.VISIBLE_EVAL_IMG_DIR = "data/cocoa-cls/val2014"
    cfg.BCNET.VISIBLE_EVAL_DATASET_NAME = "cocoa_mini_visible_eval"

    # Loss weights for the bilayer head. Boundary loss is BCE + weighted
    # BCE (~2x a plain BCE), so weights here scale that combined term.
    # 0.5/0.5 = paper's symmetric weighting; empirically best on the mini
    # set too (a 0.25/0.5 ablation regressed by ~3 segm AP).
    cfg.BCNET.LOSS = CfgNode()
    cfg.BCNET.LOSS.OCCLUDER_MASK_WEIGHT = 1.0
    cfg.BCNET.LOSS.OCCLUDEE_MASK_WEIGHT = 1.0
    cfg.BCNET.LOSS.OCCLUDER_BOUNDARY_WEIGHT = 0.5
    cfg.BCNET.LOSS.OCCLUDEE_BOUNDARY_WEIGHT = 0.5

    # Toggle features that can be turned off in early experiments.
    cfg.BCNET.HEAD = CfgNode()
    # Boundary supervision is a core BCNet component; default ON to match
    # the paper. YAMLs can flip it OFF for ablations.
    cfg.BCNET.HEAD.USE_BOUNDARY = True
    # Non-local graph reasoning in Layer 1 (the BCNet paper's signature module).
    cfg.BCNET.HEAD.USE_GCN = True


def setup_bcnet_config(config_file: str | None = None, opts: list | None = None) -> CfgNode:
    """Build a Detectron2 CfgNode for BCNet.

    Resolution order:
      1. Detectron2 defaults
      2. BCNet defaults (added by `_add_bcnet_defaults`)
      3. base Mask R-CNN R-50 FPN model-zoo YAML
      4. `configs/bcnet_train.yaml` (if `config_file` is None and it exists)
      5. `config_file` (if provided)
      6. command-line style `opts` list
    """
    cfg = get_cfg()
    _add_bcnet_defaults(cfg)

    # Start from a standard Mask R-CNN baseline so the FPN/backbone/RPN are sane.
    base = get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(base)

    target_yaml = Path(config_file) if config_file else _DEFAULT_YAML
    if target_yaml.exists():
        cfg.merge_from_file(str(target_yaml))

    if opts:
        cfg.merge_from_list(opts)

    cfg.freeze()
    return cfg
