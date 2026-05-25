"""Evaluation helpers for BCNet.

Phase 4 only ships *visible* (modal) mAP because our model has no amodal
branch yet: it predicts the occludee (visible) mask via Layer 2 and a
class-agnostic occluder mask via Layer 1. Comparing the visible
prediction against the amodal GT would penalise the model for not
predicting what it was never trained to predict, so we don't.

The visible-eval JSON (built by `tools/build_eval_anns.py`) has its
`segmentation`/`bbox`/`area` swapped to the COCOA `visible_mask`, which
is exactly what `pred_masks` from `BCNetBilayerMaskHead` represents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from detectron2.evaluation import COCOEvaluator


def build_bcnet_evaluator(
    cfg, dataset_name: str, output_folder: Optional[str] = None
) -> COCOEvaluator:
    """Build a Detectron2 COCOEvaluator for a BCNet eval split.

    Args:
        cfg: a frozen Detectron2 CfgNode.
        dataset_name: the registered dataset, e.g. cfg.BCNET.VISIBLE_EVAL_DATASET_NAME.
        output_folder: where to dump `coco_instances_results.json` etc.
            Defaults to `<cfg.OUTPUT_DIR>/eval/<dataset_name>`.

    Returns:
        A COCOEvaluator configured to score both bbox and segm tasks. Mask
        AP from this evaluator is the modal/visible AP for BCNet.
    """
    if output_folder is None:
        output_folder = str(Path(cfg.OUTPUT_DIR) / "eval" / dataset_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    return COCOEvaluator(
        dataset_name,
        tasks=("bbox", "segm"),
        distributed=False,
        output_dir=output_folder,
    )
