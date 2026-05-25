"""BCNet training entry-point.

Wraps Detectron2's `DefaultTrainer` so the standard CLI works:

    PYTHONUTF8=1 python train.py --num-gpus 1
    PYTHONUTF8=1 python train.py --num-gpus 1 SOLVER.MAX_ITER 200 OUTPUT_DIR ./output/overfit
    PYTHONUTF8=1 python train.py --eval-only MODEL.WEIGHTS ./output/model_final.pth

This script does three things on top of the stock trainer:
  1. Registers our COCOA splits into `DatasetCatalog`.
  2. Pins `cfg.DATASETS.TRAIN/TEST` to the names registered above.
  3. Swaps in `BCNetDatasetMapper` so every batch carries
     `gt_masks` / `gt_occluder_masks` / `gt_amodal_masks`.

Importing `bcnet` is what registers `BCNetBilayerMaskHead` into
`ROI_MASK_HEAD_REGISTRY`; the config's `MODEL.ROI_MASK_HEAD.NAME` picks
it up from there.
"""

from __future__ import annotations

from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger

import bcnet  # noqa: F401  (registers BCNetBilayerMaskHead)
from bcnet.data import BCNetDatasetMapper, register_cocoa_datasets
from bcnet.evaluation import build_bcnet_evaluator
from bcnet.utils import setup_bcnet_config


class BCNetTrainer(DefaultTrainer):
    """DefaultTrainer with the BCNet data mapper + COCOEvaluator wired in."""

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = BCNetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # Test-time mapper is the stock DatasetMapper, which is fine: the
        # BCNet-specific GT fields are only needed for *training* losses,
        # not for COCOEvaluator (which reads its GT from the registered JSON).
        return build_bcnet_evaluator(cfg, dataset_name, output_folder)


def _setup(args):
    """Build the cfg, register datasets, and run Detectron2's default_setup."""
    cfg = setup_bcnet_config(args.config_file or None, opts=args.opts)

    # Pin DATASETS.TRAIN/TEST to the names that `register_cocoa_datasets`
    # will create. We have to defrost/freeze because setup_bcnet_config
    # locked the cfg at the end. TEST goes against the *visible* eval
    # split (segmentation/bbox/area swapped to the COCOA visible_mask),
    # which is what BCNet's occludee branch actually predicts.
    cfg.defrost()
    cfg.DATASETS.TRAIN = (cfg.BCNET.TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (cfg.BCNET.VISIBLE_EVAL_DATASET_NAME,)
    cfg.freeze()

    register_cocoa_datasets(cfg)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, name="bcnet")
    return cfg


def main(args):
    cfg = _setup(args)

    if args.eval_only:
        # COCOEvaluator runs against cfg.DATASETS.TEST (the visible eval
        # split) via BCNetTrainer.build_evaluator. Amodal eval will be
        # added once the model grows an amodal head.
        model = BCNetTrainer.build_model(cfg)
        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return BCNetTrainer.test(cfg, model)

    trainer = BCNetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
