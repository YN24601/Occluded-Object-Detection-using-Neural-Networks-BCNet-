"""End-to-end model forward smoke test for Phase 2.

Builds the full Mask R-CNN with our BCNetBilayerMaskHead, pulls one batch
from the data loader, runs a forward pass in training mode, and prints
every loss key. Pass = no exception, every loss is a finite scalar.

Run:
    PYTHONUTF8=1 python tools/check_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage

from bcnet.data import BCNetDatasetMapper, register_cocoa_datasets
from bcnet.utils import setup_bcnet_config


def _override_train_dataset(cfg, name: str):
    """Mutate cfg.DATASETS.TRAIN to point at our registered split."""
    cfg.defrost()
    cfg.DATASETS.TRAIN = (name,)
    cfg.DATASETS.TEST = (name,)
    cfg.freeze()


def main() -> None:
    cfg = setup_bcnet_config()
    register_cocoa_datasets(cfg)
    _override_train_dataset(cfg, cfg.BCNET.VAL_DATASET_NAME)

    print(f"[build] device       = {cfg.MODEL.DEVICE}")
    print(f"[build] mask head    = {cfg.MODEL.ROI_MASK_HEAD.NAME}")
    print(f"[build] FREEZE_AT    = {cfg.MODEL.BACKBONE.FREEZE_AT}")
    print(f"[build] IMS_PER_BATCH= {cfg.SOLVER.IMS_PER_BATCH}")

    model = build_model(cfg)
    model.train()
    print(f"[build] model        = {type(model).__name__}")
    print(f"[build] mask_head    = {type(model.roi_heads.mask_head).__name__}")
    print(
        f"[build] params       = "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M total, "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M trainable"
    )

    loader = build_detection_train_loader(
        cfg, mapper=BCNetDatasetMapper(cfg, is_train=True)
    )
    batch = next(iter(loader))
    print(
        f"[batch] size={len(batch)} "
        f"image_shape={tuple(batch[0]['image'].shape)} "
        f"#inst={[len(b['instances']) for b in batch]}"
    )

    # RPN's training-mode losses log to a Detectron2 EventStorage; we have
    # to open one or `get_event_storage()` asserts. AMP-wrap to exercise the
    # same code path the real trainer will take.
    with EventStorage(0), torch.no_grad():
        autocast = torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=cfg.SOLVER.AMP.ENABLED,
        )
        with autocast:
            losses = model(batch)

    print("[loss]")
    bad = []
    for k, v in losses.items():
        scalar = v.item() if isinstance(v, torch.Tensor) else float(v)
        finite = torch.isfinite(v).all().item() if isinstance(v, torch.Tensor) else True
        flag = "OK " if finite else "BAD"
        print(f"  {flag}  {k:<30} = {scalar:.6f}")
        if not finite:
            bad.append(k)

    assert not bad, f"non-finite loss components: {bad}"
    print("\n[OK] forward+loss smoke test passed")


if __name__ == "__main__":
    main()
