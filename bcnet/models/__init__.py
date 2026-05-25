"""Model components for BCNet (Ke et al., CVPR 2021).

Importing this module registers the bilayer ROI mask head into
Detectron2's `ROI_MASK_HEAD_REGISTRY`, after which it can be selected via
`cfg.MODEL.ROI_MASK_HEAD.NAME = "BCNetBilayerMaskHead"`.
"""

from . import bilayer_head  # noqa: F401  (import side-effect: registry registration)
from .bilayer_head import BCNetBilayerMaskHead

__all__ = ["BCNetBilayerMaskHead"]
