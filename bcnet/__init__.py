"""BCNet package: bilayer occlusion-aware instance segmentation on COCOA.

Importing `bcnet` is enough to register the bilayer ROI mask head with
Detectron2; downstream code only needs to ensure `setup_bcnet_config()`
and `register_cocoa_datasets()` run before constructing the trainer.
"""

from . import models  # noqa: F401  (registers BCNetBilayerMaskHead)

__version__ = "0.0.1"
