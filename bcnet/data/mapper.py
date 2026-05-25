"""BCNet dataset mapper.

Extends Detectron2's `DatasetMapper` so that every sample also carries
the bilayer-head GT used by BCNet (Ke et al., CVPR 2021):

  - gt_masks            : occludee modal mask  (= COCOA `visible_mask`)
                          -> supervises Layer 2 (occludee branch)
  - gt_occluder_masks   : derived occluder mask
                          -> supervises Layer 1 (occluder branch)
  - gt_amodal_masks     : complete shape (= COCOA `segmentation`)
                          -> kept for evaluation only, not used in loss

The annotation JSON we consume must already have a `visible_mask`,
`occluder_mask` (added by `tools/build_occluder_anns.py`), and a
`segmentation` field, all in COCO RLE format.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks
from pycocotools import mask as mask_util


def _rle_to_mask(rle: Optional[dict], h: int, w: int) -> np.ndarray:
    """Decode a COCO RLE into a HxW uint8 mask, or return zeros if missing."""
    if rle is None:
        return np.zeros((h, w), dtype=np.uint8)
    rle = dict(rle)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


class BCNetDatasetMapper(DatasetMapper):
    """Custom DatasetMapper that attaches bilayer GT masks to each sample."""

    def __init__(self, cfg, is_train: bool = True):
        super().__init__(cfg, is_train=is_train)
        # The bilayer masks are stored as raw 2-D arrays after augmentation,
        # so we force the bitmask path in annotations_to_instances.
        if self.instance_mask_format != "bitmask":
            self.instance_mask_format = "bitmask"

    # NOTE: overrides DatasetMapper._transform_annotations to handle the
    # extra mask channels. We keep the original signature.
    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        h_orig = dataset_dict["height"]
        w_orig = dataset_dict["width"]

        annos_in = dataset_dict.pop("annotations")
        kept_annos = []
        kept_occluder = []
        kept_amodal = []

        for anno in annos_in:
            if anno.get("iscrowd", 0):
                continue

            # Pre-decode the extra masks from the raw (untransformed) RLEs.
            occluder = _rle_to_mask(anno.get("occluder_mask"), h_orig, w_orig)
            amodal = _rle_to_mask(anno.get("segmentation"), h_orig, w_orig)

            # Swap the segmentation field so the standard transform path
            # processes the *visible* (occludee modal) mask. The original
            # amodal mask is kept separately above.
            visible_rle = anno.get("visible_mask")
            if visible_rle is None:
                # No visible region: skip (object fully occluded, no signal).
                continue
            anno = copy.deepcopy(anno)
            anno["segmentation"] = visible_rle  # routed through the standard path

            # Standard Detectron2 transform: bbox + segmentation (= visible).
            new_anno = utils.transform_instance_annotations(
                anno, transforms, image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            )

            # Apply the same geometric transforms to our extra masks.
            occluder_t = transforms.apply_segmentation(occluder)
            amodal_t = transforms.apply_segmentation(amodal)

            kept_annos.append(new_anno)
            kept_occluder.append(occluder_t.astype(np.uint8))
            kept_amodal.append(amodal_t.astype(np.uint8))

        instances = utils.annotations_to_instances(
            kept_annos, image_shape, mask_format=self.instance_mask_format
        )

        # Attach BCNet-specific fields. BitMasks expects (N, H, W) uint8/bool.
        if len(kept_occluder):
            occluder_tensor = torch.as_tensor(
                np.stack(kept_occluder, axis=0).astype(np.uint8)
            )
            amodal_tensor = torch.as_tensor(
                np.stack(kept_amodal, axis=0).astype(np.uint8)
            )
        else:
            occluder_tensor = torch.zeros((0, *image_shape), dtype=torch.uint8)
            amodal_tensor = torch.zeros((0, *image_shape), dtype=torch.uint8)
        instances.gt_occluder_masks = BitMasks(occluder_tensor)
        instances.gt_amodal_masks = BitMasks(amodal_tensor)

        if self.recompute_boxes and len(kept_annos):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        # filter_empty_instances keys off gt_boxes/gt_masks; the BCNet
        # fields ride along because filter_empty_instances uses tensor
        # masking on every field.
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
