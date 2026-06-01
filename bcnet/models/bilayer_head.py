"""Bilayer ROI mask head (Ke et al., CVPR 2021).

Replaces Mask R-CNN's single mask head with two stacked branches:

  - Layer 1 (occluder, class-agnostic): predicts what is *covering* the
    target inside the ROI. Supervised by `instances.gt_occluder_masks`.
  - Layer 2 (occludee, per-class):      predicts the target's own visible
    modal mask. Layer 1's post-attention feature is fused into Layer 2's
    input via additive residual (`x_roi + f_L1`), so Layer 2 sees the
    occluder reasoning for free. Supervised by `instances.gt_masks`
    (= COCOA `visible_mask`).

Both layers run their conv stack followed by a non-local (GCN) block;
each layer has its own attention parameters. Each layer also exposes an
optional class-agnostic *boundary* predictor with an independent deconv,
so the boundary upsample does not share statistics with the mask path.

The boundary GT is derived on-the-fly from the cropped + resized mask GT
via a morphological gradient (dilation - erosion). Boundary supervision
is the paper's "contour-aware" component; loss is BCE plus a per-pixel
Gaussian-weighted BCE (`weight = 1 + gaussian(contour, sigma=1) / 50`),
mirroring `lkeab/BCNet`'s `JointLoss(BceLoss, BceLoss)`. Gate with
`cfg.BCNET.HEAD.USE_BOUNDARY`; weights live in
`cfg.BCNET.LOSS.{OCCLUDER,OCCLUDEE}_BOUNDARY_WEIGHT`.

The head registers itself into Detectron2's `ROI_MASK_HEAD_REGISTRY` so
`StandardROIHeads` picks it up automatically when
`cfg.MODEL.ROI_MASK_HEAD.NAME == "BCNetBilayerMaskHead"`.
"""

from __future__ import annotations

from typing import List, Optional

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.modeling.roi_heads.mask_head import (
    ROI_MASK_HEAD_REGISTRY,
    BaseMaskRCNNHead,
    mask_rcnn_inference,
    mask_rcnn_loss,
)
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from torch import nn


def _build_conv_stack(in_ch: int, conv_dim: int, num_conv: int, conv_norm: str = "") -> nn.ModuleList:
    """Stack `num_conv` 3x3 ConvReLU blocks: in_ch -> conv_dim -> ... -> conv_dim."""
    layers = nn.ModuleList()
    cur = in_ch
    for _ in range(num_conv):
        layers.append(
            Conv2d(
                cur,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
        )
        cur = conv_dim
    return layers


class GCNBlock(nn.Module):
    """Non-local self-attention block for ROI-level graph reasoning.

    BCNet (Ke et al., 2021) uses graph reasoning in the occluder branch to
    let each spatial position attend to every other position inside the
    ROI. The math is identical to a single scaled-dot-product attention
    layer over a flattened HxW feature map:

        affinity[i, j] = softmax_j( <theta(x)[:, i], phi(x)[:, j]> / sqrt(C') )
        out[:, i]      = sum_j affinity[i, j] * g(x)[:, j]
        y              = x + W(out)        # residual

    Output projection `W` is zero-initialised so the block is an identity
    at start of training and gradients route through the surrounding
    conv layers first — this is the standard recipe from Wang et al.,
    Non-local Neural Networks, CVPR 2018.

    Memory: the affinity tensor is (N_rois, HW, HW). For a typical 14x14
    ROI feature and ~64 ROIs/image it is well under 10 MB, so this block
    does not change the 4 GB VRAM budget noticeably.
    """

    def __init__(self, in_ch: int, reduction: int = 2):
        super().__init__()
        inter_ch = max(in_ch // reduction, 1)
        self.theta = Conv2d(in_ch, inter_ch, kernel_size=1)
        self.phi = Conv2d(in_ch, inter_ch, kernel_size=1)
        self.g = Conv2d(in_ch, inter_ch, kernel_size=1)
        self.W = Conv2d(inter_ch, in_ch, kernel_size=1)
        self.scale = inter_ch ** -0.5

        for m in (self.theta, self.phi, self.g):
            weight_init.c2_msra_fill(m)
        # Zero-init the output projection -> identity at init.
        nn.init.constant_(self.W.weight, 0)
        if self.W.bias is not None:
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Empty ROI batch (common at early inference when no proposals
        # cross the score threshold): the bmm + view(-1, ...) pipeline is
        # ambiguous on a zero-row tensor, so short-circuit to the identity.
        if x.size(0) == 0:
            return x

        # The bmm + softmax pipeline can overflow under fp16 autocast and
        # produce NaN. Force the entire block to run in fp32; the surrounding
        # graph still benefits from AMP for the conv stack and Mask R-CNN
        # head. Memory cost is negligible (1x1 convs + small attention).
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            n, _, h, w = xf.shape
            theta = self.theta(xf).flatten(2)  # (N, C', HW)
            phi = self.phi(xf).flatten(2)      # (N, C', HW)
            g = self.g(xf).flatten(2)          # (N, C', HW)

            affinity = torch.bmm(theta.transpose(1, 2), phi) * self.scale
            affinity = F.softmax(affinity, dim=-1)

            out = torch.bmm(g, affinity.transpose(1, 2)).view(n, -1, h, w)
            y = xf + self.W(out)
        return y.to(dtype=input_dtype)


def _mask_to_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Morphological-gradient boundary of a binary mask.

    For each instance, returns a HxW tensor with 1 on edge pixels and 0
    elsewhere. The two max-pool calls implement dilation and (inverted)
    erosion respectively, so `dilation - erosion` produces a thin ring
    along the mask contour.

    Args:
        mask: (N, H, W) or (N, 1, H, W) float in [0, 1]. Treated as binary
            by the downstream BCE loss.

    Returns:
        (N, 1, H, W) float in {0, 1}.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp_(0.0, 1.0)


def _gather_cropped_gt(
    pred_logits: torch.Tensor,
    instances: List[Instances],
    gt_field: str,
    mask_side_len: int,
) -> Optional[torch.Tensor]:
    """Crop and resize the named BitMasks GT field to the predictor resolution.

    Returns a stacked (N_total, mask_side_len, mask_side_len) float tensor,
    or None when there are no foreground proposals in the batch.
    """
    gts: list[torch.Tensor] = []
    for ipi in instances:
        if len(ipi) == 0:
            continue
        if not ipi.has(gt_field):
            raise RuntimeError(
                f"instances missing {gt_field!r}; check the BCNet data mapper."
            )
        gts.append(
            getattr(ipi, gt_field)
            .crop_and_resize(ipi.proposal_boxes.tensor, mask_side_len)
            .to(device=pred_logits.device)
        )
    if not gts:
        return None
    return cat(gts, dim=0).to(dtype=torch.float32)


def _occluder_mask_loss(
    pred_occ_logits: torch.Tensor, instances: List[Instances], mask_side_len: int
) -> torch.Tensor:
    """Class-agnostic BCE on Layer-1 mask logits vs cropped occluder GT."""
    assert pred_occ_logits.size(1) == 1, "occluder branch must be class-agnostic (1 channel)"
    gt = _gather_cropped_gt(pred_occ_logits, instances, "gt_occluder_masks", mask_side_len)
    if gt is None:
        return pred_occ_logits.sum() * 0.0

    pred = pred_occ_logits[:, 0]  # (N, H, W)

    # Track simple training-time accuracy so it shows up in TensorBoard.
    with torch.no_grad():
        gt_bool = gt > 0.5
        pred_pos = pred > 0.0
        storage = get_event_storage()
        storage.put_scalar("bcnet/occluder_accuracy", (pred_pos == gt_bool).float().mean().item())
        storage.put_scalar("bcnet/occluder_gt_positive_rate", gt_bool.float().mean().item())

    return F.binary_cross_entropy_with_logits(pred, gt, reduction="mean")


_GAUSSIAN_KERNEL_CACHE: dict[tuple[torch.device, torch.dtype, float, int], torch.Tensor] = {}


def _gaussian_blur(x: torch.Tensor, sigma: float = 1.0, kernel_size: int = 7) -> torch.Tensor:
    """Apply a (kernel_size, kernel_size) Gaussian blur.

    Mirrors `scipy.ndimage.gaussian_filter(..., sigma=1)` used by the source
    project to build a boundary importance map. Kernel is cached per
    (device, dtype) so the per-step cost is one conv2d.
    """
    key = (x.device, x.dtype, sigma, kernel_size)
    kernel = _GAUSSIAN_KERNEL_CACHE.get(key)
    if kernel is None:
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g1d = g1d / g1d.sum()
        kernel = (g1d[:, None] * g1d[None, :]).view(1, 1, kernel_size, kernel_size)
        _GAUSSIAN_KERNEL_CACHE[key] = kernel
    return F.conv2d(x, kernel, padding=kernel_size // 2)


def _boundary_loss(
    pred_b_logits: torch.Tensor,
    instances: List[Instances],
    gt_field: str,
    mask_side_len: int,
    log_prefix: str,
) -> torch.Tensor:
    """Class-agnostic BCE on boundary logits, with the BCNet paper's weighted
    + unweighted dual term.

    Reproduces lkeab/BCNet's `JointLoss(BceLoss, BceLoss)`:
        loss = mean( BCE(pred, gt_contour) )
             + mean( BCE(pred, gt_contour) * weight )
    where `weight = 1 + gaussian_filter(contour, sigma=1) / 50` is a
    contour-neighborhood importance map (per the source's `boundary.py`).

    Args:
        pred_b_logits: (N_total, 1, H, W) logits from a boundary predictor.
        instances:     list of FG-filtered Instances.
        gt_field:      "gt_masks" (occludee) or "gt_occluder_masks" (occluder).
        mask_side_len: spatial size of the predictor output.
        log_prefix:    tag used when writing the GT-positive rate scalar.
    """
    assert pred_b_logits.size(1) == 1
    gt_mask = _gather_cropped_gt(pred_b_logits, instances, gt_field, mask_side_len)
    if gt_mask is None:
        return pred_b_logits.sum() * 0.0

    gt_boundary = _mask_to_boundary(gt_mask)  # (N, 1, H, W) in {0, 1}

    with torch.no_grad():
        weight = 1.0 + _gaussian_blur(gt_boundary, sigma=1.0) / 50.0
        storage = get_event_storage()
        storage.put_scalar(
            f"bcnet/{log_prefix}_boundary_gt_positive_rate",
            gt_boundary.mean().item(),
        )

    bce_pix = F.binary_cross_entropy_with_logits(pred_b_logits, gt_boundary, reduction="none")
    return bce_pix.mean() + (bce_pix * weight).mean()


@ROI_MASK_HEAD_REGISTRY.register()
class BCNetBilayerMaskHead(BaseMaskRCNNHead):
    """Bilayer ROI mask head with occluder (Layer 1) and occludee (Layer 2)."""

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        conv_dim: int = 256,
        num_conv: int = 4,
        conv_norm: str = "",
        occluder_loss_weight: float = 1.0,
        occludee_loss_weight: float = 1.0,
        use_boundary: bool = False,
        occluder_boundary_weight: float = 0.0,
        occludee_boundary_weight: float = 0.0,
        use_gcn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        in_ch = input_shape.channels
        # Layer 2 mixes the ROI feature and the Layer 1 feature via
        # additive residual (`x_ori + x_layer1`, per the source). This only
        # works when both tensors share channel count.
        assert in_ch == conv_dim, (
            f"BCNetBilayerMaskHead requires input channels ({in_ch}) == "
            f"conv_dim ({conv_dim}) for the Layer 2 residual fusion."
        )

        # Layer 1: occluder branch (class-agnostic).
        self.occ_convs = _build_conv_stack(in_ch, conv_dim, num_conv, conv_norm)
        # Optional graph-reasoning block sitting between the conv stack and
        # the deconv. The post-GCN feature also becomes the conditioning
        # signal for Layer 2, so Layer 2 sees globally-reasoned context.
        self.use_gcn = use_gcn
        self.occ_gcn = GCNBlock(conv_dim) if use_gcn else None
        self.occ_deconv = ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2, padding=0)
        self.occ_deconv_relu = nn.ReLU()
        self.occ_predictor = Conv2d(conv_dim, 1, kernel_size=1, stride=1, padding=0)

        # Layer 2: occludee branch. Receives `x_ori + layer1_feat` (matches
        # the source's residual fusion), so the first conv sees `in_ch`.
        self.tgt_convs = _build_conv_stack(in_ch, conv_dim, num_conv, conv_norm)
        # Mirror Layer 1's non-local block: the source ships a second
        # attention module inside Layer 2 (`conv_norm_relus[cnt == 1]`).
        self.tgt_gcn = GCNBlock(conv_dim) if use_gcn else None
        self.tgt_deconv = ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2, padding=0)
        self.tgt_deconv_relu = nn.ReLU()
        self.tgt_predictor = Conv2d(conv_dim, num_classes, kernel_size=1, stride=1, padding=0)

        # Optional boundary predictors. The source uses an independent
        # deconv+1x1 chain per branch (`boundary_deconv_bo` / `boundary_deconv`)
        # rather than sharing the mask path's upsample, so we do the same.
        self.use_boundary = use_boundary
        if use_boundary:
            self.occ_boundary_deconv = ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2, padding=0)
            self.occ_boundary_deconv_relu = nn.ReLU()
            self.occ_boundary_predictor = Conv2d(conv_dim, 1, kernel_size=1, stride=1, padding=0)
            self.tgt_boundary_deconv = ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2, padding=0)
            self.tgt_boundary_deconv_relu = nn.ReLU()
            self.tgt_boundary_predictor = Conv2d(conv_dim, 1, kernel_size=1, stride=1, padding=0)
        else:
            self.occ_boundary_deconv = None
            self.occ_boundary_predictor = None
            self.tgt_boundary_deconv = None
            self.tgt_boundary_predictor = None

        # Init: MSRA for convs, small-std normal for the 1x1 predictors.
        msra_modules = (
            list(self.occ_convs)
            + [self.occ_deconv]
            + list(self.tgt_convs)
            + [self.tgt_deconv]
        )
        if use_boundary:
            msra_modules += [self.occ_boundary_deconv, self.tgt_boundary_deconv]
        for m in msra_modules:
            weight_init.c2_msra_fill(m)
        predictors = [self.occ_predictor, self.tgt_predictor]
        if use_boundary:
            predictors += [self.occ_boundary_predictor, self.tgt_boundary_predictor]
        for predictor in predictors:
            nn.init.normal_(predictor.weight, std=0.001)
            if predictor.bias is not None:
                nn.init.constant_(predictor.bias, 0)

        self.occluder_loss_weight = occluder_loss_weight
        self.occludee_loss_weight = occludee_loss_weight
        self.occluder_boundary_weight = occluder_boundary_weight
        self.occludee_boundary_weight = occludee_boundary_weight

    @classmethod
    def from_config(cls, cfg, input_shape: ShapeSpec):
        ret = super().from_config(cfg, input_shape)
        # Standard Mask R-CNN convention: when CLS_AGNOSTIC_MASK is True the
        # final 1x1 conv has 1 output channel; otherwise NUM_CLASSES. Both
        # mask_rcnn_loss and mask_rcnn_inference dispatch on this channel
        # count internally via `pred_mask_logits.size(1) == 1`.
        num_classes = (
            1 if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK else cfg.MODEL.ROI_HEADS.NUM_CLASSES
        )
        ret.update(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_dim=cfg.MODEL.ROI_MASK_HEAD.CONV_DIM,
            num_conv=cfg.MODEL.ROI_MASK_HEAD.NUM_CONV,
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            occluder_loss_weight=cfg.BCNET.LOSS.OCCLUDER_MASK_WEIGHT,
            occludee_loss_weight=cfg.BCNET.LOSS.OCCLUDEE_MASK_WEIGHT,
            use_boundary=cfg.BCNET.HEAD.USE_BOUNDARY,
            occluder_boundary_weight=cfg.BCNET.LOSS.OCCLUDER_BOUNDARY_WEIGHT,
            occludee_boundary_weight=cfg.BCNET.LOSS.OCCLUDEE_BOUNDARY_WEIGHT,
            use_gcn=cfg.BCNET.HEAD.USE_GCN,
        )
        return ret

    def _layer1(self, x: torch.Tensor):
        """Run the occluder branch.

        Returns:
            occ_mask_logits:     (N, 1, 2H, 2W) class-agnostic mask logits
            occ_boundary_logits: (N, 1, 2H, 2W) class-agnostic boundary logits
                                  (None when `use_boundary` is False)
            layer1_feat:         (N, conv_dim, H, W) pre-deconv features used
                                  to condition Layer 2.
        """
        f = x
        for conv in self.occ_convs:
            f = conv(f)
        if self.use_gcn:
            f = self.occ_gcn(f)
        layer1_feat = f  # Layer 2 reads this AFTER any GCN reasoning.

        f_up_mask = self.occ_deconv_relu(self.occ_deconv(f))
        occ_mask_logits = self.occ_predictor(f_up_mask)
        if self.use_boundary:
            f_up_b = self.occ_boundary_deconv_relu(self.occ_boundary_deconv(f))
            occ_boundary_logits = self.occ_boundary_predictor(f_up_b)
        else:
            occ_boundary_logits = None
        return occ_mask_logits, occ_boundary_logits, layer1_feat

    def _layer2(self, x: torch.Tensor, layer1_feat: torch.Tensor):
        """Run the occludee branch conditioned on Layer 1 features.

        Layer 2 input = `x + layer1_feat` (additive residual, matching the
        source). Layer 2 then runs its own conv stack and non-local block
        before two parallel deconv+predictor branches (mask, boundary).

        Returns:
            tgt_mask_logits:     (N, C, 2H, 2W). C=1 for class-agnostic.
            tgt_boundary_logits: (N, 1, 2H, 2W) or None.
        """
        f = x + layer1_feat
        for conv in self.tgt_convs:
            f = conv(f)
        if self.use_gcn:
            f = self.tgt_gcn(f)

        f_up_mask = self.tgt_deconv_relu(self.tgt_deconv(f))
        tgt_mask_logits = self.tgt_predictor(f_up_mask)
        if self.use_boundary:
            f_up_b = self.tgt_boundary_deconv_relu(self.tgt_boundary_deconv(f))
            tgt_boundary_logits = self.tgt_boundary_predictor(f_up_b)
        else:
            tgt_boundary_logits = None
        return tgt_mask_logits, tgt_boundary_logits

    def forward(self, x: torch.Tensor, instances: List[Instances]):
        """Standard Detectron2 mask-head contract.

        Training:
            returns a dict of losses (loss_mask, loss_occluder_mask, and
            optionally loss_occludee_boundary + loss_occluder_boundary).
        Inference:
            mutates each `Instances` in-place by adding `pred_masks` (occludee)
            and `pred_occluder_masks` (occluder), then returns `instances`.
            Boundary predictions are NOT exposed: they are a training-only
            auxiliary, and downstream consumers can derive contours from
            `pred_masks` if needed.
        """
        occ_mask_logits, occ_boundary_logits, layer1_feat = self._layer1(x)
        tgt_mask_logits, tgt_boundary_logits = self._layer2(x, layer1_feat)

        if self.training:
            mask_side_len = tgt_mask_logits.size(2)
            assert (
                tgt_mask_logits.size(2)
                == tgt_mask_logits.size(3)
                == occ_mask_logits.size(2)
                == occ_mask_logits.size(3)
            )

            losses = {
                "loss_mask": mask_rcnn_loss(tgt_mask_logits, instances, self.vis_period)
                * self.occludee_loss_weight
                * self.loss_weight,
                "loss_occluder_mask": _occluder_mask_loss(occ_mask_logits, instances, mask_side_len)
                * self.occluder_loss_weight
                * self.loss_weight,
            }
            if self.use_boundary:
                losses["loss_occludee_boundary"] = (
                    _boundary_loss(
                        tgt_boundary_logits, instances, "gt_masks", mask_side_len, "occludee"
                    )
                    * self.occludee_boundary_weight
                    * self.loss_weight
                )
                losses["loss_occluder_boundary"] = (
                    _boundary_loss(
                        occ_boundary_logits, instances, "gt_occluder_masks", mask_side_len, "occluder"
                    )
                    * self.occluder_boundary_weight
                    * self.loss_weight
                )
            return losses

        # Inference: occludee goes through Detectron2's standard helper, which
        # attaches `pred_masks` to each Instances. We additionally attach the
        # class-agnostic occluder mask under a BCNet-specific field.
        mask_rcnn_inference(tgt_mask_logits, instances)

        occ_probs = occ_mask_logits.sigmoid()  # (N_total, 1, H, W)
        num_per_image = [len(p) for p in instances]
        occ_per_image = occ_probs.split(num_per_image, dim=0)
        for inst, occ in zip(instances, occ_per_image):
            inst.pred_occluder_masks = occ  # (N_i, 1, H, W)
        return instances
