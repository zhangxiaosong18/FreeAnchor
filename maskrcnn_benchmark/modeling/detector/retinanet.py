# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.retinanet import build_retinanet
import copy


class RetinaNet(nn.Module):
    """
    Main class for RetinaNet
    It consists of three main parts:
    - backbone
    - bbox_heads: BBox prediction.
    - Mask_heads:
    """

    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.backbone = build_backbone(cfg)
        self.rpn = build_retinanet(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        # Retina RPN Output
        rpn_features = features
        if self.cfg.RETINANET.BACKBONE == "p2p7":
            rpn_features = features[1:]
        (anchors, detections), detector_losses = self.rpn(images, rpn_features, targets)
        if self.training:
            return {detector_losses}
        else:
            return detections
