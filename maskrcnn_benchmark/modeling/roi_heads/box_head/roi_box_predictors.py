# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
import math


class FastRCNNPredictor(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        if cfg.FREEANCHOR.FREEANCHOR_ON:
            num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1
            prior_prob = cfg.RETINANET.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
        else:
            num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            bias_value = 0
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.bbox_pred = nn.Linear(num_inputs, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, bias_value)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        if cfg.FREEANCHOR.FREEANCHOR_ON:
            num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1
            prior_prob = cfg.RETINANET.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
        else:
            num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            bias_value = 0

        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, bias_value)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
