from torch import nn
from torch.nn.functional import interpolate
from maskrcnn_benchmark.data.transforms.transforms import Resize
from maskrcnn_benchmark.structures.image_list import ImageList, to_image_list


class MultiScaleRetinaNet(nn.Module):
    """
    Main class for RetinaNet
    It consists of three main parts:
    - backbone
    - bbox_heads: BBox prediction.
    - Mask_heads:
    """

    def __init__(self, retinanet, scales):
        super(MultiScaleRetinaNet, self).__init__()
        self.retinanet = retinanet
        self.resizers = [Resize(min_size, max_size) for (min_size, max_size) in scales]

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
        assert self.training is False

        all_anchors, all_box_cls, all_box_regression = [], [], []
        for resizer in self.resizers:
            image_size = images.image_sizes[0]
            size = resizer.get_size(image_size[::-1])
            aug_images = interpolate(
                images.tensors[:, :, :image_size[0], :image_size[1]], size, mode='bilinear', align_corners=True
            )[0]
            aug_images = to_image_list(aug_images, size_divisible=self.retinanet.cfg.DATALOADER.SIZE_DIVISIBILITY)
            features = self.retinanet.backbone(aug_images.tensors)
            if self.retinanet.cfg.RETINANET.BACKBONE == "p2p7":
                features = features[1:]
            box_cls, box_regression = self.retinanet.rpn.head(features)
            anchors = self.retinanet.rpn.anchor_generator(aug_images, features)[0]
            all_anchors.extend(anchors), all_box_cls.extend(box_cls), all_box_regression.extend(box_regression)

        detections = self.retinanet.rpn.box_selector_test([all_anchors], all_box_cls, all_box_regression)

        return detections
