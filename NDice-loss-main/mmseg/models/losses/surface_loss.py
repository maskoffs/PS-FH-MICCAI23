# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as eucl_distance
import numpy as np

from mmseg.registry import MODELS
from .utils import get_class_weight, weighted_loss


@weighted_loss
def surface_loss(pred,
              target,
              ignore_index=255):
    assert pred.shape == target.shape
    dist_map = one_hot2dist(target)
    multipled = torch.einsum("bcwh,bcwh->bcwh", pred, dist_map)
    multipled = multipled[:, ignore_index + 1:, :, :]
    loss = torch.mean(multipled, dim=[1, 2, 3])
    return loss


# @weighted_loss
def one_hot2dist(seg, dtype='float32'):
    seg = seg.cpu().numpy()
    b, c, w, h = seg.shape
    # num_classes = seg.shape[1]
    seg = seg.reshape((-1, w, h))
    res = np.zeros_like(seg, dtype=dtype)
    pos_mask = seg.astype('bool')
    neg_mask = ~pos_mask
    for i in range(b*c):
        if pos_mask[i].any():
            res[i] = eucl_distance(neg_mask[i]) * neg_mask[i] \
                - (eucl_distance(pos_mask[i]) - 1) * pos_mask[i]
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel
    res = torch.from_numpy(res.reshape((b, c, w, h))).to('cuda:0')
    return res


@MODELS.register_module()
class SurfaceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_surface',
                 **kwards):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                **kwards):

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        one_hot_target = torch.permute(one_hot_target, [0, 3, 1, 2])

        loss = self.loss_weight * surface_loss(
            pred,
            one_hot_target,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
