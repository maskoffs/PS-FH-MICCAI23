# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from scipy.ndimage import distance_transform_edt as eucl_distance
import numpy as np


def one_hot2class_weight(seg):
    b, c, w, h = seg.shape
    class_weight = torch.sum(seg, dim=(0, 2, 3)) / (b * w * h)
    class_weight = 2 / (class_weight + 1)
    return class_weight


def one_hot2weight(seg, r=5):
    b, c, w, h = seg.shape
    seg = seg.reshape((-1, 1, w, h))
    seg_reverse = 1 - seg
    seg_reverse = seg_reverse.type(torch.float32)
    out = torch.zeros((b, w, h), device='cuda:0')
    for i in range(1, r + 1):
        weight = torch.ones([1, 1, 2 ** i + 1, 2 ** i + 1], dtype=torch.float32, device='cuda:0')
        res = F.conv2d(seg_reverse, weight, padding=2**(i-1))
        res = res / ((2 ** i + 1)**2 - 1)
        res[seg == 0] = 0
        res = torch.sum(res.reshape((b, c, w, h)), dim=1)
        out = out + res
    out = out + 1
    return out


def one_hot2weight_s(seg, r=5):
    boundary_w = one_hot2weight(seg, r) - 1
    b, c, w, h = seg.shape
    class_w = torch.sum(seg, dim=(0, 2, 3)) / (b * w * h)
    class_w = 1 / (class_w + 1)
    class_w = class_w.repeat(b, w, h, 1)
    class_w = torch.permute(class_w, dims=(0, 3, 1, 2))
    class_w[seg == 0] = 0
    class_w = torch.sum(class_w, dim=1)
    return boundary_w + class_w + 1
    # class_w = one_hot2class_weight(seg).repeat()


def one_hot2dist(seg, dtype='float32', reverse=True):
    seg = seg.cpu().numpy()
    b, c, w, h = seg.shape
    # num_classes = seg.shape[1]
    seg = seg.reshape((-1, w, h))
    res = np.zeros_like(seg, dtype=dtype)
    pos_mask = seg.astype('bool')
    # neg_mask = ~pos_mask
    for i in range(b*c):
        if pos_mask[i].any():
            res[i] = eucl_distance(pos_mask[i])
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel
    res = torch.from_numpy(res.reshape((b, c, w, h))).to('cuda:0')
    res = torch.log(torch.sum(res, dim=1))
    if reverse:
        res = torch.max(res) - res + 1
    else:
        res = res + 1

    return res


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@MODELS.register_module()
class AutoWeightedCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 reduction='mean',
                 class_weight='auto',
                 boundary_weight='neighbor',
                 loss_weight=1.0,
                 loss_name='loss_auto_weighted_ce',
                 avg_non_ignore=False):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        # assert auto_class_weight in (None, 'auto')
        self.class_weight = class_weight
        assert boundary_weight in (None, 'distance', 'neighbor', 'neighbor_s')
        self.boundary_weight = boundary_weight
        # self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')
        self.cls_criterion = cross_entropy
        if self.boundary_weight == 'distance':
            self.boundary_weight_func = one_hot2dist
        elif self.boundary_weight == 'neighbor':
            self.boundary_weight_func = one_hot2weight
        elif self.boundary_weight == 'neighbor_s':
            self.boundary_weight_func = one_hot2weight_s
        else:
            self.boundary_weight_func = None
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        num_classes = cls_score.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(label.long(), 0, num_classes - 1),
            num_classes=num_classes)
        one_hot_target = torch.permute(one_hot_target, [0, 3, 1, 2])
        if self.class_weight == 'auto':
            class_weight = one_hot2class_weight(one_hot_target)
        elif self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
            # class_weight = one_hot2class_weight(one_hot_target)
        # Note: for BCE loss, label < 0 is invalid.
        if weight is None and self.boundary_weight_func is not None:
            # weight = one_hot2dist(one_hot_target, reverse=True)
            # weight = one_hot2weight(one_hot_target)
            weight = self.boundary_weight_func(one_hot_target)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls

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
