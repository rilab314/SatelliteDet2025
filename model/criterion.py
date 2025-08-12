# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from typing import List

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)
import copy

from util.misc import build_instance


class SegmentationCriterion(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):
        matcher = build_instance(cfg.matcher.module_name, cfg.matcher.class_name, cfg)
        losses = cfg.losses.to_dict()
        losses = [k for k, v in losses.items() if k != 'focal_alpha' and v != 0 and v != False]
        return SegmentationCriterion(
            num_classes=cfg.dataset.num_classes,
            matcher=matcher,
            loss_names=losses,
            focal_alpha=cfg.losses.focal_alpha
        )

    def __init__(self, num_classes, matcher, loss_names: List[str], focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_names = loss_names
        self.focal_alpha = focal_alpha

    def forward(self, outputs, targets):
        """
        outputs: List[Dict[str, Tensor]]
            - 각 dict 안에 'pred_points', 'pred_headtail', 'pred_logits' 키 존재
            - 각각 shape: (N, C, 6), (N, 2), (N, 1)
        
        targets: List[Dict[str, Tensor]]
            - 각 dict 안에 'points', 'headtail', 'labels' 키 존재
            - 각각 shape: (N, 6), (N, 2), (N, 1)
        """

        loss = {loss_name: 0 for loss_name in self.loss_names}
        for i, (output, target) in enumerate(zip(outputs, targets)):
            if 'cls_loss' in self.loss_names:
                loss['cls_loss'] += self.classification_loss(output, target)

            mask = (target['segm_label'] > 0).squeeze(-1)
            for key in target:
                if key not in ['size', 'image_id']:
                    target[key] = target[key][mask]  # (H, W, C) -> (N,  C)
            for key in output:
                output[key] = output[key][mask]

            straight_match = self.matcher(output, target)
            if 'end_loss' in self.loss_names:
                loss['end_loss'] += self.endness_loss(output, target, straight_match)
            if 'point_loss' in self.loss_names:
                loss['point_loss'] += self.point_loss(output, target, straight_match)

        return loss

    def classification_loss(self, output, target):
        target_label = target['segm_label'].squeeze(-1)
        output_logit = output['segm_logit'].permute(2, 0, 1).contiguous()  # (H, W, C) -> (C, H, W)
        output_logit = output_logit.unsqueeze(0)  # (1, C, H, W)
        target_label = target_label.unsqueeze(0).long()  # (1, H, W)
        loss_cls = F.cross_entropy(output_logit, target_label, reduction='mean')
        return loss_cls

    def endness_loss(self, output, target, straight_match: torch.Tensor):
        stacked_original = torch.stack([output['left_end_logit'], output['right_end_logit']], dim=1)  # (N, 2, 2)
        stacked_swapped = torch.stack([output['right_end_logit'], output['left_end_logit']], dim=1)
        condition = (straight_match == 1).view(output['left_end_logit'].shape[0], 1, 1)
        aligned_logits = torch.where(condition, stacked_original, stacked_swapped)
        stacked_target = torch.stack([target['left_end'], target['right_end']], dim=1)
        loss = F.binary_cross_entropy_with_logits(aligned_logits, stacked_target, reduction='mean')
        return loss

    def point_loss(self, output, target, straight_match: torch.Tensor):
        """
        output['xxx_point']: (N,2)
        staight_match: (N,)
        """
        center_loss = F.smooth_l1_loss(output['center_point'], target['center_point'], reduction='mean')

        stacked_original = torch.stack([output['left_point'], output['right_point']], dim=1)  # (N, 2, 2)
        stacked_swapped = torch.stack([output['right_point'], output['left_point']], dim=1)
        condition = (straight_match == 1).view(output['center_point'].shape[0], 1, 1)  # (N, 1, 1)
        aligned_points = torch.where(condition, stacked_original, stacked_swapped)
        stacked_target = torch.stack([target['left_point'], target['right_point']], dim=1)
        aligned_dist = torch.norm(aligned_points - stacked_target, dim=2).sum(dim=1)     # [N]

        # 더 짧은 방향을 선택했는지 확인
        straight_dist = torch.norm(stacked_original - stacked_target, dim=2).sum(dim=1)  # [N]
        reverse_dist = torch.norm(stacked_swapped - stacked_target, dim=2).sum(dim=1)    # [N]
        min_dist = torch.minimum(straight_dist, reverse_dist)  # [N]
        correct = (aligned_dist == min_dist)                   # [N], bool
        assert correct.sum().item() == correct.shape[0], \
            f"matcher verification failed: {correct.sum().item()} != {correct.shape[0]}"

        side_loss = F.smooth_l1_loss(aligned_points, stacked_target, reduction='mean')
        return center_loss + side_loss
