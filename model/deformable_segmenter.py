# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

'''
Deformable DETR model and criterion classes.
'''
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
from dataclasses import dataclass

from util.misc import NestedTensor
from util.misc import MLP, build_instance
from util.print_util import print_data
from model.dto import LaneDetOutput, LineString

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DetrLaneDetector(nn.Module):
    ''' This is the Deformable DETR module that performs object detection '''
    @staticmethod
    def build_from_cfg(cfg):
        backbone = build_instance(cfg.backbone.module_name, cfg.backbone.class_name, cfg)
        transformer = build_instance(cfg.transformer.module_name, cfg.transformer.class_name, cfg)
        model = DetrLaneDetector(backbone, transformer, 
                                 num_classes=cfg.dataset.num_classes, 
                                 num_feature_levels=cfg.transformer.num_feature_levels, 
                                 )
        device = torch.device(cfg.runtime.device)
        model.to(device)
        return model

    def __init__(self, backbone, transformer, num_classes: int, num_feature_levels: int):
        ''' Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_feature_levels: number of feature levels
        '''
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.input_proj = self._get_input_proj(backbone, hidden_dim, num_feature_levels)
        self.output_proj = self._get_output_proj(hidden_dim, num_classes)
        self.backbone = backbone

    def _get_input_proj(self, backbone, hidden_dim, num_feature_levels):
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for k in range(num_backbone_outs):
                in_channels = backbone.num_channels[k]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            input_proj = nn.ModuleList(input_proj_list)
        else:
            input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
  
        for proj in input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        return input_proj
    
    def _get_output_proj(self, hidden_dim, num_classes):
        cls_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes + 3)  # 3: background, right endness, left endness
        )
        reg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 6: xr, yr, xc, yc, xl, yl
        )
        for proj in [cls_proj, reg_proj]:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
            nn.init.xavier_uniform_(proj[3].weight, gain=1)
            nn.init.constant_(proj[3].bias, 0)
        output_proj = nn.ModuleDict({'cls': cls_proj, 'reg': reg_proj})
        return output_proj

    def forward(self, samples: NestedTensor):
        '''
        The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        '''
        features, pos = self.backbone(samples)
        # assume that cfg.backbone.output_layers=['layer1', 'layer2', 'layer3', 'layer4'],
        # features: NestedTensor [[B, C, H/4, W/4], [B, 2C, H/8, W/8], [B, 4C, H/16, W/16], [B, 8C, H/32, W/32]], C=128 for SwinV2_384
        # pos: tensor [[B, D, H/4, W/4], [B, D, H/8, W/8], [B, D, H/16, W/16], [B, D, H/32, W/32]], D=256
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        memory = self.transformer(srcs, masks, pos)  # [B, sum(H*W), D], e.g. [2, 12240, 256]

        feat_hw = features[0].tensors.shape[2:]
        outputs = self.process_outputs(memory, feat_hw)
        return outputs
    
    def process_outputs(self, memory, feat_hw) -> LaneDetOutput:
        num_feat = feat_hw[0] * feat_hw[1]
        src = memory[:, :num_feat, :]
        side_scale = 6.0
        outputs = {}

        cls_out = self.output_proj['cls'](src)
        cls_out = cls_out.reshape(cls_out.shape[0], feat_hw[0], feat_hw[1], -1)
        outputs['segm_logit'] = cls_out[..., :-2]
        outputs['side_logits'] = [cls_out[..., -2:-1], cls_out[..., -1:]]

        reg_out = self.output_proj['reg'](src)
        reg_out = reg_out.reshape(reg_out.shape[0], feat_hw[0], feat_hw[1], -1)
        outputs['center_point'] = F.sigmoid(reg_out[..., :2]) * 1.2 - 0.1  # [-0.1, 1.1]
        outputs['side_points'] = [(F.sigmoid(reg_out[..., 2:4]) - 0.5) * side_scale, (F.sigmoid(reg_out[..., 4:6]) - 0.5) * side_scale]  # [-3, 3]
        outputs = LaneDetOutput(**outputs)
        return outputs
