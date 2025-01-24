# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from util.misc import MLP, build_instance
from util.print_util import print_data


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableSegmenter(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    @staticmethod
    def build_from_cfg(cfg):
        backbone = build_instance(cfg.backbone.module_name, cfg.backbone.class_name, cfg)
        transformer = build_instance(cfg.transformer_enc_only.module_name, cfg.transformer_enc_only.class_name, cfg)
        model = DeformableSegmenter(backbone, transformer, 
                                    num_classes=cfg.dataset.num_classes, 
                                    num_feature_levels=cfg.transformer_enc_only.num_feature_levels, 
                                    aux_loss=cfg.transformer_enc_only.aux_loss)
        device = torch.device(cfg.runtime.device)
        model.to(device)
        return model

    def __init__(self, backbone, transformer, num_classes: int, num_feature_levels: int, aux_loss: bool = True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.num_feature_levels = num_feature_levels
        self.input_proj = self._get_input_proj(backbone, hidden_dim, num_feature_levels)
        self.backbone = backbone
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
    
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

    def forward(self, samples: NestedTensor):
        '''
        The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        '''
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # assume that cfg.backbone.output_layers=['layer1', 'layer2', 'layer3', 'layer4'],
        # features: NestedTensor [[B, C, H/4, W/4], [B, 2C, H/8, W/8], [B, 4C, H/16, W/16], [B, 8C, H/32, W/32]], C=128 for SwinV2_384
        # pos: tensor [[B, D, H/4, W/4], [B, D, H/8, W/8], [B, D, H/16, W/16], [B, D, H/32, W/32]]
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        memory = self.transformer(srcs, masks, pos)
        return memory

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
