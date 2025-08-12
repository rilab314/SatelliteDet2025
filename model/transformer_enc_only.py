# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from model.deformable_transformer_todo import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from model.ops.modules import MSDeformAttn
from util.print_util import print_model, print_data


class DeformableTransformerEncoderOnly(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):    
        return DeformableTransformerEncoderOnly(
            num_classes=cfg.dataset.num_classes,
            d_model=cfg.transformer.hidden_dim,
            nhead=cfg.transformer.nheads,
            num_encoder_layers=cfg.transformer.enc_layers,
            dim_feedforward=cfg.transformer.dim_feedforward,
            dropout=cfg.transformer.dropout,
            num_feature_levels=cfg.transformer.num_feature_levels,
            enc_n_points=cfg.transformer.enc_n_points,
        )

    def __init__(self, num_classes: int, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def forward(self, srcs, masks, pos_embeds):
        '''
        assume that cfg.backbone.output_layers=['layer1', 'layer2', 'layer3', 'layer4'],
        srcs: list of tensors, [[B, C, H/4, W/4], [B, C, H/8, W/8], [B, C, H/16, W/16], [B, C, H/32, W/32]], C=256
        masks: list of tensors, [[B, H/4, W/4], [B, H/8, W/8], [B, H/16, W/16], [B, H/32, W/32]]
        pos_embeds: list of tensors, [[B, C, H/4, W/4], [B, C, H/8, W/8], [B, C, H/16, W/16], [B, C, H/32, W/32]], C=128
        '''
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # [B, C, H*W] -> [B, H*W, C]
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [B, C, H*W] -> [B, H*W, C]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # [B, H*W, C] + [C] -> [B, H*W, C]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # [B, sum(H*W), C]
        mask_flatten = torch.cat(mask_flatten, 1)  # [B, sum(H*W)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [B, sum(H*W), C]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [L, 2], L=num levels=4
        # level_start_index: [L,] [0, H0*W0, sum_i=0~1_Hi*Wi, sum_i=0~2_Hi*Wi, sum_i=0~3_Hi*Wi], e.g. [0, 9216, 11520, 12096]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder, memory: [B, sum(H*W), C]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
