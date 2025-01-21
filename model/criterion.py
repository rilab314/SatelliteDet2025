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


class SetCriterion(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):
        matcher = build_instance(cfg.matcher.module_name, cfg.matcher.class_name, cfg)
        losses = cfg.losses.to_dict()
        losses = [k for k, v in losses.items() if k != 'focal_alpha' and v != 0 and v != False]
        if cfg.transformer.segmentation is False:
            losses.remove('mask_loss')
            losses.remove('dice_loss')
        return SetCriterion(
            num_classes=cfg.dataset.num_classes,
            matcher=matcher,
            loss_names=losses,
            focal_alpha=cfg.losses.focal_alpha
        )

    ''' This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    '''
    def __init__(self, num_classes, matcher, loss_names: List[str], focal_alpha=0.25):
        ''' Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            loss_names: list containing the names of the losses
            focal_alpha: alpha in Focal Loss
        '''
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_names = loss_names
        self.focal_alpha = focal_alpha  # TODO check

    def forward(self, outputs, targets):
        ''' This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        '''
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = self.get_losses(self.loss_names, outputs, targets, indices, num_boxes)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                loss_names = [k for k in self.loss_names if k != 'masks']  # mask loss is too costly for intermediate layers
                kwargs = {'log': False} if 'accuracy' in loss_names else {}
                aux_losses = self.get_losses(loss_names, aux_outputs, targets, indices, num_boxes, **kwargs)
                aux_losses = {k + f'_{i}': v for k, v in aux_losses.items()}
                losses.update(aux_losses)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            loss_names = [k for k in self.loss_names if k != 'masks']  # mask loss is too costly for intermediate layers
            kwargs = {'log': False} if 'accuracy' in loss_names else {}
            enc_losses = self.get_losses(loss_names, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
            enc_losses = {k + f'_enc': v for k, v in enc_losses.items()}
            losses.update(enc_losses)

        return losses

    def get_losses(self, loss_names: List[str], outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'cls_loss': self.loss_labels,
            'accuracy': self.loss_labels,
            'bbox_loss': self.loss_boxes,
            'giou_loss': self.loss_boxes,
            'mask_loss': self.loss_masks,
            'dice_loss': self.loss_masks,
            'cardinality': self.loss_cardinality,
        }
        losses = {}
        for loss_name in loss_names:
            if loss_name not in losses:
                loss_fn = loss_map[loss_name]
                losses.update(loss_fn(outputs, targets, indices, num_boxes, **kwargs))

        losses = {k: v for k, v in losses.items() if k in losses}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        ''' 
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        '''
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'cls_loss': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['accuracy'] = accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, log=True):
        ''' 
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        '''
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, log=True):
        ''' 
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        '''
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['bbox_loss'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['giou_loss'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, log=True):
        '''
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        '''
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {"mask_loss": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
                  "dice_loss": dice_loss(src_masks, target_masks, num_boxes),
                  }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
