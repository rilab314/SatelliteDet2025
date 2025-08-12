import os
import json
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from util.misc import get_sizes_and_ids, build_instance


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            break
    return out


class LitDeformableDETR(pl.LightningModule):
    @staticmethod
    def build_from_cfg(cfg):
        model = build_instance(cfg.core_model.module_name, cfg.core_model.class_name, cfg)
        criterion = build_instance(cfg.criterion.module_name, cfg.criterion.class_name, cfg)
        postproc_cfg = cfg.postprocessors.to_dict()
        postprocessors = {}
        for key, val in postproc_cfg.items():
            postproc = build_instance(val['module_name'], val['class_name'], cfg)
            postprocessors[key] = postproc
        model = LitDeformableDETR(cfg, model, criterion)
        device = torch.device(cfg.runtime.device)
        model.to(device)
        return model

    def __init__(self, cfg, model=None, criterion=None, postprocessors=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.cfg = cfg
        self.loss_weights = {k: v for k, v in cfg.losses.to_dict().items() if k.endswith('_loss')}
        self.save_hyperparameters(ignore=['model', 'criterion'])
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[LitDeformableDETR] Number of params: {n_parameters}")

    def forward(self, samples):
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        for k, v in loss_dict.items():
            factor = self.loss_weights.get(k, 1.0)
            self.log(f"train_{k}", v * factor, prog_bar=False, batch_size=self.cfg.training.batch_size)
        total_loss = sum(loss_dict[k] * self.loss_weights.get(k, 0) for k in loss_dict)
        self.log(f"train_total_loss", total_loss, prog_bar=False, batch_size=self.cfg.training.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        for k, v in loss_dict.items():
            factor = self.loss_weights.get(k, 1.0)
            self.log(f"val_{k}", v * factor, prog_bar=False, batch_size=self.cfg.training.batch_size)
        total_loss = sum(loss_dict[k] * self.loss_weights.get(k, 0) for k in loss_dict)
        self.log(f"val_total_loss", total_loss, prog_bar=False, batch_size=self.cfg.training.batch_size)
        # TODO eval per frame performance

        # TODO : LineStringInstanceGenerator 로 lane instance 추출해서 수집 or segmentation map 수집
        return total_loss

    def on_validation_epoch_end(self):
        # TODO implement performance eval
        pass

    def configure_optimizers(self):
        lr = self.cfg.training.lr
        lr_backbone = self.cfg.training.lr_backbone
        lr_backbone_names = self.cfg.training.lr_backbone_names
        lr_linear_proj_names = self.cfg.training.lr_linear_proj_names
        lr_linear_proj_mult = self.cfg.training.lr_linear_proj_mult
        weight_decay = self.cfg.training.weight_decay
        lr_drop = self.cfg.training.lr_drop
        sgd = getattr(self.cfg.training, "sgd", False)

        param_dicts = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, lr_backbone_names)
                       and not match_name_keywords(n, lr_linear_proj_names)
                       and p.requires_grad
                ],
                "lr": lr,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, lr_backbone_names) and p.requires_grad
                ],
                "lr": lr_backbone,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad
                ],
                "lr": lr * lr_linear_proj_mult,
            },
        ]

        if sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

        lr_scheduler = StepLR(optimizer, step_size=lr_drop, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"  # or "coco/AP"
            }
        }
