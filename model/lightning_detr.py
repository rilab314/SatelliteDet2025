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
        box_postprocessor = build_instance(cfg.postprocessors.bbox.module_name, cfg.postprocessors.bbox.class_name, cfg)
        postprocessors = {"bbox": box_postprocessor}
        model = LitDeformableDETR(cfg, model, criterion, postprocessors)
        device = torch.device(cfg.runtime.device)
        model.to(device)
        return model

    def __init__(self, cfg, model=None, criterion=None, postprocessors=None):
        super().__init__()
        if model is None:
            model = build_instance(cfg.core_model.module_name, cfg.core_model.class_name, cfg)
            criterion = build_instance(cfg.criterion.module_name, cfg.criterion.class_name, cfg)
            box_postprocessor = build_instance(cfg.postprocessors.bbox.module_name, cfg.postprocessors.bbox.class_name, cfg)
            postprocessors = {"bbox": box_postprocessor}
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.cfg = cfg
        self.loss_weights = {k: v for k, v in cfg.losses.to_dict().items() if k.endswith('_loss')}
        self.save_hyperparameters(ignore=['model', 'criterion'])
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[LitDeformableDETR] Number of params: {n_parameters}")
        val_split = "val"
        self.val_annotation_path = os.path.join(cfg.dataset.path, val_split, cfg.dataset[val_split].coco_ann_file)
        self._outputs_buffer = []  # COCO detection 결과 저장

    def forward(self, samples):
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.loss_weights.get(k, 1.0) for k in loss_dict)
        for k, v in loss_dict.items():
            factor = self.loss_weights.get(k, 1.0)
            self.log(f"train_{k}", v * factor, prog_bar=False, batch_size=self.cfg.training.batch_size)
        return losses

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        losses = sum(loss_dict[k] * self.loss_weights.get(k, 1.0) for k in loss_dict)
        for k, v in loss_dict.items():
            factor = self.loss_weights.get(k, 1.0)
            self.log(f"val_{k}", v * factor, prog_bar=False, batch_size=self.cfg.training.batch_size)

        # metric evaluation을 위한 형식 변환 및 버퍼 저장
        target_sizes, image_ids = get_sizes_and_ids(targets, outputs["pred_logits"].device)
        coco_dets = self.postprocessors["bbox"](outputs, target_sizes, image_ids)
        self._outputs_buffer.extend(coco_dets)
        return losses

    def on_validation_epoch_end(self):
        if len(self._outputs_buffer) == 0:
            print("No predictions for validation, skip COCO eval.")
            return

        assert self.logger is not None and hasattr(self.logger, "log_dir"), "Logger is not initialized"
        pred_json_path = os.path.join(self.logger.log_dir, f"val_predictions.json")
        with open(pred_json_path, 'w') as f:
            json.dump(self._outputs_buffer, f)

        coco_gt = COCO(self.val_annotation_path)
        coco_dt = coco_gt.loadRes(pred_json_path)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_50_95 = coco_eval.stats[0]
        map_50    = coco_eval.stats[1]
        self.log("coco/AP", map_50_95, sync_dist=True)
        self.log("coco/AP50", map_50, sync_dist=True)
        print(f"[Epoch {self.current_epoch}] COCO AP: {map_50_95:.4f}, AP50: {map_50:.4f}")

        # 3) 버퍼 초기화
        self._outputs_buffer.clear()

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
