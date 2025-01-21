import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import numpy as np
import torch
from torch.utils.data import DataLoader

import settings
from configs.config import CfgNode
from util.print_util import print_model, print_data
from util.misc import NestedTensor, nested_tensor_from_batch_data, build_instance
from pipeline.dataloader import create_dataloader


def create_modules():
    print('\n========== config ==========\n')
    cfg = CfgNode.from_file('defm_detr_base')
    # print(cfg)
    print('\n========== backbone ==========\n')
    backbone = build_instance(cfg.backbone.module_name, cfg.backbone.class_name, cfg)
    print_model(backbone, max_depth=4)
    print('\n========== detr ==========\n')
    detr = build_instance(cfg.transformer.module_name, cfg.transformer.class_name, cfg)
    print_model(detr, max_depth=3)
    print('\n========== model ==========\n')
    model = build_instance(cfg.core_model.module_name, cfg.core_model.class_name, cfg)
    print_model(model, max_depth=4)
    print('\n========== criterion ==========\n')
    criterion = build_instance(cfg.criterion.module_name, cfg.criterion.class_name, cfg)
    print_model(criterion, max_depth=4)
    print('\n========== postprocessors ==========\n')
    box_postprocessor = build_instance(cfg.postprocessors.bbox.module_name, cfg.postprocessors.bbox.class_name, cfg)
    print_model(box_postprocessor, max_depth=4)


def check_backbone_outputs():
    cfg = CfgNode.from_file('defm_detr_base')
    dataloader = create_dataloader(cfg, 'val')
    backbone = build_instance(cfg.backbone.module_name, cfg.backbone.class_name, cfg)
    for k, batch in enumerate(dataloader):
        print(f"===== Batch {k + 1}/{len(dataloader)} =====")
        samples, targets = batch
        print_data(samples, title='samples')
        print_data(targets, title='targets')
        outputs = backbone(samples)
        print_data(outputs, title='outputs')
        break


def check_defm_detr_outputs():
    cfg = CfgNode.from_file('defm_detr_base')
    dataloader = create_dataloader(cfg, 'val')
    model = build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)
    criterion = build_instance(cfg.criterion.module_name, cfg.criterion.class_name, cfg)
    for k, batch in enumerate(dataloader):
        print(f"===== Batch {k + 1}/{len(dataloader)} =====")
        samples, targets = batch
        print_data(samples, title='samples')
        print_data(targets, title='targets')
        outputs = model(samples)
        print_data(outputs, title='outputs')
        loss = criterion(outputs, targets)
        print_data(loss, title='loss')
        break


if __name__ == "__main__":
    create_modules()
    check_backbone_outputs()
    check_defm_detr_outputs()
