import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('medium')

import settings
from configs.config import CfgNode
from pipeline.dataloader import create_dataloader
from util.misc import build_instance


def train():
    cfg = CfgNode.from_file('defm_detr_base')
    pl.seed_everything(cfg.runtime.seed, workers=True)  # reproducibility
    tb_logger = TensorBoardLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)

    train_loader = create_dataloader(cfg, 'train')
    val_loader = create_dataloader(cfg, 'val')
    model = build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=tb_logger,
        # 등등 필요한 옵션(ex. 로그, 체크포인트, GradientClipVal 등)
    )
    # 학습 시작
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
