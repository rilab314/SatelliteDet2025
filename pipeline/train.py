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


def train_bef():
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

def train():
    cfg = CfgNode.from_file('satellite_detr')
    pl.seed_everything(cfg.runtime.seed, workers=True)  # reproducibility
    tb_logger = TensorBoardLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)
    
    # 체크포인트 콜백 설정
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.runtime.output_dir, 'checkpoints'),
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    # 조기 종료 콜백 설정
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )
    
    # 학습 진행률 표시 콜백
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    
    train_loader = create_dataloader(cfg, 'train')
    val_loader = create_dataloader(cfg, 'val')
    model = build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.training.get('num_gpus', 1),
        strategy='ddp' if cfg.training.get('num_gpus', 1) > 1 else None,
        precision=cfg.training.get('precision', 32),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
        log_every_n_steps=50,
        val_check_interval=0.25,  # 25% 마다 검증 수행
        deterministic=True,
        # 자동 배치 크기 찾기 (선택 사항)
        # auto_scale_batch_size='binsearch',
        # 자동 학습률 찾기 (선택 사항)
        # auto_lr_find=True,
    )
    
    # 학습 시작
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
