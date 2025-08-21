import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import os
import torch
torch.set_float32_matmul_precision('medium')

import settings
from configs.config import CfgNode
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pipeline.dataloader import create_dataloader
from util.misc import build_instance


def train():
    torch.use_deterministic_algorithms(False)
    cfg = CfgNode.from_file('satellite_detr')
    pl.seed_everything(cfg.runtime.seed, workers=True)  # reproducibility
    tb_logger = TensorBoardLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)
    csv_logger = CSVLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)
    
    # 체크포인트 콜백 설정
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.runtime.output_dir, 'checkpoints'),
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor='train_total_loss',
        mode='min',
        save_top_k=10,
        save_last=True,
    )
    
    # 조기 종료 콜백 설정
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='train_loss_total',
        patience=5,
        mode='min',
        verbose=True
    )
    
    # 학습 진행률 표시 콜백
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    train_dataset = build_instance(cfg.dataset.module_name, cfg.dataset.class_name, cfg, split='train')
    train_loader = create_dataloader(cfg, train_dataset, 'train')
    print('train__loader fin')
    exit()
    val_dataset = build_instance(cfg.dataset.module_name, cfg.dataset.class_name, cfg, split='validation')
    val_loader = create_dataloader(cfg, val_dataset, 'validation')
    model = build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, progress_bar],
        accelerator='gpu',
        devices=2,
        strategy='ddp_find_unused_parameters_true',
        precision=cfg.training.get('precision', 32),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
        log_every_n_steps=50,
        val_check_interval=0.25,
        deterministic=False,
    )
    
    # 학습 시작
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
