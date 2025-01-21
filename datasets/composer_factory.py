import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from detectron2.structures import Instances, Boxes
import albumentations as A
from albumentations.pytorch import ToTensorV2
from detectron2.config import get_cfg, CfgNode


def composer_factory(cfg, split: str):
    """
    cfg.dataset.augmentation에 정의된 설정을 기반으로 albumentations.Compose 객체를 생성합니다.
    """
    if cfg.dataset[split].augmentation:
        augmentations = create_augmentations(cfg)
    else:
        augmentations = []

    augmentations.append(A.Resize(
        height=cfg.dataset.image_height,
        width=cfg.dataset.image_width,
        interpolation=cv2.INTER_LINEAR,
        p=1.0
    ))
    augmentations.append(A.ToFloat(max_value=255.0))
    augmentations.append(ToTensorV2())

    bbox_params = A.BboxParams(
        format='yolo',  # 데이터셋의 바운딩 박스 형식에 따라 설정
        label_fields=['category_ids']
    )
    transform = A.Compose(augmentations, bbox_params=bbox_params)
    return transform


def create_augmentations(cfg):
    aug_cfg = cfg.dataset.augmentation
    augmentations = []
    if 'horizontal_flip' in aug_cfg:
        params = aug_cfg.horizontal_flip
        augmentations.append(A.HorizontalFlip(p=params.get('p', 0.5)))

    if 'random_resized_crop' in aug_cfg:
        params = aug_cfg.random_resized_crop
        augmentations.append(A.RandomResizedCrop(
            size=params.get('size', (384, 384)),
            scale=params.get('scale', (0.5, 1.5)),
            ratio=params.get('ratio', (0.75, 1.333)),
            p=params.get('p', 1.0)
        ))

    if 'random_brightness_contrast' in aug_cfg:
        params = aug_cfg.random_brightness_contrast
        augmentations.append(A.RandomBrightnessContrast(
            brightness_limit=params.get('brightness_limit', 0.2),
            contrast_limit=params.get('contrast_limit', 0.2),
            p=params.get('p', 0.5)
        ))

    if 'hue_saturation_value' in aug_cfg:
        params = aug_cfg.hue_saturation_value
        augmentations.append(A.HueSaturationValue(
            hue_shift_limit=params.get('hue_shift_limit', 20),
            sat_shift_limit=params.get('sat_shift_limit', 30),
            val_shift_limit=params.get('val_shift_limit', 20),
            p=params.get('p', 0.5)
        ))

    if 'random_gamma' in aug_cfg:
        params = aug_cfg.random_gamma
        augmentations.append(A.RandomGamma(
            gamma_limit=params.get('gamma_limit', (80, 120)),
            p=params.get('p', 0.5)
        ))

    if 'gauss_noise' in aug_cfg:
        params = aug_cfg.gauss_noise
        augmentations.append(A.GaussNoise(
            std_range=params.get('std_range', (0.2, 0.44)),
            mean_range=params.get('mean_range', (0.0, 0.0)),
            p=params.get('p', 0.5)
        ))

    return augmentations
