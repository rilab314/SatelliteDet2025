import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets.composer_factory import composer_factory


class SoccerPlayersDataset(Dataset):
    def __init__(self, cfg, split: str):
        self.root_path = cfg.dataset.path
        self.split = split
        self.device = torch.device(cfg.runtime.device)
        self.image_dir = str(os.path.join(cfg.dataset.path, self.split, 'images'))
        self.label_dir = str(os.path.join(cfg.dataset.path, self.split, 'labels'))
        image_files = sorted(os.listdir(self.image_dir))
        self.image_files = [file for file in image_files if file.endswith('.jpg')]
        self.augment = composer_factory(cfg, split)
        self.cfg = cfg  # cfg 객체를 저장하여 다른 메서드에서 사용

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        bboxes, category_ids = self.load_labels(idx)
        image, bboxes, category_ids = self.apply_augmentation(
            image, bboxes, category_ids
        )
        image = image.to(self.device)
        bboxes = torch.tensor(bboxes, dtype=torch.float32, device=self.device)
        category_ids = torch.tensor(category_ids, dtype=torch.int64, device=self.device)
        return {
            'image': image,
            'targets': {'boxes': bboxes, 'labels': category_ids},
            'height': image.shape[1],
            'width': image.shape[2],
            'filename': os.path.join(self.image_dir, self.image_files[idx]),
        }

    def load_image(self, idx):
        """
        Returns:
            image (numpy.ndarray): 로드된 이미지 (RGB)
        """
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_labels(self, idx):
        """
        Returns:
            bboxes (list): 바운딩 박스 리스트 ([[center_x, center_y, width, height], ...])
            category_ids (list): 클래스 레이블 리스트
        """
        image_filename = self.image_files[idx]
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_filename)

        bboxes = []
        category_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.strip().split())
                    bboxes.append([center_x, center_y, bbox_width, bbox_height])
                    category_ids.append(int(class_id))
        else:
            print(f"Label file not found: {label_path}")
        return np.array(bboxes, dtype=np.float32), np.array(category_ids, dtype=np.int64)

    def apply_augmentation(self, image, bboxes, category_ids):
        transformed = self.augment(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        image = transformed['image']
        bboxes = transformed['bboxes']
        category_ids = transformed['category_ids']
        return image, bboxes, category_ids
