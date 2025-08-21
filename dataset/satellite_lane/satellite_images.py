import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.composer_factory import composer_factory


class SatelliteImagesDataset(Dataset):
    @staticmethod
    def build_from_cfg(cfg, split):
        dataset = SatelliteImagesDataset(cfg.dataset.path, cfg.dataset.num_classes, cfg.runtime.device, split=split)
        return dataset

    def __init__(self, dataset_path, num_classes, device, split: str):
        self.root_path = dataset_path
        self.split = split
        self.device = torch.device(device)
        self.image_dir = str(os.path.join(dataset_path, self.split, 'image')    )
        self.label_dir = str(os.path.join(dataset_path, self.split, 'label'))
        image_files = sorted(os.listdir(self.image_dir))
        self.image_files = [file for file in image_files if file.endswith(('.png'))]
        # self.augment = composer_factory(cfg, split)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image = self.load_image(idx)
        labels = self.load_numpy_labels(idx)        
        # if self.augment:
        #     image, labels = self.apply_augmentation(image, labels)
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)

        target_dict = dict()
        target_dict["center_point"] = labels_tensor[:, :, :2]
        target_dict["left_point"] = labels_tensor[:, :, 2:4]
        target_dict["right_point"] = labels_tensor[:, :, 4:6]
        target_dict["left_end"] = labels_tensor[:, :, 6:7]
        target_dict["right_end"] = labels_tensor[:, :, 7:8]
        target_dict["segm_label"] = labels_tensor[:, :, 8:]
        
        height, width = image.shape[1], image.shape[2]
        return {
            'image': image,
            'targets': target_dict,
            'height': height,
            'width': width,
            'filename': os.path.join(self.image_dir, image_filename)
        }

    def load_image(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_numpy_labels(self, idx):
        """
        9차원 넘파이 라벨 파일을 로드합니다.
        라벨 차원 설명
        0, 1: 현재 블록의 좌표 정보 (x, y)
        2, 3: 이전 블록의 좌표 정보 (prev_x, prev_y)
        4, 5: 다음 블록의 좌표 정보 (next_x, next_y)
        6: 현재 블록이 라인의 시작 블록인지 여부 (is_start)
        7: 다음 블록이 라인의 끝 블록인지 여부 (is_end)
        8: 카테고리 정보 (category_id)

        Returns:
            labels (np.ndarray): 라벨 정보 (N x 9 배열)
        """
        image_filename = self.image_files[idx]
        label_filename = os.path.splitext(image_filename)[0] + '.npy'
        label_path = os.path.join(self.label_dir, label_filename)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        return np.load(label_path)

    def apply_augmentation(self, image, labels):
        line_blocks = labels[:, :8]
        category_ids = labels[:, 8]
        
        transformed = self.augment(
            image=image,
            line_blocks=line_blocks,
            category_ids=category_ids
        )
        
        image = transformed['image']
        line_blocks = transformed['line_blocks']
        category_ids = transformed['category_ids']
        
        augmented_labels = np.zeros((len(category_ids), 9), dtype=np.float32)
        augmented_labels[:, :8] = line_blocks
        augmented_labels[:, 8] = category_ids
        
        return image, augmented_labels
    