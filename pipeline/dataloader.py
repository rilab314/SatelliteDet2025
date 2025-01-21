import torch
from typing import List
from util.misc import nested_tensor_from_tensor_list
from torch.utils.data import DataLoader
from datasets import build_dataset


def custom_collate_fn(batch: List[dict]):
    """
    batch는 Dataset에서 반환한 dict들의 리스트
    [
      {
        'image': Tensor(3, H, W),
        'targets': { 'boxes': Tensor(N,4), 'labels': Tensor(N,) },
        'height': int,
        'width': int,
        'filename': str
      },
      ...
    ]

    이걸 Deformable DETR가 원하는 형태로 맞춘다:
      samples = NestedTensor(batch_images, batch_masks)
      targets = List[Dict], 각 이미지별 box, label, size 등
    """
    # NestedTensor 생성
    images = [item['image'] for item in batch]
    samples = nested_tensor_from_tensor_list(images)

    # targets는 List[Dict[str, Tensor]] 형태
    targets = []
    for i, item in enumerate(batch):
        t = {}
        t["boxes"] = item["targets"]["boxes"]
        t["labels"] = item["targets"]["labels"]
        # Deformable DETR에서는 'orig_size', 'size' 등을 사용
        # 여기서는 이미지의 최종크기(증강 뒤)를 동일하게 쓴다고 가정
        height = item["height"]
        width = item["width"]
        t["orig_size"] = torch.tensor([height, width])
        t["size"] = torch.tensor([height, width])
        # image_id를 넣으면 나중에 coco-eval 같은 곳에서 활용 가능
        t["image_id"] = torch.tensor([i])  # 임시로 batch 내 index를 ID처럼 사용
        targets.append(t)

    return samples, targets


def create_dataloader(cfg, split='train'):
    dataset = build_dataset(cfg, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    return dataloader
