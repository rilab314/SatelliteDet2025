import os
import sys
import json
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from configs.config import CfgNode
from datasets import build_dataset


def create_coco_annotations(cfg, split='train', output_json='coco_annotations.json'):
    """
    Dataset을 순회하여,
    COCO style의 annotation.json 파일을 생성합니다.
    단, targets['boxes']는 [cx, cy, w, h] (0~1 normalize) 형식임.

    Args:
        cfg: config 객체 (cfg.dataset.path, cfg.dataset.num_classes 등 참조)
        split (str): 'train', 'val' 등
        output_json (str): 저장할 json 파일 경로
    """
    dataset = build_dataset(cfg, split=split)
    
    images = []
    annotations = []
    ann_id = 0  # annotation 고유 ID를 0부터 순차 부여

    print(f"Creating COCO annotations for {split}")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]  # {'image', 'targets': {...}, 'height', 'width', 'filename'}
        filename = sample['filename']
        img_height = sample['height']  # 실제 이미지 세로 픽셀
        img_width  = sample['width']   # 실제 이미지 가로 픽셀

        images.append({
            "file_name": os.path.basename(filename),
            "height": img_height,
            "width": img_width,
            "id": i  # image_id
        })
        boxes = sample["targets"]["boxes"]   # shape: (N, 4) -> [cx, cy, w, h], 0~1 normalized
        labels = sample["targets"]["labels"] # shape: (N,)
        if boxes.numel() == 0:
            continue

        # boxes를 COCO 형식 ([x, y, w, h]) (픽셀 단위)로 변환
        # [cx, cy, w_norm, h_norm] → x_min, y_min, w_pix, h_pix
        for j in range(boxes.shape[0]):
            cx = boxes[j, 0].item()  # normalized center x
            cy = boxes[j, 1].item()  # normalized center y
            bw = boxes[j, 2].item()  # normalized width
            bh = boxes[j, 3].item()  # normalized height
            # 픽셀 단위로 변환
            x_min = (cx - bw/2) * img_width
            y_min = (cy - bh/2) * img_height
            w_box = bw * img_width
            h_box = bh * img_height
            area = w_box * h_box
            category_id = labels[j].item()  # 클래스 ID
            ann = {
                "id": ann_id,
                "image_id": i,
                "category_id": category_id,
                "bbox": [float(x_min), float(y_min), float(w_box), float(h_box)],
                "area": float(area),
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1

    # categories 필드 (각 클래스 id, name)
    # cfg.dataset.num_classes만큼 단순 생성 (id=0..N-1)
    categories = []
    num_classes = getattr(cfg.dataset, "num_classes", 1)
    for cat_id in range(num_classes):
        categories.append({
            "id": cat_id,
            "name": f"class_{cat_id}",
            "supercategory": "none"
        })

    # 최종 COCO 딕셔너리
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)
    print(f"[create_coco_annotations] Saved {len(images)} images, {len(annotations)} annotations -> {output_json}")


if __name__ == "__main__":
    cfg = CfgNode.from_file('defm_detr_base')
    out_file = os.path.join(cfg.dataset.path, 'train', 'instances_train.json')
    create_coco_annotations(cfg=cfg, split='train', output_json=out_file)
    out_file = os.path.join(cfg.dataset.path, 'val', 'instances_val.json')
    create_coco_annotations(cfg=cfg, split='val', output_json=out_file)

