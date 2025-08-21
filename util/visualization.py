import sys
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.satellite_images import SatelliteImagesDataset


def visualize_dataset(dataset, scale_factor=4, idx=0):
    data = dataset[idx]

    torch_image = data['image']
    image = torch_image.cpu().numpy().astype('uint8')
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    CATEGORY = [{'id': 0, 'name': 'None', 'color':(0, 0, 0)},
               {'id': 1, 'name': 'ignore_label', 'color':(0, 0, 0)},
               {'id': 2, 'name': 'center_line', 'color': (77, 77, 255)},
               {'id': 3, 'name': 'u_turn_zone_line', 'color':(77, 178, 255)},
               {'id': 4, 'name': 'lane_line', 'color':(77, 255, 77)},
               {'id': 5, 'name': 'bus_only_lane', 'color':(255, 153, 77)},
               {'id': 6, 'name': 'edge_line', 'color':(255, 77, 77)},
               {'id': 7, 'name': 'path_change_restriction_line', 'color':(178, 77, 255)},
               {'id': 8, 'name': 'no_parking_stopping_line', 'color':(77, 255, 178)},
               {'id': 9, 'name': 'guiding_line', 'color':(255, 178, 77)},
               {'id': 10, 'name': 'stop_line', 'color':(77, 102, 255)},
               {'id': 11, 'name': 'safety_zone', 'color':(255, 77, 128)},
               {'id': 12, 'name': 'bicycle_lane', 'color':(128, 255, 77)},
               ]

    label_pts, label_cat = data['targets']['line_blocks'].cpu().numpy(), data['targets']['labels'].cpu().numpy()

    image_with_pts = draw_all_points_and_lines(image_bgr, label_pts, label_cat, CATEGORY)

    cv2.imshow('result', image_with_pts)
    cv2.waitKey()


def draw_all_points_and_lines(image, label_pts, mask, category_info, radius=3, thickness=2):
    result_image = image.copy()
    pts = label_pts.reshape(-1, 8)
    mask_flat = mask.reshape(-1)

    xys = pts[:, 0:2].astype(int)
    nexts = pts[:, 4:6].astype(int)
    is_start = pts[:, 6] > 0.5
    is_end = pts[:, 7] > 0.5

    for idx in range(len(pts)):
        x, y = tuple(xys[idx])
        nx, ny = tuple(nexts[idx])
        color = category_info[int(mask_flat[idx])]['color']
        cv2.line(result_image, (x, y), (nx, ny), color, thickness)
        if is_start[idx]:
            cv2.circle(result_image, (x, y), radius + 1, color, -1)
        if is_end[idx]:
            cv2.drawMarker(result_image, (nx, ny), color, markerType=cv2.MARKER_SQUARE, markerSize=6, thickness=thickness)

    return result_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='위성 이미지 라벨 시각화')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    
    # 설정 생성
    class SimpleConfig:
        pass
    
    cfg = SimpleConfig()
    cfg.dataset = SimpleConfig()
    cfg.dataset.path = args.dataset_path
    cfg.dataset.num_classes = args.num_classes
    cfg.runtime = SimpleConfig()
    cfg.runtime.device = args.device
    
    # 데이터셋 생성 및 시각화
    dataset = SatelliteImagesDataset(cfg, split='train')
    visualize_dataset(dataset, scale_factor=args.scale)
