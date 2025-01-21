import cv2
import numpy as np

import settings
from datasets import build_dataset
from configs.config import CfgNode
from util.print_util import print_data


class DatasetVisualizer:
    def __init__(self, cfg, split: str):
        self.dataset = build_dataset(cfg, split)
    
    def display_dataset_frames(self):
        total_frames = len(self.dataset)
        for idx in range(total_frames):
            data = self.dataset[idx]
            print(f"===== Frame {idx}/{total_frames} =====")
            print_data(data, title='frame')
            self._print_frame_info(data)
            if self._show_frame(data):
                break
        cv2.destroyAllWindows()
    
    def _print_frame_info(self, data):
        print(f"file name: {data['filename']}")
        print(f"image size: {data['width']}x{data['height']}")
        print(f"first box: {data['targets']['boxes'][0]}")
        print(f"class IDs: {data['targets']['labels']}")
    
    def _show_frame(self, data):
        """이미지에 바운딩 박스와 클래스 ID를 표시합니다.
        
        Args:
            data (dict): 이미지와 바운딩 박스 정보를 포함하는 딕셔너리
        """
        image_tensor = data['image']
        targets = data['targets']
        image = self._to_numpy_image(image_tensor)
        image = self._draw_instances(image, targets)
        cv2.imshow('Frame', image)
        return cv2.waitKey(0) & 0xFF == ord('q')

    def _to_numpy_image(self, image_tensor):
        image = image_tensor.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = image * 255.
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _draw_instances(self, image, targets):
        height, width = image.shape[:2]
        boxes = targets['boxes']
        classes = targets['labels']
        for box, class_id in zip(boxes, classes):
            x_min, y_min, x_max, y_max = box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2
            x_min, y_min = int(x_min * width), int(y_min * height)
            x_max, y_max = int(x_max * width), int(y_max * height)
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{class_id}"
            image = cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image


def visualize_detection_dataset():
    cfg = CfgNode.from_file('defm_detr_base')
    visualizer = DatasetVisualizer(cfg, 'train' )
    visualizer.display_dataset_frames()


if __name__ == "__main__":
    visualize_detection_dataset()
