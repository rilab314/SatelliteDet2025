import os
import glob
import cv2
import json
import numpy as np
from typing import List


class ConvertDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.scale = 4  # 다운샘플링 비율 (예: 4 → 원본은 4×4 블록)

    def convert(self):
        image_list = glob.glob(os.path.join(self.dataset_path, 'image', '*.jpg'))
        for image_path in image_list:
            image, annot = self.load_image_and_annot(image_path)
            gt_map_list = []
            for line_string in annot['lines']:
                # line_string를 numpy array로 변환 (shape: (N,2))
                line_string_np = np.array(line_string, dtype=np.int32)
                line_splits = self.split_line_string(line_string_np)
                xy_vec_list = []
                for line_split in line_splits:
                    mask = self.generate_lane_mask(image, line_split)
                    # line_split를 인자로 투영 계산 등 수행
                    xy_vec = self.find_line_points_on_block(mask, line_split)
                    xy_vec = self.sort_xy_from_start_to_end(xy_vec, line_split)
                    xy_vec_list.append(xy_vec)
                xy_vec = self.connect_xy_vecs(xy_vec_list)
                gt_map = self.generate_gt_map(image, xy_vec)
                gt_map_list.append(gt_map)
            gt_map = self.merge_gt_maps(gt_map_list)

    def load_image_and_annot(self, image_path: str):
        image = cv2.imread(image_path)
        annot_path = image_path.replace('image', 'label').replace('.jpg', '.json')
        with open(annot_path, 'r') as f:
            annot = json.load(f)
        return image, annot

    def split_line_string(self, line_string: np.ndarray) -> List[np.ndarray]:
        '''
        line_string: a list of (x,y) coordinates on line string / type: np.ndarray, shape: (N, 2)
        return: a list of line splits, each split is a list of (x,y) coordinates / type: List[np.ndarray], shape: (M, 2)
        
        line_string의 각 점과 시작점(line_string[0]) 사이의 거리를 계산하여,
        거리가 가장 큰 점을 기준으로 두 개의 선분으로 나눈다. 
        만약 가장 먼 점이 끝점(line_string[-1])이면 [line_string]을 반환한다.
        '''
        # 시작점과의 유클리드 거리를 계산
        dists = np.linalg.norm(line_string - line_string[0], axis=1)
        split_idx = np.argmax(dists)
        if split_idx == (len(line_string) - 1):
            return [line_string]
        else:
            # split_idx를 기준으로 앞부분과 뒷부분(중복되는 split_idx 포함)으로 분리
            split1 = line_string[:split_idx + 1]
            split2 = line_string[split_idx:]
            return [split1, split2]

    def generate_lane_mask(self, image: np.ndarray, line_split: np.ndarray) -> np.ndarray:
        """
        이전에 만든 generate_lane_mask() 함수를 사용합니다.
        입력 인자는 아래 고정값을 사용:
          - downsample_factor=self.scale, thickness=1, method='resize'
        """
        if len(image.shape) == 3:
            H, W, _ = image.shape
        else:
            H, W = image.shape

        mask_full = np.zeros((H, W), dtype=np.uint8)
        pts = line_split.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask_full, [pts], isClosed=False, color=255, thickness=1)

        new_H = H // self.scale
        new_W = W // self.scale
        mask_down = cv2.resize(mask_full, (new_W, new_H), interpolation=cv2.INTER_AREA)
        mask_binary = (mask_down > 0).astype(np.uint8)
        return mask_binary

    def find_line_points_on_block(self, mask: np.ndarray, line_split: np.ndarray) -> np.ndarray:
        """
        mask: binary mask of shape (H/scale, W/scale), 값이 1인 블록은 차선이 지나감.
        line_split: 원본 해상도의 line string 좌표, shape: (N, 2)
        
        각 mask의 1인 블록(4×4 영역)에서, 블록 중심에서 선분에 투영한 점과
        블록 내부에 존재하는 line string의 노드 후보 중 중심과 더 가까운 점을 선택하여
        유효한 좌표들만 추출하여 반환합니다.
        최종 반환되는 좌표들은 원본 해상도 기준이며, mask에 해당하는 블록 순서대로 모아집니다.
        """
        scale_factor = self.scale
        # mask에서 1인 블록의 인덱스를 추출 (각 행: [row, col])
        block_idx = np.argwhere(mask == 1)
        if block_idx.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        block_rows = block_idx[:, 0].astype(np.float32)
        block_cols = block_idx[:, 1].astype(np.float32)
        # 각 블록의 중심 좌표 (원본 해상도)
        centers = np.empty((block_idx.shape[0], 2), dtype=np.float32)
        centers[:, 0] = block_cols * scale_factor + (scale_factor - 1) / 2.0  # x 좌표
        centers[:, 1] = block_rows * scale_factor + (scale_factor - 1) / 2.0  # y 좌표

        # 각 블록의 경계 (원본 해상도)
        x_min = block_cols * scale_factor
        x_max = (block_cols + 1) * scale_factor
        y_min = block_rows * scale_factor
        y_max = (block_rows + 1) * scale_factor

        # line_split의 노드가 2개 미만이면 선분 후보가 없으므로 노드 후보만 사용
        if line_split.shape[0] < 2:
            best_nodes, node_dists = ConvertDataset._get_dist_point_to_nodes(
                centers, block_cols, block_rows, line_split, scale_factor)
            valid = node_dists < np.inf
            return best_nodes[valid]

        # 선분 구성: 인접 노드 쌍
        seg_A = line_split[:-1].astype(np.float32)
        seg_B = line_split[1:].astype(np.float32)

        foot, proj_dists = ConvertDataset._project_points_onto_segments(
            centers, seg_A, seg_B, x_min, x_max, y_min, y_max)

        best_seg_idx = np.argmin(proj_dists, axis=1)
        best_proj_dists = proj_dists[np.arange(foot.shape[0]), best_seg_idx]
        best_proj = foot[np.arange(foot.shape[0]), best_seg_idx, :]

        best_nodes, node_dists = ConvertDataset._get_dist_point_to_nodes(
            centers, block_cols, block_rows, line_split, scale_factor)

        final_dists = np.minimum(best_proj_dists, node_dists)
        use_proj = best_proj_dists <= node_dists
        final_points = np.where(use_proj[:, None], best_proj, best_nodes)
        valid = final_dists < np.inf
        final_points = final_points[valid]
        return final_points

    def sort_xy_from_start_to_end(self, xy_vec: np.ndarray, line_split: np.ndarray) -> np.ndarray:
        '''
        xy_vec: 추출된 (x,y) 좌표들, shape: (M, 2)
        line_split: 원본 line string의 노드들, shape: (N, 2)
        반환: line_split 상의 순서를 참고하여 시작점에서 끝점 순으로 정렬된 (x,y) 좌표들.
        '''
        if xy_vec.shape[0] == 0:
            return xy_vec
        # 각 xy에 대해, line_split의 각 노드와의 거리를 계산하여 가장 가까운 노드의 인덱스를 구함
        indices = []
        for xy in xy_vec:
            dists = np.linalg.norm(line_split - xy, axis=1)
            idx = np.argmin(dists)
            indices.append(idx)
        indices = np.array(indices)
        order = np.argsort(indices)
        return xy_vec[order]

    def connect_xy_vecs(self, xy_vec_list: List[np.ndarray]) -> np.ndarray:
        '''
        xy_vec_list: List of xy vectors, each of shape (M, 2)
        반환: 여러 xy vector가 있다면 concatenate()한 후,
                같은 블록(중복되는 좌표)이 있으면 제거하여 연결한 (L, 2) 좌표 배열.
                만약 하나의 vector만 있다면 그대로 반환.
        '''
        if len(xy_vec_list) == 0:
            return np.empty((0, 2), dtype=np.float32)
        if len(xy_vec_list) == 1:
            return xy_vec_list[0]
        concatenated = np.concatenate(xy_vec_list, axis=0)
        # 중복 제거: 순서를 유지하면서 중복 좌표(거의 동일한 좌표)를 제거합니다.
        unique_list = []
        for pt in concatenated:
            if len(unique_list) == 0 or not np.allclose(pt, unique_list[-1], atol=1e-3):
                unique_list.append(pt)
        return np.array(unique_list, dtype=np.float32)

    def generate_gt_map(self, image: np.ndarray, xy_vec: np.ndarray):
        H, W = image.shape[:2]
        gt_map = np.zeros((H//self.scale, W//self.scale, 8))
        scaled_xy = (xy_vec / self.scale).astype(np.int32)
        # 0,1: point in this block
        gt_map[scaled_xy[:, 1], scaled_xy[:, 0], :2] = xy_vec
        # 2,3: point in the previous block
        gt_map[scaled_xy[1:, 1], scaled_xy[1:, 0], 2:4] = xy_vec[:-1]
        gt_map[scaled_xy[0, 1], scaled_xy[0, 0], 2:4] = xy_vec[0]
        # 4,5: point in the next block
        gt_map[scaled_xy[:-1, 1], scaled_xy[:-1, 0], 4:6] = xy_vec[1:]
        gt_map[scaled_xy[-1, 1], scaled_xy[-1, 0], 4:6] = xy_vec[-1]
        # 6: whether previous block has a starting point
        gt_map[scaled_xy[1, 1], scaled_xy[1, 0], 6] = 1
        # 7: whether next block has an ending point
        gt_map[scaled_xy[-2, 1], scaled_xy[-2, 0], 7] = 1
        return gt_map

    # ==============================
    # 내부 헬퍼 함수들 (정적 메서드)
    # ==============================
    @staticmethod
    def _project_points_onto_segments(points: np.ndarray, seg_A: np.ndarray, seg_B: np.ndarray,
                                      x_min: np.ndarray, x_max: np.ndarray,
                                      y_min: np.ndarray, y_max: np.ndarray):
        """
        여러 점(points, shape: (P,2))에서 선분(seg_A→seg_B, shape: (S,2))으로의 투영(수선의 발)을 계산.
        투영 파라미터 t가 [0,1]에 있지 않거나,
        투영된 점이 각 블록의 내부 [x_min, x_max) 및 [y_min, y_max) 영역에 있지 않으면 거리를 np.inf로 처리.
        
        반환:
            foot: (P, S, 2) – 각 점에 대해 각 선분상의 투영점
            dists: (P, S) – 각 투영점까지의 거리 (유효하지 않으면 np.inf)
        """
        AB = seg_B - seg_A  # (S,2)
        AB_norm_sq = np.sum(AB ** 2, axis=1)
        AB_norm_sq = np.where(AB_norm_sq == 0, 1e-8, AB_norm_sq)
        AP = points[:, None, :] - seg_A[None, :, :]  # (P,S,2)
        t = np.sum(AP * AB[None, :, :], axis=2) / AB_norm_sq[None, :]
        foot = seg_A[None, :, :] + t[..., None] * AB[None, :, :]
        valid_seg = (t >= 0) & (t <= 1)
        diff = foot - points[:, None, :]  # (P,S,2)
        dists = np.linalg.norm(diff, axis=2)  # (P,S)
        dists[~valid_seg] = np.inf

        # 블록 내부 조건 검사: 각 점의 투영이 블록 내 [x_min, x_max) 및 [y_min, y_max) 에 있어야 함
        valid_in_block = (foot[:, :, 0] >= x_min[:, None]) & (foot[:, :, 0] < x_max[:, None]) & \
                         (foot[:, :, 1] >= y_min[:, None]) & (foot[:, :, 1] < y_max[:, None])
        dists[~valid_in_block] = np.inf
        return foot, dists

    @staticmethod
    def _get_dist_point_to_nodes(centers: np.ndarray, block_cols: np.ndarray, block_rows: np.ndarray,
                                 line_string: np.ndarray, scale_factor: int):
        """
        각 블록 중심(centers, shape: (P,2))과 line_string의 각 노드 간의 거리를 계산.
        단, 각 노드가 해당 블록 내부 (원본에서 scale_factor×scale_factor 영역)에 있지 않으면 np.inf로 처리.
        
        반환:
            best_nodes: (P, 2) 각 블록마다 내부에 있는 노드 중 중심과 가장 가까운 노드 좌표
            dists: (P,) 각 블록 중심과 해당 노드 사이의 거리
        """
        P = centers.shape[0]
        # 각 블록의 경계 (원본 좌표)
        x_min = block_cols * scale_factor
        x_max = (block_cols + 1) * scale_factor
        y_min = block_rows * scale_factor
        y_max = (block_rows + 1) * scale_factor

        diff = centers[:, None, :] - line_string[None, :, :]  # (P, N, 2)
        dists = np.linalg.norm(diff, axis=2)  # (P, N)
        nodes_x = line_string[:, 0][None, :]  # (1, N)
        nodes_y = line_string[:, 1][None, :]  # (1, N)
        valid = (nodes_x >= x_min[:, None]) & (nodes_x < x_max[:, None]) & \
                (nodes_y >= y_min[:, None]) & (nodes_y < y_max[:, None])
        dists[~valid] = np.inf
        best_idx = np.argmin(dists, axis=1)  # (P,)
        best_dists = dists[np.arange(P), best_idx]
        best_nodes = line_string[best_idx].astype(np.float32)
        return best_nodes, best_dists
