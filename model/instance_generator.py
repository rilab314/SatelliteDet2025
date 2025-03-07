from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from model.dto import LaneDetOutput, LineString


# TODO : 구현은 했지만 검증이 필요함
class LineStringInstanceGenerator(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):
        return LineStringInstanceGenerator()

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, output: LaneDetOutput) -> List[LineString]:
        points, class_ids, scores = self._find_peaks(output.segm_logit)
        output.line_strings = self._find_lines(output, points, class_ids, scores)
        return output

    def _find_peaks(self, segm_logit: torch.Tensor, threshold: float = 0.5, kernel_size: int = 3):
        '''
        segm_logit: shape=(B, H, W, C)  (raw logits for each class)
        threshold : prob threshold for local peaks
        kernel_size : used for local maxima detection (3=>8-방향)
        returns:
            points: List[Tensor], length=B, shape=(N, 2)
            class_ids: List[Tensor], length=B, shape=(N,)
            scores: List[Tensor], length=B, shape=(N,)
        '''
        B, H, W, C = segm_logit.shape
        prob = F.softmax(segm_logit, dim=-1)  # (B, H, W, C)

        points_list = []
        class_ids_list = []
        scores_list = []

        for b in range(B):
            # prob[b]: (H,W,C), transpose to (C,H,W), unsqueeze to (1,C,H,W) for max_pool2d
            prob_b = prob[b].permute(2,0,1).unsqueeze(0)   # (1,C,H,W)
            
            # local maxima detection, 8-방향 이웃 비교
            local_max_b = F.max_pool2d(prob_b, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            peak_mask_b = (prob_b == local_max_b)
            peak_mask_b = peak_mask_b.squeeze(0)  # (C,H,W)
            prob_b_squeezed = prob_b.squeeze(0)  # (C,H,W)
            
            # threshold -> (N_found, 3) each row => [c, h, w]
            peak_coords = (peak_mask_b & (prob_b_squeezed > threshold)).nonzero(as_tuple=False)

            # (y_idx, x_idx, c_idx, score_val) 형태 후보를 모음
            candidates = []
            for c_y_x in peak_coords:
                c_idx, y_idx, x_idx = c_y_x.tolist()
                score_val = prob_b_squeezed[c_idx, y_idx, x_idx].item()
                candidates.append((y_idx, x_idx, c_idx, score_val))

            # 중복 제거: 동일 (y,x)에 대해서 가장 점수 높은 클래스만
            candidates.sort(key=lambda x: x[3], reverse=True)
            used_y_x = set()
            peaks = []
            for (y_idx, x_idx, c_idx, score_val) in candidates:
                if (y_idx, x_idx) not in used_y_x:
                    used_y_x.add((y_idx, x_idx))
                    peaks.append((y_idx, x_idx, c_idx, score_val))

            # peaks -> (N, 2) + (N,) + (N,)
            if len(peaks) == 0:
                # No peaks => empty Tensors
                points_list.append(torch.empty((0,2), dtype=torch.long))
                class_ids_list.append(torch.empty((0,), dtype=torch.long))
                scores_list.append(torch.empty((0,), dtype=torch.float32))
                continue

            # peaks: list of (y_idx, x_idx, c_idx, score_val)
            points = []
            class_ids = []
            scores = []
            for (y_idx, x_idx, c_idx, score_val) in peaks:
                points.append([y_idx, x_idx])
                class_ids.append(c_idx)
                scores.append(score_val)

            points = torch.tensor(points, dtype=torch.long)       # (N,2)
            class_ids = torch.tensor(class_ids, dtype=torch.long) # (N,)
            scores = torch.tensor(scores, dtype=torch.float32)    # (N,)

            points_list.append(points)
            class_ids_list.append(class_ids)
            scores_list.append(scores)
        return points_list, class_ids_list, scores_list

    def _find_lines(self, output: LaneDetOutput, points: List[torch.Tensor], class_ids: List[torch.Tensor], scores: List[torch.Tensor]) -> List[List[LineString]]:
        '''
        output: LaneDetOutput
        points: List[Tensor], length=B, shape=(N, 2)
        class_ids: List[Tensor], length=B, shape=(N,)
        scores: List[Tensor], length=B, shape=(N,)
        '''
        B = len(points)
        line_strings = []
        for b in range(B):
            points_b = points[b]
            class_ids_b = class_ids[b]
            scores_b = scores[b]
            segm_logit_b = output.segm_logit[b]
            side_endness_b = [F.sigmoid(output.side_logits[0][b]), F.sigmoid(output.side_logits[1][b])]
            center_point_b = output.center_point[b]
            side_points_b = [output.side_points[0][b], output.side_points[1][b]]
            lines_b = []
            for i in range(points_b.shape[0]):
                grid_points = self._find_lines_in_grid(points_b[i], class_ids_b[i], segm_logit_b, side_endness_b, center_point_b, side_points_b)
                line = LineString(class_id=class_ids_b[i].item(), points=grid_points, scores=scores_b[i])  # TODO scores
                lines_b.append(line)
            line_strings.append(lines_b)
        return line_strings

    def _find_lines_in_grid(self, point: torch.Tensor, class_id: torch.Tensor, segm_logit: torch.Tensor, side_endness: List[torch.Tensor], 
                            center_point: torch.Tensor, side_points: List[torch.Tensor]) -> torch.Tensor:
        '''
        % all coordinates are in grid space
        point: (2,), grid coordinates of peaks (y, x), int
        class_id: (1,), class id of line
        segm_logit: (H, W, K), logits for each class
        side_endness: [(H, W, 1), (H, W, 1)], logits for each side
        center_point: (H, W, 2), point on line in this grid, 0~1
        side_points: [(H, W, 2), (H, W, 2)], points on line in side grids, 0~1
        '''
        class_mask = (torch.argmax(segm_logit, dim=-1) == class_id).squeeze(0)  # (H, W)
        start_point = point + center_point[point[0], point[1]]  # point means left-top corner of grid
        left_point = self._init_side_point(point, class_mask, side_points[0], side_endness[0])
        right_point = self._init_side_point(point, class_mask, side_points[1], side_endness[1])
        left_points = [start_point, left_point]
        right_points = [start_point, right_point]
        while True:
            left_points = self._extend_line(left_points, class_mask, side_points, side_endness)
            right_points = self._extend_line(right_points, class_mask, side_points, side_endness)
            if left_points[-1] is None and right_points[-1] is None:
                break
        grid_points = left_points[::-1] + right_points[1:]  # left points, center point, right points
        grid_points = torch.stack(grid_points, dim=0)
        return grid_points

    def _init_side_point(self, grid_point: torch.Tensor, class_mask: torch.Tensor, side_point_map: torch.Tensor, side_endness_map: torch.Tensor) -> torch.Tensor:
        side_point = grid_point + 0.5 + side_point_map[grid_point[0], grid_point[1]]  # +-3 grids from the center of 'grid_point' cell
        side_point_int = side_point.to(torch.long)
        same_class = class_mask[side_point_int[0], side_point_int[1]]
        if not same_class:
            return None
        side_endness = side_endness_map[side_point_int[0], side_point_int[1]]
        if side_endness > 0.5:
            return None
        return side_point

    def _extend_line(self, grid_points: List[torch.Tensor], class_mask: torch.Tensor, side_point_maps: List[torch.Tensor], side_endness_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        last_point = grid_points[-1]
        if last_point is None:
            return grid_points
        direction = grid_points[-1] - grid_points[-2]

        left_point = self._init_side_point(last_point, class_mask, side_point_maps[0], side_endness_maps[0])
        right_point = self._init_side_point(last_point, class_mask, side_point_maps[1], side_endness_maps[1])
        
        if left_point is None and right_point is None:
            grid_points.append(None)
            return grid_points

        left_dot = torch.dot(left_point, direction) if left_point is not None else -1
        right_dot = torch.dot(right_point, direction) if right_point is not None else -1
        if left_point is None:
            if right_dot > 0:
                grid_points.append(right_point)
            else:
                grid_points.append(None)
        elif right_point is None:
            if left_dot > 0:
                grid_points.append(left_point)
            else:
                grid_points.append(None)
        else:
            if left_dot > right_dot and left_dot > 0:
                grid_points.append(left_point)
            elif right_dot > left_dot and right_dot > 0:
                grid_points.append(right_point)
            else:
                grid_points.append(None)
        return grid_points
