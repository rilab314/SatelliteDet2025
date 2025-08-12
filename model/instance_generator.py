from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from model.dto import LaneDetOutput, LineString


class LineStringInstanceGenerator(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):
        return LineStringInstanceGenerator()

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, output: LaneDetOutput) -> List[List[LineString]]:
        points, class_ids, scores = self._select_by_argmax(output.segm_logit, self.threshold)
        line_strings = self._make_lines(output, points, class_ids, scores)
        output.line_strings = line_strings
        return line_strings

    def _select_by_argmax(self, segm_logit: torch.Tensor, threshold: float):
        """
        segm_logit: (B,H,W,C)
        returns:
          points_list:    List[Tensor], (N,2) int64  (y,x)
          class_ids_list: List[Tensor], (N,)  int64
          scores_list:    List[Tensor], (N,)  float32
        """
        B, H, W, C = segm_logit.shape
        prob = F.softmax(segm_logit, dim=-1)            # (B,H,W,C)
        max_scores, max_cls = prob.max(dim=-1)          # (B,H,W), (B,H,W)
        mask = max_scores > threshold                   # (B,H,W)

        points_list, class_ids_list, scores_list = [], [], []
        for b in range(B):
            ys, xs = torch.nonzero(mask[b], as_tuple=True)     # (N,), (N,)
            if ys.numel() == 0:
                points_list.append(torch.empty((0,2), dtype=torch.long))
                class_ids_list.append(torch.empty((0,), dtype=torch.long))
                scores_list.append(torch.empty((0,), dtype=torch.float32))
                continue

            pts = torch.stack([ys, xs], dim=1).long()          # (N,2)
            cls = max_cls[b, ys, xs].long()                    # (N,)
            scs = max_scores[b, ys, xs].float()                # (N,)

            points_list.append(pts)
            class_ids_list.append(cls)
            scores_list.append(scs)

        return points_list, class_ids_list, scores_list

    def _make_lines(
        self,
        output: LaneDetOutput,
        points: List[torch.Tensor],
        class_ids: List[torch.Tensor],
        scores: List[torch.Tensor],
    ) -> List[List[LineString]]:
        """
        output.center_point: (B,H,W,2)
        """
        B = len(points)
        line_strings_per_batch: List[List[LineString]] = []

        for b in range(B):
            pts_b = points[b]
            cls_b = class_ids[b]
            sco_b = scores[b]
            center_b = output.center_point[b]  # (H,W,2)

            lines_b: List[LineString] = []
            for i in range(pts_b.shape[0]):
                y, x = pts_b[i].tolist()
                center_offset = center_b[y, x]                 # (2,)
                start_point = torch.tensor([y, x], dtype=center_offset.dtype) + center_offset  # (2,)
                line_points = start_point.unsqueeze(0)         # (1,2)

                line = LineString(
                    class_id=int(cls_b[i].item()),
                    points=line_points,
                    scores=float(sco_b[i].item())
                )
                lines_b.append(line)

            line_strings_per_batch.append(lines_b)

        return line_strings_per_batch
