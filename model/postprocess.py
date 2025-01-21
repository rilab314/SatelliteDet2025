import torch
from torch import nn
from util import box_ops


class BoxPostProcess(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):
        return BoxPostProcess(
            topk=cfg.postprocessors.bbox.topk,
            score_threshold=cfg.postprocessors.bbox.score_threshold
        )

    def __init__(self, topk=100, score_threshold=0.05):
        super().__init__()
        self.topk = topk
        self.score_threshold = score_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, image_ids):
        '''
        Args:
            outputs (dict):
              - pred_logits: (batch_size, num_queries, num_classes)
              - pred_boxes:  (batch_size, num_queries, 4) in [0,1] (cx,cy,w,h)
            target_sizes (Tensor): (batch_size, 2) -> (h, w)
            image_ids (Tensor): (batch_size,) -> image_id for each sample

        Returns:
            List[dict]: COCO-style detection results across the batch
                [
                  {
                    "image_id": int,
                    "category_id": int,
                    "bbox": [x, y, w, h],
                    "score": float
                  },
                  ...
                ]
        '''
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        assert len(image_ids) == out_logits.shape[0]

        # 1) Compute predicted probabilities
        #    shape: [batch_size, num_queries, num_classes]
        prob = out_logits.sigmoid()
        batch_size, num_queries, num_classes = prob.shape

        # 2) Flatten the scores per image to shape [num_queries * num_classes]
        #    so we can find which queries+classes are topk
        #    - Another approach: directly do topk on each row (as in original code).
        #      We'll do a threshold FIRST, then topk among the remaining, to strictly
        #      remove very low confidence boxes.
        results = []

        for b in range(batch_size):
            scores_b = prob[b]  # shape [num_queries, num_classes]
            boxes_b  = out_bbox[b]  # shape [num_queries, 4] (cx,cy,w,h)
            # Expand scores to shape [num_queries * num_classes]
            scores_b = scores_b.view(-1)  
            # Index for label: 0..(num_classes-1)
            labels_b = torch.arange(num_classes, device=boxes_b.device)
            labels_b = labels_b.unsqueeze(0).repeat(num_queries, 1).view(-1)  # shape [num_queries*num_classes]

            # Confidence threshold
            keep_mask = scores_b > self.score_threshold
            scores_b  = scores_b[keep_mask]
            labels_b  = labels_b[keep_mask]

            # keep_mask와 동일하게 boxes_b에서 query 인덱스만 골라야 함
            #   keep_mask는 (num_queries*num_classes,) -> query_idx = idx // num_classes
            keep_indices = keep_mask.nonzero(as_tuple=True)[0]
            # 실제 query index
            query_indices = keep_indices // num_classes

            # boxes_b(cxcywh)를 gather
            boxes_b = boxes_b[query_indices]  # shape [N, 4]

            # 만약 threshold가 굉장히 낮아 많은 박스가 남으면, topk를 적용
            if len(scores_b) > self.topk:
                topk_vals, topk_idx = torch.topk(scores_b, self.topk)
                scores_b = topk_vals
                labels_b = labels_b[topk_idx]
                boxes_b  = boxes_b[topk_idx]

            # boxes_b: (cx, cy, w, h) in [0,1]
            # convert to xyxy
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_b.unsqueeze(0))  # shape [1, N, 4]
            boxes_xyxy = boxes_xyxy[0]  # [N,4]

            # scale to absolute coords
            h, w = target_sizes[b]  # original height, width
            scale_fct = torch.tensor([w, h, w, h], device=boxes_xyxy.device)
            boxes_xyxy = boxes_xyxy * scale_fct

            # convert xyxy -> (x, y, w, h)
            x1y1x2y2 = boxes_xyxy
            xywh = torch.zeros_like(x1y1x2y2)
            xywh[:, 0] = x1y1x2y2[:, 0]
            xywh[:, 1] = x1y1x2y2[:, 1]
            xywh[:, 2] = x1y1x2y2[:, 2] - x1y1x2y2[:, 0]
            xywh[:, 3] = x1y1x2y2[:, 3] - x1y1x2y2[:, 1]

            image_id = int(image_ids[b].item())
            for idx_det in range(len(scores_b)):
                score = scores_b[idx_det].item()
                label = labels_b[idx_det].item()
                bbox = xywh[idx_det].tolist()  # [x, y, w, h]

                det = {
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [float(x) for x in bbox],
                    "score": float(score)
                }
                results.append(det)

        return results
