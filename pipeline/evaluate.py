import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import re
import os
import json
import torch
from pytorch_lightning import seed_everything
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import settings
from configs.config import CfgNode
from model.lightning_detr import LitDeformableDETR
from pipeline.dataloader import create_dataloader
from util.misc import get_sizes_and_ids


def evaluate(version: int = -1):
    cfg = CfgNode.from_file('defm_detr_base')
    seed_everything(cfg.runtime.seed, workers=True)
    
    log_dir = get_log_dir(cfg, version)
    ckpt_path = get_ckpt_path(log_dir)
    print(f"Loading checkpoint from {ckpt_path}")
    model = LitDeformableDETR.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()
    model.freeze()

    split = "val"
    val_loader = create_dataloader(cfg, split=split)
    annotation_path = os.path.join(cfg.dataset.path, split, f"instances_{split}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = []

    for batch_idx, batch in enumerate(val_loader):
        samples, targets = batch
        samples = samples.to(device)
        with torch.no_grad():
            outputs = model(samples)

        # COCO 평가를 위한 형식 변환
        target_sizes, image_ids = get_sizes_and_ids(targets, outputs["pred_logits"].device)
        coco_dets = model.postprocessors["bbox"](outputs, target_sizes, image_ids)
        predictions.extend(coco_dets)

    pred_json = os.path.join(log_dir, "val_predictions.json")
    with open(pred_json, 'w') as f:
        json.dump(predictions, f)

    coco_gt = COCO(annotation_path)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = coco_eval.stats[0]
    map_50    = coco_eval.stats[1]
    print(f"[Evaluate] COCO AP: {map_50_95:.4f}, AP50: {map_50:.4f}")


def get_log_dir(cfg, version: int):
    if version >= 0:
        log_dir = os.path.join(cfg.runtime.output_dir, cfg.runtime.logger_name, f"version_{version}")
        assert os.path.isdir(log_dir), f"Log directory not found: {log_dir}"
        return log_dir

    versions = []
    log_root = os.path.join(cfg.runtime.output_dir, cfg.runtime.logger_name)
    for name in os.listdir(log_root):
        full_path = os.path.join(log_root, name)
        if os.path.isdir(full_path) and name.startswith("version_"):
            try:
                ver_num = int(name.split("_")[1])
                versions.append(ver_num)
            except ValueError:
                pass

    if not versions:
        raise ValueError(f"No version directories found in the output directory: {log_root}")

    max_ver = max(versions)
    return os.path.join(log_root, f"version_{max_ver}")


def get_ckpt_path(log_dir):
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    # 정규표현식: epoch=숫자-step=숫자.ckpt
    pattern = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt")
    max_step = -1
    best_ckpt = None

    for fname in os.listdir(ckpt_dir):
        if fname.endswith(".ckpt"):
            match = pattern.match(fname)
            if match:
                # group(1)=epoch, group(2)=step
                step_val = int(match.group(2))
                if step_val > max_step:
                    max_step = step_val
                    best_ckpt = fname

    if best_ckpt is None:
        raise ValueError(f"No checkpoint files found in the checkpoint directory: {ckpt_dir}")
    return os.path.join(ckpt_dir, best_ckpt)


if __name__ == "__main__":
    evaluate()
