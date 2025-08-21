# examples/visualize_json_vs_npy.py
import os
import os.path as op
import sys
import json
import cv2
import numpy as np

# --- 프로젝트 루트 import 보정 ---
ROOT = op.dirname(op.dirname(op.abspath(__file__)))  # .../SatelliteDet2025
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import CfgNode


def _get_palette_bgr_and_map(cfg):
    """cfg.dataset.labels[*].name / .color(RGB) → BGR 팔레트 & 이름→BGR 맵"""
    palette_bgr, name2bgr = [], {}
    for item in cfg.dataset.labels:
        name = item.get('name', 'unknown')
        rgb = tuple(item.get('color', (0, 0, 0)))
        if len(rgb) != 3:
            rgb = (0, 0, 0)
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        palette_bgr.append(bgr)
        name2bgr[name] = bgr
    if not palette_bgr:
        palette_bgr = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    return palette_bgr, name2bgr


def _first_existing_path(paths):
    for p in paths:
        if p and op.exists(p):
            return p
    return None


def draw_json_as_points(image_bgr, json_path, name2bgr, valid_names,
                        point_radius=3, point_thickness=-1, step=1, strong=True):
    """
    JSON의 image_points를 '선'이 아니라 '점'으로 찍어 그립니다.
      - step>1 이면 간격 샘플링(드문드문 찍기)
      - strong=True 면 알파블렌딩 없이 바로 덮어그림(진하게)
    """
    vis = image_bgr.copy()
    if not (json_path and op.exists(json_path)):
        return vis

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            objs = json.load(f)
    except Exception as e:
        print(f"[json] skip {json_path}: {e}")
        return vis

    if not isinstance(objs, list):
        print(f"[json] invalid format (not list): {json_path}")
        return vis

    canvas = vis if strong else vis.copy()
    for obj in objs:
        if obj.get('class') != 'RoadObject':
            continue
        cat = obj.get('category', None)
        color = name2bgr.get(cat, (200, 200, 200)) if (cat in valid_names) else (200, 200, 200)

        pts = np.array(obj.get('image_points', []), dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 1:
            continue

        if step > 1:
            pts = pts[::step]
        cv2.polylines(canvas, [pts], False, color, 1)

        x_s, y_s = map(int, pts[0])
        x_e, y_e = map(int, pts[-1])

        cv2.rectangle(
            canvas,
            (x_s - point_radius - 1, y_s - point_radius - 1),
            (x_s + point_radius + 1, y_s + point_radius + 1),
            (0, 255, 255),
            2
        )

        cv2.drawMarker(
            canvas, (x_e, y_e),
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=point_radius * 3,
            thickness=2
        )

    return canvas if strong else cv2.addWeighted(canvas, 0.7, vis, 0.3, 0)


def draw_npy_as_points(image_bgr, npy_path, palette_bgr,
                       point_radius=3, point_thickness=-1, strong=True):
    """
    NPY 라벨을 '더 진하고 굵게' 점으로 찍어 그립니다.
    채널 해석(존재하는 것만 사용):
      0..1: center(norm x,y), 2..3: prev, 4..5: next, 6: start, 7: end, 8: category_id
    """
    vis = image_bgr.copy()
    H, W = vis.shape[:2]
    if not (npy_path and op.exists(npy_path)):
        return vis

    try:
        lm = np.load(npy_path)
    except Exception as e:
        print(f"[npy] skip {npy_path}: {e}")
        return vis

    if lm.ndim != 3 or lm.shape[-1] < 2:
        print(f"[npy] invalid shape: {npy_path}, shape={lm.shape}")
        return vis

    C = lm.shape[-1]
    ctr = lm[..., 0:2]
    cat = lm[..., 8].astype(np.int32) if C >= 9 else np.zeros(lm.shape[:2], dtype=np.int32)

    # 유효 포인트(0,0 제외)
    valid = (ctr[..., 0] > 0) & (ctr[..., 1] > 0)
    is_prev_start = lm[..., 6] > 0.5
    is_next_end   = lm[..., 7] > 0.5
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return vis

    px = np.clip((ctr[..., 0] * W).round().astype(np.int32), 0, W - 1)
    py = np.clip((ctr[..., 1] * H).round().astype(np.int32), 0, H - 1)

    canvas = vis if strong else vis.copy()
    for y, x in zip(ys, xs):
        cidx = cat[y, x]
        color = palette_bgr[cidx] if 0 <= cidx < len(palette_bgr) else (180, 180, 180)
        cv2.circle(canvas, (px[y, x], py[y, x]), point_radius, color, point_thickness)

    ys_s, xs_s = np.where(valid & is_prev_start)
    for y, x in zip(ys_s, xs_s):
        cv2.rectangle(canvas,
                        (px[y, x] - point_radius - 1, py[y, x] - point_radius - 1),
                        (px[y, x] + point_radius + 1, py[y, x] + point_radius + 1),
                        (0, 255, 255), 1)

    ys_e, xs_e = np.where(valid & is_next_end)
    for y, x in zip(ys_e, xs_e):
        cv2.drawMarker(canvas, (px[y, x], py[y, x]),
                        (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=point_radius*3, thickness=1)

    return canvas if strong else cv2.addWeighted(canvas, 0.8, vis, 0.2, 0)


def visualize_compare(cfg_name='satellite_detr', split='train', limit=None,
                      raw_label_dir=None, out_dirname='_vis_json_vs_npy_points',
                      json_point_radius=3, json_step=1,
                      npy_point_radius=3,
                      strong_color=True):
    """
    좌: JSON 점 오버레이 / 우: NPY 점 오버레이 → 가로 결합 저장
      - strong_color=True  → 알파블렌딩 없이 바로 그려 "진하게"
      - 반경(point_radius)으로 '굵기' 조절
    """
    cfg = CfgNode.from_file(cfg_name)
    dataset_path = cfg.dataset.path
    img_dir = op.join(dataset_path, split, 'image')
    npy_dir = op.join(dataset_path, split, 'label')
    out_dir = op.join(dataset_path, split, out_dirname)
    os.makedirs(out_dir, exist_ok=True)

    # JSON 라벨 디렉터리 추정
    default1 = op.join(op.dirname(dataset_path), 'satellite_images', 'label')
    default2 = op.join(op.dirname(dataset_path), 'satellite_images_', 'label')
    json_dir = _first_existing_path([raw_label_dir, default1, default2])
    if not json_dir:
        print(f"[warn] cannot locate raw JSON label dir (tried: {raw_label_dir}, {default1}, {default2})")

    palette_bgr, name2bgr = _get_palette_bgr_and_map(cfg)
    valid_names = {item.get('name', '') for item in cfg.dataset.labels}

    img_names = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith('.png')]
    if limit is not None:
        img_names = img_names[:limit]

    for name in img_names:
        stem = op.splitext(name)[0]
        img_path = op.join(img_dir, name)
        npy_path = op.join(npy_dir, stem + '.npy')
        json_path = op.join(json_dir, stem + '.json') if json_dir else None

        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] cannot read image: {img_path}")
            continue

        left = draw_json_as_points(
            img, json_path, name2bgr, valid_names,
            point_radius=json_point_radius, point_thickness=-1,
            step=json_step, strong=strong_color
        ) if json_path and op.exists(json_path) else img.copy()

        right = draw_npy_as_points(
            img, npy_path, palette_bgr,
            point_radius=npy_point_radius, point_thickness=-1,
            strong=strong_color
        ) if op.exists(npy_path) else img.copy()

        combo = cv2.hconcat([left, right])
        out_path = op.join(out_dir, stem + '_cmp_pts.png')
        cv2.imwrite(out_path, combo)
        print(f"[saved] {out_path}")


if __name__ == '__main__':
    # 예시 실행
    # python -m examples.visualize_json_vs_npy  (파일명 그대로면 모듈명 변경)
    visualize_compare(
        cfg_name='satellite_detr',
        split='train',
        limit=10,
        raw_label_dir=None,        # JSON 경로가 다르면 절대경로 지정
        json_point_radius=4,       # JSON 점 굵기
        json_step=1,               # JSON 포인트 간격(>1이면 듬성듬성)
        npy_point_radius=2,        # NPY 점 굵기
        strong_color=True          # True면 진하게(알파블렌딩 없음)
    )
