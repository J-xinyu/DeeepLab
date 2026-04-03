# -*- coding: utf-8 -*-
"""
批处理整夹的坑洞几何量计算（专业可视化：背景变暗 + 绿色掩膜）
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# =========================
# 配置路径
# =========================
INPUT_IMAGE_DIR = r"/home/robot/Code/VOCdevkit/VOC20073/JPEGImages"
INPUT_MASK_DIR  = r"/home/robot/Code/VOCdevkit/VOC20073/SegmentationClass"
OUTPUT_DIR       = r"/home/robot/Code/result_voc3"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
MASK_EXTS  = [".png"]
MIN_AREA_PX = 50.0
PIXEL_SIZE_M = None
DRAW_MINAREA_RECT = False
# =========================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_mask(mask_path):
    """读取掩膜并返回二值图：0/255"""
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, binm = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    return binm.astype(np.uint8)


def contour_metrics(cnt, pixel_size_m=None):
    cnt2d = cnt.reshape(-1, 2)

    area_px = float(cv2.contourArea(cnt))
    perimeter_px = float(cv2.arcLength(cnt, True))

    x, y, w, h = cv2.boundingRect(cnt)
    aabb_w_px = float(w)
    aabb_h_px = float(h)

    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (rw, rh), angle = rect
    width_minrect_px  = float(min(rw, rh))
    length_minrect_px = float(max(rw, rh))

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx = float(np.mean(cnt2d[:, 0]))
        cy = float(np.mean(cnt2d[:, 1]))

    if pixel_size_m is not None:
        area_m2        = area_px * (pixel_size_m ** 2)
        perimeter_m    = perimeter_px * pixel_size_m
        aabb_w_m       = aabb_w_px * pixel_size_m
        aabb_h_m       = aabb_h_px * pixel_size_m
        width_minrect_m  = width_minrect_px * pixel_size_m
        length_minrect_m = length_minrect_px * pixel_size_m
    else:
        area_m2 = perimeter_m = None
        aabb_w_m = aabb_h_m = None
        width_minrect_m = length_minrect_m = None

    return {
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "centroid_x_px": cx,
        "centroid_y_px": cy,
        "aabb_x": int(x),
        "aabb_y": int(y),
        "aabb_w_px": aabb_w_px,
        "aabb_h_px": aabb_h_px,
        "minrect_center_x_px": float(rcx),
        "minrect_center_y_px": float(rcy),
        "minrect_width_px": width_minrect_px,
        "minrect_length_px": length_minrect_px,
        "minrect_angle_deg": float(angle),
        "area_m2": area_m2,
        "perimeter_m": perimeter_m,
        "aabb_w_m": aabb_w_m,
        "aabb_h_m": aabb_h_m,
        "minrect_width_m": width_minrect_m,
        "minrect_length_m": length_minrect_m,
    }


def draw_results(image_bgr, contours, results, draw_minarea_rect=False):
    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, (cnt, res) in enumerate(zip(contours, results), start=1):
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

        x, y, w, h = res["aabb_x"], res["aabb_y"], int(res["aabb_w_px"]), int(res["aabb_h_px"])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if draw_minarea_rect:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(vis, [box], True, (255, 255, 0), 1)

        cxy = (int(res["centroid_x_px"]), int(res["centroid_y_px"]))
        cv2.circle(vis, cxy, 3, (0, 0, 255), -1)
        cv2.putText(vis, f"ID {idx}", (cxy[0] + 5, cxy[1] - 5),
                    font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        rc = (x + w // 2, y + h // 2)
        cv2.putText(
            vis,
            f"{res['aabb_w_px']:.1f}px x {res['aabb_h_px']:.1f}px",
            (rc[0] + 5, rc[1] + 15),
            font, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )

    return vis


def list_files_with_exts(folder, exts):
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def build_pairs(image_dir, mask_dir, image_exts, mask_exts):
    def stem(p): return os.path.splitext(os.path.basename(p))[0]

    imgs = list_files_with_exts(image_dir, image_exts)
    msks = list_files_with_exts(mask_dir, mask_exts)

    img_map = {stem(p): p for p in imgs}
    msk_map = {stem(p): p for p in msks}

    pairs = []
    missing = []
    for name, img_path in img_map.items():
        if name in msk_map:
            pairs.append((img_path, msk_map[name], name))
        else:
            missing.append(name)

    if missing:
        print(f"[Warning] 原图缺少对应掩膜：{missing[:5]}{'...' if len(missing)>5 else ''}")
    if not pairs:
        print("[Error] 没有可配对的图片和掩膜")
    return pairs

def make_mask_overlay(img, mask):
    """
    背景变暗（0.6 倍亮度）+ 坑洞绿色叠加（0,255,0）
    """
    # 1) 背景整体变暗
    dark_bg = (img * 0.6).astype(np.uint8)

    # 2) 掩膜转二值
    mask_bin = (mask > 0).astype(np.uint8)

    # 3) 创建绿色掩膜
    color_mask = np.zeros_like(img)
    color_mask[mask_bin == 1] = (255, 180, 80)  # 绿色

    # 4) 半透明叠加（绿色 30%）
    overlay = cv2.addWeighted(dark_bg, 1.0, color_mask, 0.3, 0)
    return overlay


def process_one(image_path, mask_path, out_dir_image, out_dir_overlay,
                pixel_size_m=None, min_area_px=50.0,
                draw_minarea_rect=False):

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    mask = load_mask(mask_path)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]

    if len(contours) == 0:
        overlay = make_mask_overlay(img, mask)
        base = os.path.splitext(os.path.basename(image_path))[0]
        overlay_path = os.path.join(out_dir_overlay, f"{base}_overlay_none.jpg")
        cv2.imwrite(overlay_path, overlay)
        return pd.DataFrame(), None, overlay_path

    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    results = [contour_metrics(cnt, pixel_size_m=pixel_size_m) for cnt in contours]

    # 主结果图
    vis = draw_results(img, contours, results, draw_minarea_rect=draw_minarea_rect)

    # 专业版掩膜叠加图
    overlay = make_mask_overlay(img, mask)

    base = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(out_dir_image, f"{base}_overlay_results.jpg")
    msk_overlay_path = os.path.join(out_dir_overlay, f"{base}_overlay_mask.jpg")

    cv2.imwrite(vis_path, vis)
    cv2.imwrite(msk_overlay_path, overlay)

    df = pd.DataFrame(results)
    df.insert(0, "pothole_id", [i + 1 for i in range(len(results))])
    df.insert(0, "image_name", os.path.basename(image_path))

    return df, vis_path, msk_overlay_path


def main():
    ensure_dir(OUTPUT_DIR)
    out_vis_dir = ensure_dir(os.path.join(OUTPUT_DIR, "overlays"))
    out_mask_overlay_dir = ensure_dir(os.path.join(OUTPUT_DIR, "mask_overlays"))

    pairs = build_pairs(INPUT_IMAGE_DIR, INPUT_MASK_DIR, IMAGE_EXTS, MASK_EXTS)
    if not pairs:
        return

    all_rows = []
    for idx, (img_path, msk_path, name) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] Processing: {os.path.basename(img_path)}")
        try:
            df_one, vis_path, overlay_path = process_one(
                img_path, msk_path, out_vis_dir, out_mask_overlay_dir,
                pixel_size_m=PIXEL_SIZE_M, min_area_px=MIN_AREA_PX,
                draw_minarea_rect=DRAW_MINAREA_RECT
            )
            if not df_one.empty:
                all_rows.append(df_one)
        except Exception as e:
            print(f"[Error] {name}: {e}")

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        csv_path = os.path.join(OUTPUT_DIR, "pothole_metrics_all.csv")
        df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 汇总CSV已保存：{csv_path}")
    else:
        print("[Info] 无有效坑洞结果")

    print(f"[OK] 可视化已保存：{out_vis_dir}")
    print(f"[OK] 掩膜叠加已保存：{out_mask_overlay_dir}")


if __name__ == "__main__":
    main()
