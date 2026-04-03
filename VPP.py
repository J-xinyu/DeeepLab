# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle

# ===================== 核心配置（改这里就行） =====================
# 选择要处理的数据集：1/2/3
DATASET_NUM = 2  
# 结果保存根目录（自动创建）
OUT_ROOT = f"/home/robot/Code/VOCdevkit/VOC2007{DATASET_NUM}/VPP_Results"
# 可视化参数（不用改）
ALPHA_OVERLAY = 0.65
CMAP_NAME     = "turbo"
NORMALIZE_BY  = "max"  # "max" 或 "p99"

# ===================== 自动计算路径（不用改） =====================
# 数据集根路径
DATASET_ROOT = f"/home/robot/Code/VOCdevkit/VOC2007{DATASET_NUM}"
# 图片和掩码路径（按VOC格式）
IMAGE_DIR = os.path.join(DATASET_ROOT, "JPEGImages")
MASK_DIR  = os.path.join(DATASET_ROOT, "SegmentationClass")

# 创建结果目录
os.makedirs(OUT_ROOT, exist_ok=True)

# ===================== 核心处理函数 =====================
def process_image(image_path, mask_path, out_path, out_overlay_path):
    """处理单张图片：生成掩码可视化+叠加图"""
    # 读取图片和掩码
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        print(f"跳过：{image_path} 或 {mask_path} 读取失败")
        return
    
    # 归一化掩码
    if NORMALIZE_BY == "max":
        norm = Normalize(vmin=0, vmax=np.max(mask))
    elif NORMALIZE_BY == "p99":
        norm = Normalize(vmin=0, vmax=np.percentile(mask, 99))
    else:
        norm = Normalize(vmin=0, vmax=255)
    
    # 获取颜色映射
    cmap = get_cmap(CMAP_NAME)
    mask_colored = cmap(norm(mask))[:, :, :3]  # 去掉alpha通道
    mask_colored = (mask_colored * 255).astype(np.uint8)
    
    # 生成叠加图
    overlay = img.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (ALPHA_OVERLAY * mask_colored[mask_bool] + 
                          (1 - ALPHA_OVERLAY) * img[mask_bool]).astype(np.uint8)
    
    # 保存结果
    cv2.imwrite(out_path, cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"已处理：{os.path.basename(image_path)} → 保存到 {OUT_ROOT}")

# ===================== 批量处理主逻辑 =====================
if __name__ == "__main__":
    # 获取所有图片文件名
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    
    if not image_files:
        print(f"错误：{IMAGE_DIR} 里没有图片！")
    else:
        print(f"开始处理数据集{DATASET_NUM}，共 {len(image_files)} 张图片...")
        
        for img_file in image_files:
            # 拼接路径
            img_name = os.path.splitext(img_file)[0]
            image_path = os.path.join(IMAGE_DIR, img_file)
            mask_path  = os.path.join(MASK_DIR, f"{img_name}.png")  # 掩码默认是png
            
            # 结果路径
            out_path        = os.path.join(OUT_ROOT, f"{img_name}_mask.png")
            out_overlay_path = os.path.join(OUT_ROOT, f"{img_name}_overlay.png")
            
            # 处理单张图片
            process_image(image_path, mask_path, out_path, out_overlay_path)
        
        print(f"\n处理完成！所有结果保存在：{OUT_ROOT}")
