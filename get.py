import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.utils_metrics import compute_mIoU, show_results
from deeplab import DeeplabV3

# 设置评估模式
miou_mode = 0  # 0代表获取预测结果并计算miou，1代表仅获取预测结果，2代表仅计算miou
num_classes = 2
name_classes = ["_background_", "pothole"]  # 定义分类名称

# VOC数据集路径
VOCdevkit_path = 'VOCdevkit'
image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
miou_out_path = "miou_out"
pred_dir = os.path.join(miou_out_path, 'detection-results')

if miou_mode == 0 or miou_mode == 1:
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
        
    print("Load model.")
    deeplab = DeeplabV3()
    print("Load model done.")
    
    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image = Image.open(image_path)
        image = deeplab.get_miou_png(image)
        image.save(os.path.join(pred_dir, image_id + ".png"))
    print("Get predict result done.")

if miou_mode == 0 or miou_mode == 2:
    print("Get miou.")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
    print("Get miou done.")
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

