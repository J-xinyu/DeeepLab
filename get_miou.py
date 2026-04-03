import os
import time
import csv

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results


def to_percent(x):
    """将 0.8734 转成 '87.34' 的字符串"""
    return f"{x * 100:.2f}"


if __name__ == "__main__":
    miou_mode       = 0
    num_classes     = 2
    name_classes    = ["_background_", "pothole"]
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC20071/ImageSets/Segmentation/val.txt"),
                           'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "VOC20071/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    fps         = 0.0
    avg_time    = 0.0
    total_time  = 0.0

    #-------------------------------------------#
    #   Step 1：预测阶段 + FPS 测量
    #-------------------------------------------#
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        start_time = time.time()

        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC20071/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))

        end_time   = time.time()
        total_time = end_time - start_time
        num_images = len(image_ids)
        fps        = num_images / total_time if total_time > 0 else 0.0
        avg_time   = total_time / num_images if num_images > 0 else 0.0

        print("----------- Speed -----------")
        print("Images      :", num_images)
        print("Total time  : {:.4f} s".format(total_time))
        print("Avg / image : {:.4f} s".format(avg_time))
        print("FPS         : {:.2f}".format(fps))
        print("-----------------------------")

    #-------------------------------------------#
    #   Step 2：计算 mIoU + 各类指标
    #-------------------------------------------#
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        (
            hist,
            IoUs, PA_Recall, Precision, F_scores,
            mPrecision, mRecall, mFscore, mIoU, acc
        ) = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)

        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        #-------------------------------------------#
        #   Step 3：写入 metrics.csv（全部百分制字符串）
        #-------------------------------------------#
        metrics_path = os.path.join(miou_out_path, "metrics.csv")

        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([
                "Type", "Class",
                "Precision(%)", "Recall(%)", "F_score(%)", "IoU(%)",
                "mPrecision(%)", "mRecall(%)", "mF_score(%)", "mIoU(%)",
                "FPS", "Avg_time(s)", "Total_time(s)", "Images", "Global_Accuracy(%)"
            ])

            # —— 每类一行 —— #
            for i in range(num_classes):
                writer.writerow([
                    "class",
                    name_classes[i],
                    to_percent(Precision[i]),
                    to_percent(PA_Recall[i]),
                    to_percent(F_scores[i]),
                    to_percent(IoUs[i]),
                    "", "", "", "",
                    "", "", "", "",
                    ""
                ])

            # —— 整体一行 —— #
            writer.writerow([
                "overall",
                "all",
                to_percent(mPrecision),
                to_percent(mRecall),
                to_percent(mFscore),
                to_percent(mIoU),
                to_percent(mPrecision),
                to_percent(mRecall),
                to_percent(mFscore),
                to_percent(mIoU),
                fps,
                avg_time,
                total_time,
                len(image_ids),
                to_percent(acc)
            ])

        print("Save metrics to:", metrics_path)
