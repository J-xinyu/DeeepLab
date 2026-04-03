import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fast_hist(a, b, n):
    # a: GT 一维数组 (H*W,)
    # b: 预测 一维数组 (H*W,)
    k = (a >= 0) & (a < n)
    return np.bincount(
        n * a[k].astype(int) + b[k],
        minlength=n ** 2
    ).reshape(n, n)


def per_class_iu(hist):
    # IoU = TP / (TP + FP + FN)
    return np.diag(hist) / np.maximum(
        (hist.sum(1) + hist.sum(0) - np.diag(hist)),
        1
    )


def per_class_PA_Recall(hist):
    # Recall = TP / (TP + FN) = diag / row_sum
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    # Precision = TP / (TP + FP) = diag / col_sum
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    # 全局像素精度
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    """
    计算各类 IoU / Precision / Recall / F-score 以及整体 mIoU / mPrecision / mRecall / mFscore
    """
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()),
                    gt_imgs[ind], pred_imgs[ind]
                )
            )
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIoU-{:.2f}%; mPA-{:.2f}%; Accuracy-{:.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            ))

    IoUs = per_class_iu(hist)          # 每类 IoU
    PA_Recall = per_class_PA_Recall(hist)  # 每类 Recall(=PA)
    Precision = per_class_Precision(hist)  # 每类 Precision

    # ----------------- 每类 F-score ----------------- #
    F_scores = []
    print("\n========== Per-Class Metrics ==========")
    if name_classes is None:
        name_classes = [str(i) for i in range(num_classes)]

    for i in range(num_classes):
        precision_i = Precision[i]
        recall_i = PA_Recall[i]
        f_score_i = (2 * precision_i * recall_i) / (precision_i + recall_i + 1e-10)
        iou_i = IoUs[i]
        F_scores.append(f_score_i)

        print(f"Class: {name_classes[i]}")
        print(f"  Precision : {precision_i * 100:.2f}%")
        print(f"  Recall    : {recall_i * 100:.2f}%")
        print(f"  F-score   : {f_score_i * 100:.2f}%")
        print(f"  IoU       : {iou_i * 100:.2f}%\n")

    F_scores = np.array(F_scores)

    # ----------------- 整体指标 ----------------- #
    mPrecision = np.nanmean(Precision)
    mRecall = np.nanmean(PA_Recall)
    mFscore = np.nanmean(F_scores)
    mIoU = np.nanmean(IoUs)
    acc = per_Accuracy(hist)

    print("========== Overall Metrics ==========")
    print(f"mPrecision     : {mPrecision * 100:.2f}%")
    print(f"mRecall        : {mRecall * 100:.2f}%")
    print(f"mF_score       : {mFscore * 100:.2f}%")
    print(f"mIoU           : {mIoU * 100:.2f}%")
    print(f"Global Accuracy: {acc * 100:.2f}%\n")

    # 把整体指标也返回，方便主程序写 Excel
    return (
        np.array(hist, np.int32),
        IoUs,
        PA_Recall,
        Precision,
        F_scores,
        mPrecision,
        mRecall,
        mFscore,
        mIoU,
        acc
    )


# ----------------- 画图相关 ----------------- #
def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path,
                   tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " {0:.2f}%".format(val * 100)
        t = plt.text(val, i, str_val, color='royalblue',
                     va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes,
                 tick_font_size=12):
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    draw_plot_func(
        IoUs,
        name_classes,
        "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100),
        "Intersection over Union",
        os.path.join(miou_out_path, "mIoU.png"),
        tick_font_size=tick_font_size,
        plt_show=True
    )
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(
        PA_Recall,
        name_classes,
        "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100),
        "Recall",
        os.path.join(miou_out_path, "Recall.png"),
        tick_font_size=tick_font_size,
        plt_show=False
    )
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(
        Precision,
        name_classes,
        "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
        "Precision",
        os.path.join(miou_out_path, "Precision.png"),
        tick_font_size=tick_font_size,
        plt_show=False
    )
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    # 保存混淆矩阵
    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
