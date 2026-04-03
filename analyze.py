import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(df_all):
    

    # 1. 面积分布直方图
    plt.figure(figsize=(8, 6))
    plt.hist(df_all['area_px'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Pothole Area Distribution (px²)')
    plt.xlabel('Area (px²)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./pothole_area_distribution.png', dpi=300)
    plt.close()

    # 2. 周长分布直方图
    plt.figure(figsize=(8, 6))
    plt.hist(df_all['perimeter_px'], bins=20, color='lightcoral', edgecolor='black')
    plt.title('Pothole Perimeter Distribution (px)')
    plt.xlabel('Perimeter (px)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./pothole_perimeter_distribution.png', dpi=300)
    plt.close()

    # 3. 长宽比散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(df_all['aabb_w_px'], df_all['aabb_h_px'], c='green', alpha=0.6)
    plt.title('Pothole Width vs Height (px)')
    plt.xlabel('Width (px)')
    plt.ylabel('Height (px)')
    plt.tight_layout()
    plt.savefig('./pothole_width_vs_height.png', dpi=300)
    plt.close()

    # 4. 最小外接矩形长宽比散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(df_all['minrect_width_px'], df_all['minrect_length_px'], c='orange', alpha=0.6)
    plt.title('Min Rectangle Width vs Length (px)')
    plt.xlabel('Width (px)')
    plt.ylabel('Length (px)')
    plt.tight_layout()
    plt.savefig('./minrect_width_vs_length.png', dpi=300)
    plt.close()

    # 5. 周长与面积的关系图（新增）
    plt.figure(figsize=(8, 6))
    plt.scatter(df_all['perimeter_px'], df_all['area_px'], c='purple', alpha=0.6)
    plt.title('Pothole Perimeter vs Area (px)')
    plt.xlabel('Perimeter (px)')
    plt.ylabel('Area (px²)')
    plt.tight_layout()
    plt.savefig('./perimeter_vs_area.png', dpi=300)
    plt.close()

    print("[OK] 所有图表已保存！")

def main():
    # 读取 CSV 文件
    df_all = pd.read_csv(r'./result_voc3/pothole_metrics_all.csv')

    # 绘制图表
    plot_metrics(df_all)

if __name__ == "__main__":
    main()
