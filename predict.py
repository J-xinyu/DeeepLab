import os
from PIL import Image

from deeplab import DeeplabV3


if __name__ == "__main__":
    # 初始化模型
    deeplab = DeeplabV3()

    name_classes = ["background", "pothole"]

    while True:
        img_path = input('Input image filename (or "exit" to quit): ')
        if img_path.lower() in ["exit", "quit"]:
            break

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("Open Error! Try again!", e)
            continue

        deep_red_img = deeplab.get_red_black_image(
            image,
            pothole_class_index=1,
            color=(113, 0, 0)   
        )

        deep_red_img.show()

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_name = base_name + "_pothole_deep_red.png"
        deep_red_img.save(save_name)

        print("✅ 已保存：", save_name)
