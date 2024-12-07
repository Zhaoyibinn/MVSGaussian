import cv2
from PIL import Image
import os
import numpy as np
import sys
# 设置你的文件夹路径

try:
    folder_path = sys.argv[1]

except:
    folder_path = '/home/zhaoyibin/3DRE/3DGS/MVSGaussian/dtu_data/dtu_colmap/scan24/images'
    print("请传参")
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # 确保文件是PNG格式
        file_path = os.path.join(folder_path, filename)

        # 打开图片
        with Image.open(file_path) as img:
            # 检查图片是否为四通道
            img_np = np.asarray(img)
            # if img_np.shape != (800, 800,3):
            #     print("1channel Image")
            #     img_np = cv2.resize(img_np,(800,800),interpolation=cv2.INTER_NEAREST)
            #     cv2.imwrite(os.path.join(folder_path, filename),img_np)
                # img = img.resize((800, 800),)
                # img.save(os.path.join(folder_path, filename))
            if img_np.shape == (800,800):
                repeated_arr = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
                cv2.imwrite(os.path.join(folder_path, filename), repeated_arr)

            if img.mode == 'RGBA':

                print("RGBA")
                # 将背景设置为黑色
                background = Image.new('RGB', img.size, 'black')
                # 将原始图片粘贴到背景上
                background.paste(img, (0, 0), img)
                # 保存转换后的图片
                background.save(os.path.join(folder_path, filename))
            else:
                print(f"{filename} is not a four-channel PNG image.")