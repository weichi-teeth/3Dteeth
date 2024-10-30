import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


input_root = 'gt_examine'                
output_root = 'wxc_contour_examine_1023'  

if not os.path.exists(output_root):
    os.makedirs(output_root)


def multi_img():
    # 遍历gt目录下的所有子文件夹
    for subdir, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.npy'):
                input_file_path = os.path.join(subdir, file)
                # 构建对应的输出目录结构
                relative_path = os.path.relpath(subdir, input_root)
                output_dir = os.path.join(output_root, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                data = np.load(input_file_path)

                # 提取第0个通道
                channel_0 = data[:, :, 0]

                # 确保数据是二维的
                if len(channel_0.shape) != 2:
                    print(f"Warning: Expected 2D data in {input_file_path}, got shape {channel_0.shape}")
                    continue

                print(f"Channel 0 min value: {channel_0.min()}, max value: {channel_0.max()}")

                # 使用动态阈值
                threshold = (channel_0.max() + channel_0.min()) / 2
                binary_image = (channel_0 > threshold).astype(np.uint8) * 255

                # binary_image = (channel_0 > 0).astype(np.uint8) * 255

                # plt.imshow(binary_image, cmap='binary')
                # plt.title('Before Cropping')
                # plt.show()

                H, W = binary_image.shape

                # 目标比例为4:3
                target_ratio = 4 / 3
                current_ratio = W / H

                if current_ratio > target_ratio:
                    # 图像过宽，需要裁剪宽度
                    new_W = int(H * target_ratio)
                    start_W = (W - new_W) // 2
                    cropped_image = binary_image[:, start_W:start_W + new_W]
                else:
                    # 图像过高，需要裁剪高度
                    new_H = int(W / target_ratio)
                    start_H = (H - new_H) // 2
                    cropped_image = binary_image[start_H:start_H + new_H, :]

                # 确保裁剪后的图像仍然是二值的
                cropped_image = (cropped_image > 0).astype(np.uint8) * 255

                # plt.imshow(cropped_image, cmap='binary')
                # plt.title('After Cropping')
                # plt.show()

                # 保存为PNG格式的二值图
                output_image = Image.fromarray(cropped_image)
                output_image = output_image.convert('1')  # 转换为灰度图
                output_file_path = os.path.join(output_dir, file.replace('.npy', '.png'))
                output_image.save(output_file_path)


def single_img():
    input_file_path = "0001_EFPPKK_口内左侧位像_segmentation.npy"
    data = np.load(input_file_path)

    # 提取第0个通道
    channel_0 = data[:, :, 0]

    # 确保数据是二维的
    if len(channel_0.shape) != 2:
        print(f"Warning: Expected 2D data in {input_file_path}, got shape {channel_0.shape}")

    print(f"Channel 0 min value: {channel_0.min()}, max value: {channel_0.max()}")

    # 使用动态阈值
    threshold = (channel_0.max() + channel_0.min()) / 2
    binary_image = (channel_0 > threshold).astype(np.uint8) * 255

    # binary_image = (channel_0 > 0).astype(np.uint8) * 255

    plt.imshow(binary_image, cmap='gray')
    plt.title('Before Cropping')
    plt.show()

    H, W = binary_image.shape

    # 目标比例为4:3
    target_ratio = 4 / 3
    current_ratio = W / H

    if current_ratio > target_ratio:
        # 图像过宽，需要裁剪宽度
        new_W = int(H * target_ratio)
        start_W = (W - new_W) // 2
        cropped_image = binary_image[:, start_W:start_W + new_W]
    else:
        # 图像过高，需要裁剪高度
        new_H = int(W / target_ratio)
        start_H = (H - new_H) // 2
        cropped_image = binary_image[start_H:start_H + new_H, :]

    # 确保裁剪后的图像仍然是二值的
    cropped_image = (cropped_image > 0).astype(np.uint8) * 255

    plt.imshow(cropped_image, cmap='gray')
    plt.title('After Cropping')
    plt.show()

    # 保存为PNG格式的二值图
    output_image = Image.fromarray(cropped_image)
    output_image = output_image.convert('1')  # 转换为灰度图
    output_file_path = os.path.join(output_root, input_file_path.replace('.npy', '.png'))
    output_image.save(output_file_path)


if __name__ == "__main__":
    multi_img()
    # single_img()