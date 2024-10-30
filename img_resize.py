from PIL import Image


def resize_and_crop_image(input_image_path, output_image_path, crop_width=800, crop_height=600):
    # 打开图片
    with Image.open(input_image_path) as img:
        # 获取图片的原始尺寸
        original_width, original_height = img.size

        bili = 0.25

        # 计算缩放后的尺寸
        resized_width = int(original_width * bili)
        resized_height = int(original_height * bili)

        # 缩放图片
        resized_img = img.resize((resized_width, resized_height))

        # 计算裁剪区域的左上角坐标
        left = (resized_width - crop_width) // 2
        top = (resized_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # 裁剪图片
        cropped_img = resized_img.crop((left, top, right, bottom))

        # if 'A' in cropped_img.mode:
        #     # 将图片转换为RGBA，然后填充透明区域为黑色
        #     cropped_img = cropped_img.convert('RGBA')
        #     data = cropped_img.getdata()
        #     newData = []
        #     for item in data:
        #         # 如果alpha为0（透明），则替换为黑色（255, 0, 0, 255）
        #         if item[3] == 0:
        #             newData.append((255, 0, 0, 255))  # 你可以改为(0, 0, 0, 255)来得到纯黑色
        #         else:
        #             newData.append(item)
        #
        #     cropped_img.putdata(newData)

            # 保存裁剪后的图片
        cropped_img.save(output_image_path)

    # 使用示例


input_image_path = '/media/cx206/work/teeth/seg/valid/Case402/Case432/╓╨╞┌╜╫╢╬1/5.png'  # 输入图片路径
output_image_path = '/media/cx206/work/teeth/seg/valid/Case402/Case432/╓╨╞┌╜╫╢╬1/55.png'  # 输出图片路径
resize_and_crop_image(input_image_path, output_image_path)