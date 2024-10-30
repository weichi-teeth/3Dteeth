import cv2
import json
import numpy as np

def json_to_contours(json_path, image_folder):
    # 读取 JSON 文件
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # 遍历每张图像
    for image_filename, image_data in data.items():

        image_filename = image_filename[:12]
        image_path = f"{image_folder}/{image_filename}"
        #print(image_data)

        image = cv2.imread(image_path)


        contour_image = np.zeros_like(image)

        reg = image_data['regions']
        print(reg)

        for da in reg:
            print(da)
            da = da['shape_attributes']
            print(da)

            all_points_x = da.get('all_points_x', [])
            all_points_y = da.get('all_points_y', [])
            all_points_x = da['all_points_x']
            all_points_y = da['all_points_y']
            print(all_points_x)
            # 使用多边形的所有点构建轮廓线
            contour = np.array(list(zip(all_points_x, all_points_y)))

            # 将轮廓线添加到轮廓图像中
            cv2.drawContours(contour_image, [contour], 0, (255, 255, 255), thickness=3)


        output_path = f"contour_{image_filename}"
        cv2.imwrite(output_path, contour_image)
        print(f"保存成功：{output_path}")

if __name__ == "__main__":

    json_path = "黄歆婷.json"
    image_folder = "an"


    json_to_contours(json_path, image_folder)
