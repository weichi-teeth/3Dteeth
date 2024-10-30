import os
import re
import cv2
import json
import numpy as np

output_dir1 = r'D:\summer23\segdata\image'
output_dir2 = r'D:\summer23\segdata\label'

num = 166

def json_to_contours(json_path, image_folder,name1):
    # 读取 JSON 文件
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # 遍历每张图像
    for image_filename, image_data in data.items():

        image_filename = image_filename[:12]
        image_filename1 = str(num)+image_filename
        image_path = f"{image_folder}\\{image_filename}"
        #print(image_data)
        print(image_path)
        #image = cv2.imread(image_path)
        image = cv2.imdecode(np.fromfile(file=image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # print(image)

        zc = image_filename1.split('.')
        output_path = f"{output_dir1}\\{zc[0]}.png"
        print(output_path)
        cv2.imwrite(output_path, image)

        contour_image = np.zeros_like(image)

        reg = image_data['regions']
        #print(reg)

        for da in reg:
            print(da)
            da = da['shape_attributes']
            #print(da)

            all_points_x = da.get('all_points_x', [])
            all_points_y = da.get('all_points_y', [])
            all_points_x = da['all_points_x']
            all_points_y = da['all_points_y']
            #print(all_points_x)
            # 使用多边形的所有点构建轮廓线
            contour = np.array(list(zip(all_points_x, all_points_y)))

            # 将轮廓线添加到轮廓图像中
            cv2.drawContours(contour_image, [contour], 0, (255, 255, 255), thickness=3)

        output_path = f"{output_dir2}\\{zc[0]}.png"
        cv2.imwrite(output_path, contour_image)
        print(f"保存成功：{output_path}")


folder_path = r'D:\summer23\口内标注1\口内标注1\5不标准'


files = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]

for file in files:
    print(file)



folder_path1 = r'D:\summer23\无附件'

rawp = [name for name in os.listdir(folder_path1) ]

for pic in rawp:
    print(pic)


name_regex = r'^([\u4e00-\u9fa5]+)'

json_name = []

for file in files:
    match = re.match(name_regex, file)
    if match:
        name = match.group(1)
        for pic in rawp:
            if name in pic:
                num+=1
                print(pic)
                new_dir = folder_path1 + "\\" + pic + "\\口内照"
                # print(os.listdir(new_dir))
                jdir = folder_path+"\\"+file
                print(jdir)
                idir = folder_path1+"\\"+pic+"\\"+"口内照"
                json_to_contours(jdir,idir,name)
                # print(new_dir)
                break

        # json_name.append(name)

    else:
        print("未找到姓名部分")

print(num)



