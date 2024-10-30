import os
import numpy as np
import matplotlib.pyplot as                                            plt


# 指定路径
#directory = '/media/cx206/work/teeth/gt/0390H3'
directory = '/media/cx206/work/teeth/2X1AJ0_jky'

# 遍历目录中的文件
for filename in os.listdir(directory):
    if filename.endswith('.npy'):  # 只处理 .npy 文件
        file_path = os.path.join(directory, filename)
        
        # 加载 NumPy 数组
        np_array = np.load(file_path)
        plt.imshow(np_array[:,:,0])
        plt.show()
        
        # 打印文件名和数组的 shape
        print(f'{filename}: {np_array.shape}')
