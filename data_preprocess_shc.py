import numpy as np
import open3d as o3d
import copy
# import cycpd
import numpy as np
import open3d as o3d
import trimesh
from const import *

def farthestPointDownSample(vertices, num_point_sampled, return_flag=False):
    """Farthest Point Sampling (FPS) algorithm
    Input:
        vertices: numpy array, shape (N,3) or (N,2)
        num_point_sampled: int, the number of points after downsampling, should be no greater than N
        return_flag: bool, whether to return the mask of the selected points
    Output:
        selected_vertices: numpy array, shape (num_point_sampled,3) or (num_point_sampled,2)
        [Optional] flags: boolean numpy array, shape (N,3) or (N,2)"""
    N = len(vertices)
    n = num_point_sampled
    assert (
        n <= N
    ), "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0)  # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    farthest = np.argmax(_d)
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_)
    for i in range(n):
        flags[farthest] = True
        distances[farthest] = 0.0
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)
    if return_flag == True:
        return vertices[flags], flags
    else:
        return vertices[flags]


def process_stl_files(base_dir):
    for dir_name in os.listdir('new/'):
        new_case_dir = os.path.join(base_dir, dir_name, "新病例阶段", "ExportSTLs")
        if os.path.exists(new_case_dir):
            for file_name in os.listdir(new_case_dir):
                if file_name.endswith(".stl"):
                    try:
                        stl_path = os.path.join(new_case_dir, file_name)
                        # 读取 STL 文件
                        msh = o3d.io.read_triangle_mesh(stl_path)
                        X_src = np.asarray(msh.vertices, np.float64)
                        ds_X_src = farthestPointDownSample(X_src, num_point_sampled=1500, return_flag=False)

                        # 保存原始顶点数据
                        npy_path = os.path.join(new_case_dir, os.path.splitext(file_name)[0] + ".npy")
                        np.save(npy_path, X_src)

                        # 保存采样后的顶点数据
                        ds_npy_path = os.path.join(new_case_dir, os.path.splitext(file_name)[0] + "-ds.npy")
                        np.save(ds_npy_path, ds_X_src)
                        print(f"Processed {stl_path}, saved to {npy_path} and {ds_npy_path}")
                    except:
                        print('error!')
                        print(stl_path)
        new_case_dir = os.path.join(base_dir, dir_name, "中期阶段1", "ExportSTLs")
        if os.path.exists(new_case_dir):
            for file_name in os.listdir(new_case_dir):
                if file_name.endswith(".stl"):
                    try:
                        stl_path = os.path.join(new_case_dir, file_name)
                        # 读取 STL 文件
                        msh = o3d.io.read_triangle_mesh(stl_path)
                        X_src = np.asarray(msh.vertices, np.float64)
                        ds_X_src = farthestPointDownSample(X_src, num_point_sampled=1500, return_flag=False)

                        # 保存原始顶点数据
                        npy_path = os.path.join(new_case_dir, os.path.splitext(file_name)[0] + ".npy")
                        np.save(npy_path, X_src)

                        # 保存采样后的顶点数据
                        ds_npy_path = os.path.join(new_case_dir, os.path.splitext(file_name)[0] + "-ds.npy")
                        np.save(ds_npy_path, ds_X_src)
                        print(f"Processed {stl_path}, saved to {npy_path} and {ds_npy_path}")
                    except:
                        print('error!')
                        print(stl_path)


if __name__ == "__main__":
    base_dir = './seg/train/new_dataset/口腔/20240918STLs'
    process_stl_files(base_dir)
