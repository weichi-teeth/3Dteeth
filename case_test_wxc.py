from case_test import *


# 分别处理中期阶段和新病例阶段的stl文件，并将它们的顶点数据进行对比
def load_and_process_mesh(mesh_dir, FDIs):
    """加载和处理STL文件，返回处理后的顶点数据"""
    mesh_vertices_by_FDI = []
    for fdi in FDIs:
        mshf = os.path.join(mesh_dir, f"crown{fdi}.stl")
        msh = o3d.io.read_triangle_mesh(mshf)
        msh_v = np.asarray(msh.vertices, np.float64)
        msh_v = farthestPointDownSample(msh_v, num_point_sampled=1500)
        # 进行坐标轴转换
        x = -msh_v[:, 0]
        y = -msh_v[:, 2]
        z = -msh_v[:, 1]
        msh_v[:, 0] = x
        msh_v[:, 1] = y
        msh_v[:, 2] = z
        mesh_vertices_by_FDI.append(msh_v)
    return np.array(mesh_vertices_by_FDI)


# 使用现有的evaluation函数框架，分别处理中期阶段和新病例阶段的数据
def compare_stages(h5File, X_Ref_Upper, X_Ref_Lower, FDIs_upper, FDIs_lower, mid_stage_mesh_dir, new_case_mesh_dir, log_dir):
    # 分别加载中期阶段和新病例阶段的数据
    X_Pred_Upper_MidStage = load_and_process_mesh(mid_stage_mesh_dir, FDIs_upper)
    X_Pred_Upper_NewCase = load_and_process_mesh(new_case_mesh_dir, FDIs_upper)

    # 计算并对比 RMSD, ASSD, HD等指标
    print("Comparing middle stage and new case stage...")
    
    # 对比中期阶段的结果
    evaluation(h5File, X_Ref_Upper, X_Ref_Lower, FDIs_upper, FDIs_lower, mid_stage_mesh_dir, log_dir)
    
    # 对比新病例阶段的结果
    evaluation(h5File, X_Ref_Upper, X_Ref_Lower, FDIs_upper, FDIs_lower, new_case_mesh_dir, log_dir)


# 三维可视化
def visualize_3d_data(mid_stage_data, new_case_data):
    fig = plt.figure(figsize=(10, 5))

    # 中期阶段
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('中期阶段')
    ax1.plot_surface(np.arange(mid_stage_data.shape[1]), np.arange(mid_stage_data.shape[0]), mid_stage_data[0], cmap='viridis')

    # 新病例阶段
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('新病例阶段')
    ax2.plot_surface(np.arange(new_case_data.shape[1]), np.arange(new_case_data.shape[0]), new_case_data[0], cmap='plasma')

    plt.show()


if __name__ == "__main__":
    # 假设已有中期阶段和新病例阶段的 mesh 目录路径
    mid_stage_mesh_dir = "seg/train/500RealCases1/Case2/中期阶段1/ExportSTLs/"
    new_case_mesh_dir = "seg/train/500RealCases1/Case2/新病例阶段/ExportSTLs/"
    
    # 定义其他必要的变量，如FDI, log目录等
    FDIs_upper = [11, 12, 13, 14, 15, 16, 21]  # 示例数据
    FDIs_lower = [22, 23, 24, 25, 26, 31, 32]
    # h5File = "weights-teeth-boundary-model.h5"
    X_Ref_Upper = []  # 假设加载了参考数据
    X_Ref_Lower = []
    log_dir = "log_directory_wxc.log"
    
    compare_stages(h5File, X_Ref_Upper, X_Ref_Lower, FDIs_upper, FDIs_lower, mid_stage_mesh_dir, new_case_mesh_dir, log_dir)