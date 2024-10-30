import functools
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"  # run on CPU
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import psutil
import ray

import pcd_mesh_utils as pm_util
import recons_eval_metric as metric
from const import *
from emopt5views import EMOpt5Views
from seg.seg_const import IMG_SHAPE
from seg.seg_model import ASPP_UNet
from seg.utils import predict_teeth_contour, gt_teeth_contour
import cv2
from stl import mesh
from pcd_mesh_utils import farthestPointDownSample
from natsort import natsorted
import time
import gc
import copy


def count_unique_vertices(stl_file):
    # 加载STL文件
    your_mesh = mesh.Mesh.from_file(stl_file)

    unique_vertices = set()
    for vector3 in your_mesh.vectors:
        # vector3是一个包含所有三角形顶点的列表，每个三角形由三个顶点组成
        for vertex in vector3:
            # 将顶点（作为元组）添加到集合中，集合会自动去重
            unique_vertices.add(tuple(vertex))

            # 返回不重复顶点的数量
    return len(unique_vertices)

def evaluation(h5File, X_Ref_Upper, X_Ref_Lower, FDIs_upper, FDIs_lower, mesh_dir, log_dir):
    """
    h5file: emopt result saved in h5 format
    X_Ref_Upper, X_Ref_Lower: List of numpy arrays
    """
    with h5py.File(h5File, "r") as f:
        grp = f["EMOPT"]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]

    # 使用调整后的预测结果/使用新病例阶段作为X_Pred_Upper进行evaluation
    # mesh_dir = "seg/valid/500RealCases1/Case424/中期阶段1-modified/"
    # mesh_dir = "seg/valid/500RealCases1/Case415/新病例阶段/ExportSTLs/"
    old_stdout = sys.stdout
    LogFile = os.path.join(log_dir, "compare.txt")
    log = open(LogFile, "a", encoding="utf-8")
    sys.stdout = log
    mesh_vertices_by_FDI = []
    for fdi in FDIs_upper:
        mshf = os.path.join(
            # mesh_dir, f"Pred_Lower_Mesh_toothid={fdi}.stl"
            mesh_dir, f"crown{fdi}.stl"
        )
        msh = o3d.io.read_triangle_mesh(mshf)
        mesh_nu = count_unique_vertices(mshf)
        msh_v = np.asarray(msh.vertices, np.float64)
        # if mesh_nu < 1500:
        #     msh_v = farthestPointDownSample(
        #         msh_v, num_point_sampled=mesh_nu
        #     )
        # else:
        #     msh_v = farthestPointDownSample(
        #         msh_v, num_point_sampled=1500
        #     )
        msh_v = farthestPointDownSample(
            msh_v, num_point_sampled=1500
        )
        x = -msh_v[:, 0]
        y = -msh_v[:, 2]
        z = -msh_v[:, 1]
        msh_v[:, 0] = x
        msh_v[:, 1] = y
        msh_v[:, 2] = z
        mesh_vertices_by_FDI.append(msh_v)
    X_Pred_Upper = np.array(mesh_vertices_by_FDI)

    # 使用调整后的预测结果/使用新病例阶段作为X_Pred_Upper进行evaluation
    mesh_vertices_by_FDI = []
    for fdi in FDIs_lower:
        mshf = os.path.join(
            # mesh_dir, f"Pred_Lower_Mesh_toothid={fdi}.stl"
            mesh_dir, f"crown{fdi}.stl"
        )
        msh = o3d.io.read_triangle_mesh(mshf)
        msh_v = np.asarray(msh.vertices, np.float64)
        msh_v = farthestPointDownSample(
            msh_v, num_point_sampled=1500
        )
        x = -msh_v[:, 0]
        y = -msh_v[:, 2]
        z = -msh_v[:, 1]
        msh_v[:, 0] = x
        msh_v[:, 1] = y
        msh_v[:, 2] = z
        mesh_vertices_by_FDI.append(msh_v)
    X_Pred_Lower = np.array(mesh_vertices_by_FDI)

    try:
        _X_Ref = X_Ref_Upper + X_Ref_Lower  # List concat
        print(
            "Compare prediction shape aligned by similarity registration with ground truth."
        )
        with_scale = True
        TX_Upper = pm_util.getAlignedSrcPointCloud(
            X_Pred_Upper.reshape(-1, 3), np.concatenate(X_Ref_Upper), with_scale=with_scale
        )
        TX_Lower = pm_util.getAlignedSrcPointCloud(
            X_Pred_Lower.reshape(-1, 3), np.concatenate(X_Ref_Lower), with_scale=with_scale
        )

        TX_Pred_Upper = TX_Upper.reshape(-1, NUM_POINT, 3)
        TX_Pred_Lower = TX_Lower.reshape(-1, NUM_POINT, 3)
        _TX_Pred = np.concatenate([TX_Pred_Upper, TX_Pred_Lower])

        RMSDs = [-9.9, -9.9]
        RMSD_T_pred, RMSD_T_pred1, RMSD_T_pred2, RMSDs, RMSDs1, RMSDs2 = metric.computeRMSD(_X_Ref, _TX_Pred)
        ASSD_T_pred = metric.computeASSD(_X_Ref, _TX_Pred)
        HD_T_pred = metric.computeHD(_X_Ref, _TX_Pred)
        CD_T_pred = metric.computeChamferDistance(_X_Ref, _TX_Pred)
        print("[RMSD] Root Mean Squared surface Distance (mm): {:.4f}".format(RMSD_T_pred))
        print(RMSDs)
        print("[RMSD1] Root Mean Squared surface Distance (mm): {:.4f}".format(RMSD_T_pred1))
        print(RMSDs1)
        print("[RMSD2] Root Mean Squared surface Distance (mm): {:.4f}".format(RMSD_T_pred2))
        print(RMSDs2)
        print("[ASSD] average symmetric surface distance (mm): {:.4f}".format(ASSD_T_pred))
        print("[HD] Hausdorff distance (mm): {:.4f}".format(HD_T_pred))
        print("[CD] Chamfer distance (mm^2): {:.4f}".format(CD_T_pred))
        log.close()
        sys.stdout = old_stdout
        # 检测是否出现异常
        if len(RMSDs) < 20:
            print(f"evaluation errors! teeth number only {len(RMSDs)}")
    except:
        log.close()
        sys.stdout = old_stdout
        print('error!')
        print(mesh_dir)



    # Dice_VOE_lst = [
    #     metric.computeDiceAndVOE(_x_ref, _x_pred, pitch=0.2)
    #     for _x_ref, _x_pred in zip(_X_Ref, _TX_Pred)
    # ]
    # avg_Dice, avg_VOE = np.array(Dice_VOE_lst).mean(axis=0)
    # print("[DC] Volume Dice Coefficient: {:.4f}".format(avg_Dice))
    # print("[VOE] Volumetric Overlap Error: {:.2f} %".format(100.0 * avg_VOE))

    # 将TX_Pred_Upper，TX_Pred_Lower的每个牙齿重建成mesh并保存
    # for pg in TX_Pred_Upper:
    #     x = -pg[:, 0]
    #     y = -pg[:, 2]
    #     z = -pg[:, 1]
    #     pg[:, 0] = x
    #     pg[:, 1] = y
    #     pg[:, 2] = z
    # for pg in TX_Pred_Lower:
    #     x = -pg[:, 0]
    #     y = -pg[:, 2]
    #     z = -pg[:, 1]
    #     pg[:, 0] = x
    #     pg[:, 1] = y
    #     pg[:, 2] = z
    # demoMeshDir = os.path.join(DEMO_MESH_ALIGNED_DIR, "{}/".format(tag))
    # os.makedirs(demoMeshDir, exist_ok=True)
    # for idx, t_i in enumerate(UPPER_INDICES):
    #     tooth_Mesh = pm_util.surfaceVertices2WatertightO3dMesh(TX_Pred_Upper[idx])
    #     pm_util.exportTriMeshObj(
    #         np.asarray(tooth_Mesh.vertices),
    #         np.asarray(tooth_Mesh.triangles),
    #         os.path.join(demoMeshDir, "Pred_Upper_Mesh_toothid={}.obj".format(str(t_i))),
    #     )
    # for idx, t_i in enumerate(LOWER_INDICES):
    #     tooth_Mesh = pm_util.surfaceVertices2WatertightO3dMesh(TX_Pred_Lower[idx])
    #     pm_util.exportTriMeshObj(
    #         np.asarray(tooth_Mesh.vertices),
    #         np.asarray(tooth_Mesh.triangles),
    #         os.path.join(demoMeshDir, "Pred_Lower_Mesh_toothid={}.obj".format(str(t_i))),
    #     )

def read_demo_mesh_vertices_by_FDI_npy(mesh_dir, tag, FDIs):
    mesh_vertices_by_FDI = []
    for fdi in FDIs:
        mshf = os.path.join(
        mesh_dir, f"crown{fdi}.npy"
        )
        msh = np.load(mshf)
        msh2 = np.zeros_like(msh)
        msh2[:, 0] = -msh[:, 0]
        msh2[:, 1] = -msh[:, 2]
        msh2[:, 2] = -msh[:, 1]
        mesh_vertices_by_FDI.append(msh2)
    return mesh_vertices_by_FDI

def teeth_check(u, l, path1, path2):
    for root, dirs, files in os.walk(path1):
        if not "crown11.npy" in files:
            u[0] = False
        if not "crown12.npy" in files:
            u[1] = False
        if not "crown13.npy" in files:
            u[2] = False
        if not "crown14.npy" in files:
            u[3] = False
        if not "crown15.npy" in files:
            u[4] = False
        if not "crown16.npy" in files:
            u[5] = False
        if not "crown17.npy" in files:
            u[6] = False
        if not "crown21.npy" in files:
            u[7] = False
        if not "crown22.npy" in files:
            u[8] = False
       
