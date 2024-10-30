import functools
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # run on CPU
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import psutil
import ray

import pcd_mesh_utils as pm_util
import recons_eval_metric as metric
from const_new_shc import *
from emopt5views_shc import EMOpt5Views
from seg.seg_const import IMG_SHAPE
# from seg.seg_model import ASPP_UNet
from seg.utils_shc import predict_teeth_contour, gt_teeth_contour
import cv2
from PIL import Image

from pcd_mesh_utils import farthestPointDownSample
import time


TEMP_DIR = r"./demo_shc/_temp/"
os.makedirs(TEMP_DIR, exist_ok=True)

NUM_CPUS = psutil.cpu_count(logical=False)
print = functools.partial(print, flush=True)


def getToothIndex(f):
    return int(os.path.basename(f).split(".")[0].split("_")[-1])


def loadMuEigValSigma(ssmDir, numPC):
    """Mu.shape=(28,1500,3), sqrtEigVals.shape=(28,1,100), Sigma.shape=(28,4500,100)"""
    muNpys = glob.glob(os.path.join(ssmDir, "meanAlignedPG_*.npy"))
    muNpys = sorted(muNpys, key=lambda x: getToothIndex(x))
    Mu = np.array([np.load(x) for x in muNpys])
    eigValNpys = glob.glob(os.path.join(ssmDir, "eigVal_*.npy"))
    eigValNpys = sorted(eigValNpys, key=lambda x: getToothIndex(x))
    sqrtEigVals = np.sqrt(np.array([np.load(x) for x in eigValNpys]))
    eigVecNpys = glob.glob(os.path.join(ssmDir, "eigVec_*.npy"))
    eigVecNpys = sorted(eigVecNpys, key=lambda x: getToothIndex(x))
    Sigma = np.array([np.load(x) for x in eigVecNpys])
    return Mu, sqrtEigVals[:, np.newaxis, :numPC], Sigma[..., :numPC]


def run_emopt(emopt: EMOpt5Views, verbose: bool = False):
    # 3d teeth reconstruction by optimization
    print("-" * 100)
    print("Start optimization.")

    # grid search parallelled by Ray
    print("-" * 100)
    print("Start Grid Search.")

    # parallel function supported by Ray
    emopt.searchDefaultRelativePoseParams()
    emopt.gridSearchExtrinsicParams()
    emopt.gridSearchRelativePoseParams()

    emopt.expectation_step_5Views(-1, verbose)

    for phtype in PHOTO_TYPES:
        canvas, canvas_gt, canvas_pred = emopt.showEdgeMaskPredictionWithGroundTruth(photoType=phtype)
        canvas_gt = canvas_gt.astype(np.uint8)[:, :, 0]
        canvas_gt = np.stack((canvas_gt, canvas_gt, canvas_gt), axis=-1)
        canvas_gt_img = Image.fromarray(canvas_gt)
        # canvas_gt_img.save(f"canvas_gt_{str(phtype.value)}.png")
        canvas_pred = (canvas_pred * 255)
        canvas_pred = np.stack((canvas_pred, np.zeros((canvas_pred.shape[0], canvas_pred.shape[1])), np.zeros((canvas_pred.shape[0], canvas_pred.shape[1]))), axis=-1).astype(np.uint8)
        canvas_pred = canvas_pred + canvas_gt
        canvas_pred_img = Image.fromarray(canvas_pred)
        canvas_pred_img.save(f"demo_shc/int/{tag}/canvas_{tag}_0_{str(phtype.value)}.png")

    min_e_loss = emopt.get_e_loss()
    optParamDict = emopt.get_current_e_step_result()

    # stage0initMatFile = os.path.join(TEMP_DIR, "E-step-result-stage0-init.mat")
    # stage0finalMatFile = os.path.join(TEMP_DIR, "E-step-result-stage0-final.mat")

    # emopt.save_expectation_step_result(stage0initMatFile) # save checkpoint

    maxiter = 20
    # stageIter = [10, 5, 10]
    stageIter = [10, 0, 5]
    # stageIter = [10, 0, 5]
    # stageIter = [0, 0, 0]
    # stage 0 & 1 optimization

    print("-" * 100)
    print("Start Stage 0.")
    stage = 0

    # # Continue from checkpoint "E-step-result-stage0-init.mat"
    # emopt.load_expectation_step_result(stage0initMatFile, stage)
    # emopt.expectation_step_5Views(stage, verbose)

    E_loss = []
    for it in range(stageIter[0]):
        emopt.maximization_step_5Views(stage, step=-1, maxiter=maxiter, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage, verbose)
        e_loss = emopt.get_e_loss()
        if e_loss < min_e_loss:
            optParamDict = emopt.get_current_e_step_result()
            min_e_loss = e_loss
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        # E_loss.append(e_loss)
        # 设置终止条件（e_loss比上一次更大）
        if len(E_loss) >= 2 and e_loss >= np.mean(E_loss[-2:]):
            print(
                "Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(
                    E_loss[-2], E_loss[-1], e_loss
                )
            )
            E_loss.append(e_loss)
            break
        else:
            E_loss.append(e_loss)

    # Load best result of stage 0
    emopt.load_e_step_result_from_dict(optParamDict)
    # emopt.expectation_step_5Views(stage, verbose)
    E_loss.append(min_e_loss)

    # emopt.save_expectation_step_result(stage0finalMatFile)  # save checkpoint

    for phtype in PHOTO_TYPES:
        canvas, canvas_gt, canvas_pred = emopt.showEdgeMaskPredictionWithGroundTruth(photoType=phtype)
        canvas_gt = canvas_gt.astype(np.uint8)[:, :, 0]
        canvas_gt = np.stack((canvas_gt, canvas_gt, canvas_gt), axis=-1)
        canvas_gt_img = Image.fromarray(canvas_gt)
        # canvas_gt_img.save(f"canvas_gt_{str(phtype.value)}.png")
        canvas_pred = (canvas_pred * 255)
        canvas_pred = np.stack((canvas_pred, np.zeros((canvas_pred.shape[0], canvas_pred.shape[1])), np.zeros((canvas_pred.shape[0], canvas_pred.shape[1]))), axis=-1).astype(np.uint8)
        canvas_pred = canvas_pred + canvas_gt
        canvas_pred_img = Image.fromarray(canvas_pred)
        canvas_pred_img.save(f"demo_shc/int/{tag}/canvas_{tag}_1_{str(phtype.value)}.png")

    skipStage1Flag = False
    print("-" * 100)
    print("Start Stage 1.")

    stage = 1
    for it in range(stageIter[1]):
        emopt.maximization_step_5Views(stage, step=-1, maxiter=maxiter, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage, verbose)
        e_loss = emopt.get_e_loss()
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        if e_loss >= E_loss[-1]:
            if it == 0:
                skipStage1Flag = True  # first optimization with rowScaleXZ gets worse result compared with optimziaiton without rowScaleXZ
            print(
                "Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(
                    E_loss[-2], E_loss[-1], e_loss
                )
            )
            break
        else:
            E_loss.append(e_loss)

    # whether to skip stage1 to avoid extreme deformation
    if skipStage1Flag == True:
        print("Skip Stage 1; Reverse to Stage 0 final result.")
        emopt.rowScaleXZ = np.ones((2,))
        emopt.load_e_step_result_from_dict(optParamDict)
        # # Continue from checkpoint "E-step-result-stage0-final.mat"
        # emopt.load_expectation_step_result(stage0finalMatFile, stage=2)
    else:
        print("Accept Stage 1.")
        print("emopt.rowScaleXZ: ", emopt.rowScaleXZ)
        print("approx tooth scale: ", np.prod(emopt.rowScaleXZ) ** (1 / 3))
        emopt.anistropicRowScale2ScalesAndTransVecs()

    # Load best result of stage 1
    # emopt.expectation_step_5Views(stage, verbose)
    e_loss = emopt.get_e_loss()
    optParamDict = emopt.get_current_e_step_result()

    for phtype in PHOTO_TYPES:
        canvas, canvas_gt, canvas_pred = emopt.showEdgeMaskPredictionWithGroundTruth(photoType=phtype)
        canvas_gt = canvas_gt.astype(np.uint8)[:, :, 0]
        canvas_gt = np.stack((canvas_gt, canvas_gt, canvas_gt), axis=-1)
        canvas_gt_img = Image.fromarray(canvas_gt)
        # canvas_gt_img.save(f"canvas_gt_{str(phtype.value)}.png")
        canvas_pred = (canvas_pred * 255)
        canvas_pred = np.stack((canvas_pred, np.zeros((canvas_pred.shape[0], canvas_pred.shape[1])), np.zeros((canvas_pred.shape[0], canvas_pred.shape[1]))), axis=-1).astype(np.uint8)
        canvas_pred = canvas_pred + canvas_gt
        canvas_pred_img = Image.fromarray(canvas_pred)
        canvas_pred_img.save(f"demo_shc/int/{tag}/canvas_{tag}_2_{str(phtype.value)}.png")

    # Stage = 2 and 3
    print("-" * 100)
    print("Start Stage 2 and 3.")
    stage = 2
    E_loss = [
        min_e_loss,
    ]
    for it in range(stageIter[2]):
        emopt.maximization_step_5Views(stage, step=2, maxiter=maxiter, verbose=False)
        emopt.maximization_step_5Views(stage, step=3, maxiter=maxiter, verbose=False)
        emopt.maximization_step_5Views(stage=3, step=-1, maxiter=maxiter, verbose=False)
        emopt.maximization_step_5Views(stage, step=1, maxiter=maxiter, verbose=False)
        print("M-step loss: {:.4f}".format(emopt.loss_maximization_step))
        emopt.expectation_step_5Views(stage=3, verbose=verbose)
        e_loss = emopt.get_e_loss()
        if e_loss < min_e_loss:
            optParamDict = emopt.get_current_e_step_result()
            min_e_loss = e_loss
        print("Sum of expectation step loss: {:.4f}".format(e_loss))
        if len(E_loss) >= 2 and (e_loss >= np.mean(E_loss[-2:])):
            print(
                "Early stop with last 3 e-step loss {:.4f}, {:.4f}, {:.4f}".format(
                    E_loss[-2], E_loss[-1], e_loss
                )
            )
            break
        else:
            E_loss.append(e_loss)

    # Load best result of stage 2 and 3
    emopt.load_e_step_result_from_dict(optParamDict)
    # emopt.expectation_step_5Views(stage=3, verbose=verbose)

    return emopt


def evaluation(h5File, X_Ref_Upper, X_Ref_Lower, FDIs_upper, FDIs_lower, mask_u, mask_l, tag):
    """
    h5file: emopt result saved in h5 format
    X_Ref_Upper, X_Ref_Lower: List of numpy arrays
    """
    with h5py.File(h5File, "r") as f:
        grp = f["EMOPT"]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]

    # # 使用调整后的预测结果/使用新病例阶段作为X_Pred_Upper进行evaluation
    # mesh_dir = "seg/valid/500RealCases/Case" + tag + "/新病例阶段/ExportSTLs/"
    # mesh_vertices_by_FDI = []
    # for fdi in FDIs_upper:
    #     msh_v = np.load(f'{mesh_dir}crown{fdi}-ds.npy')
    #     x = -msh_v[:, 0]
    #     y = -msh_v[:, 2]
    #     z = -msh_v[:, 1]
    #     msh_v[:, 0] = x
    #     msh_v[:, 1] = y
    #     msh_v[:, 2] = z
    #     mesh_vertices_by_FDI.append(msh_v)
    # X_Pred_Upper = np.array(mesh_vertices_by_FDI)
    #
    # # 使用调整后的预测结果/使用新病例阶段作为X_Pred_Upper进行evaluation
    # mesh_vertices_by_FDI = []
    # for fdi in FDIs_lower:
    #     msh_v = np.load(f'{mesh_dir}crown{fdi}-ds.npy')
    #     x = -msh_v[:, 0]
    #     y = -msh_v[:, 2]
    #     z = -msh_v[:, 1]
    #     msh_v[:, 0] = x
    #     msh_v[:, 1] = y
    #     msh_v[:, 2] = z
    #     mesh_vertices_by_FDI.append(msh_v)
    # X_Pred_Lower = np.array(mesh_vertices_by_FDI)

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

    # Dice_VOE_lst = [
    #     metric.computeDiceAndVOE(_x_ref, _x_pred, pitch=0.2)
    #     for _x_ref, _x_pred in zip(_X_Ref, _TX_Pred)
    # ]
    # avg_Dice, avg_VOE = np.array(Dice_VOE_lst).mean(axis=0)
    # print("[DC] Volume Dice Coefficient: {:.4f}".format(avg_Dice))
    # print("[VOE] Volumetric Overlap Error: {:.2f} %".format(100.0 * avg_VOE))

    # 将TX_Pred_Upper，TX_Pred_Lower的每个牙齿重建成mesh并保存
    for pg in TX_Pred_Upper:
        x = -pg[:, 0]
        y = -pg[:, 2]
        z = -pg[:, 1]
        pg[:, 0] = x
        pg[:, 1] = y
        pg[:, 2] = z
    for pg in TX_Pred_Lower:
        x = -pg[:, 0]
        y = -pg[:, 2]
        z = -pg[:, 1]
        pg[:, 0] = x
        pg[:, 1] = y
        pg[:, 2] = z
    demoMeshDir = os.path.join(DEMO_MESH_ALIGNED_DIR, "{}/".format(tag))
    os.makedirs(demoMeshDir, exist_ok=True)

    # 依据mask_u依次生成上牙的每一颗非缺失牙齿
    for idx, t_i in enumerate(np.array(UPPER_INDICES)[mask_u]):
        tooth_Mesh = pm_util.surfaceVertices2WatertightO3dMesh(TX_Pred_Upper[idx])
        pm_util.exportTriMeshObj(
            np.asarray(tooth_Mesh.vertices),
            np.asarray(tooth_Mesh.triangles),
            os.path.join(demoMeshDir, "Pred_Upper_Mesh_toothid={}.obj".format(str(t_i))),
        )
    # 依据mask_l依次生成下牙的每一颗非缺失牙齿
    for idx, t_i in enumerate(np.array(LOWER_INDICES)[mask_l]):
        tooth_Mesh = pm_util.surfaceVertices2WatertightO3dMesh(TX_Pred_Lower[idx])
        pm_util.exportTriMeshObj(
            np.asarray(tooth_Mesh.vertices),
            np.asarray(tooth_Mesh.triangles),
            os.path.join(demoMeshDir, "Pred_Lower_Mesh_toothid={}.obj".format(str(t_i))),
        )


def create_mesh_from_emopt_h5File(h5File, meshDir, save_name, mask_u, mask_l):
    with h5py.File(h5File, "r") as f:
        grp = f["EMOPT"]
        X_Pred_Upper = grp["UPPER_PRED"][:]
        X_Pred_Lower = grp["LOWER_PRED"][:]
    print('here1')

    # # 改回原始坐标系
    # for pg in X_Pred_Upper:
    #     x = -pg[:, 0]
    #     y = -pg[:, 2]
    #     z = -pg[:, 1]
    #     pg[:, 0] = x
    #     pg[:, 1] = y
    #     pg[:, 2] = z
    # for pg in X_Pred_Lower:
    #     x = -pg[:, 0]
    #     y = -pg[:, 2]
    #     z = -pg[:, 1]
    #     pg[:, 0] = x
    #     pg[:, 1] = y
    #     pg[:, 2] = z

    demoMeshDir = os.path.join(meshDir, "{}/".format(save_name))
    os.makedirs(demoMeshDir, exist_ok=True)
    print('here2')

    # 依据mask_u依次生成上牙的每一颗非缺失牙齿
    for idx, t_i in enumerate(np.array(UPPER_INDICES)[mask_u]):
        tooth_Mesh = pm_util.surfaceVertices2WatertightO3dMesh(X_Pred_Upper[idx])
        pm_util.exportTriMeshObj(
            np.asarray(tooth_Mesh.vertices),
            np.asarray(tooth_Mesh.triangles),
            os.path.join(demoMeshDir, "Pred_Upper_Mesh_toothid={}.obj".format(str(t_i))),
        )
    # 依据mask_l依次生成下牙的每一颗非缺失牙齿
    for idx, t_i in enumerate(np.array(LOWER_INDICES)[mask_l]):
        tooth_Mesh = pm_util.surfaceVertices2WatertightO3dMesh(X_Pred_Lower[idx])
        pm_util.exportTriMeshObj(
            np.asarray(tooth_Mesh.vertices),
            np.asarray(tooth_Mesh.triangles),
            os.path.join(demoMeshDir, "Pred_Lower_Mesh_toothid={}.obj".format(str(t_i))),
        )

    X_Pred_Upper_Meshes = [
        pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Upper
    ]
    X_Pred_Lower_Meshes = [
        pm_util.surfaceVertices2WatertightO3dMesh(pg) for pg in X_Pred_Lower
    ]
    Pred_Upper_Mesh = pm_util.mergeO3dTriangleMeshes(X_Pred_Upper_Meshes)
    Pred_Lower_Mesh = pm_util.mergeO3dTriangleMeshes(X_Pred_Lower_Meshes)

    pm_util.exportTriMeshObj(
        np.asarray(Pred_Upper_Mesh.vertices),
        np.asarray(Pred_Upper_Mesh.triangles),
        os.path.join(demoMeshDir, "Pred_Upper_Mesh_Tag={}.obj".format(save_name)),
    )
    pm_util.exportTriMeshObj(
        np.asarray(Pred_Lower_Mesh.vertices),
        np.asarray(Pred_Lower_Mesh.triangles),
        os.path.join(demoMeshDir, "Pred_Lower_Mesh_Tag={}.obj".format(save_name)),
    )


def read_demo_mesh_vertices_by_FDI(mesh_dir, tag, FDIs):
    mesh_vertices_by_FDI = []
    for fdi in FDIs:
        mshf = os.path.join(
            mesh_dir, str(tag), "byFDI", f"Ref_Mesh_Tag={tag}_FDI={fdi}.obj"
        )
        msh = o3d.io.read_triangle_mesh(mshf)
        mesh_vertices_by_FDI.append(np.asarray(msh.vertices, np.float64))
    return mesh_vertices_by_FDI


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

def expand_array(arr):
    """
    将(a,b,c)大小的数组扩展成(a2,b2,c)大小的数组，a2为大于等于a且满足a2=3*n(n为整数），b2为大于等于b且满足b2=4*n。
    """
    a, b, c = arr.shape  # 获取原始数组的大小

    # 计算满足条件的 a2 和 b2
    a2 = ((a + 2) // 3) * 3  # 找到大于等于 a 的最近的 3 的倍数
    b2 = ((b + 3) // 4) * 4  # 找到大于等于 b 的最近的 4 的倍数

    # 创建新的填充数组，并将原数组的值复制到新数组中
    expanded_arr = np.zeros((a2, b2, c), dtype=arr.dtype)
    expanded_arr[:a, :b, :] = arr

    return expanded_arr


def main(tag="0"):
    start_time = time.time()
    Mu0, SqrtEigVals, Sigma = loadMuEigValSigma(SSM_DIR, numPC=NUM_PC)
    Mu = Mu0

    # dir_Mu2 = "seg/train/500RealCases/Case" + '83' + "/新病例阶段/ExportSTLs/"
    dir_Mu2 = "seg/train/new_dataset/口腔/20240918STLs/" + tag + "/新病例阶段/ExportSTLs/"
    dir_Mu2_mid = "seg/train/new_dataset/口腔/20240918STLs/" + tag + "/中期阶段1/ExportSTLs/"
    file_names = [f'{dir_Mu2}crown{d1}{d2}-ds.npy' for d1 in range(1, 5) for d2 in range(1, 8)]
    data_list = []
    for file_name in file_names:
        try:
            data = np.load(file_name)
            data_list.append(data)
        except:
            # print(file_name)
            data_list.append(np.random.uniform(low=-20, high=20, size=(1500, 3)))
    Mu2 = np.stack(data_list, axis=0)
    # 计算 Mu_mean
    Mu_mean = np.zeros((28, 3))
    Mu2_mean = np.zeros((28, 3))
    Mu_vec = np.zeros((28, 3))
    Mu3 = np.zeros_like(Mu2)

    Mu2_2 = np.zeros_like(Mu2)
    Mu2_2[:, :, 0] = -Mu2[:, :, 0]
    Mu2_2[:, :, 1] = -Mu2[:, :, 2]
    Mu2_2[:, :, 2] = -Mu2[:, :, 1]

    Mu_mean = np.array([np.mean(Mu[:, :, 0]), np.mean(Mu[:, :, 1]), np.mean(Mu[:, :, 2])])
    Mu2_2_mean = np.array([np.mean(Mu2_2[:, :, 0]), np.mean(Mu2_2[:, :, 1]), np.mean(Mu2_2[:, :, 2])])
    Mu_vec = Mu_mean - Mu2_2_mean
    for i in range(28):
        Mu3[i, :, 0] = Mu2_2[i, :, 0] + Mu_vec[0]
        Mu3[i, :, 1] = Mu2_2[i, :, 1] + Mu_vec[1]
        Mu3[i, :, 2] = Mu2_2[i, :, 2] + Mu_vec[2]
    Mu = Mu3

    np.save('points_vis/Mu.npy', Mu)

    # 是否使用初始参数模型
    # Mu = Mu0

    Mu_normals = EMOpt5Views.computePointNormals(Mu)

    transVecStd = 1.1463183505325343  # obtained by SSM
    rotVecStd = 0.13909168140778128  # obtained by SSM
    PoseCovMats = np.load(
        os.path.join(REGIS_PARAM_DIR, "PoseCovMats.npy")
    )  # Covariance matrix of tooth pose for each tooth, shape=(28,6,6)
    ScaleCovMat = np.load(
        os.path.join(REGIS_PARAM_DIR, "ScaleCovMat.npy")
    )  # Covariance matrix of scales for each tooth, shape=(28,28)
    tooth_exist_mask = TOOTH_EXIST_MASK['1']
    LogFile = os.path.join(TEMP_DIR, "Tag={}.log".format(tag))
    if os.path.exists(LogFile):
        os.remove(LogFile)
    log = open(LogFile, "a", encoding="utf-8")
    sys.stdout = log
    # teeth boundary segmentation model
    # weight_ckpt = r".\seg\weights\weights-teeth-boundary-model.h5"
    weight_ckpt = r"model_weights.h5"
    # weight_ckpt = r"./seg/weights/model_weights.h5"
    # weight_ckpt = r"seg\weights\weights-teeth-boundary-model.h5"
    # model = ASPP_UNet(IMG_SHAPE, filters=[16, 32, 64, 128, 256])
    # model.load_weights(weight_ckpt)
    print('hi3')
    # 保存模型预测轮廓线图
    if not os.path.exists(f"demo_shc/int/{tag}"):
        os.makedirs(f"demo_shc/int/{tag}")
    # predcit teeth boundary in each photo
    edgeMasks = []
    edgeMasks_sizes = []
    for phtype in PHOTO_TYPES:
        print(phtype)
        imgfile = os.path.join(PHOTO_DIR, f"{tag}-{phtype.value}.png")

        # edge_mask = predict_teeth_contour(
        #     model, imgfile, resized_width=RECONS_IMG_WIDTH
        # )  # resize image to (800,~600)
        # cv2.imwrite(f"{tag}-{phtype.value}-predict.png", edge_mask)

        # np.count_nonzero(edge_mask[:, :, 1] == 31)

        # edge_mask = cv2.imread(f"seg/valid/label/{tag}-{phtype.value}.png", 0)
        edge_mask = np.load(f'new1/{tag}/output_npy/{phtype.value}.npy')
        # edge_mask = np.load(f'gt/{tag}/{phtype.value}.npy')
        # 上视图和下视图需要翻转
        if phtype == PHOTO.UPPER or phtype == PHOTO.LOWER:
            edge_mask = np.flipud(edge_mask)
        # 上视图不需要牙齿17，27；下视图不需要牙齿37，47；左视图不需要牙齿41，37；右视图不需要牙齿31，47；
        # 正视图不需要牙齿15，16，17，25，26，27，35，36，37，45，46，47；
        edge_mask[edge_mask[:, :, 1] == 17, 0] = 0
        edge_mask[edge_mask[:, :, 1] == 27, 0] = 0
        edge_mask[edge_mask[:, :, 1] == 37, 0] = 0
        edge_mask[edge_mask[:, :, 1] == 47, 0] = 0
        if phtype == PHOTO.LEFT:
            edge_mask[edge_mask[:, :, 1] == 41, 0] = 0
        if phtype == PHOTO.RIGHT:
            edge_mask[edge_mask[:, :, 1] == 31, 0] = 0
        if phtype == PHOTO.FRONTAL:
            edge_mask[edge_mask[:, :, 1] == 15, 0] = 0
            edge_mask[edge_mask[:, :, 1] == 16, 0] = 0
            edge_mask[edge_mask[:, :, 1] == 25, 0] = 0
            edge_mask[edge_mask[:, :, 1] == 26, 0] = 0
            edge_mask[edge_mask[:, :, 1] == 35, 0] = 0
            edge_mask[edge_mask[:, :, 1] == 36, 0] = 0
            edge_mask[edge_mask[:, :, 1] == 45, 0] = 0
            edge_mask[edge_mask[:, :, 1] == 46, 0] = 0


        # 扩充轮廓线图，增加固定的空白边缘
        pad_width = 200
        edge_mask = np.pad(edge_mask, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)

        # 扩充轮廓线图，增加空白边缘，满足长宽比例为800x600
        edge_mask = expand_array(edge_mask)

        edge_mask = edge_mask[:, :, 0] * 255

        # 二值图像骨架提取以及resize成(600,800)大小
        edge_mask = gt_teeth_contour(edge_mask)

        # 图像的大小

        size = (edge_mask.shape[0], edge_mask.shape[1])

        cv2.imwrite(f"demo_shc/int/{tag}/{tag}-{phtype.value}-label.png", edge_mask)
        edgeMasks.append(edge_mask)
        edgeMasks_sizes.append(size)

        # plt.imshow(edge_mask)
        # plt.show()
    print('hi4')
    # del model # to release memory

    mask_u, mask_l = np.split(tooth_exist_mask, 2)
    # X_Ref_Upper = read_demo_mesh_vertices_by_FDI(
    #     mesh_dir=REF_MESH_DIR, tag=tag, FDIs=np.array(UPPER_INDICES)[mask_u]
    # )
    # X_Ref_Lower = read_demo_mesh_vertices_by_FDI(
    #     mesh_dir=REF_MESH_DIR, tag=tag, FDIs=np.array(LOWER_INDICES)[mask_l]
    # )
    X_Ref_Upper = read_demo_mesh_vertices_by_FDI_npy(
        mesh_dir=dir_Mu2_mid, tag=tag, FDIs=np.array(UPPER_INDICES)[mask_u]
    )
    X_Ref_Lower = read_demo_mesh_vertices_by_FDI_npy(
        mesh_dir=dir_Mu2_mid, tag=tag, FDIs=np.array(LOWER_INDICES)[mask_l]
    )
    print('hi5')
    # run deformation-based 3d reconstruction
    emopt = EMOpt5Views(
        edgeMasks,
        PHOTO_TYPES,
        VISIBLE_MASKS["1"],
        tooth_exist_mask,
        Mu,
        Mu_normals,
        SqrtEigVals,
        Sigma,
        PoseCovMats,
        ScaleCovMat,
        transVecStd,
        rotVecStd,
        edgeMasks_sizes,
        # pad_width,
    )
    emopt = run_emopt(emopt)
    demoh5File = os.path.join(DEMO_H5_DIR, f"demo-tag={tag}.h5")
    emopt.saveDemo2H5(demoh5File)
    print('hi6')

    for phtype in PHOTO_TYPES:
        canvas, canvas_gt, canvas_pred = emopt.showEdgeMaskPredictionWithGroundTruth(photoType=phtype)
        try:
            canvas_gt = canvas_gt.astype(np.uint8)[:, :, 0]
            canvas_gt = np.stack((canvas_gt, canvas_gt, canvas_gt), axis=-1)
            canvas_gt_img = Image.fromarray(canvas_gt)
            canvas_gt_img.save(f"canvas_gt_{str(phtype.value)}.png")

            canvas_pred = (canvas_pred * 255)
            canvas_pred = np.stack((canvas_pred, np.zeros((canvas_pred.shape[0], canvas_pred.shape[1])), np.zeros((canvas_pred.shape[0], canvas_pred.shape[1]))), axis=-1).astype(np.uint8)
            canvas_pred = canvas_pred + canvas_gt
            canvas_pred_img = Image.fromarray(canvas_pred)
            canvas_pred_img.save(f"demo_shc/int/{tag}/canvas_{tag}_pred_{str(phtype.value)}.png")

        except:
            l = 1

    create_mesh_from_emopt_h5File(demoh5File, meshDir=DEMO_MESH_DIR, save_name=tag, mask_u=mask_u, mask_l=mask_l)

    evaluation(demoh5File, X_Ref_Upper, X_Ref_Lower, FDIs_upper=np.array(UPPER_INDICES)[mask_u],
               FDIs_lower=np.array(LOWER_INDICES)[mask_l], mask_u=mask_u, mask_l=mask_l, tag=tag)

    print('end')
    end_time = time.time()
    print("代码运行时间为：", end_time - start_time, "秒")
    log.close()
    # print('end')


if __name__ == "__main__":
    ray.init(num_cpus=4, num_gpus=1)
    # tags = ["83", "100", "108", "125", "134", "143", "402", "404", "418", "424", "432"]
    # tags = ["51", "54", "89", "419", "427", "437", "457", "465"]
    # tags = ["83", "100", "108", "125", "134", "143", "402", "404", "418", "424", "432", "51", "54", "89", "419", "427", "437", "457", "465"]
    tags = os.listdir('new1/')
    for tag in tags:
        try:
            main(tag)
        except:
            l=1
