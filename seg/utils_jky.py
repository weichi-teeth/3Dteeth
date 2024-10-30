import numpy as np
import skimage
import cv2
from seg.seg_const import IMG_SHAPE
import matplotlib.pyplot as plt
from skimage.transform import resize

def get_contour_from_raw_pred(pred_label, mask_shape, thresh=0.5):
    pred_prob_map = skimage.transform.resize(pred_label, mask_shape)
    pred_mask = pred_prob_map > thresh
    pred_mask = skimage.morphology.skeletonize(pred_mask.astype(np.uint8))
    pred_edge_img = (255.0 * pred_mask).astype(np.uint8)
    return pred_edge_img


def predict_teeth_contour(model, imgfile, resized_width=800):
    img = skimage.io.imread(imgfile)
    h, w = img.shape[:2]
    scale = resized_width / w
    rimg = skimage.transform.resize(img, IMG_SHAPE)
    raw_pred = model.predict(rimg[None, :])
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.title('Original Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(raw_pred[0], cmap='gray')
    # plt.title('Predicted Mask')
    # plt.savefig(f'28prediction_1.png')
    # plt.close()
    raw_pred = np.squeeze(raw_pred)
    edge_pred = get_contour_from_raw_pred(
        raw_pred, (int(scale * h), int(scale * w)), thresh=0.5
    )
    return edge_pred


def gt_teeth_contour(img, resized_width=800):
    edge_pred = get_contour_from_raw_pred(
        img/255.0, (600, 800), thresh=0.5
    )
    return edge_pred

def get_contour_from_raw_pred_jky(pred_label, mask_shape, thresh=0.5):
    #resize后像素变成了小数，不要加入注释的，会导致两个不匹配
    pred_label = pred_label > thresh
    #pred_label = skimage.morphology.skeletonize(pred_label.astype(np.uint8))
    pred_edge_img = (255.0 * pred_label.astype(np.uint8))
    return pred_edge_img
def gt_teeth_contour_jky(img, resized_width=800,resized_height=600):
    #edge_pred = get_contour_from_raw_pred_jky(
    #    img, (600, 800), thresh=0.5
    #)
    img1=img[:,:,0]
    img2=img[:,:,1]
    #plt.imshow(img2)
    #plt.show()
    resized_img1=resize(img1,(resized_height,resized_width),anti_aliasing=False,preserve_range=True,order=0)
    resized_img2 = resize(img2, (resized_height, resized_width), anti_aliasing=False, preserve_range=True,order=0)

    resized_imgs=np.stack([resized_img1,resized_img2],axis=-1)
    edge_pred1=get_contour_from_raw_pred_jky(resized_imgs[:,:,0],(resized_height,resized_width),thresh=0)
    #plt.imshow(edge_pred1)
    #plt.show()
    #edge_pred2 = get_contour_from_raw_pred_jky(resized_imgs[:, :, 1], (resized_height, resized_width), thresh=0.5)
    #plt.imshow(resized_img2)
    #plt.show()
    edge_pred = np.stack([edge_pred1, resized_img2], axis=-1)

    return edge_pred

def reset_by_indices_jky(edge_mask, indices):
    """
    根据edge_mask_jky[:,:,1]的值，如果这些值等于indices中的任何一个数值，
    那么在edge_mask_jky的两个通道中对应的位置都设置为0。

    :param edge_mask_jky: 一个三维NumPy数组，假设至少有两个通道。
    :param indices: 一个包含序号的列表。
    """
    # 检查输入是否为NumPy数组
    if not isinstance(edge_mask, np.ndarray):
        raise ValueError("输入必须是一个NumPy数组")

    # 检查数组是否有足够的通道
    if edge_mask.shape[2] < 2:
        raise ValueError("数组必须至少有两个通道")
    
    # 创建一个掩码，用于标识第二通道中等于indices中任何一个值的位置
    mask = np.isin(edge_mask[:,:,1], indices)
    
    # 使用掩码将第一个通道和第二个通道的相应位置设置为0
    edge_mask[:,:,0][mask] = 0
    edge_mask[:,:,1][mask] = 0

    return edge_mask


def reset_by_indices_predmask_jky(edge_mask, indices):
    """
    根据edge_mask_jky[:,:,1]的值，如果这些值等于indices中的任何一个数值，
    那么在edge_mask_jky的两个通道中对应的位置都设置为0。

    :param edge_mask_jky: 一个三维NumPy数组，假设至少有两个通道。
    :param indices: 一个包含序号的列表。
    """
    # 检查输入是否为NumPy数组


    # 创建一个掩码，用于标识第二通道中等于indices中任何一个值的位置
    mask = np.isin(edge_mask, indices)

    # 使用掩码将第一个通道和第二个通道的相应位置设置为0
    edge_mask[mask] = 0


    return edge_mask
    