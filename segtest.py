import functools
import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import skimage
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from seg.seg_const import IMG_SHAPE, LOW_MEMORY, ROOT_DIR, TRAIN_PATH, VALID_PATH, TEST_PATH
from seg.seg_data import DataGenerator, get_data_filenames, read_data
from seg.seg_loss import Dice_SSIM_loss
from seg.seg_model import ASPP_UNet

def calc_recall_precision_F1score(y, py):
    _TP = np.count_nonzero(np.logical_and(y, py))
    _TN = np.count_nonzero(np.logical_and(1.0 - y, 1.0 - py))
    _FP = np.count_nonzero(np.logical_and(1.0 - y, py))
    _FN = np.count_nonzero(np.logical_and(y, 1.0 - py))
    _recall = _TP / (_TP + _FN)
    _precision = _TP / (_TP + _FP)
    _f1 = 2 * _TP / (2 * _TP + _FN + _FP)
    return _recall, _precision, _f1


def compute_avg_recall_precision_F1score(
    masks, pred_prob_map, thre=0.5, from_logits=False
):
    pred_masks = pred_prob_map.copy()
    if from_logits == True:
        pred_masks = np.exp(pred_masks) / (1.0 + np.exp(pred_masks))
    pred_masks = pred_masks > thre
    ret_list = [
        calc_recall_precision_F1score(masks[i], pred_masks[i])
        for i in range(len(masks))
    ]
    return tuple(np.array(ret_list).mean(axis=0))


def evaluate(model):
    valid_image, valid_label = get_data_filenames(TEST_PATH)
    if not LOW_MEMORY:
        valid_image, valid_label = read_data(TEST_PATH)
    valid_dg = DataGenerator(valid_image, valid_label, batch_size=1, train=False)


    valid_pred_labels = model.predict(valid_dg)
    valid_labels = np.concatenate(
        [img_lbl_pair[1] for img_lbl_pair in valid_dg], axis=0
    )
    # _recall, _precision, _f1 = compute_avg_recall_precision_F1score(
    #     valid_labels, valid_pred_labels, thre=0.5, from_logits=False
    # )
    # print(
    #     "[Validation Data] Average Recall: {:.4f}, Average precision: {:.4f}, Average F1-score: {:.4f}".format(
    #         _recall, _precision, _f1
    #     )
    # )

    for i in range(len(valid_image)):


        img = valid_image[i]
        pred_mask = valid_pred_labels[i]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.savefig(f'./test/prediction_{i}.png')
        plt.close()
        


if __name__ == "__main__":
    weight_ckpt = r"model_weights.h5"
    # weight_ckpt = r"seg\weights\weights-teeth-boundary-model.h5"
    model = ASPP_UNet(IMG_SHAPE, filters=[16, 32, 64, 128, 256])
    model.load_weights(weight_ckpt)



    evaluate(model)

