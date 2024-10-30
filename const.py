import enum
import os

import numpy as np

# tooth existence of the patient
tem_1 = np.ones((28,), np.bool_)
tem_1[6] = False
tem_3 = np.ones((28,), np.bool_)
tem_4 = np.ones((28,), np.bool_)
tem_5 = np.ones((28,), np.bool_)
tem_6 = np.ones((28,), np.bool_)
tem_7 = np.ones((28,), np.bool_)
tem_51 = np.ones((28,), np.bool_)
# TOOTH_EXIST_MASK = {"0": np.ones((28,), np.bool_), "1": np.ones((28,), np.bool_), "3": tem_3}
TOOTH_EXIST_MASK = {"0": np.ones((28,), np.bool_), "51": tem_51, "1": tem_1, "3": tem_3, "4": tem_4,"5": tem_5,"6": tem_6,"7": tem_7}



# Mask used to project the contour of the selected teeth in photos of different views
MASK_UPPER = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #17
        True,
        True,
        True,
        True,
        True,
        True,
        False, #27
        False,
        False,
        False,
        False,
        False,
        False,
        False, #37
        False,
        False,
        False,
        False,
        False,
        False,
        False, #47
    ]
)

MASK_LOWER = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False, #17
        False,
        False,
        False,
        False,
        False,
        False,
        False, #27
        True,
        True,
        True,
        True,
        True,
        True,
        False, #37
        True,
        True,
        True,
        True,
        True,
        True,
        False, #47
    ]
)

MASK_LEFT = np.array(
    [
        True,
        False,
        False,
        False,
        False,
        False,
        False, #17
        True,
        True,
        True,
        True,
        True,
        True,
        False, #27
        True,
        True,
        True,
        True,
        True,
        True,
        False, #37
        True,
        False,
        False,
        False,
        False,
        False,
        False, #47
    ]
)

MASK_RIGHT = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #17
        True,
        False,
        False,
        False,
        False,
        False,
        False, #27
        True,
        False,
        False,
        False,
        False,
        False,
        False, #37
        True,
        True,
        True,
        True,
        True,
        True,
        False, #47
    ]
)

MASK_FRONTAL = np.array(
    [
        True,
        True,
        True,
        True,
        False,
        False,
        False, #17
        True,
        True,
        True,
        True,
        False,
        False,
        False, #27
        True,
        True,
        True,
        True,
        False,
        False,
        False, #37
        True,
        True,
        True,
        True,
        False,
        False,
        False, #47
    ]
)


@enum.unique
class PHOTO(enum.Enum):
    # Enum values must be 0,1,2,3,4
    UPPER = 0
    LOWER = 1
    LEFT = 2
    RIGHT = 3
    FRONTAL = 4


PHOTO_TYPES = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
VISIBLE_MASKS = [MASK_UPPER, MASK_LOWER, MASK_LEFT, MASK_RIGHT, MASK_FRONTAL]
RECONS_IMG_WIDTH = 800

PHOTO_DIR = r"./seg/valid/image"
# PHOTO_DIR = r"./seg/valid/image"
# GT_DIR = r"./seg/valid/500RealCases1/Case415/中期阶段1"

NUM_PC = 10  # num of modes of deformation for each tooth used in reconstruction
NUM_POINT = 1500  # num of points to represent tooth surface used in SSM
PG_SHAPE = (NUM_POINT, 3)

# FDI TOOTH NUMEBRING
UPPER_INDICES = [
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
]  # ignore wisdom teeth
LOWER_INDICES = [
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
]  # ignore wisdom teeth


SSM_DIR = r"./ssm/eigValVec/"
REGIS_PARAM_DIR = r"./ssm/cpdGpParams/"
DEMO_H5_DIR = r"./demo/h5/"
DEMO_MESH_DIR = r"./demo/mesh/"
DEMO_MESH_ALIGNED_DIR = r"./demo/mesh_aligned/"
REF_MESH_DIR = r"./demo/ref_mesh/"
VIS_DIR = r"./demo/visualization"
os.makedirs(DEMO_H5_DIR, exist_ok=True)
os.makedirs(DEMO_MESH_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

