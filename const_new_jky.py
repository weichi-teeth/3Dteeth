import enum
import os

import numpy as np

# tooth existence of the patient
tem_1 = np.ones((28,), np.bool_)
# tem_1[6] = False
# tem_1[13] = False
# tem_1[20] = False
# tem_1[27] = False
tem_83 = np.ones((28,), np.bool_)
tem_100 = np.ones((28,), np.bool_)
tem_108 = np.ones((28,), np.bool_)
tem_108[22] = False
tem_125 = np.ones((28,), np.bool_)
tem_134 = np.ones((28,), np.bool_)
tem_143 = np.ones((28,), np.bool_)
tem_402 = np.ones((28,), np.bool_)
tem_402[6] = False
tem_402[13] = False
tem_402[20] = False
tem_402[27] = False
tem_404 = np.ones((28,), np.bool_)
tem_418 = np.ones((28,), np.bool_)
tem_424 = np.ones((28,), np.bool_)
tem_432 = np.ones((28,), np.bool_)

tem_3 = np.ones((28,), np.bool_)
tem_4 = np.ones((28,), np.bool_)
tem_5 = np.ones((28,), np.bool_)
tem_6 = np.ones((28,), np.bool_)
tem_7 = np.ones((28,), np.bool_)
tem_9 = np.ones((28,), np.bool_)
tem_184 = np.ones((28,), np.bool_)
tem_218 = np.ones((28,), np.bool_)
tem_261 = np.ones((28,), np.bool_)
tem_269 = np.ones((28,), np.bool_)
tem_19 = np.ones((28,), np.bool_)
tem_16 = np.ones((28,), np.bool_)
tem_51 = np.ones((28,), np.bool_)
tem_54 = np.ones((28,), np.bool_)
tem_89 = np.ones((28,), np.bool_)
tem_419 = np.ones((28,), np.bool_)
tem_419[6] = False
tem_419[13] = False
tem_427 = np.ones((28,), np.bool_)
tem_427[6] = False
tem_427[13] = False
tem_427[27] = False
tem_437 = np.ones((28,), np.bool_)
tem_457 = np.ones((28,), np.bool_)
tem_465 = np.ones((28,), np.bool_)
tem_jiajun = np.ones((28,), np.bool_)
# TOOTH_EXIST_MASK = {"0": np.ones((28,), np.bool_), "1": np.ones((28,), np.bool_), "3": tem_3}
TOOTH_EXIST_MASK = {"0": np.ones((28,), np.bool_), "1": tem_1, "3": tem_3, "4": tem_4, "5": tem_5, "6": tem_6, "7": tem_7, "9": tem_9, "19": tem_19, "16": tem_16, "218": tem_218, "261": tem_261, "269": tem_269, "184": tem_184, "83": tem_83, "100": tem_100, "108": tem_108, "125": tem_125, "134": tem_134, "143": tem_143, "402": tem_402, "404": tem_404, "418": tem_418, "424": tem_424, "432": tem_432, "51": tem_51, "54": tem_54, "89": tem_89, "419": tem_419, "427": tem_427, "437": tem_437, "457": tem_457, "465": tem_465, "jiajun": tem_jiajun}



# Mask used to project the contour of the selected teeth in photos of different views
MASK_UPPER_4 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_4 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_4 = np.array(
    [
        True,
        True,
        True,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        True, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_4 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True, #7
        True,
        True,
        True,
        False,
        False,
        False,
        False, #14
        True,
        True,
        True,
        False,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_FRONTAL_4 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)


MASK_UPPER_7 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_7 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_7 = np.array(
    [
        True,
        False,
        False,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_7 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        False,
        False,
        False,
        False,
        False,
        False,
        False, #14
        True,
        False,
        False,
        False,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL_7 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)



MASK_UPPER_19 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_19 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_19 = np.array(
    [
        True,
        True,
        True,
        True,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        True, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_RIGHT_19= np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_FRONTAL_19 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_UPPER_9 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_9 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_9 = np.array(
    [
        True,
        True,
        True,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_9= np.array(
    [
        True,
        True,
        True,
        True,
        True,
        False,
        False, #7
        True,
        True,
        True,
        False,
        False,
        False,
        False, #14
        True,
        True,
        True,
        True,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL_9 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)
MASK_UPPER_16 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_16 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_16 = np.array(
    [
        True,
        True,
        True,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_16= np.array(
    [
        True,
        True,
        True,
        True,
        True,
        False,
        False, #7
        True,
        True,
        True,
        False,
        False,
        False,
        False, #14
        True,
        True,
        True,
        True,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL_16 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_UPPER_218 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_218 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_218 = np.array(
    [
        True,
        True,
        True,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        False, #21
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_218 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True, #7
        True,
        True,
        True,
        False,
        False,
        False,
        False, #14
        True,
        True,
        True,
        False,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL_218 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
)

MASK_UPPER_261 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,#14
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_261 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_261 = np.array(
    [
        True,
        True,
        False,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        False, #21
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_261 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        True,
        True,
        False,
        False,
        False,
        False, #14
        True,
        True,
        True,
        False,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL_261 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        True, #21
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_UPPER_269 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,#14
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_269 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
)

MASK_LEFT_269 = np.array(
    [
        True,
        False,
        False,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        False, #21
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_269 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        False,
        False,
        False,
        False,
        False,
        False, #14
        True,
        False,
        False,
        False,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL_269 = np.array(
    [
        True,
        True,
        True,
        True,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        False,
        False,
        False, #14
        True,
        True,
        True,
        True,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
)

MASK_UPPER_184 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,#14
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_LOWER_184 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,#14
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_LEFT_184 = np.array(
    [
        False,
        False,
        False,
        False,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        True,
        True,
        False, #14
        True,
        True,
        True,
        True,
        True,
        True,
        False, #21
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
)

MASK_RIGHT_184 = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        False, #7
        True,
        False,
        False,
        False,
        False,
        False,
        False, #14
        True,
        True,
        False,
        False,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
)

MASK_FRONTAL_184 = np.array(
    [
        True,
        True,
        True,
        True,
        False,
        False,
        False, #7
        True,
        True,
        True,
        True,
        False,
        False,
        False, #14
        True,
        True,
        True,
        True,
        False,
        False,
        False, #21
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
)

MASK_UPPER_1 = np.array(
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

MASK_LOWER_1 = np.array(
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

MASK_LEFT_1 = np.array(
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

MASK_RIGHT_1 = np.array(
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

MASK_FRONTAL_1 = np.array(
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

MASK_UPPER_83 = MASK_UPPER_1.copy()
MASK_LOWER_83 = MASK_LOWER_1.copy()
MASK_LEFT_83 = MASK_LEFT_1.copy()
MASK_RIGHT_83 = MASK_RIGHT_1.copy()
MASK_FRONTAL_83 = MASK_FRONTAL_1.copy()

MASK_UPPER_100 = MASK_UPPER_1.copy()
MASK_LOWER_100 = MASK_LOWER_1.copy()
MASK_LEFT_100 = MASK_LEFT_1.copy()
MASK_LEFT_100[0] = False
MASK_RIGHT_100 = MASK_RIGHT_1.copy()
MASK_FRONTAL_100 = MASK_FRONTAL_1.copy()

MASK_UPPER_108 = MASK_UPPER_1.copy()
MASK_LOWER_108 = MASK_LOWER_1.copy()
MASK_LOWER_108[22] = False
MASK_LEFT_108 = MASK_LEFT_1.copy()
MASK_RIGHT_108 = MASK_RIGHT_1.copy()
MASK_RIGHT_108[22] = False
MASK_FRONTAL_108 = MASK_FRONTAL_1.copy()
MASK_FRONTAL_108[22] = False

MASK_UPPER_125 = MASK_UPPER_1.copy()
MASK_LOWER_125 = MASK_LOWER_1.copy()
MASK_LEFT_125 = MASK_LEFT_1.copy()
MASK_RIGHT_125 = MASK_RIGHT_1.copy()
MASK_FRONTAL_125 = MASK_FRONTAL_1.copy()

MASK_UPPER_134 = MASK_UPPER_1.copy()
MASK_LOWER_134 = MASK_LOWER_1.copy()
MASK_LEFT_134 = MASK_LEFT_1.copy()
MASK_LEFT_134[13] = False
MASK_RIGHT_134 = MASK_RIGHT_1.copy()
MASK_RIGHT_134[6] = False
MASK_FRONTAL_134 = MASK_FRONTAL_1.copy()

MASK_UPPER_143 = MASK_UPPER_1.copy()
MASK_LOWER_143 = MASK_LOWER_1.copy()
MASK_LEFT_143 = MASK_LEFT_1.copy()
MASK_RIGHT_143 = MASK_RIGHT_1.copy()
MASK_FRONTAL_143 = MASK_FRONTAL_1.copy()

MASK_UPPER_402 = MASK_UPPER_1.copy()
MASK_LOWER_402 = MASK_LOWER_1.copy()
MASK_LEFT_402 = MASK_LEFT_1.copy()
MASK_RIGHT_402 = MASK_RIGHT_1.copy()
MASK_FRONTAL_402 = MASK_FRONTAL_1.copy()

MASK_UPPER_404 = MASK_UPPER_1.copy()
MASK_LOWER_404 = MASK_LOWER_1.copy()
MASK_LEFT_404 = MASK_LEFT_1.copy()
MASK_RIGHT_404 = MASK_RIGHT_1.copy()
MASK_FRONTAL_404 = MASK_FRONTAL_1.copy()

MASK_UPPER_418 = MASK_UPPER_1.copy()
MASK_LOWER_418 = MASK_LOWER_1.copy()
MASK_LEFT_418 = MASK_LEFT_1.copy()
MASK_RIGHT_418 = MASK_RIGHT_1.copy()
MASK_FRONTAL_418 = MASK_FRONTAL_1.copy()

MASK_UPPER_424 = MASK_UPPER_1.copy()
MASK_LOWER_424 = MASK_LOWER_1.copy()
MASK_LEFT_424 = MASK_LEFT_1.copy()
MASK_RIGHT_424 = MASK_RIGHT_1.copy()
MASK_FRONTAL_424 = MASK_FRONTAL_1.copy()

MASK_UPPER_432 = MASK_UPPER_1.copy()
MASK_LOWER_432 = MASK_LOWER_1.copy()
MASK_LEFT_432 = MASK_LEFT_1.copy()
MASK_RIGHT_432 = MASK_RIGHT_1.copy()
MASK_FRONTAL_432 = MASK_FRONTAL_1.copy()

MASK_UPPER_51 = MASK_UPPER_1.copy()
MASK_LOWER_51 = MASK_LOWER_1.copy()
MASK_LEFT_51 = MASK_LEFT_1.copy()
MASK_RIGHT_51 = MASK_RIGHT_1.copy()
MASK_FRONTAL_51 = MASK_FRONTAL_1.copy()

MASK_UPPER_54 = MASK_UPPER_1.copy()
MASK_LOWER_54 = MASK_LOWER_1.copy()
MASK_LEFT_54 = MASK_LEFT_1.copy()
MASK_RIGHT_54 = MASK_RIGHT_1.copy()
MASK_FRONTAL_54 = MASK_FRONTAL_1.copy()

MASK_UPPER_89 = MASK_UPPER_1.copy()
MASK_LOWER_89 = MASK_LOWER_1.copy()
MASK_LEFT_89 = MASK_LEFT_1.copy()
MASK_RIGHT_89 = MASK_RIGHT_1.copy()
MASK_FRONTAL_89 = MASK_FRONTAL_1.copy()

MASK_UPPER_419 = MASK_UPPER_1.copy()
MASK_LOWER_419 = MASK_LOWER_1.copy()
MASK_LEFT_419 = MASK_LEFT_1.copy()
MASK_RIGHT_419 = MASK_RIGHT_1.copy()
MASK_FRONTAL_419 = MASK_FRONTAL_1.copy()

MASK_UPPER_427 = MASK_UPPER_1.copy()
MASK_LOWER_427 = MASK_LOWER_1.copy()
MASK_LEFT_427 = MASK_LEFT_1.copy()
MASK_RIGHT_427 = MASK_RIGHT_1.copy()
MASK_FRONTAL_427 = MASK_FRONTAL_1.copy()

MASK_UPPER_437 = MASK_UPPER_1.copy()
MASK_LOWER_437 = MASK_LOWER_1.copy()
MASK_LEFT_437 = MASK_LEFT_1.copy()
MASK_RIGHT_437 = MASK_RIGHT_1.copy()
MASK_FRONTAL_437 = MASK_FRONTAL_1.copy()

MASK_UPPER_457 = MASK_UPPER_1.copy()
MASK_LOWER_457 = MASK_LOWER_1.copy()
MASK_LEFT_457 = MASK_LEFT_1.copy()
MASK_RIGHT_457 = MASK_RIGHT_1.copy()
MASK_FRONTAL_457 = MASK_FRONTAL_1.copy()

MASK_UPPER_465 = MASK_UPPER_1.copy()
MASK_LOWER_465 = MASK_LOWER_1.copy()
MASK_LEFT_465 = MASK_LEFT_1.copy()
MASK_RIGHT_465 = MASK_RIGHT_1.copy()
MASK_FRONTAL_465 = MASK_FRONTAL_1.copy()

MASK_UPPER_jiajun = MASK_UPPER_1.copy()
MASK_LOWER_jiajun = MASK_LOWER_1.copy()
MASK_LEFT_jiajun = MASK_LEFT_1.copy()
MASK_RIGHT_jiajun = MASK_RIGHT_1.copy()
MASK_FRONTAL_jiajun = MASK_FRONTAL_1.copy()

MASK_UPPER = {"1": MASK_UPPER_1,"4": MASK_UPPER_4, "7": MASK_UPPER_7, "9": MASK_UPPER_9, "19": MASK_UPPER_19, "16": MASK_UPPER_16, "218": MASK_UPPER_218, "261": MASK_UPPER_261, "269": MASK_UPPER_269, "184": MASK_UPPER_184, "83": MASK_UPPER_83, "100": MASK_UPPER_100, "108": MASK_UPPER_108, "125": MASK_UPPER_125, "134": MASK_UPPER_134, "143": MASK_UPPER_143, "402": MASK_UPPER_402, "404": MASK_UPPER_404, "418": MASK_UPPER_418, "424": MASK_UPPER_424, "432": MASK_UPPER_432, "51": MASK_UPPER_51, "54": MASK_UPPER_54, "89": MASK_UPPER_89, "419": MASK_UPPER_419, "427": MASK_UPPER_427, "437": MASK_UPPER_437, "457": MASK_UPPER_457, "465": MASK_UPPER_465, "jiajun": MASK_UPPER_jiajun}
MASK_LOWER = {"1": MASK_LOWER_1,"4": MASK_LOWER_4, "7": MASK_LOWER_7, "9": MASK_LOWER_9, "19": MASK_LOWER_19, "16": MASK_LOWER_16, "218": MASK_LOWER_218, "261": MASK_LOWER_261, "269": MASK_LOWER_269, "184": MASK_LOWER_184, "83": MASK_LOWER_83, "100": MASK_LOWER_100, "108": MASK_LOWER_108, "125": MASK_LOWER_125, "134": MASK_LOWER_134, "143": MASK_LOWER_143, "402": MASK_LOWER_402, "404": MASK_LOWER_404, "418": MASK_LOWER_418, "424": MASK_LOWER_424, "432": MASK_LOWER_432, "51": MASK_LOWER_51, "54": MASK_LOWER_54, "89": MASK_LOWER_89, "419": MASK_LOWER_419, "427": MASK_LOWER_427, "437": MASK_LOWER_437, "457": MASK_LOWER_457, "465": MASK_LOWER_465, "jiajun": MASK_LOWER_jiajun}
MASK_LEFT = {"1": MASK_LEFT_1,"4": MASK_LEFT_4, "7": MASK_LEFT_7, "9": MASK_LEFT_9, "19": MASK_LEFT_19, "16": MASK_LEFT_16, "218": MASK_LEFT_218, "261": MASK_LEFT_261, "269": MASK_LEFT_269, "184": MASK_LEFT_184, "83": MASK_LEFT_83, "100": MASK_LEFT_100, "108": MASK_LEFT_108, "125": MASK_LEFT_125, "134": MASK_LEFT_134, "143": MASK_LEFT_143, "402": MASK_LEFT_402, "404": MASK_LEFT_404, "418": MASK_LEFT_418, "424": MASK_LEFT_424, "432": MASK_LEFT_432, "51": MASK_LEFT_51, "54": MASK_LEFT_54, "89": MASK_LEFT_89, "419": MASK_LEFT_419, "427": MASK_LEFT_427, "437": MASK_LEFT_437, "457": MASK_LEFT_457, "465": MASK_LEFT_465, "jiajun": MASK_LEFT_jiajun}
MASK_RIGHT = {"1": MASK_RIGHT_1,"4": MASK_RIGHT_4, "7": MASK_RIGHT_7, "9": MASK_RIGHT_9, "19": MASK_RIGHT_19, "16": MASK_RIGHT_16, "218": MASK_RIGHT_218, "261": MASK_RIGHT_261, "269": MASK_RIGHT_269, "184": MASK_RIGHT_184, "83": MASK_RIGHT_83, "100": MASK_RIGHT_100, "108": MASK_RIGHT_108, "125": MASK_RIGHT_125, "134": MASK_RIGHT_134, "143": MASK_RIGHT_143, "402": MASK_RIGHT_402, "404": MASK_RIGHT_404, "418": MASK_RIGHT_418, "424": MASK_RIGHT_424, "432": MASK_RIGHT_432, "51": MASK_RIGHT_51, "54": MASK_RIGHT_54, "89": MASK_RIGHT_89, "419": MASK_RIGHT_419, "427": MASK_RIGHT_427, "437": MASK_RIGHT_437, "457": MASK_RIGHT_457, "465": MASK_RIGHT_465, "jiajun": MASK_RIGHT_jiajun}
MASK_FRONTAL = {"1": MASK_FRONTAL_1,"4": MASK_FRONTAL_4, "7": MASK_FRONTAL_7, "9": MASK_FRONTAL_9, "19": MASK_FRONTAL_19, "16": MASK_FRONTAL_16, "218": MASK_FRONTAL_218, "261": MASK_FRONTAL_261, "269": MASK_FRONTAL_269, "184": MASK_FRONTAL_184, "83": MASK_FRONTAL_83, "100": MASK_FRONTAL_100, "108": MASK_FRONTAL_108, "125": MASK_FRONTAL_125, "134": MASK_FRONTAL_134, "143": MASK_FRONTAL_143, "402": MASK_FRONTAL_402, "404": MASK_FRONTAL_404, "418": MASK_FRONTAL_418, "424": MASK_FRONTAL_424, "432": MASK_FRONTAL_432, "51": MASK_FRONTAL_51, "54": MASK_FRONTAL_54, "89": MASK_FRONTAL_89, "419": MASK_FRONTAL_419, "427": MASK_FRONTAL_427, "437": MASK_FRONTAL_437, "457": MASK_FRONTAL_457, "465": MASK_FRONTAL_465, "jiajun": MASK_FRONTAL_jiajun}

VISIBLE_MASKS_1 = [MASK_UPPER["1"], MASK_LOWER["1"], MASK_LEFT["1"], MASK_RIGHT["1"], MASK_FRONTAL["1"]]
VISIBLE_MASKS_4 = [MASK_UPPER["4"], MASK_LOWER["4"], MASK_LEFT["4"], MASK_RIGHT["4"], MASK_FRONTAL["4"]]
VISIBLE_MASKS_7 = [MASK_UPPER["7"], MASK_LOWER["7"], MASK_LEFT["7"], MASK_RIGHT["7"], MASK_FRONTAL["7"]]
VISIBLE_MASKS_9 = [MASK_UPPER["9"], MASK_LOWER["9"], MASK_LEFT["9"], MASK_RIGHT["9"], MASK_FRONTAL["9"]]
VISIBLE_MASKS_19 = [MASK_UPPER["19"], MASK_LOWER["19"], MASK_LEFT["19"], MASK_RIGHT["19"], MASK_FRONTAL["19"]]
VISIBLE_MASKS_16 = [MASK_UPPER["16"], MASK_LOWER["16"], MASK_LEFT["16"], MASK_RIGHT["16"], MASK_FRONTAL["16"]]
VISIBLE_MASKS_218 = [MASK_UPPER["218"], MASK_LOWER["218"], MASK_LEFT["218"], MASK_RIGHT["218"], MASK_FRONTAL["218"]]
VISIBLE_MASKS_261 = [MASK_UPPER["261"], MASK_LOWER["261"], MASK_LEFT["261"], MASK_RIGHT["261"], MASK_FRONTAL["261"]]
VISIBLE_MASKS_269 = [MASK_UPPER["269"], MASK_LOWER["269"], MASK_LEFT["269"], MASK_RIGHT["269"], MASK_FRONTAL["269"]]
VISIBLE_MASKS_184 = [MASK_UPPER["184"], MASK_LOWER["184"], MASK_LEFT["184"], MASK_RIGHT["184"], MASK_FRONTAL["184"]]
VISIBLE_MASKS_83 = [MASK_UPPER["83"], MASK_LOWER["83"], MASK_LEFT["83"], MASK_RIGHT["83"], MASK_FRONTAL["83"]]
VISIBLE_MASKS_100 = [MASK_UPPER["100"], MASK_LOWER["100"], MASK_LEFT["100"], MASK_RIGHT["100"], MASK_FRONTAL["100"]]
VISIBLE_MASKS_108 = [MASK_UPPER["108"], MASK_LOWER["108"], MASK_LEFT["108"], MASK_RIGHT["108"], MASK_FRONTAL["108"]]
VISIBLE_MASKS_125 = [MASK_UPPER["125"], MASK_LOWER["125"], MASK_LEFT["125"], MASK_RIGHT["125"], MASK_FRONTAL["125"]]
VISIBLE_MASKS_134 = [MASK_UPPER["134"], MASK_LOWER["134"], MASK_LEFT["134"], MASK_RIGHT["134"], MASK_FRONTAL["134"]]
VISIBLE_MASKS_143 = [MASK_UPPER["143"], MASK_LOWER["143"], MASK_LEFT["143"], MASK_RIGHT["143"], MASK_FRONTAL["143"]]
VISIBLE_MASKS_402 = [MASK_UPPER["402"], MASK_LOWER["402"], MASK_LEFT["402"], MASK_RIGHT["402"], MASK_FRONTAL["402"]]
VISIBLE_MASKS_404 = [MASK_UPPER["404"], MASK_LOWER["404"], MASK_LEFT["404"], MASK_RIGHT["404"], MASK_FRONTAL["404"]]
VISIBLE_MASKS_418 = [MASK_UPPER["418"], MASK_LOWER["418"], MASK_LEFT["418"], MASK_RIGHT["418"], MASK_FRONTAL["418"]]
VISIBLE_MASKS_424 = [MASK_UPPER["424"], MASK_LOWER["424"], MASK_LEFT["424"], MASK_RIGHT["424"], MASK_FRONTAL["424"]]
VISIBLE_MASKS_432 = [MASK_UPPER["432"], MASK_LOWER["432"], MASK_LEFT["432"], MASK_RIGHT["432"], MASK_FRONTAL["432"]]
VISIBLE_MASKS_51 = [MASK_UPPER["51"], MASK_LOWER["51"], MASK_LEFT["51"], MASK_RIGHT["51"], MASK_FRONTAL["51"]]
VISIBLE_MASKS_54 = [MASK_UPPER["54"], MASK_LOWER["54"], MASK_LEFT["54"], MASK_RIGHT["54"], MASK_FRONTAL["54"]]
VISIBLE_MASKS_89 = [MASK_UPPER["89"], MASK_LOWER["89"], MASK_LEFT["89"], MASK_RIGHT["89"], MASK_FRONTAL["89"]]
VISIBLE_MASKS_419 = [MASK_UPPER["419"], MASK_LOWER["419"], MASK_LEFT["419"], MASK_RIGHT["419"], MASK_FRONTAL["419"]]
VISIBLE_MASKS_427 = [MASK_UPPER["427"], MASK_LOWER["427"], MASK_LEFT["427"], MASK_RIGHT["427"], MASK_FRONTAL["427"]]
VISIBLE_MASKS_437 = [MASK_UPPER["437"], MASK_LOWER["437"], MASK_LEFT["437"], MASK_RIGHT["437"], MASK_FRONTAL["437"]]
VISIBLE_MASKS_457 = [MASK_UPPER["457"], MASK_LOWER["457"], MASK_LEFT["457"], MASK_RIGHT["457"], MASK_FRONTAL["457"]]
VISIBLE_MASKS_465 = [MASK_UPPER["465"], MASK_LOWER["465"], MASK_LEFT["465"], MASK_RIGHT["465"], MASK_FRONTAL["465"]]
VISIBLE_MASKS_jiajun = [MASK_UPPER["jiajun"], MASK_LOWER["jiajun"], MASK_LEFT["jiajun"], MASK_RIGHT["jiajun"], MASK_FRONTAL["jiajun"]]


VISIBLE_MASKS = {"1": VISIBLE_MASKS_1, "184": VISIBLE_MASKS_184, "4": VISIBLE_MASKS_4, "7": VISIBLE_MASKS_7,  "9": VISIBLE_MASKS_9,  "19": VISIBLE_MASKS_19,  "16": VISIBLE_MASKS_16,  "218": VISIBLE_MASKS_218,  "261": VISIBLE_MASKS_261,  "269": VISIBLE_MASKS_269, "83": VISIBLE_MASKS_83, "100": VISIBLE_MASKS_100, "108": VISIBLE_MASKS_108, "125": VISIBLE_MASKS_125, "134": VISIBLE_MASKS_134, "143": VISIBLE_MASKS_143, "402": VISIBLE_MASKS_402, "404": VISIBLE_MASKS_404, "418": VISIBLE_MASKS_418, "424": VISIBLE_MASKS_424, "432": VISIBLE_MASKS_432, "51": VISIBLE_MASKS_51, "54": VISIBLE_MASKS_54, "89": VISIBLE_MASKS_89, "419": VISIBLE_MASKS_419, "427": VISIBLE_MASKS_427, "437": VISIBLE_MASKS_437, "457": VISIBLE_MASKS_457, "465": VISIBLE_MASKS_465, "jiajun": VISIBLE_MASKS_jiajun}


@enum.unique
class PHOTO(enum.Enum):
    # Enum values must be 0,1,2,3,4
    UPPER = 0
    LOWER = 1
    LEFT = 2
    RIGHT = 3
    FRONTAL = 4


PHOTO_TYPES = [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
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
DEMO_MESH_DIR2 = r"./demo/mesh2/"
DEMO_MESH_ALIGNED_DIR = r"./demo/mesh_aligned/"
REF_MESH_DIR = r"./demo/ref_mesh/"
VIS_DIR = r"./demo/visualization"
os.makedirs(DEMO_H5_DIR, exist_ok=True)
os.makedirs(DEMO_MESH_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

