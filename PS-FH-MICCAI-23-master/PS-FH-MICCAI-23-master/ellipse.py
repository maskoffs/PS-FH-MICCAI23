import cv2
import math
import numpy as np


def drawline_AOD(element_, element_1):
    element = (element_[0], (element_[1][1], element_[1][0]), element_[2] - 90)
    element1 = (element_1[0], (element_1[1][1], element_1[1][0]), element_1[2] - 90)

    [d11, d12] = [element1[0][0] - element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] - element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    [d21, d22] = [element1[0][0] + element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] + element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    # cv2.line(background, (round(d11), round(d12)), (round(d21), round(d22)), (255, 255, 255), 2)
    a = element[1][0] / 2
    b = element[1][1] / 2
    angel = 2 * math.pi * element[2] / 360
    dp21 = d21 - element[0][0]
    dp22 = d22 - element[0][1]

    dp2 = np.array([[dp21], [dp22]])
    Transmat1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                          [math.sin(-angel), math.cos(-angel)]])
    Transmat2 = np.array([[math.cos(angel), -math.sin(angel)],
                          [math.sin(angel), math.cos(angel)]])
    dpz2 = Transmat1 @ dp2
    dpz21 = dpz2[0][0]
    dpz22 = dpz2[1][0]
    if dpz21 ** 2 - a ** 2 == 0:
        dpz21 += 1
    if (b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2) >= 0:
        xielv_aod = (dpz21 * dpz22 - math.sqrt(b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2)) / (
                dpz21 ** 2 - a ** 2)
    else:
        xielv_aod = 0
    bias_aod = dpz22 - xielv_aod * dpz21
    qiepz1 = (-2 * xielv_aod * bias_aod / b ** 2) / (2 * (1 / a ** 2 + xielv_aod ** 2 / b ** 2))
    qiepz2 = qiepz1 * xielv_aod + bias_aod
    qiepz = np.array([[qiepz1], [qiepz2]])
    qiep = list(Transmat2 @ qiepz)
    qie1 = qiep[0][0] + element[0][0]
    qie2 = qiep[1][0] + element[0][1]

    ld1d3 = math.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2)
    ld3x4 = math.sqrt((d21 - qie1) ** 2 + (d22 - qie2) ** 2)
    ld1x4 = math.sqrt((d11 - qie1) ** 2 + (d12 - qie2) ** 2)

    aod = math.acos((ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / (2 * ld1d3 * ld3x4)) / math.pi * 180  ##余弦定理
    # cv2.line(background, (round(d21), round(d22)), (int(qie1), int(qie2)), (255, 255, 255), 2)
    return aod
