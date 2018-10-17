
# code from
# https://github.com/vamsiramakrishnan/AdvancedLaneLines/blob/master/Advanced_Lane_lines_Final.ipynb


import glob
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.utils import resample
from scipy.optimize import curve_fit
import skimage.filters as filters
from collections import deque
from math import ceil
warped_size = np.array([1280, 720])
original_size =np.array([720, 1280])
OFFSET =0


isPerspectiveCompute=0;
##################################

# Calculate Source and Destination points
def calc_warp_points():
    """
    :return: Source and Destination pointts
    """
    src = 4*np.float32 ([
        [220, 651],
        [350, 577],
        [828, 577],
        [921, 651]
    ])

    dst = 4*np.float32 ([
            [220, 651],
            [220, 577],
            [921, 577],
            [921, 651]
        ])
    return src, dst


# Calculate Transform
def calc_transform(src_, dst_):
    """
    Calculate Perspective and Inverse Perspective Transform Matrices
    :param src_: Source points
    :param dst_: Destination Points
    :return: Perspective Matrix and Inverse Perspective Transform Matrix
    """
    M_ = cv2.getPerspectiveTransform(src_, dst_)
    Minv_ = cv2.getPerspectiveTransform(dst_, src_)
    return M_, Minv_


# Get perspective transform
def perspective_transform(M_, img_):
    """

    :param M_: Perspective Matrix
    :param img_ : Input Image
    :return: Transformed Image
    """
    img_size = (img_.shape[1],img_.shape[0])
    transformed = cv2.warpPerspective(
        img_,
        M_, img_size,
        flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    return transformed


# Inverse Perspective Transform
def inv_perspective_transform(Minv_, img_):
    """

    :param M_: Inverse Perspective Transform Matrix
    :param img_: Input Image
    :return: Transformed Image
    """
    img_size = (img_.shape[1], img_.shape[0])
    transformed = cv2.warpPerspective(
        img_,
        Minv_, img_size,
        flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    return transformed

    #####################################


    # Calculate Bird' Eye Transform #
if not isPerspectiveCompute:
    src_, dst_ = calc_warp_points()
    m, minv = calc_transform(src_, dst_)
    isPerspectiveCompute = True
undistorted_img = cv2.imread('images/IMG-2402.JPG',0)

    # Get Bird's Eye View #
warped = perspective_transform(m, undistorted_img)
resized_image = cv2.resize(warped, (2000, 2000))

cv2.imshow('detected circles',resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
