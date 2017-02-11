# Imports
import cv2
from utils import Utils
import matplotlib.pyplot as plt

# Pipeline
utils = Utils()
# Calibrate Camera
ret, mtx, dist, rvecs, tvecs = utils.calibrate_camera()

# read stuff in
img_from_camera = cv2.imread('../test_images/test2.jpg')

# undistort image
undistorted_img = cv2.undistort(img_from_camera, mtx, dist, None, mtx)

# perform color and gradient thresholding
binary_img = utils.create_threshold_binary(undistorted_img)

# warp image perspective
top_down, perspective_M = utils.warp(binary_img, mtx, dist)

# window margin
margin = 100

# Fit a second order polynomial to each
left_fit, right_fit = utils.find_lanes_with_rectangle(top_down, margin)

utils.find_lanes_with_fit(top_down, left_fit, right_fit, margin)

# Fit a second order polynomial to each
# with warnings.catch_warnings():
#     try:
#         left_fit, right_fit = utils.find_lanes_with_fit(img, left_fit, right_fit, margin)
#     except (np.RankWarning, UnboundLocalError):
#         left_fit, right_fit = utils.find_lanes_with_rectangle(img, margin)