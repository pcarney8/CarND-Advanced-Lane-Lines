# Imports
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from Line import Line
from utils import Utils


class Pipeline:
    def __init__(self):
        self.utils = Utils()

    def run(self, image, right_line, left_line, fit_with_rectangle=True):
        # undistort image
        undistorted_img = cv2.undistort(image, self.utils.mtx, self.utils.dist, None, self.utils.mtx)

        # perform color and gradient thresholding
        binary_img = self.utils.create_threshold_binary(undistorted_img)

        # warp image perspective
        top_down, perspective_M = self.utils.warp(binary_img)
        inverse_img, Minv = self.utils.warp(binary_img, True)

        # window margin
        margin = 100

        # Fit a second order polynomial to each depending on whether we fit to a rectangle or not
        if fit_with_rectangle:
            left_line.recent_xfitted, right_line.recent_xfitted = self.utils.find_lanes_with_rectangle(top_down, margin)
        else:
            left_line.recent_xfitted, right_line.recent_xfitted = self.utils.find_lanes_with_fit(top_down, left_line.current_fit, right_line.current_fit, margin)

        left_line.radius_of_curvature, right_line.radius_of_curvature = self.utils.radius_of_curve(top_down, left_line.recent_xfitted, right_line.recent_xfitted)

        offset = self.utils.calculate_distance_to_center(top_down, left_line.recent_xfitted, right_line.recent_xfitted)
        left_line.line_base_pos = offset
        right_line.line_base_pos = offset

        left_line.current_fit, right_line.current_fit = self.utils.extract_polynomial(top_down, left_line.recent_xfitted, right_line.recent_xfitted, False)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        left_fitx = left_line.current_fit[0]*ploty**2 + left_line.current_fit[1]*ploty + left_line.current_fit[2]
        right_fitx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(top_down).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        # TODO: should this be the undistorted image or the raw one?
        result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

        return left_line, right_line, result
