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

    def run(self, image, left_line, right_line):
        # undistort image
        undistorted_img = cv2.undistort(image, self.utils.mtx, self.utils.dist, None, self.utils.mtx)

        # perform color and gradient thresholding
        binary_img = self.utils.create_threshold_binary(undistorted_img)

        # warp image perspective, and get the inverse (for plotting later)
        top_down, Minv = self.utils.warp(binary_img)

        # window margin
        margin = 100

        # Fit a second order polynomial to each depending on whether we fit to a rectangle or not
        if right_line.detected == False | left_line.detected == False:
            print('rectangle')
            left_indices, right_indices = self.utils.find_lanes_with_rectangle(top_down, margin)
        else:
            left_indices, right_indices = self.utils.find_lanes_with_fit(top_down, left_line.current_fit, right_line.current_fit, margin)

        # Set the Radius of the curvature in meters for each line
        left_radius_curvature, right_radius_curvature = self.utils.radius_of_curve(top_down, left_indices, right_indices)

        # Find the offset from the center of the lane
        offset = self.utils.calculate_distance_to_center(top_down, left_indices, right_indices)

        # Get the polynomial from the indices, in pixels
        left_fit, right_fit = self.utils.extract_polynomial(top_down, left_indices, right_indices, False)

        # Generate x and y values based on the polynomial
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Figure out distance between the lines, in meters
        avg_distance_between_lines = np.asscalar((np.mean(np.absolute(left_fitx - right_fitx)) * self.utils.xm_per_pixel))

        # initialize booleans for sanity check
        distance_bool = False
        similar_A = False
        similar_B = False

        # Beginning of the sanity checks

        # Check to see if the lanes are between 3 & 4 meters, they should be 3.7m by regulation
        if 3. < avg_distance_between_lines < 4.:
            distance_bool = True
        else:
            print('fail, not good lane')
            distance_bool = False

        # Figure out if it's similar curve
        # take the differences in the coefficents
        similar_curve = np.absolute(left_fit - right_fit)

        # check to see if the A coefficient is similar
        if 0. < similar_curve[0] < 0.2:
            similar_A = True
        else:
            print('fail, not similar')
            similar_A = False

        # check to see if the B coefficient is similar
        if 0. < similar_curve[1] < 0.5:
            similar_B = True
        else:
            print('fail, not similar')
            similar_B = False

        # resolve sanity checks
        if distance_bool & similar_A & similar_B:
            # Because we found a line, and passed the sanity checks,
            #  set everything that will be passed back into the video loop
            left_line.detected = True
            right_line.detected = True

            left_line.recent_xfitted.append(left_fitx)
            right_line.recent_xfitted.append(right_fitx)

            left_line.recent_poly.append(left_fit)
            right_line.recent_poly.append(right_fit)

            left_line.allx = left_fitx
            left_line.ally = ploty
            right_line.allx = right_fitx
            right_line.ally = ploty

            left_line.radius_of_curvature = left_radius_curvature
            right_line.radius_of_curvature = right_radius_curvature

            left_line.line_base_pos = offset
            right_line.line_base_pos = offset

            left_line.diffs = np.absolute(left_fit - left_line.current_fit)
            right_line.diffs = np.absolute(right_fit - right_line.current_fit)

            left_line.current_fit = left_fit
            right_line.current_fit = right_fit

        elif left_line.allx is None:
            # This is in case the first frame is not good,
            # otherwise we'll throw an error
            left_line.recent_xfitted.append(left_fitx)
            right_line.recent_xfitted.append(right_fitx)

            left_line.recent_poly.append(left_fit)
            right_line.recent_poly.append(right_fit)

            left_line.allx = left_fitx
            left_line.ally = ploty
            right_line.allx = right_fitx
            right_line.ally = ploty

            left_line.radius_of_curvature = left_radius_curvature
            right_line.radius_of_curvature = right_radius_curvature

            left_line.line_base_pos = offset
            right_line.line_base_pos = offset

            left_line.diffs = np.absolute(left_fit - left_line.current_fit)
            right_line.diffs = np.absolute(right_fit - right_line.current_fit)

            left_line.current_fit = left_fit
            right_line.current_fit = right_fit

        else:
            # Sanity Checks failed,
            # Retain the previous positions and step to the next frame by not setting anything new
            left_line.detected = False
            right_line.detected = False

        # Cacluate the best fit average
        left_line.best_fit = np.mean(left_line.recent_poly, axis=0)
        right_line.best_fit = np.mean(right_line.recent_poly, axis=0)

        # Calculate the best x values average
        left_line.bestx = np.mean(left_line.recent_xfitted, axis=0)
        right_line.bestx = np.mean(right_line.recent_xfitted, axis=0)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(top_down).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line.bestx, left_line.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, right_line.ally])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

        cv2.putText(result, "Radius of Curvature: " + str(left_line.radius_of_curvature) + "m", (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 255)
        cv2.putText(result, "Offset: " + str(left_line.line_base_pos) + "m", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 255)

        # DEBUGGING: Diagnostic screen for looking at various views at once
        # middle panel text example
        # using cv2 for drawing text in diagnostic pipeline.
        # font = cv2.FONT_HERSHEY_COMPLEX
        # middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        # cv2.putText(middlepanel, 'Estimated lane curvature: ERROR!', (30, 60), font, 1, (255,0,0), 2)
        # cv2.putText(middlepanel, 'Estimated Meters right of center: ERROR!', (30, 90), font, 1, (255,0,0), 2)
        #
        # binary_img = np.dstack((binary_img, binary_img, binary_img))
        # top_down = np.dstack((top_down, top_down, top_down))
        #
        # # assemble the screen example
        # diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # diagScreen[0:720, 0:1280] = result
        # diagScreen[0:240, 1280:1600] = cv2.resize(binary_img, (320,240), interpolation=cv2.INTER_AREA)
        # diagScreen[0:240, 1600:1920] = cv2.resize(top_down, (320,240), interpolation=cv2.INTER_AREA)
        # diagScreen[240:480, 1280:1600] = cv2.resize(color_warp, (320,240), interpolation=cv2.INTER_AREA)
        # diagScreen[240:480, 1600:1920] = cv2.resize(newwarp, (320,240), interpolation=cv2.INTER_AREA)*4
        # diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
        # diagScreen[720:840, 0:1280] = middlepanel
        # diagScreen[840:1080, 0:320] = cv2.resize(diag5, (320,240), interpolation=cv2.INTER_AREA)
        # diagScreen[840:1080, 320:640] = cv2.resize(diag6, (320,240), interpolation=cv2.INTER_AREA)
        # diagScreen[840:1080, 640:960] = cv2.resize(diag9, (320,240), interpolation=cv2.INTER_AREA)
        # diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320,240), interpolation=cv2.INTER_AREA)
        return left_line, right_line, result
