import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in all the calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y


for

# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    img = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found:
    if (ret == True):
        # a) draw corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # corners that were automatically detected during the undistortion steps
        # We recommend using the automatic detection of corners in your code
        topRightCorner = nx - 1
        bottomLeftCorner = (nx * ny) - nx
        bottomRightCorner = (nx * ny) - 1
        print(corners[0])
        print(topRightCorner)
        print(corners[topRightCorner])
        print(bottomLeftCorner)
        print(corners[bottomLeftCorner])
        print(bottomRightCorner)
        print(corners[bottomRightCorner])
        print(corners.shape)
        src = np.float32([
            corners[0],
            corners[topRightCorner],
            corners[bottomLeftCorner],
            corners[bottomRightCorner]
        ])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([
            [100, 100],
            [1200, 100],
            [100, 850],
            [1200, 850]
        ])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # M = None
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        img_size = (gray.shape[1], gray.shape[0])
        print(img_size)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        # warped = img.copy()
    return warped, M


top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
