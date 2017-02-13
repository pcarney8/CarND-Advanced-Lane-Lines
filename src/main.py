# Imports
import cv2
from pipeline import Pipeline
import glob
import matplotlib.pyplot as plt
import numpy as np
from Line import Line

# TODO: DOUBLE CHECK THAT THE FIRST LEFT TURN IS ACTUALLY 1KM, I'M GETTING ABOUT 0.5KM
# TODO: might have to play with src & dst in warp, bridge seems to mess things up
# Instantiate utils (calibrates camera)
pipeline = Pipeline()
right_line = Line(10)
left_line = Line(10)
lines = [left_line, right_line]
counter = 0
max_lost_frames = 2
lost_frame = 0

# # for debugging
# # read images in
# images = glob.glob('../video_output/frame*.jpg')
# img_from_camera = cv2.imread('../test_images/test2.jpg')
#
# for image in images:
#     print(image)
#     image = cv2.imread(image)
#     left_line, right_line, result = pipeline.run(image, left_line, right_line)
#
#     if left_line.detected == False | right_line.detected == False:
#         lost_frame += 1
#     else:
#         lost_frame = 0
#
#     # print('lost_frame:', lost_frame, 'offset:', left_line.line_base_pos, 'left radius:', left_line.radius_of_curvature, 'right radius:', right_line.radius_of_curvature)
#
#     if lost_frame < max_lost_frames:
#         left_line.detected = True
#         right_line.detected = True
#     else:
#         left_line.detected = False
#         right_line.detected = False
#
#     result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#     plt.imshow(result)
#     plt.show()


counter = 0
# read video in
video = cv2.VideoCapture('../project_video.mp4')
while video.isOpened():
    success, image = video.read()

    left_line, right_line, result = pipeline.run(image, left_line, right_line)

    if left_line.detected == False | right_line.detected == False:
        lost_frame += 1
    else:
        lost_frame = 0

    print('lost_frame:', lost_frame,
          'offset:', left_line.line_base_pos,
          'left radius:', left_line.radius_of_curvature,
          'right radius:', right_line.radius_of_curvature
          )

    if lost_frame < max_lost_frames:
        left_line.detected = True
        right_line.detected = True
    else:
        left_line.detected = False
        right_line.detected = False

    # show the frames with the lane marked
    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
