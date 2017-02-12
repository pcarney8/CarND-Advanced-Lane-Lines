# Imports
import cv2
from pipeline import Pipeline
import glob
import matplotlib.pyplot as plt
from Line import Line

# TODO: DOUBLE CHECK THAT THE FIRST LEFT TURN IS ACTUALLY 1KM, I'M GETTING ABOUT 0.5KM
# TODO: might have to play with src & dst in warp, bridge seems to mess things up
# Instantiate utils (calibrates camera)
pipeline = Pipeline()
right_line = Line()
left_line = Line()

# for debugging
# read images in
images = glob.glob('../test_images/test*.jpg')
img_from_camera = cv2.imread('../test_images/test2.jpg')

for image in images:
    print(image)
    image = cv2.imread(image)
    left_line, right_line, result = pipeline.run(image, left_line, right_line)
    print(left_line.radius_of_curvature, 'm', right_line.radius_of_curvature, 'm')
    print('offset from center: ', left_line.line_base_pos, 'm')
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.show()


# # read video in
# video = cv2.VideoCapture('../project_video.mp4')
# while video.isOpened():
#     success, image = video.read()
#
#     result = utils.pipeline(image)
#
#     # cv2.imshow('frame', image)
#     # show the frames with the lane marked
#     cv2.imshow('frame', result)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video.release()
# cv2.destroyAllWindows()

    # Fit a second order polynomial to each
    # with warnings.catch_warnings():
    #     try:
    #         left_fit, right_fit = utils.find_lanes_with_fit(img, left_fit, right_fit, margin)
    #     except (np.RankWarning, UnboundLocalError):
    #         left_fit, right_fit = utils.find_lanes_with_rectangle(img, margin)