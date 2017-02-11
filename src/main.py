# Imports
import cv2
from utils import Utils
import glob
import matplotlib.pyplot as plt

# TODO: DOUBLE CHECK THAT THE FIRST LEFT TURN IS ACTUALLY 1KM, I'M GETTING ABOUT 0.5KM
# TODO: might have to play with src & dst in warp, bridge seems to mess things up
# Instantiate utils (calibrates camera)
utils = Utils()

# # for debugging
# # read images in
# images = glob.glob('../test_images/test*.jpg')
# img_from_camera = cv2.imread('../test_images/test2.jpg')
#
# for image in images:
#     print(image)
#     image = cv2.imread(image)
#     result = utils.pipeline(image)
#     plt.imshow(result)
#     plt.show()


# read video in
video = cv2.VideoCapture('../project_video.mp4')
while video.isOpened():
    success, image = video.read()

    result = utils.pipeline(image)

    # cv2.imshow('frame', image)
    # show the frames with the lane marked
    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
