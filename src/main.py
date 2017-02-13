# Imports
import cv2
from pipeline import Pipeline
from Line import Line

# Instantiate utils (calibrates camera)
pipeline = Pipeline()
right_line = Line(10)
left_line = Line(10)
lines = [left_line, right_line]
counter = 0
max_lost_frames = 2
lost_frame = 0

# read video in
video = cv2.VideoCapture('../project_video.mp4')
# to output video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('../output_project_video.mp4', fourcc, 20.0, (1280, 720))

# while the video is open, process images
while video.isOpened():
    # read each frame
    success, image = video.read()

    # run the pipeline on the frame
    left_line, right_line, result = pipeline.run(image, left_line, right_line)

    # see if the line was detected
    if left_line.detected == False | right_line.detected == False:
        # if not add to lost (unusable) frame count
        lost_frame += 1
    else:
        # reset the count, good frame
        lost_frame = 0

    # print line information, offset, and lost frame count
    print('lost_frame:', lost_frame,
          'offset:', left_line.line_base_pos,
          'left radius:', left_line.radius_of_curvature,
          'right radius:', right_line.radius_of_curvature
          )

    # if lost_frame is under the threshold, just pretend like we don't notice it
    if lost_frame < max_lost_frames:
        left_line.detected = True
        right_line.detected = True
    else:
        left_line.detected = False
        right_line.detected = False

    # Write the output
    out.write(result)
    # show the frames with the lane marked
    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
