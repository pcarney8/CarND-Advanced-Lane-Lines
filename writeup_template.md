##Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted-images/calibration1.jpg "Undistorted"
[image2]: ./camera_cal/calibration1.jpg "Normal"
[image3]: ./output_images/undistorted-images/undistorted-road.jpg "Road Transformed"
[image4]: ./output_images/threshold-images/color_and_gradient_threshold.png "Binary Example"
[image5]: ./output_images/threshold-images/warped.png "Warp Example"
[image6]: ./output_images/threshold-images/curve.png "Rectangle Fit Visual"
[image7]: ./output_images/threshold-images/curve_with_polynomial.png "Poly Fit Visual"
[image8]: ./output_images/threshold-images/output.jpg "Output"
[image9]: ./output_images/threshold-images/output-bad-src-dst.jpg "Output"
[image10]: ./output_images/threshold-images/histogram.png "Histogram"
[video1]: ./output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in a method `calibrate_camera` on lines 15 through 61 of the file called `src/utils.py`) and is initialized whenever a new `Utils` class is instantiated.

I started by creating the object points, which will be in 3-dimensions (x, y, z) of the chessboard corners. As in the lessons, I am assuming the chessboard is fixed on the (x, y) plane at z=0, and the object points will be the same for all the calibration images.  

`objp`, the replicated array of coordinates, and `objpoints` are appended everytime the corners are successfully detected.  `imgpoints` are appended with the (x, y) pixel position of each corner when successful.  This is the same method that we followed during our lesson plans.

I used the `objpoints` and `imgpoints` to compute and return the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I put these in the initialization of the `Utils` class because it made sense to do this only once when the program booted up. If necessary, the camera could be re-calibrated with a direct call to `calibrate_camera` and then setting all of the coefficients. I use the values in line 17 of the `pipeline.py` class when the `cv2.undistort()` function is called. You can see around the corners of the below images where it is undistorted: 

![alt text][image1]![alt text][image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
After the `Utils` class was initialized all of the coefficients needed to undistort the image are there, and is called in line 17 of the `pipeline.py` file.
![alt text][image3]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used the Sobel x (gradient) and Saturation Channel fo the HLS color transforms to generate the binary image. These steps are called line 20 of `pipeline.py` and executed in lines 65 through 107 in `utils.py`. This is an example of the output:

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 111 through 147 in the file `utils.py` (src/utils.py).  The `warp()` function takes as inputs an image (`img`), but it doesn't take an `src` and `dst` since they are fixed for this project.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
I originally tried to put the `src` in front of the car hood and out as far into the distance as possible. This resulted in some very poor radius of curvature and jumpy lanes. Notice the difference between this image and the final output image (at the bottom):
![alt text][image9]

After reviewing this template, I decided to switch to what was used here (including the bottom of the image with the hood) and it resulted in a much smoother experience. These are the actual calculated source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by inspecting the warped image and that the lames appeared parallel in curved and straight-lined images.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In `utils.py` on lines 224 through 316, I calculated the "sliding window" fit to then extract a 2nd order polynomial. I took in the image `img` and the margin. I output the image to verify it was performed correctly:

![alt text][image6]

In `utils.py` on lines 173 to 222, I calculated the polynomial fit with a previous second order polynomial. I took in the image `img`, left line polynomial coefficients `left_fit`, and right line polynomial coefficients `right_fit`. I output the image to verify it was performed correctly:

![alt text][image7]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In `utils.py` on lines 319 to 330, I calculated the radius of curvature in meters. I originally thought it might be a bit off, but that was due to my `src` and `dst` not being in the right place.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In `pipeline.py` on lines 155 to 169 I plotted the result back onto the road. Here is an example of my result on a test image:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

For the most part, the construction of my pipeline and utilities went smoothly. I went through a built each utility function at a time, tested the output to verify and then started to stitch together a pipeline class. I chose to use standard python instead of a jupyter notebook because of all the scrolling, and I wanted to start pulling things out into the Object-Oriented world, since that is where my background lies. 

I had some small issues that I faced.
* My line indices were jumping back and forth, and all of the sudden my `bestx` indices would shrink my lane line to nothing and grow back to normal size, it was very peculiar. For example, left_line_indices was showing 350 in frame 1 and then 950 in frame 2. This was a stupid mistake on my part, I flip-flopped the inputs to the pipeline `run()` function. Once that was resolved, all of my left line x values stayed right around the mid to high 300 pixels.
* Another issue I faced was the first and second bridges. When I wasn't using my best x values, thing got very wobbly, and probably would have made the car swerve a lot, potentially being unsafe. Once I implemented the sanity checks, and used the best x values, the lanes went pretty smoothly across the bridge, although there were a good number of not parallel lines detected. I tuned my binary threshold just a little bit to see if it would help, and it didn't seem to make as much of a difference as getting the best x values to work correctly. This method of taking the previous good values, did not work very well in the challenge video. I would need to think about potentially using the Lightness color channel or some other type of thresholding to work better in the harder videos.
