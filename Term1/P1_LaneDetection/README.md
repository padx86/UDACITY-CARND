# Project Reflection

## 1. Pipeline description
My pipeline consisted of 6 steps. First of all the image was smoothed to remove noise using Gaussian filter(Step 1). The next step was a grayscale conversion of the image(Step 2). Then the Image was masked by applying a triangle to retrieve the region of interest(Step 3). In order to make the lane canny edge detection easier gray color below a threshold was set to black(Step 4). This Step was continued by the canny edge detection(Step 5). Finally houghlines were calculated. As 7th step (excluded from the 6 steps) a function for the display of the lane marks was implemented to show the results.

## 2. Potential shortcomings

There are multiple shortcomings in the few lines of code. At first there is no error handling. 
Furthermore there is no routine in case there is an object on the same lane covering camera sight.
In Step 4 light areas wont be set to black and cause additional houghlines. Another shortcoming would be the lack of reconstruction of the lane markings and center

## 3. Possible improvements

A possible improvement would be to calculate road curvature from lane markings.
To guarantee a working system some kind of memory should be implemented to check the actual lane marks against lane marks in an earlier frame. Another potential improvement could be to reconstruct lane marks and lane centers to have a basis for trajectory planning with focus on autonomous driving

## Remarks
All images and Steps can be found in the jupyter notebook P1_submit in this repository

