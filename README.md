## Vehicle Detection Project

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---


### Histogram of Oriented Gradients (HOG)

The training images have been provided in two sets: vehicle images and non-vehicle images.
The training set consisted of 8795 Vehicle images and 8968 Non-vehicle images.

A random image from each set has been displayed:

![Random vehicle and non-vehicle photos](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/example_images.JPG)

In order to have a better sense of the HOG features in each set of pictures, random images from each set have been displayed, with their respective HOG features.
The example HOG features have been extracted by using the following sets of parameters:

Column | Orientation | Pixels per cell 
------------ | ------------- | ------------- 
2 | 9 | 8
3 | 9 | 16
4 | 11 | 8
5 | 11 | 16

Vehicle image | Non-vehicle image
------------ | ------------- 
![Random vehicle photos with HOG](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/hog_vehicles.JPG) | ![Random non-vehicle photos with HOG](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/hog_nonvehicles.JPG)

The features have been extracted by using the method **get_hog_features()** from the file **tools.py**.
The method uses the hog method of skimage.feature, with the parameters listed above.


In order to determine the best set of HOG parameters, extensive testing have been made. During these tests, the **extract_features()** method from the **tools.py** file has been used.
The method extracts gradient, color and spatial features from images and merges them together. 
The features are treated in the **tunner1()** and **tunner2()** methods of the file **tunning.py**. 
The vehicle and non-vehicle features are stacked together and standardized.
A **Linear Support Vector Classifier** has been selected for the training.

1. A first bacth of tests has been made, using the following parameters:
```
  -Color spaces: ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
  -HOG orientations: [9, 11, 13] 
  -HOG pixels per cell: [8, 16] 
  -HOG cells per block: 2 
  -HOG channels: [0, 1, 2, "ALL"]
  -Spatial binning dimensions: [(16, 16), (32, 32)]  
  -Number of histogram bins: [16, 32] 
```

The first 30 lines of the results have been attached below

![Results first batch of tests](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/tunning_results.JPG)


2. Using the results of the first batch of tests, the following pairs of color-space and hog channels have been selected for further testing:
```
  -Color space and HOG channel: [('HSV', 'ALL'), ('YUV', 1), ('YUV', 'ALL'), ('YCrCb', 'ALL'), ('LUV', 'ALL'), ('HSV', 2)]  
  
  The other selected parameters are:
  -HOG orientations: [9, 11] 
  -HOG pixels per cell: [8, 16] 
  -HOG cells per block: 2 
  -HOG channels: [0, 1, 2, "ALL"]
  -Spatial binning dimensions: [(16, 16), (32, 32)]
  -Number of histogram bins: [16, 32] 
```

The results of the second set of tests has been attached below:
  
![Results second batch of tests](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/tunning_results_2nd_lvl.JPG)
  
3. The final set of HOG parameters have been selected with respect to the accuracy but also with respect to the training time:

```
  -Color spaces: 'YCrCb'
  -HOG orientations: 11
  -HOG pixels per cell: 16
  -HOG cells per block: 2 
  -HOG channels: "ALL"
  -Spatial binning dimensions: (32, 32)
  -Number of histogram bins: 16
  
```

The parameter C of the LinearSVC() has also been tunned, using a sing a GridSearchCV(). Test have been made using the following possible values: [0.0001, 0.1, 10, 100]. 
The best value has been returned as being 0.0001. 

```
LinearSVC(C=0.0001, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0) with an accuracy of 0.9918
```

The training of the LinearSVC classifier, with the selected parameters has been done in the **cell 4** of the file **vehicle_detection.ipynb**

``` python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split



### After extensive testing of different parameter combinations, the following have been selected
color_space = 'YCrCb' 
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" 
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

car_features = extract_features(veh_images, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(nonveh_images, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:',color_space,'color space',orient,'orientations',pix_per_cell,
                        'pixels per cell and', cell_per_block,'cells per block', 
                          hog_channel, 'hog channel', spatial_size, 'spatial size', 
                         hist_bins, 'hist bins')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC(C=0.0001)

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

```

```
Using: YCrCb color space 11 orientations 16 pixels per cell and 2 cells per block ALL hog channel (32, 32) spatial size 16 hist bins
Feature vector length: 4308
5.72 Seconds to train SVC...
Test Accuracy of SVC =  0.9916

```

---


### Sliding Window Search

The sliding window search has been implemented by dividing the area of interest of an image (bounded on the Y scale between 400 and 656 pixels) into small rectangles with a size of 64X64, overlapping by 0.5. The values had been defined during the courses and they seemed to respond well to the needs of the project.

The method **search_windows()** in the file **tools.py** iterates over all the windows in the picture, extracts the test window and the features from the image being processed and returns a set of windows that have been evaluated as containing a car.

Windows | Hot Windows
------------ | ------------- 
![Window display on test images](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/windows.jpg) | ![Hot window display on test images](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/hot_windows.jpg)

In order to remove the false positives, a heat map has been added. The results of the heat map as well as the vehicle detection in the test image after applying the heat map are displayed below (one of the heatmaps images was completely black due to lack of vehicles, so it has been ignored):

Heatmap | Sliding windows
------------ | ------------- 
![Window display on test images](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/heatmaps.jpg) | ![Hot window display on test images](https://raw.githubusercontent.com/sorix6/CarND-Vehicle-Detection/master/output_images/sliding_windows.jpg)

---

### Video Implementation

The complete pipeline for processing video inputs can be found in the file **vehicle_detection.ipynb**, **cell 11**

``` python
def process_frame(image):
    
    bboxes = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel)
    
    heat = np.zeros_like(image[:,:,0])

    # Add heat to each box in box list
    heat = add_heat(heat,bboxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,0.5)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return draw_img
```

The pipeline applies the following steps to each of the frames in the video:
* detect boxes containing cars by calling the method **find_cars()** from the file **tools.py**. The process of detecting areas with cars has been detailed above.
* apply a heat map in order to remove the false positives
* display the boxes on the image
* return the final image


The final video processing can be found in the file project_video_YCrCb.mp4, in the folder output_videos.


---

### Discussion

The pipeline seems to have small glitches, in very dark regions of the road. Although tests have been run toimprove the accuracy of the classifier, no solution for improving it further has been found until now.

Most of the time, the vehicles are correctly detected, although, the bounding box is not always completely stable.
A way to solve this problem could be to save differences in the position of the car with respect to the previous frames and try to create a sort of continuous transition.

Investigating other parameter combinations that had similar performances during tests could be another way of trying to improve accuracy.

To date, no other classifier or combination of classifier has appeared to provide better results than the LinearSVC, during the development of this project.





