## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Submission Contents & Execution

My project includes the following directories/files:

1. model_training directory containing data sets, pickled objects, training and test set preparation code, feature extraction code, and model training code

2. output_images directory containing a final output image from the pipeline (it was difficult to auto-generate each stage’s output to this directory, so I took snapshots of the stages and put them in the writeup report 

3. output_video directory containing final output video: processed_project_video.mp4

4. sdcp4 directory containing all dependencies from project 4

5. test_images directory containing Udacity-provided pipeline test images (and a few I exported from the project video)

6. test_video directory containing Udacity-provided project videos: project_video.mp4 and test_video.mp4

7. main.py containing initialization, test (export final output image), and production (export processed video) pipeline execution code

8. production_pipeline.py containing code that processes each frame of the project video through my pipeline and produces a video saved to the output_video directory

9.  test_pipeline.py containing code that saves final output image from the pipeline to the output_images directory

10. vehicle_processor.py containing code that performs a vehicle search using a sliding window technique and a trained classifier, also thresholding code to deal with false positives and object labeling using heatmaps

11. writeup_template.md summarizing the results 

first, clone the project, Once cloned, an example image of the final stage of my pipeline is auto-generated and saved in the output_images directory and a processed version of the provided project video is also auto-generated and saved in the output_video directory by executing:

```python
python main.py  
```

### Overview

I have incorporated my vehicle detection and tracking process into the pipeline I delivered in the advanced lane finding project (please refer to my detailed write-up from P4 for details on its inner-workings). I won’t revisit those portions but will address the deltas I’ve added to meet the requirements of this current project.

As before, my submission includes two pipelines: The first is the test pipeline (test_pipeline.py) which saves an example of the final output of the pipeline to the output_images directory (see lines: 112–152 related to this current project). The second is the production pipeline (production_pipeline.py) which processes each frame of the Udacity-provided project video (project_video.mp4) through each stage (0-part-2 through 4 of my P4 submission) and the newly added vehicle detection and tracking capability I’ve added with P5 (see lines: 28–41, 61–67, 183–209). Then finally, it saves a processed video (processed_project_video.mp4) to the output_video directory. I’ll describe the sequence of processing of the data preparation, model training, and finally the pipeline processing the project video.

