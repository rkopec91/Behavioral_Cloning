# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
In order to run the file you execute the following
```sh
python model.py
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My architecture can be seen between lines 91-110.  The model consists of an input layer (image) that normalizes the image, then crops the image to only view the road and removes the front of the car and the horizon.  The following 5 layers are all convolutional layers, each having a relu activation function.  The first three have a kernal size of 5x5 and a stride of 2x2 and a filter of 24, 36, and 48.  The next two have a filter of 64 and a 3x3 kernal.  I added a dropout of 0.8 and a flattening layer.  THe next 4 layers are all dense layers with 100, 50, 10, and 1 units.  The output is the turning angle

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 102).   The dropout that I chose was 0.8.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 123-125). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually and used mean square error (model.py line 110).  I used a batch size of 32 and a learning rate of 0.001.  The validation data split was 0.15.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  I flipped each of the images to create more data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach and Final Model

I decided to create a convolutional neural network similar to Nvidias architecture that they use in their autonomous vehicles.  More on this architecture can be seen in https://devblogs.nvidia.com/deep-learning-self-driving-cars/.  This approach proved to be very accurate so I didn't ave to change the model much at all.  One thing that I did add to this was the cropping layer to crop the images as they came in so the model couldn't see the hood of the car or anything above the horizon.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay in the middle of the road a bit better. 

Then I decided to go around the track twice but in the opposite direction just to get more data..

To augment the data sat, I also flipped images and angles thinking that this would help create more data for the model to be trained on.

After the collection process, I had 11,000 number of data points.  


I finally randomly shuffled the data set and put 15% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 32. I used an adam optimizer so that manually training the learning rate wasn't necessary.
