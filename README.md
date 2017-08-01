# CarND-Behavioral-Cloning-Project
This repository contains files for the Behavioral Cloning Project from Udacity.

In this project, it was learned about deep neural networks and convolutional neural networks to clone driving behavior. It trains, validate and test a model using Keras. The model outputs a steering angle to an autonomous vehicle.

It was provided a simulator where a car can be steer around a track for data collection. It was use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

To meet specifications, the project requires submitting five files:

* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)


![CarNd]( /images/2017_07_26_11_18_46_self_driving_car_nanodegree_program.png "")


Youtube link:
https://youtu.be/RyA0zojd2bY

My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.

There is python scripts: model.py that:
* load images
* flip some of images in order to get more input data
* calculate neural network

# The Project #

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Dependencies ##

This lab requires resources described here:
https://github.com/udacity/CarND-Behavioral-Cloning-P3


# Project Specification #

Use Deep Learning to Clone Driving Behavior

The main purpose of this script is to train the model using the data saved from the above python script.
First, it imports the pickle file from the local drive and train the data using model that I built.
The detail of the model can be found in the script.
When the training is done, the model is saved as model.h5.

# Image processing #

The images are cropped so that the model won’t be trained with the sky and the car front parts
The images are resized to 66x200 (3 YUV channels) as per NVIDIA model
The images are normalized (image data divided by 127.5 and subtracted 1.0). 

As stated in the Model Architecture section, this is to avoid saturation and make gradients work better)

Original image

![CarNd]( /images/image_screenshot_31.07.2017.png "")

Processed image - cropped, resized and blured

![CarNd]( /images/image2_screenshot_31.07.2017.png "")

# Training #
The design of the network is based on the NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test. As such, it is well suited for the project.

It is a deep convolution network which works well with supervised image classification / regression problems. As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

I've added the following adjustments to the model.

* I used Lambda layer to normalized input images to avoid saturation and make gradients work better.
* I've added an additional dropout layer to avoid overfitting after the convolution layers.
* I've also included RELU for activation function for every layer except for the output layer to introduce non-linearity.

In the end, the model looks like as follows:

![CarNd]( /images/model.png "")

```
Total params: 2,882,619
Trainable params: 2,882,619
Non-trainable params: 0
```
