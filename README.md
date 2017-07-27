# CarND-Behavioral-Cloning-Project
This repository contains files for the Behavioral Cloning Project.

![CarNd]( /2017_07_26_11_18_46_self_driving_car_nanodegree_program.png "")

My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.

There is python scripts: model.py that:
* load images
* flip some of images in order to get more input data
* calculate neural network


# Project Specification #

Use Deep Learning to Clone Driving Behavior

The main purpose of this script is to train the model using the data saved from the above python script.
First, it imports the pickle file from the local drive and train the data using model that I built.
The detail of the model can be found in the script.
When the training is done, the model is saved as model.h5.

# Training #
Below is the summary of the model I implemented to train the data.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 78, 158, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 37, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 17, 37, 48)    43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 15, 35, 64)    27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 13, 33, 64)    36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 13, 33, 64)    0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 27456)         0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           2745700     flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 2,882,619
Trainable params: 2,882,619
Non-trainable params: 0
