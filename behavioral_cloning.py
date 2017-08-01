'''
Created on 19. 7. 2017

@author: Martin Koubek
'''
import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot

'''
Parameters to tune model
'''

UDACITY = False
LEARNING_RATE=1.0e-4
EPOCHS = 7
BATCH_SIZE = 32
IMG_HEIGHT = 66
IMG_WIDTH = 200
INPUT_DIMENSIONS = (IMG_HEIGHT,IMG_WIDTH,3)
TEST_SIZE = 0.2


class BehavioralCloning(object):
    '''
    The class calculates model from images
    '''


    def __init__(self):
        """
        Init the model
        """
        self.prepare()
        self.neural_network()
        
    def prepare(self):
        """
        Put images paths and angles to variables lists
        """
        samples = []
        image_paths = []
        angles = []

        if UDACITY:
            path = "../../simulator_datacollection/udacity/driving_log.csv"
        else:
            path = "../../simulator_datacollection/driving_log.csv"

                            
        with open(path) \
           as csvfile:
            reader = csv.reader(csvfile)

            for line in reader:
                samples.append(line)
                
                if float(line[6]) < 0.1 :
                    continue
                
                for i in range(3):
                    if UDACITY:
                        name = './IMG/'+ line[i].split('/')[-1]
                        folder = os.getcwd()
                        path = os.path.join(folder, "../../simulator_datacollection/udacity/", name)
                        path = os.path.abspath(path)
                    else:
                        path = line[i]
                        
                    
                    if (i == 0):
                        offset = 0
                    elif (i == 1):
                        offset = 0.25
                    elif (i == 2):
                        offset = -0.25

                    image_paths.append(path)
                    angles.append(float(line[3]) + offset)


        image_paths = np.array(image_paths)
        angles = np.array(angles)

        #self.train_samples, self.validation_samples = \
#             train_test_split(samples, test_size=TEST_SIZE)
            
        self.image_paths_train, self.image_paths_test, \
            self.angles_train, self.angles_test = \
            train_test_split(image_paths, angles,\
                              test_size=TEST_SIZE)

    def neural_network(self):
        """
        Set the neural network
        """
        model = Sequential()
        
        train_generator = self.generator(self.image_paths_train, self.angles_train, validation_flag=False, batch_size=BATCH_SIZE)
        validation_generator = self.generator(self.image_paths_train, self.angles_train, validation_flag=True, batch_size=BATCH_SIZE)

        model.add(Lambda(lambda x: x/127.5 - 1.0, \
                         input_shape=(INPUT_DIMENSIONS)))
        
        model.add(Convolution2D(24,5,5, subsample=(2,2), \
                                activation="elu"))                  
        model.add(Convolution2D(36,5,5, subsample=(2,2), \
                                activation="elu"))
        model.add(Convolution2D(48,5,5, subsample=(2,2), \
                                activation="elu"))
        model.add(Convolution2D(64,3,3,activation="relu"))
        model.add(Convolution2D(64,3,3,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Flatten())    
        model.add(Dense(100,activation='elu'))
        model.add(Dense(50,activation='elu'))
        model.add(Dense(10,activation='elu'))
        model.add(Dense(1))
        print(model.summary())
        
        
        model.compile(loss = 'mse', optimizer=Adam(lr=LEARNING_RATE))
        plot(model, to_file='model.png', show_shapes=True, \
                   show_layer_names=True)
        
        checkpoint = ModelCheckpoint('model{epoch:02d}.h5', \
                                 monitor='val_loss', \
                                 verbose=0, \
                                 save_best_only=True, \
                                 mode='auto')
        
        model.fit_generator(train_generator, 
                            samples_per_epoch= 23040, \
                            validation_data=validation_generator, \
                            nb_val_samples=2560, \
                            nb_epoch=EPOCHS, \
                            callbacks=[checkpoint], \
                            verbose = 1)
            
            
       
        model.save('model.h5')
        print ("Model saved")
        
     
    def generator(self, image_paths, angles, batch_size=64, validation_flag=False):
        """
        This is generator in order to NOT store all data in memory 
        """
        image_paths, angles = sklearn.utils.shuffle(image_paths, angles)
        self.X,self.y = ([],[])
        
        while 1: # Loop forever so the generator never terminates
            for i in range(len(angles)):           
                image = cv2.imread(image_paths[i])
                angle = angles[i]
#                 cv2.imshow('image',image)

                image = self.imagePreprocess(image)
#                 cv2.imshow('image2',image)
#                 image, angle = self.distort(image, angle);
#                 cv2.imshow('image3',image)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
                
                self.X.append(image)
                self.y.append(angle)
                
                if len(self.X) == batch_size:
                    yield (np.array(self.X), np.array(self.y))
                    self.X, self.y = ([],[])
                    image_paths, angles = sklearn.utils.shuffle \
                        (image_paths, angles)
                        
                if abs(angle) > 0.33:
                    image = cv2.flip(image, 1)
                    angle *= -1
                    self.X.append(image)
                    self.y.append(angle)
                    if len(self.X) == batch_size:
                        yield (np.array(self.X), np.array(self.y))
                        self.X, self.y = ([],[])
                        image_paths, angles = sklearn.utils.shuffle \
                            (image_paths, angles)
       
    @staticmethod 
    def imagePreprocess(image):
        image = image[60:-25,:,:]        
        image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
        image = cv2.GaussianBlur(image, (3,3), 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        return image
    
    def distort(self, image, angle):
        new_img = image.astype(float)
        value = np.random.randint(-28, 28)
        # random brightness
        if value > 0:
            mask = (new_img[:,:,0] + value) > 255 
        if value <= 0:
            mask = (new_img[:,:,0] + value) < 0
            
        new_img[:,:,0] += np.where(mask, 0, value)
        # random shadow
        h,w = new_img.shape[0:2]
        mid = np.random.randint(0,w)
        factor = np.random.uniform(0.6,0.8)
        if np.random.rand() > .5:
            new_img[:,0:mid,0] *= factor
        else:
            new_img[:,mid:w,0] *= factor
            
        # randomly shift horizon
        h,w,_ = new_img.shape
        horizon = 2*h/5        
        v_shift = np.random.randint(-h/8,h/8)        
        pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
        pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
        new_img = new_img.astype(np.uint8)
        return (image,angle)
            
if __name__ == '__main__':
    b = BehavioralCloning()
    b.neural_network()