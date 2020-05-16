# General libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, LeakyReLU, Activation, GlobalAveragePooling2D, Multiply, Conv2D, MaxPool2D
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
%matplotlib inline

# Change this
path_to_dir = "/content/drive/My Drive/Spring 2020/SAAS/car_classification/"

######### Subset size Variables #############
train_size = 1000
test_size = int(train_size//2)
num_classes = 68 # web: 2004 # sv: 281
#################################

# File paths to get input data
web_path = path_to_dir + 'combined_data/'

# web_data
train_names = list(pd.read_csv(web_path+"train_test_split/classification/train.txt", sep="\t", header=None)[0])
test_names = list(pd.read_csv(web_path+"train_test_split/classification/test.txt", sep="\t", header=None)[0])

# tabke a subset of the names
train_names_sub = train_names[:train_size]
test_names_sub = test_names[:test_size]

######################################################################
########################### Generate y ###############################

# classify model
y_train = np.array([train_names_sub[i].split("/")[1] for i in range(len(train_names_sub))])
y_train = [int(i) for i in y_train]
y_test = np.array([test_names_sub[i].split("/")[1] for i in range(len(test_names_sub))])
y_test = [int(i) for i in y_test]

from keras.utils import np_utils
# one-hot encoding for categorical values.
y_train = np_utils.to_categorical(y_train)[:, 1:] 
y_test = np_utils.to_categorical(y_test)[:, 1:]

y_train = np.pad(y_train, [(0, 0), (0, num_classes - y_train.shape[1])], mode='constant')
y_test = np.pad(y_test, [(0, 0), (0, num_classes - y_test.shape[1])], mode='constant')


######################################################################
########################### Read in images ###########################

# takes a good while to run
def fetch_images(file_path, jpg_names):
    arr = []
    for name in jpg_names:
        img = cv2.imread(file_path+"image/" + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #crop
        with open(file_path+"label/" + name[:-4]+".txt") as f:
            x1, y1, x2, y2 = map(int, f.readlines()[2][:-1].split(" "))
        img = img[y1:y2, x1:x2]
        
        old_size = img.shape[:2] # (height, width)
        desired_size = 250
        ratio = desired_size/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))

        # padding for squaring
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        arr.append(padded_img)
        
    # Turn into numpy array and normalize pixel values
    arr = np.array(arr).astype('float32')
    arr = arr / 255 # pixel values are integers between [0, 255]
    return np.array(arr)

# web data
x_train = fetch_images(web_path, train_names_sub)
x_test = fetch_images(web_path, test_names_sub)

# Get dimensions of each image
img_dim = x_train.shape[1:]


(trainX, valX, trainY, valY) = train_test_split(x_train, y_train, stratify = y_train, test_size=0.2, random_state = 42)


#######################################################################
########################### Build Model ###############################

######## Hyperparameters ###############
batch_size = 40
epochs = 100
steps = trainX.shape[0] // batch_size
########################################

# Inputs
inputs = Input(shape=img_dim)
# DenseNet
densenet121 = DenseNet121(weights='imagenet', include_top=False)(inputs)

def fc_layer(x, units = 256, reg = False):
    if reg == True:
      kernel_reg = regularizers.l2(0.01)
    else:
      kernel_reg = None
    x = Dense(units,  use_bias=True, kernel_regularizer=kernel_reg)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x

# # Our FC layer
flat1 = Flatten()(densenet121)
fc1 = fc_layer(flat1)
drop1 = Dropout(rate=0.5)(fc1)
# # Output
out = Dense(num_classes, activation='softmax')(drop1)
# Create Model
model = Model(inputs=inputs, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
img_aug = ImageDataGenerator(rotation_range=5, horizontal_flip=True) # brightness_range=[0.5,1.5]
val_img_aug = ImageDataGenerator()
model_history = model.fit(img_aug.flow(trainX, trainY, batch_size=30), 
                    epochs=epochs, 
                    validation_data=val_img_aug.flow(valX, valY, batch_size=30),
                    callbacks=[early_stop, reduce_lr])
model.evaluate(x_test, y_test)
model.save(path_to_dir + 'models/model_web_'+str(train_size)+'.h5')






