import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM,GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten,RepeatVector,Permute
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras.layers import Dropout
from keras import backend as K
import sklearn.metrics as metrics
from keras import optimizers
from keras.models import Model
from keras.models import model_from_json
from keras.backend import manual_variable_initialization
from keras.layers import Input,concatenate, activations, Wrapper,merge,Lambda,Activation
from keras.engine import InputSpec
import math

import json,codecs
def saveHist(path,history):

    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float64:
               new_hist[key] = list(map(float, history.history[key]))

    #print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)

def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n


steps_per_epoch = 512
nb_validation_samples = 8041
classes = 196
width = 256
length = 256
batch_size = 64
car_data_dir = './input/car_data/'
train_dir = car_data_dir + 'train/'
test_dir = car_data_dir + 'test/'

train_datagen = ImageDataGenerator(
    featurewise_std_normalization = True,
    rescale=1./ 255,
    zoom_range=0.2,
    rotation_range = 8,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(width, length),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(width, length),
    batch_size=batch_size,
    class_mode='categorical')


model = Sequential()

#CONV-RELU-POOL 1
model.add(Conv2D(filters = 16,kernel_size = (7,7),strides = 1,activation="relu",batch_input_shape=(batch_size, width, length,3)))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2, border_mode='same'))

#CONV-RELU-POOL 2
model.add(Conv2D(filters = 32,kernel_size = (5,5),strides = 1,activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2, border_mode='same'))

#CONV-RELU-POOL 3
model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = 1,activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2, border_mode='same'))

#CONV-RELU-POOL 4
model.add(Conv2D(filters = 128,kernel_size = (3,3),strides = 1,activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2, border_mode='same'))
#CONV-RELU-POOL 5
model.add(Conv2D(filters = 128,kernel_size = (3,3),strides = 1,activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3),strides=2, border_mode='same'))
#CONV-RELU-POOL 6
model.add(Conv2D(filters = 256,kernel_size = (3,3),strides = 1,activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(3,2), border_mode='same'))
#DENSE 1
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu',name="Dense_2"))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))



rm = optimizers.RMSprop(lr = 0.001)
sgd = optimizers.SGD(lr=0.01, momentum=0.9)



model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit_generator(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = nb_validation_samples // batch_size)
model.save_weights('weights/weights.hdf5')

saveHist('weights/VGGtans_h_uf',history)
