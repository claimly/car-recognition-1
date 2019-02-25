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
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten,RepeatVector,Permute,GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras.layers import Dropout
from keras import backend as K
import sklearn.metrics as metrics
from keras import optimizers
from keras.models import Model
from keras.models import model_from_json
from keras.backend import manual_variable_initialization
from keras.layers import Input,concatenate, activations, Wrapper,Add,Lambda,Activation
from keras.engine import InputSpec
from keras.utils import plot_model
from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
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

classes = 196
width = 256
length = 256
batch_size = 64
G = 4
car_data_dir = './input/car_data/'
train_dir = car_data_dir + 'train/'
test_dir = car_data_dir + 'test/'

train_datagen = ImageDataGenerator(
    featurewise_std_normalization = True
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

#base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(width,length,3), pooling=None, classes=classes)
#model = Model(inputs = main_input, outputs = out)
#model = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(width,length,3), pooling=None, classes=classes)

base_model = VGG19(weights='imagenet', include_top=False, pooling=None, input_shape=(width, length, 3),classes=classes)
'''
for layer in base_model.layers:
    layer.trainable = False
'''
x = base_model.output
x = GlobalAveragePooling2D()(x)
BatchNormalization()
Dropout(0.25)
x = Dense(1024, activation='relu')(x)
BatchNormalization()
Dropout(0.5)
predictions = Dense(classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

rm = optimizers.RMSprop(lr = 0.001)
sgd = optimizers.SGD(lr=0.01, momentum=0.9)



#plot_model(model, to_file="output/resnet.png", show_shapes=True)

print("[INFO] training with {} GPUs...".format(G))
# we'll store a copy of the model on *every* GPU and then combine
# the results from the gradient updates on the CPU
with tf.device("/cpu:0"):
    # initialize the model
    model1 = model
    # make the model parallel(if you have more than 2 GPU)
model = multi_gpu_model(model1, gpus=G)


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
steps_per_epoch = 256
nb_validation_samples = 8041
history = model.fit_generator(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = nb_validation_samples // batch_size)
#history = model.fit(train_set,train_label,epochs=20,batch_size = batch_size,validation_data = (test_set,test_label),verbose = 1)
model.save_weights('weights/VGG_translearning.hdf5')
saveHist('weights/VGGtans_h',history)
