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

#parameters
freeze = True
save_dir = "weights/"
Model_name = "VGG_trans"
nb_validation_samples = 8041
classes = 196
width = 256
length = 256
batch_size = 32
car_data_dir = './input/car_data/'
train_dir = car_data_dir + 'train/'
test_dir = car_data_dir + 'test/'
steps_per_epoch = 256*32*2//batch_size #8144//32->256
G = 4
epochs = 50

#data loader and data augmentation
train_datagen = ImageDataGenerator(
    # set rescaling factor (applied before any other transformation)
    rescale=1./ 255,
    # set range for random zoom
    zoom_range=0.2,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range = 10,
    # randomly flip images
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


#model

base_model = VGG19(weights='imagenet', include_top=False, pooling=None, input_shape=(width, length, 3),classes=classes)

if freeze:
    for layer in base_model.layers:
        layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = BatchNormalization()(x)
#x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
predictions = Dense(classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

rm = optimizers.RMSprop(lr = 0.001)
sgd = optimizers.SGD(lr=0.01, momentum=0.9)



print("[INFO] training with {} GPUs...".format(G))
with tf.device("/cpu:0"):
    # initialize the model
    model1 = model
    # make the model parallel(if you have more than 2 GPU)
model = multi_gpu_model(model1, gpus=G)


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = nb_validation_samples // batch_size)

#save
def saveHist(path,history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float64:
               new_hist[key] = list(map(float, history.history[key]))
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4) 

def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n

weight_dir = os.path.join(save_dir, Model_name+".h5")
model.save_weights(weight_dir)
history_dir = os.path.join(save_dir, Model_name+".history")
saveHist(history_dir,history)