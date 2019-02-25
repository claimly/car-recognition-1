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
from keras.layers import Input,concatenate, activations, Wrapper,Add,Lambda,Activation
from keras.engine import InputSpec
from keras.utils import multi_gpu_model   
import math

classes = 196
width = 256
length = 256
batch_size = 64
car_data_dir = './input/car_data/'
train_dir = car_data_dir + 'train/'
test_dir = car_data_dir + 'test/'

train_datagen = ImageDataGenerator(
    rescale=1./ width-1,
    zoom_range=0.2,
    rotation_range = 8,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / width-1)

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

def identity_block(x,nb_filter,kernel_size=5):
    k1,k2,k3 = nb_filter
    out = Conv2D(k1,1,1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(k2,kernel_size,kernel_size,border_mode='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(k3,1,1)(out)
    out = BatchNormalization()(out)


    out = Add()([out, x])
    out = Activation('relu')(out)
    return out

def conv_block(x,nb_filter,kernel_size=5):
    k1,k2,k3 = nb_filter

    out = Conv2D(k1,1,1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = out = Conv2D(k2,kernel_size,kernel_size,border_mode='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(k3,1,1)(out)
    out = BatchNormalization()(out)

    x = Conv2D(k3,1,1)(x)
    x = BatchNormalization()(x)
    out = Add()([out, x])
    out = Activation('relu')(out)
    return out


main_input = Input(shape=(width, length,3),name='main_input')
#CONV-RELU-POOL 1
conv_1 = Conv2D(filters = 16,kernel_size = (5,5),strides = 1,activation="relu",padding='same')(main_input)
pooling1 = MaxPooling2D(pool_size=(3, 3),strides=2, border_mode='same')(conv_1)

out = conv_block(pooling1,[64,64,5])
out = identity_block(out,[64,64,5])
out = identity_block(out,[64,64,5])

out = conv_block(out,[128,128,5])
out = identity_block(out,[128,128,5])
out = identity_block(out,[128,128,5])

out = AveragePooling2D((3,3))(out)
out = Flatten()(out)
out = Dense(512, activation='relu')(out)
out = Dense(256, activation='relu')(out)
out = Dense(classes,activation='softmax')(out)

model = Model(inputs = main_input, outputs = out)


rm = optimizers.RMSprop(lr = 0.001)
sgd = optimizers.SGD(lr=0.01, momentum=0.9)



print("[INFO] training with {} GPUs...".format(4))
# we'll store a copy of the model on *every* GPU and then combine
# the results from the gradient updates on the CPU
with tf.device("/cpu:0"):
    # initialize the model
    model1 = model
    # make the model parallel(if you have more than 2 GPU)
model = multi_gpu_model(model1, gpus=4)


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
model.save_weights('weights/ResNetweights.hdf5')

import matplotlib.pyplot as plt

model_id = "CNN_padding"
fig = plt.figure()#新建一张图
plt.plot(history.history['acc'],label='training acc')
plt.plot(history.history['val_acc'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('VGG16'+str(model_id)+'acc.png')
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('VGG16'+str(model_id)+'loss.png')