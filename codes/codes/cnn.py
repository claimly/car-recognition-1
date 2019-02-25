import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

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

train_set =np.load("./DATA/padding_train.npy")
test_set =np.load("./DATA/padding_test.npy")
train_label =np.load("./DATA/padding_train_label.npy")
test_label =np.load("./DATA/padding_test_label.npy")

print(train_set.shape)
print(train_label.shape)
train_set = train_set.reshape(train_set.shape[0],train_set.shape[1],train_set.shape[2],1)
test_set = test_set.reshape(test_set.shape[0],test_set.shape[1],test_set.shape[2],1)
print(train_set.shape)
classes = 196
width = 500
length = 500
batch_size = 64

def batchappend(l, batch_size):
    for i in range(l.shape[0],math.ceil(l.shape[0]/batch_size)*batch_size):
        l = np.append(l,l[i-l.shape[0]:i-l.shape[0]+1],axis = 0)
    return l
train_set = batchappend(train_set,batch_size)
test_set = batchappend(test_set,batch_size)
train_label = batchappend(train_label,batch_size)
test_label = batchappend(test_label,batch_size)

cnn = Sequential()
cnn.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (500,500,1)))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.22))
cnn.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.22))
cnn.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters = 96, kernel_size = (3,3), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(BatchNormalization(axis = 1))
cnn.add(Flatten())
cnn.add(Dropout(0.18))
cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dense(512, activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(Dense(196, activation = 'softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = cnn.fit(train_set,train_label,epochs=100,batch_size = batch_size,validation_data = (test_set,test_label),verbose = 1)
cnn.save_weights('weights/weights.hdf5')

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






