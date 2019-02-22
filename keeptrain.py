import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import json,codecs


#parameters
nb_validation_samples = 8041
classes = 196
width = 256
length = 256
batch_size = 64
car_data_dir = './input/car_data/'
train_dir = car_data_dir + 'train/'
test_dir = car_data_dir + 'test/'
steps_per_epoch = 256*32*2//batch_size #8144//32->256
G = 8
save_dir = "weights/"
Model_name = 'ResNet50_trans'
epochs = 10

#load data

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

history_dir = os.path.join(save_dir, Model_name+".history")

weight_dir = os.path.join(save_dir, Model_name+".h5")


#load model
model = load_model(weight_dir)


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


#keep training
history = model.fit_generator(
    train_generator,
    epochs=epochs, verbose=1, workers=4,
    validation_data=validation_generator,
    steps_per_epoch = steps_per_epoch,
    validation_steps = nb_validation_samples // batch_size)

#save
weight_dir = os.path.join(save_dir, Model_name+".h5")
model.save_weights(weight_dir)
history_dir = os.path.join(save_dir, Model_name+".history")
saveHist(history_dir,history)