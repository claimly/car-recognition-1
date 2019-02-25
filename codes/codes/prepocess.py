import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from PIL import Image

anno_train_df = pd.read_csv('./input/anno_train.csv', header=None)
anno_test_df = pd.read_csv('./input/anno_test.csv', header=None)
names_df = pd.read_csv('./input/names.csv', header=None)


car_data_dir = './input/car_data/'
train_dir = car_data_dir + 'train/'
test_dir = car_data_dir + 'test/'
train_imgs, train_labels, test_imgs, test_labels = [], [], [], []
outwidth = 500
outlength = 500
classes = 196

def crop_images(img_lst, label_lst, input_dir):
    for subdir, dirs, files in os.walk(input_dir):
        for f in files:
            if f == '.DS_Store': 
                continue
            img_path = os.path.join(subdir, f)
            img = Image.open(img_path)
            row = anno_train_df.loc[anno_train_df[0] == f]
            #box = (row[1], row[2], row[3], row[4])
            label = row[5]
            #img = img.crop(box)
            img_lst.append(img)
            label_lst.append(label)

def img_to_np_arr(img, box=None):
    if box is not None:
        img = img.crop(box)
    img_arr = np.array(img)
    try:
        greyscale_img = img_arr[:, :, 0]
        return greyscale_img
    except IndexError:
        # image is greyscale already
        return img_arr


def images_to_np_array(img_lst, label_lst, input_dir):
    for subdir, dirs, files in os.walk(input_dir):
        for f in files:
            if f == '.DS_Store':
                continue
            img_path = os.path.join(subdir, f)
            img = Image.open(img_path)
            row = anno_train_df.loc[anno_train_df[0] == f]
            label = row[5]
            # turn image to greyscale 2d numpy array
            img_arr = img_to_np_arr(img)
            img_lst.append(img_arr)
            label_lst.append(label)

MAX_LEN = 500
MAX_WIDTH = 500

train_imgs_pad, train_labels_pad, test_imgs_pad, test_labels_pad = [], [], [], []

def one_hot(label,classes):
    res = [0]*classes
    res[label-1] = 1
    return res

def crop_images_and_pad(img_lst, label_lst, input_dir, classes):
    for subdir, dirs, files in os.walk(input_dir):
        for f in files:
            if f == '.DS_Store': 
                continue
            img_path = os.path.join(subdir, f)
            img = Image.open(img_path)
            row = anno_train_df.loc[anno_train_df[0] == f]
            box = (row[1], row[2], row[3], row[4])
            label = one_hot(int(row[5]),classes)
            img = img.crop(box)
            shape = img.size
            times = max(shape[0],shape[1])/500
            width = int(shape[0]/times)-1
            lenth = int(shape[1]/times)-1
            
            img = img.resize((width,lenth))
            img_arr = img_to_np_arr(img)
            shape = img_arr.shape
            # pad the image with zeroes
            img_arr = np.pad(img_arr, ((0, MAX_LEN - shape[0]), (0, MAX_WIDTH - shape[1])), 'constant', constant_values=(0))
            img_lst.append(img_arr)
            label_lst.append(label)

crop_images_and_pad(train_imgs_pad, train_labels_pad, train_dir, classes)
crop_images_and_pad(test_imgs_pad, test_labels_pad, test_dir, classes)

np.save("DATA/padding_train",train_imgs_pad)
np.save("DATA/padding_test",train_imgs_pad)
train_labels_pad = np.array(train_labels_pad)  
test_labels_pad = np.array(test_labels_pad)  
np.save("DATA/padding_train_label",train_labels_pad)
np.save("DATA/padding_test_label",test_labels_pad)