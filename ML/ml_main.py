import os
import random
from PIL import Image
import shutil
import csv
import sys
import numpy as np

def create_set(path):
    train_set = []
    validation_set = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            if random.random() < 0.7:
                ### train_set.append(parent foldeer name and filename):
                train_set.append(filename)
            else:
                validation_set.append(filename)
    return train_set, validation_set

print(create_set('./ML/image/log1'))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K

batch_size = 32
num_classes = 1
epochs = 10

# input image dimensions from ./data/log1/001-rgb.png
img_rows, img_cols = 480, 640

train_data_dir = './ML/image/log1'
validation_data_dir = './ML/mask/log1'
train_set, validation_set = create_set('./ML/image/log1')

x_train_files = []
y_train_files = []
for filename in train_set:
    x_train_files.append('./ML/image/'+filename)
    y_train_files.append('./ML/mask/'+filename)

x_train = [Image.open(filename) for filename in x_train_files]
y_train = [Image.open(filename) for filename in y_train_files]