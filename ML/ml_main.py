import os
import random
from PIL import Image
import numpy as np

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def create_set(path):
    train_set = []
    validation_set = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            if random.random() < 0.5:
                train_set.append(filename)
            else:
                validation_set.append(filename)
    return train_set, validation_set


# input image dimensions from ./data/log1/001-rgb.png
img_rows, img_cols = 448, 800

batch_size = 32
input_shape = (img_rows, img_cols, 3)
output_shape = (img_rows, img_cols, 1)
epochs = 10

train_data_dir = './ML/image'
validation_data_dir = './ML/mask'
train_set, validation_set = create_set('./ML/image')

x_train_files = []
y_train_files = []
for filename in train_set:
    x_train_files.append('./ML/image/'+filename)
    y_train_files.append('./ML/mask/'+filename)

x_train = [Image.open(filename) for filename in x_train_files]
y_train = [np.array(Image.open(filename).convert('L')) for filename in y_train_files]

x_train = np.array(x_train)
y_train = np.array(y_train)

threshold = 128
value = 1
y_train = (y_train > threshold) * value

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))

def add_pool(mo, filters):
    mo.add(Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
    mo.add(Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
    mo.add(MaxPooling2D(pool_size=(2, 2)))
    return

def add_up(mo, filters):
    mo.add(Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
    mo.add(Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same'))
    mo.add(UpSampling2D((2, 2)))
    return

add_pool(model, 32)
add_pool(model, 64)
add_pool(model, 128)
add_pool(model, 256)

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))

add_up(model, 256)
add_up(model, 128)
add_up(model, 64)
add_up(model, 32)

model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

model.summary()

model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                optimizer=tensorflow.keras.optimizers.Adadelta(),
                metrics=['accuracy'])

model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

x_test_files = []
y_test_files = []
for filename in validation_set:
    x_test_files.append('./ML/image/'+filename)
    y_test_files.append('./ML/mask/'+filename)

x_test = [Image.open(filename) for filename in x_test_files]
y_test = [np.array(Image.open(filename).convert('L')) for filename in y_test_files]

x_test = np.array(x_test)
y_test = np.array(y_test)

threshold = 128
value = 1
y_test = (y_test > threshold) * value


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

