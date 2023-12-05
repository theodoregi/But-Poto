import os
import random
from PIL import Image
import numpy as np

import tensorflow.keras
from tensorflow.keras import Input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import InputLayer
import tensorflow.keras.layers as layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Dense, Reshape


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
num_classes = 2
output_shape = (img_rows, img_cols, 3)
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
y_train = [Image.open(filename) for filename in y_train_files]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = normalize(x_train, axis=1)
y_train = normalize(y_train, axis=1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size=(10, 10)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(10, 10)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(np.prod(output_shape), activation='sigmoid'))
model.add(Reshape(output_shape))

model.summary()

model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                optimizer=tensorflow.keras.optimizers.Adadelta(),
                metrics=['accuracy'])

model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_train, y_train))

x_test_files = []
y_test_files = []
for filename in validation_set:
    x_test_files.append('./ML/image/'+filename)
    y_test_files.append('./ML/mask/'+filename)

x_test = [Image.open(filename) for filename in x_test_files]
y_test = [Image.open(filename) for filename in y_test_files]

x_test = np.array(x_test)
y_test = np.array(y_test)

x_test = normalize(x_test, axis=1)
y_test = normalize(y_test, axis=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

