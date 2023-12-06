# use the saved model to predict the result of one image
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def create_mask(img_name, path_to_save):
    model = load_model('./ML/model/model.h5')

    image = Image.open('./data/'+img_name)
    image = np.array(image)
    image = image.reshape((1, 448, 800, 3))

    y_pred = model.predict(image)[0]

    #convert y_pred to a binary image
    y_pred = (y_pred > 0.5) * 255
    y_pred = y_pred.reshape((448, 800))
    y_pred = y_pred.astype(np.uint8)

    # save the result in ./ML/result
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    Image.fromarray(y_pred).save(path_to_save+img_name[4:])
    return

# create_mask('log1/001-rgb.png', './ML/result/')