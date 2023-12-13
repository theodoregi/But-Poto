# use the saved model to predict the result of one image
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def create_mask(img_name, path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # if os.path.exists(path_to_save+img_name[5:]):
    #     print("Mask already exists.")
    #     return path_to_save+img_name[5:]
    if  os.path.exists('./data/mask/'+img_name):
        print("Mask already exists in ./data/mask/ .")
        Image.open('./data/mask/'+img_name).save(path_to_save+img_name[5:])
        return path_to_save+img_name[5:]
    if not os.path.exists('./ML/model/model.h5'):
        print("Model does not exist.")
        raise FileNotFoundError
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
    Image.fromarray(y_pred).save(path_to_save+img_name[5:])
    return path_to_save+img_name[5:]


def clear_all_masks(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(path+filename)
    return

if __name__ == '__main__':
    create_mask('log1/001-rgb.png', './ML/result/')