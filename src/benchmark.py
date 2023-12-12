import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from detection import main_detection
from prediction import create_mask,clear_all_masks
from lib_poto_detection import MASK_GENERATION_REPO

def get_time_for_detection(image_name):
    start_time = time.time()
    main_detection(image_name,flag_display=False,flag_debug=False)
    return time.time() - start_time


def get_time_for_prediction(image_name):
    start_time = time.time()
    create_mask(image_name, MASK_GENERATION_REPO)
    return time.time() - start_time


def build_graph_of_benchmark(N = 10, repo = 'log1/', remove_masks = True):
    max_image = 0
    if repo == 'log1/':
        max_image = 102
    elif repo == 'log2/':
        max_image = 50
    elif repo == 'log3/':
        max_image = 10
    elif repo == 'log4/':
        max_image = 10
    detection_times = []
    prediction_times = []
    create_mask(repo+str(1).zfill(3)+'-rgb.png', MASK_GENERATION_REPO) # the first prediction is always longer
    if remove_masks:
        clear_all_masks(MASK_GENERATION_REPO)
    for i in range(N):
        image_name = repo+str(random.randint(1, max_image)).zfill(3)+'-rgb.png' # after log1/102, there is no more images with goals
        while not os.path.exists('./data/'+image_name):
            print(image_name, 'does not exist. Trying another one.')
            image_name = repo+str(random.randint(1, max_image)).zfill(3)+'-rgb.png'
        if not remove_masks:
            create_mask(image_name, MASK_GENERATION_REPO)
        detection_times.append(get_time_for_detection(image_name))
        if remove_masks:
            clear_all_masks(MASK_GENERATION_REPO)
        prediction_times.append(get_time_for_prediction(image_name))
    x = np.arange(N)
    plt.plot(x, detection_times, label='prediction + detection')
    plt.plot(x, prediction_times, label='prediction')
    plt.xlabel('image number')
    plt.ylabel('time (s)')
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    return


if __name__ == "__main__" :
    build_graph_of_benchmark(remove_masks = True)