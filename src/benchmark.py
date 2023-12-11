import os
import time
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


def build_graph_of_benchmark(repo = 'log1/', remove_masks = True):
    detection_times = []
    prediction_times = []
    create_mask(repo+str(1).zfill(3)+'-rgb.png', MASK_GENERATION_REPO) # the first prediction is always longer
    if remove_masks:
        clear_all_masks(MASK_GENERATION_REPO)
    for i in range(10):
        image_name = repo+str(i+1).zfill(3)+'-rgb.png'
        if not remove_masks:
            create_mask(image_name, MASK_GENERATION_REPO)
        detection_times.append(get_time_for_detection(image_name))
        if remove_masks:
            clear_all_masks(MASK_GENERATION_REPO)
        prediction_times.append(get_time_for_prediction(image_name))
    x = np.arange(10)
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