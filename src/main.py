import os
from cv2 import imread
from prediction import create_mask
from detection import main_detection
from lib_poto_detection import display
from goal_statistics import show_statistics
from benchmark import build_graph_of_benchmark
from compute_goal_surface import compute_all_goals_surface, create_accuracy_file, ACCURACY_FILE
from register_goals import create_register_file, main_register_goals, get_next_goal, GOALS_REGISTRY_FILE


if __name__ == '__main__':
    # Choose the image to work on
    image_name = 'log1/015-rgb.png'

    # Create the mask for the image
    if not os.path.exists('./ML/result/'):
        os.makedirs('./ML/result/')
    if os.path.exists('./ML/result/'+image_name[5:]):
        os.remove('./ML/result/'+image_name[5:])
    mask_path=create_mask(image_name, './ML/result/')
    print(mask_path)
    mask = imread(mask_path, 0)
    display(mask)

    # Do the detection on an image
    main_detection(image_name,flag_display = 1, flag_debug = 0)

    # Do the benchmark on N images
    build_graph_of_benchmark(N=10)

    # Manual detection of a goal. It will then serve as a reference for the accuracy of the detection
    create_register_file(GOALS_REGISTRY_FILE)
    img_name = get_next_goal(GOALS_REGISTRY_FILE)
    main_register_goals(img_name)

    # Another one to make at least two statistics
    img_name = get_next_goal(GOALS_REGISTRY_FILE)
    main_register_goals(img_name)

    # Compute the accuracy of the detection
    create_accuracy_file(ACCURACY_FILE)
    compute_all_goals_surface()

    # Show statistics of accuracy with curves
    show_statistics()