import os
import cv2
from detection import main_detection
from lib_poto_detection import draw_rectangle, rotate_image
from register_goals import GOALS_REGISTRY_FILE

ACCURACY_FILE = "./src/accuracy_registry.csv"

def compute_goal_surface(x_max, x_min, y_max, y_min):
    surface = abs(x_max - x_min) * abs(y_max - y_min)
    return surface

def main_compute_goal_surface(image_name, horizon_angle_manu, x_max_manu, x_min_manu, y_max_manu, y_min_manu):
    # Get goal position from detection
    [x_max_det, x_min_det, y_max_det, y_min_det, image, horizon_angle] = main_detection(image_name,flag_display=False,flag_debug=False)
    if horizon_angle_manu != horizon_angle:
        print("!!! ERROR IN HORIZON ANGLE DETECTION !!!")
        print("    NOT THE SAME IN DETECTION/MANUAL")
        return
    
    image = rotate_image(image, horizon_angle)
    print("Image: ", image_name)
    
    # Compute surfaces
    manual_surface = compute_goal_surface(x_max_manu, x_min_manu, y_max_manu, y_min_manu)
    detected_surface = compute_goal_surface(x_max_det, x_min_det, y_max_det, y_min_det)
    # print("Manual surface: ", manual_surface)
    # print("Detected surface: ", detected_surface)
    # erreur = abs(detected_surface - manual_surface) / manual_surface * 100
    # print("Surface value error (%): ", erreur)
    
    # Draw rectangles
    image = draw_rectangle(image, x_min_det, y_min_det, x_max_det, y_max_det, 255, 0, 0)
    image = draw_rectangle(image, x_min_manu, y_min_manu, x_max_manu, y_max_manu, 0, 0, 255)
    image = rotate_image(image, -horizon_angle)

    # compute the intersection of the two rectangles
    xA = max(x_min_det, x_min_manu)
    yA = max(y_min_det, y_min_manu)
    xB = min(x_max_det, x_max_manu)
    yB = min(y_max_det, y_max_manu)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    jaccard = (interArea / (detected_surface + manual_surface - interArea)) *100
    print("Jaccard index - Intersection Over Union (%): ", jaccard)

    intersection_error = abs(1-interArea/manual_surface)*100
    print("Surface intersection error (%): ", intersection_error)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if horizon_angle_manu == horizon_angle:
        #open accuracy file and write errors
        with open(ACCURACY_FILE, "a") as file:
            file.write(image_name + "," + str(jaccard) + "," + str(intersection_error) + "\n")
    return


def create_accuracy_file(accuracy_path):
    if not os.path.exists(accuracy_path):
        with open(accuracy_path, "w") as file:
            pass  # Empty block to create the file
        print("Accuracy file created successfully.")
    else:
        print("Accuracy file already exists.")
    return

def clear_accuracy_file(accuracy_path):
    with open(accuracy_path, "w") as file:
        pass
    return

def compute_all_goals_surface():
    clear_accuracy_file(ACCURACY_FILE)
    with open(GOALS_REGISTRY_FILE, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            [image_name, horizon_angle, x_max, x_min, y_max, y_min] = line.split(",")
            main_compute_goal_surface(image_name, int(float(horizon_angle)//1), int(x_max), int(x_min), int(y_max), int(y_min))
    return

if __name__ == "__main__":
    create_accuracy_file(ACCURACY_FILE)
    compute_all_goals_surface()