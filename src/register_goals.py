import os
import cv2
from lib_poto_detection import *
from prediction import create_mask
from detection import main_detection
from query_yes_no import query_yes_no

GOALS_REGISTRY_FILE = "./src/goals_registry.csv"

def get_manual_goal_position(image):

    def choosePoints(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 5, (0,255,0), -1)
            # print(f'{x}, {y}')
            points.append((x,y))
    points = []
    cv2.imshow('please select 4 points', image)
    cv2.setMouseCallback('please select 4 points', choosePoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points

def main_register_goals(image_name):
    create_mask(image_name, MASK_GENERATION_REPO)
    mask_name = MASK_GENERATION_REPO+image_name[5:]
    mask = cv2.imread(mask_name, 0)
    horizon_angle = detect_horizon_angle(mask)

    image = cv2.imread('./data/'+image_name, 1)
    image = rotate_image(image, horizon_angle)
    
    # Get goal position from manual selection
    points = get_manual_goal_position(image)
    if len(points) != 4:
        print("!!! ERROR IN MANUAL SELECTION : you must select 4 points !!!")
        return
    x_max_manu = max(points[0][0], points[1][0], points[2][0], points[3][0])
    x_min_manu = min(points[0][0], points[1][0], points[2][0], points[3][0])
    y_max_manu = max(points[0][1], points[1][1], points[2][1], points[3][1])
    y_min_manu = min(points[0][1], points[1][1], points[2][1], points[3][1])
    # print("Manual points :", x_min_manu, y_min_manu, x_max_manu, y_max_manu)
    
    # Draw rectangles
    image = draw_rectangle(image, x_min_manu, y_min_manu, x_max_manu, y_max_manu, 0, 0, 255)
    image = rotate_image(image, -horizon_angle)

    display(image, "Manual detection")
    # wait for key : if not OK, just return
    response = query_yes_no("Is the goal placed correctly?")
    if response:
        register_new_goal(GOALS_REGISTRY_FILE, image_name, horizon_angle, x_max_manu, x_min_manu, y_max_manu, y_min_manu)
        print("Goal registered successfully.")
    else:
        print("Goal not registered.")
    return

def create_register_file(registry_path):
    if not os.path.exists(registry_path):
        with open(registry_path, "w") as file:
            pass  # Empty block to create the file
        print("Registry file created successfully.")
    else:
        print("Registry file already exists.")
    return

def register_new_goal(registry_path, name, horizon_angle, xmax, xmin, ymax, ymin):
    if not os.path.exists(registry_path):
        print("Registry does not exist.")
        raise FileNotFoundError
    # read the contents of the file
    with open(registry_path, "r") as file:
        lines = file.readlines()
    # check if the line already exists in the file
    for i, line in enumerate(lines):
        line_parts = line.strip().split(",")
        if len(line_parts) == 6:
            file_name = line_parts[0]
            if file_name != name:
                continue
            else:
                print("Line already exists in the file.")
                return
    # append the new line to the file
    with open(registry_path, "a") as file:
        file.write(f"{name},{horizon_angle},{xmax},{xmin},{ymax},{ymin}")
        file.write('\n')
    return

def get_last_goal(registry_path):
    if not os.path.exists(registry_path):
        print("Registry does not exist.")
        raise FileNotFoundError
    with open(registry_path, "r") as file:
        lines = file.readlines()
    if len(lines) > 0:
        for i in range(len(lines)-1, -1, -1):
            if lines[i] != "\n":
                last_line_index = i
                break
        last_line = lines[last_line_index]
        line_parts = last_line.strip().split(",")
        if len(line_parts) == 6:
            file_name, horizon_angle, x_max, x_min, y_max, y_min = line_parts
            return file_name
    return

def get_next_goal(registry_path, default="log1/011-rgb.png"):
    if not os.path.exists(registry_path):
        print("Registry does not exist.")
        raise FileNotFoundError
    last_goal = get_last_goal(registry_path)
    print("Last image: ", last_goal)
    if last_goal is None:
        return default
    last_num = last_goal[5:8]
    last_num = int(last_num)
    last_num += 1
    last_num = str(last_num)
    additional_zeros = "0"*(3-len(last_num))
    next_goal = last_goal[:5] + additional_zeros + last_num + last_goal[8:]
    print("Next image: ", next_goal)
    if not os.path.exists("./data/" + next_goal):
        print("Next image does not exist.")
        return
    return next_goal

if __name__ == "__main__":
    create_register_file(GOALS_REGISTRY_FILE)
    img_name = get_next_goal(GOALS_REGISTRY_FILE)
    main_register_goals(img_name)