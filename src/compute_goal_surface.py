import cv2
from detection import main

def compute_goal_surface(x_max, x_min, y_max, y_min):
    # compute surface
    surface = abs(x_max - x_min) * abs(y_max - y_min)
    print(surface)
    return surface


def get_manual_goal_position(image_name, log_name):

    def choosePoints(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 5, (255,0,0), -1)
            print(f'{x}, {y}')
            points.append((x,y))
    points = []
    image = cv2.imread('./data/'+ log_name +'/' + image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', choosePoints)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points

[x_max_det, x_min_det, y_max_det, y_min_det] = detection.main() 
points = get_manual_goal_position('020-rgb.png', "log1")
x_max_manu = max(points[0][0], points[1][0], points[2][0], points[3][0])
x_min_manu = min(points[0][0], points[1][0], points[2][0], points[3][0])
y_max_manu = max(points[0][1], points[1][1], points[2][1], points[3][1])
y_min_manu = min(points[0][1], points[1][1], points[2][1], points[3][1])
manual_surface = compute_goal_surface(x_max_manu, x_min_manu, y_max_manu, y_min_manu)
detected_surface = compute_goal_surface(x_max_det, x_min_det, y_max_det, y_min_det)
print("Manual surface: ", manual_surface)
print("Detected surface: ", detected_surface)
