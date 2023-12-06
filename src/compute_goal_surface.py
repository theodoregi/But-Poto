import cv2
from detection import main_detection
from lib_poto_detection import draw_rectangle

def compute_goal_surface(x_max, x_min, y_max, y_min):
    # compute surface
    surface = abs(x_max - x_min) * abs(y_max - y_min)
    print(surface)
    return surface


def get_manual_goal_position(image_name):

    def choosePoints(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 5, (0,255,0), -1)
            print(f'{x}, {y}')
            points.append((x,y))
    points = []
    image = cv2.imread('./data/' + image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', choosePoints)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points

def main_compute_goal_surface(image_name):
    # Get goal position from detection
    [x_max_det, x_min_det, y_max_det, y_min_det] = main_detection(image_name)
    # Get goal position from manual selection
    points = get_manual_goal_position(image_name)
    x_max_manu = max(points[0][0], points[1][0], points[2][0], points[3][0])
    x_min_manu = min(points[0][0], points[1][0], points[2][0], points[3][0])
    y_max_manu = max(points[0][1], points[1][1], points[2][1], points[3][1])
    y_min_manu = min(points[0][1], points[1][1], points[2][1], points[3][1])
    # Compute surfaces
    manual_surface = compute_goal_surface(x_max_manu, x_min_manu, y_max_manu, y_min_manu)
    detected_surface = compute_goal_surface(x_max_det, x_min_det, y_max_det, y_min_det)
    print("Manual surface: ", manual_surface)
    print("Detected surface: ", detected_surface)
    erreur = abs(detected_surface - manual_surface) / manual_surface * 100
    print("Erreur: ", erreur)
    # Display surfaces to see differences
    image = cv2.imread('./data/' + img_name, cv2.IMREAD_COLOR)
    image = draw_rectangle(image, x_min_manu, y_min_manu, x_max_manu, y_max_manu, 0, 0, 255)
    image = draw_rectangle(image, x_min_det, y_min_det, x_max_det, y_max_det, 255, 0, 0)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_name = "log1/020-rgb.png"
    main_compute_goal_surface(img_name)