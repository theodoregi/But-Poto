import cv2
import numpy as np
import os

def create_mask(image_name):

    def choosePoints(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 5, (255,0,0), -1)
            print(f'{x}, {y}')
            points.append((x,y))

    points = []
    image = cv2.imread('./data/log1/' + image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', choosePoints)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(points)
    # create mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
    cv2.imwrite('./data/all_mask/log1/' + image_name, mask)

# for i in range (1,5):
#     images_directory = './data/log' + str(i) + '/'
#     image_files = [f for f in os.listdir(images_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
#     for image_file in image_files:
#         create_mask(image_file)

create_mask('004-rgb.png')