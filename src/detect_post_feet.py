import cv2
import numpy as np

kernel_size = 5

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img = cv2.imread('But-Poto/data/log1/010-rgb.png', cv2.IMREAD_GRAYSCALE)
terrain_mask = cv2.imread('But-Poto/data/mask/log1/010-rgb.png', cv2.IMREAD_GRAYSCALE)
assert(terrain_mask.shape == img.shape)

for row in range(len(img)):
    for col in range(len(img[row])):
        if (terrain_mask[row][col] == 0):
            img[row][col] = 0

img_eroded = cv2.erode(img, kernel, iterations=1)
img -= img_eroded

img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

img = cv2.erode(img, kernel_2, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel_2, iterations=4)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()