import cv2
import numpy as np


er = 2
erosion_size = (er,er)
di = 2
dilate_size = (di,di)
cl = 3
closing_size = (di,di)

img = cv2.imread('./data/log1/001-rgb.png', 0)
img_color = cv2.imread('./data/log1/001-rgb.png', 1)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closing_size)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

def adaptative_binarization(img):
    m = max(2*np.median(img), np.mean(img))
    print(m)
    ret, bin = cv2.threshold(img, m, 255, cv2.THRESH_BINARY)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if bin[i][j] == 255:
                img[i][j] = img[i][j]
            else:
                img[i][j] = 0
    return img


for i in range(5):
    img = adaptative_binarization(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_size)
    img = cv2.dilate(img, kernel, iterations=1)


    img = cv2.equalizeHist(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_size)
    img = cv2.erode(img, kernel, iterations=1)


# contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (255,255,255), 3)

lines = cv2.HoughLinesP(img, 1.3, np.pi, 100, minLineLength=20, maxLineGap=1)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_color, (x1,y1), (x2,y2), (255,0,0), 2)

for i in range(len(img)):
    for j in range(len(img[i])):
        if img_color[i][j][0] != 0 and img_color[i][j][1] != 0 and img_color[i][j][2] != 255:
            img_color[i][j] = img[i][j]

cv2.imshow('image', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()