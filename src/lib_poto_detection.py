from math import *
import sys
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

def raise_error_size_diff(img1,img2):
    if len(img1) != len(img2) or len(img1[0]) != len(img2[0]):
        raise ValueError("The two images must have the same size but have sizes {}x{} and {}x{}".format(len(img1),len(img1[0]),len(img2),len(img2[0])))
    return
    

def adaptative_binarization(img):
    m = max(2*np.median(img), np.mean(img))
    # print(m)
    ret, bin = cv2.threshold(img, m, 255, cv2.THRESH_BINARY)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if bin[i][j] == 255:
                img[i][j] = img[i][j]
            else:
                img[i][j] = 0
    return img


def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def search_and_draw_lines(img_search,img_draw,width_research,angle,line_number,minLength,maxGap):
    raise_error_size_diff(img_search,img_draw)
    lines = cv2.HoughLinesP(img_search, width_research, angle, line_number, minLineLength=minLength, maxLineGap=maxGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_draw, (x1,y1), (x2,y2), (255,0,0), 2)
    return img_draw,lines


def apply_mask(img, mask): # also keep the chosen color
    raise_error_size_diff(img,mask)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 0:
                img[i][j] = [0,0,0]
    return img


def keep_only_color(img,img_replace,blue,green,red):
    raise_error_size_diff(img,img_replace)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j][0] != blue or img[i][j][1] != green or img[i][j][2] != red:
                img[i][j] = img_replace[i][j]
    return img


def draw_rectangle(img, xmin, ymin, xmax, ymax, blue, green, red):
    cv2.line(img, (xmin,ymax), (xmax,ymax), (blue, green, red), 2)
    cv2.line(img, (xmin,ymin), (xmax,ymin), (blue, green, red), 2)

    cv2.line(img, (xmin,ymin), (xmin,ymax), (blue, green, red), 2)
    cv2.line(img, (xmax,ymin), (xmax,ymax), (blue, green, red), 2)
    return img


def copy_and_keep(img_base, img_w_draw, img_res, blue, green,red):
    raise_error_size_diff(img_base,img_w_draw)
    raise_error_size_diff(img_base,img_res)
    for i in range(len(img_base)):
        for j in range(len(img_base[i])):
            if img_w_draw[i][j][0] == blue and img_w_draw[i][j][1] == green and img_w_draw[i][j][2] == red:
                img_res[i][j] = img_w_draw[i][j]
            else :
                img_res[i][j] = img_base[i][j]
    return img_res


def register_lines(img,lines):
    reg = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if img[y1][x1][0] == 255 and img[y1][x1][1] == 0 and img[y1][x1][2] == 0:
            if img[y2][x2][0] == 255 and img[y2][x2][1] == 0 and img[y2][x2][2] == 0:
                cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
                reg.append([[x1,y1],[x2,y2]])
            else:
                cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
                reg.append([[x1,y1],[x2,y2]])
        else:
            if img[y2][x2][0] == 255 and img[y2][x2][1] == 0 and img[y2][x2][2] == 0:
                cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
                reg.append([[x1,y1],[x2,y2]])
            else:
                cv2.line(img, (x1,y1), (x2,y2), (0,0,0), 2)
    return img,reg


def detect_horizon_angle(mask, scharr=False):
    # set the kernel size, depending on whether we are using the Sobel
    # operator of the Scharr operator, then compute the gradients along
    # the x and y axis, respectively

    # print(mask.shape)
    mask_height = mask.shape[0]
    mask_width = mask.shape[1]
    ksize = -1 if scharr else 3
    gX = cv2.Sobel(mask, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(mask, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

    # the gradient magnitude images are now of the floating point data
    # type, so we need to take care to convert them back a to unsigned
    # 8-bit integer representation so other OpenCV functions can operate
    # on them and visualize them

    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)

    # combine the gradient representations into a single image
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    
    # show our output images
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Sobel/Scharr Combined", combined)
    # cv2.waitKey(0)

    # Compute the angle of the horizon with linear regression
    points_x = []
    points_y = []
    for i in range(mask_height):
        for j in range(mask_width):
            if combined[i][j] > 0:
                points_x.append(j)
                points_y.append(i)

    if len(points_x) == 0:
        return None
    else:
        X = np.array(points_x).reshape(-1, 1)
        Y = np.array(points_y)
        model = LinearRegression(n_jobs=-1).fit(X, Y)
        return atan(model.coef_[0]) * 180 / pi


def get_test_mask_horizontal(shape=(448, 800)):
    mask = np.zeros(shape, dtype = "uint8")
    mask[shape[0]//2:, :] = 255
    return mask


def get_test_mask_vertical(shape=(448, 800)):
    mask = np.zeros(shape, dtype = "uint8")
    mask[:, shape[1]//2:] = 255
    return mask


def rotate_image(image, angle):
    # grab the dimensions of the image and calculate the center of the
    # image
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

if __name__ == "__main__" :
    sys.exit(0)