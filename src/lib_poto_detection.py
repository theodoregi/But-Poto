import cv2
import numpy as np

def raise_error_size_diff(img1,img2):
    if len(img1) != len(img2) or len(img1[0]) != len(img2[0]):
        raise ValueError("The two images must have the same size")
    return
    


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


def copy_and_keep(img_base, img_w_draw, img_color, blue, green,red):
    raise_error_size_diff(img_base,img_w_draw)
    raise_error_size_diff(img_base,img_color)
    for i in range(len(img_base)):
        for j in range(len(img_base[i])):
            if img_w_draw[i][j][0] == blue and img_w_draw[i][j][1] == green and img_w_draw[i][j][2] == red:
                img_color[i][j] = img_w_draw[i][j]
            else :
                img_color[i][j] = img_base[i][j]
    return img_color


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