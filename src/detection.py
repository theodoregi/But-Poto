from lib_poto_detection import *
from copy import deepcopy
from prediction import create_mask

# # MODIFIER CE PARAMETRES POUR RENDRE LA DETECTION MEILLEURE
EROSION_ITERATIONS = 30 # # 30
DILATION_ITERATIONS = 30 # # 30
MIN_LINE_LENGTH = 50 # # 50
NUMBER_OF_LINES = 100 # # 100
WIDTH_RESEARCH = 1.3 # # 1.3
ANGLE_RESOLUTION = np.pi # # pi

def main_detection(image_name, flag_display = 1, flag_debug = 0):
    mask_name = create_mask(image_name, MASK_GENERATION_REPO)
    mask = cv2.imread(mask_name, 0)

    # définition des paramètres de pré traitement
    er = 2
    erosion_size = (er,er)
    di = 2
    dilate_size = (di,di)
    cl = 2
    closing_size = (cl,cl)

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, closing_size)
    kernel_dilate  = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_size)
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_size)

    preprocessing = er+di+cl


    # pré traitement
    img_grey = cv2.imread('./data/'+image_name, 0)
    img_grey = cv2.morphologyEx(img_grey, cv2.MORPH_CLOSE, kernel_closing, iterations=1)

    img_base = cv2.imread('./data/'+image_name, 1)
    img_color = deepcopy(img_base)

    size_x, size_y = len(img_grey), len(img_grey[0])

    # encore pré traitement
    for i in range(5):
        img_grey = adaptative_binarization(img_grey)
        img_grey = cv2.dilate(img_grey, kernel_dilate, iterations=1)
        img_grey = cv2.equalizeHist(img_grey)
        img_grey = cv2.erode(img_grey, kernel_erosion, iterations=1)

    ########################################################################################

    #### on calcule l'angle de l'horizon
    horizon_angle = detect_horizon_angle(mask)
    if flag_display:
        print("Horizon angle :", horizon_angle)

    #### on effectue une rotation de l'image
    img_grey = rotate_image(img_grey, horizon_angle) 
    img_color = rotate_image(img_color, horizon_angle)
    img_grey_3_channels = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

    #### on recherche les lignes et on les copie sur img_grey_3_channels
    img_grey_3_channels,lines=search_and_draw_lines(img_grey,img_grey_3_channels,WIDTH_RESEARCH,ANGLE_RESOLUTION,NUMBER_OF_LINES,MIN_LINE_LENGTH,1)
    
    #### on garde la couleur bleue et on l'applique à img_grey_3_channels. Sinon, on garde du niveau de gris
    img_grey_3_channels=keep_only_color(img_grey_3_channels,img_grey,255,0,0)
    if flag_debug:
        display(img_grey_3_channels)

    #### on crée et applique le masque (soustraction de dilatation et erosion)
    mask = rotate_image(mask, horizon_angle)
    eroded_mask = deepcopy(mask)
    dilated_mask = deepcopy(mask)
    eroded_mask = cv2.erode(eroded_mask, kernel_erosion, iterations=EROSION_ITERATIONS)
    dilated_mask = cv2.dilate(dilated_mask, kernel_dilate, iterations=DILATION_ITERATIONS)
    sub_mask = cv2.subtract(dilated_mask, eroded_mask)
    if flag_debug:
        display(sub_mask)
    
    img_grey_3_channels=apply_mask(img_grey_3_channels,sub_mask)

    #### on garde la couleur bleue. Sinon, on met du noir.
    #### avec ça, on ne garde que le bas des poteaux et ils sont colorés en bleu.
    img_black = np.zeros((size_x, size_y, 3), dtype = "uint8")
    img_black = rotate_image(img_black, horizon_angle)
    img_black=keep_only_color(img_black,img_grey_3_channels,255,0,0)
    if flag_debug:
            display(img_black)

    #### on recherche les lignes et on les copie sur img_black
    img_black,reg=register_lines(img_black,lines)

    #### on garde les lignes les plus grandes et on trace les buts
    try:
        xmax = max(reg, key=lambda x: x[0])[0][0] -preprocessing
        xmin = min(reg, key=lambda x: x[0])[0][0] -preprocessing
        ymax = max(max(reg, key=lambda x: x[1][1])[1][1], max(reg, key=lambda x: x[0][1])[0][1]) -preprocessing
        ymin = min(min(reg, key=lambda x: x[1][1])[1][1], min(reg, key=lambda x: x[0][1])[0][1]) -preprocessing
    except ValueError:
        print("No goal detected")
        return 0, 0, 0, 0, img_base, horizon_angle
    
    # correction of goal height if the detection is incorrect
    deltaX = abs(xmax - xmin)
    deltaY = abs(ymax - ymin)
    if deltaY > deltaX:
        ymin += int(deltaY / 2)

    img_color=draw_rectangle(img_color, xmin, ymin, xmax, ymax, 255, 0, 0)
    if flag_display:
        print("Detected points (xmax, xmin, ymax, ymin):", xmax, xmin, ymax, ymin)

    #### on rétablit la rotation originale
    img_color = rotate_image(img_color, -horizon_angle)

    # display(img_base)
    if flag_display:
        display(img_color)
    return xmax, xmin, ymax, ymin, img_base, horizon_angle


if __name__ == '__main__':
    img_name = 'log1/015-rgb.png'
    main_detection(img_name,flag_display = 1, flag_debug = 0)