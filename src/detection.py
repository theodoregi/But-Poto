from lib_poto_detection import *

img_name = '020-rgb.png'

def main():
    er = 2
    erosion_size = (er,er)
    di = 2
    dilate_size = (di,di)
    cl = 3
    closing_size = (di,di)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closing_size)

    img_grey1 = cv2.imread('./data/log1/'+img_name, 0)
    img_grey1 = cv2.morphologyEx(img_grey1, cv2.MORPH_CLOSE, kernel, iterations=1)

    img_color1 = cv2.imread('./data/log1/'+img_name, 1)

    img_color2 = cv2.imread('./data/log1/'+img_name, 1)


    #pré traitement
    for i in range(5):
        img_grey1 = adaptative_binarization(img_grey1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_size)
        img_grey1 = cv2.dilate(img_grey1, kernel, iterations=1)


        img_grey1 = cv2.equalizeHist(img_grey1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_size)
        img_grey1 = cv2.erode(img_grey1, kernel, iterations=1)

    ####################

    # contours, hierarchy = cv2.findContours(img_grey1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img_grey1, contours, -1, (255,255,255), 3)

    #### on recherche les lignes et on les copie sur img_color1
    img_color1,lines=search_and_draw_lines(img_grey1,img_color1,1.3,np.pi,100,20,1)

    #### on garde la couleur bleue et on l'applique à img_color1. Sinon, on garde du niveau de gris
    img_color1=keep_only_color(img_color1,img_grey1,255,0,0)

    #### on crée et applique le masque
    mask = cv2.imread('./data/mask/log1/'+img_name, 0)
    img_color1=apply_mask(img_color1,mask)

    #### on garde la couleur bleue et on l'applique à img_color1. Sinon, on met du noir.
    #### avec ça, on ne garde que le bas des poteaux et cela est colorié en bleu.
    black = np.zeros((len(img_color1), len(img_color1[0]), 3), dtype = "uint8")
    img_color1=keep_only_color(img_color1,black,255,0,0)

    #### on recherche les lignes et on les copie sur img_color1
    img_color1,reg=register_lines(img_color1,lines)

    #### on garde les lignes les plus grandes et on trace les buts
    xmax = max(reg, key=lambda x: x[0])[0][0]
    xmin = min(reg, key=lambda x: x[0])[0][0]
    ymax = max(max(reg, key=lambda x: x[1][1])[1][1], max(reg, key=lambda x: x[0][1])[0][1])
    ymin = min(min(reg, key=lambda x: x[1][1])[1][1], min(reg, key=lambda x: x[0][1])[0][1])
    img_color1=draw_rectangle(img_color1, xmin, ymin, xmax, ymax, 255, 0, 0)

    #### on garde la couleur bleue et on l'applique à img_color2.
    img_color2 = copy_and_keep(img_grey1, img_color1, img_color2, 255, 0, 0)

    display(img_color2)
    return

main()