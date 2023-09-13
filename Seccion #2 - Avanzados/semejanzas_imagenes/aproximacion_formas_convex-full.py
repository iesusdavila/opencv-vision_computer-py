import numpy as np
import cv2 as cv

def encontrar_contornos(image, thesh_inf, thresh_sup, funcion_contornos, aproximacion):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    contorno_gray = cv.Canny(gray, thesh_inf, thresh_sup)
    
    contornos, hierachies = cv.findContours(contorno_gray.copy(), funcion_contornos, aproximacion)
    print(f'{len(contornos)} encontrados en la imagen de grises!')
    
    return contornos

image = cv.imread('../../Resources/Images/house.jpg')
orig_image = image.copy()
# cv.imshow('Original Image', orig_image)
# cv.waitKey(0)

contours = encontrar_contornos(image, 127, 255, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# Siempre hacer esto: Ordenar los contornos por area y eliminar el mayor
# Porque toma como contorno la ventana de windows donde se abre la imagen
n = len(contours) - 1
contours = sorted(contours, key=cv.contourArea, reverse=False)[:n]

# Iterar alrededor de todos los contornos para dibujar de manera delimitada el Convex Hull 
for c in contours:
    hull = cv.convexHull(c)
    cv.drawContours(image, [hull], 0, (0, 255, 0), 2)
    cv.imshow('Convex Hull', image)

cv.waitKey(0)    
cv.destroyAllWindows()