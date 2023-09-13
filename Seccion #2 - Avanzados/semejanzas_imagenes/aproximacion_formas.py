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

# Iterar alrededor de todos los contornos para dibujar los contornos en rectangulos
for c in contours:
    # Encuentra el rectangulo mas pequeño del contorno
    # Devuelve la posicion inicial en x-y asi como el ancho-alto
    x,y,w,h = cv.boundingRect(c)

    cv.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    
    cv.imshow('Rectangulo con Bounding', orig_image)

cv.waitKey(0) 
    
# Iterar alrededor de todos los contornos para dibujar de manera delimitada el poligono de contornos 
for c in contours:
    # arcLength calcula la longitud del contorno
    # Recibe el contorno y si la figura es o no es cerrada
    # 1% de la longitud del contorno como precision
    accuracy = 0.01 * cv.arcLength(c, True)
    
    # approxPolyDP recibe 3 parametros
    # contorno, aproximacion de exactitud: 5% es buen datos
    # closed: indica si la figura es cerrada o abierta
    approx = cv.approxPolyDP(c, accuracy, True)
    cv.drawContours(image, [approx], 0, (0, 255, 0), 2)
    
    cv.imshow('Aproximacion Poly DP', image)
    
cv.waitKey(0)   
cv.destroyAllWindows()