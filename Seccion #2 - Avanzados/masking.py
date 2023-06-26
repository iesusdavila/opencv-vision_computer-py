import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cats 2.jpg')
#cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
#cv.imshow('Imagen blanco', blank)

# creacion de un circulo centrado con radio de 100px, color blanco y que rellene la imagen
circle = cv.circle(blank.copy(), (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
# creacion de un rectangulo que inicio en (30,30) y termine en (370,370), color blanco y que rellene la imagen
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

forma_mascara = cv.bitwise_and(circle,rectangle)
#cv.imshow('Interseccion Circulo y Cuadrado', forma_mascara)

mascara = cv.bitwise_and(img,img,mask=forma_mascara)
cv.imshow('Mascara a los gatos', mascara)

cv.waitKey(0)