import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/park.jpg')
#cv.imshow('Park', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

# ------------- Metodo de Laplacian -------------
# Recibe 3 parametros: la imagen a tratar, profundidad de la imagen (normalmente CV_64F),
# el tamaño del kernel (1, 3, 5 o 7)
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Metodo Laplacian 1', lap)

# ------------- Metodo de Sobel -------------
# Recibe 5 parametros: imagen a tratar, profundidad de la imagen (normalmente CV_64F)
# ordenes de las derivadas en x, ordenes de las derivadas en y,
# tamaño del kernel (1,3,5 o 7)
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Sobel Combinado', combined_sobel)

# ------------- Metodo de Canny -------------
# recibe 4 parametros: imagen a tratar, umbral inferior, umbral superior
# tamaño del kernel y booleano que especifica si se debe utilizar la norma L2 para calcular el gradiente.
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)
cv.waitKey(0)