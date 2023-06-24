import cv2 as cv
import sklearn as skl

img = cv.imread('../Resources/Photos/park.jpg')

cv.imshow('Imagen sin cambios', img)

# Convertir a escala de grises
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grises', gray)

# Desenfocar
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Limitar bordes - Revisar mas info
canny = cv.Canny(img, 125, 175)
cv.imshow('Limitar',canny)

# Dilatar bordes - Revisar mas info
dilated = cv.dilate(img, (8,8), iterations=20)
cv.imshow('Dilatacion',dilated)

# Recortar imagen
recorte = img[200:300, 200:400]
cv.imshow('Recorte', recorte)

cv.waitKey(0)