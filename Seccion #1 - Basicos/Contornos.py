import cv2 as cv

img = cv.imread('../Resources/Photos/cat.jpg')

cv.imshow('Gatos',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Gatos Grises',gray)

contorno = cv.Canny(img,125,175)
cv.imshow('Gatos contorno', contorno)

contorno_gray = cv.Canny(gray,125,175)
cv.imshow('Gatos contorno grises', contorno_gray)

contornos, hierachies = cv.findContours(contorno, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contornos)} encontrados en la imagen!')

cv.waitKey(0)