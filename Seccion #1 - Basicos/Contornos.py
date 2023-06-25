import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cat.jpg')
cv.imshow('Gatos', img)

# ---- Primera forma de encontrar bordes ----
contorno = cv.Canny(img, 125, 175)
cv.imshow('Gatos contorno', contorno)

contornos, hierachies = cv.findContours(contorno, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contornos)} encontrados en la imagen sin ningun efecto!')

# ---- Segunda forma de encontrar bordes ----
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gatos Grises', gray)

contorno_gray = cv.Canny(gray, 125, 175)
cv.imshow('Gatos contorno grises', contorno_gray)

contornos_grises, hierachies_grises = cv.findContours(contorno_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contornos_grises)} encontrados en la imagen de grises!')

# ---- Tercera forma de encontrar bordes ----
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Contornos por Umbral', thresh)

contornos_thres, hierachies_thres = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contornos_thres)} encontrados en la imagen aplicado threshold!')

# ---- Dibujar los bordes ----
blank = np.zeros(img.shape, dtype='uint8')
blank_grises = np.zeros(img.shape, dtype='uint8')
blank_thres = np.zeros(img.shape, dtype='uint8')

cv.drawContours(blank, contornos, -1, (0, 255, 0), 1)
cv.imshow('Dibujo de contornos imagen normal', blank)

cv.drawContours(blank_grises, contornos_grises, -1, (0, 0, 255), 1)
cv.imshow('Dibujo de contornos imagen en grises', blank_grises)

cv.drawContours(blank_thres, contornos_thres, -1, (255, 0, 0), 1)
cv.imshow('Dibujo de contornos imagen aplicado threshold', blank_thres)

cv.waitKey(0)
