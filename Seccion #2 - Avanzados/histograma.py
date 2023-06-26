import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('../Resources/Photos/cats.jpg')
#cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# circulo centrado de radio 100, color blanco y que rellene su fondo
mascara = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

# --------------- HISTOGRAMA PARA IMAGEN EN GRISES ---------------
# transformacion de imagen en gris
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# aplica mascara sobre la imagen en grises
img_mascara = cv.bitwise_and(gray,gray,mask=mascara)
cv.imshow('Mascara sobre gatos grises', img_mascara)

# Histograma de la imagen en grises
# calcHist recibe: lista de imagenes, lista de canales, la mascara
gray_hist = cv.calcHist([gray], [0], img_mascara, [256], [0,256] )

# Esta seccion es para graficar
plt.figure()
plt.title('Histograma de imagen en grises')
plt.xlabel('Intensidad')
plt.ylabel('# pixeles')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

# --------------- HISTOGRAMA PARA IMAGEN SIN EFECTOS ---------------
# aplica mascara sobre la imagen
img_mascara = cv.bitwise_and(img,img,mask=mascara)
cv.imshow('Mascara sobre gatos sin transformaciones', img_mascara)

# Esta seccion es para graficar
plt.figure()
plt.title('Histograma de imagen a color')
plt.xlabel('Intensidad')
plt.ylabel('# pixeles')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    # Histograma de la imagen
    # calcHist recibe: lista de imagenes, lista de canales, la mascara
    hist = cv.calcHist([img], [i], mascara, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])
plt.show()

cv.waitKey(0)