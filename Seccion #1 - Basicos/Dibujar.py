import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')

# ----------------------------------------
# Dibujar 
blank[:100,200:300] = 0,0,255
blank[100:200,100:400] = 0,150,255
blank[200:300,100:400] = 0,255,255
blank[300:400,100:400] = 150,255,255
blank[400:500,200:300] = 255,255,255

# ----------------------------------------
# Dibujar un rectangulo 
cv.rectangle(blank, (100,0), (400,500), (150,150,150), thickness=3)
cv.rectangle(blank, (200,0), (300,100),(0,0,255), thickness=-1)
cv.rectangle(blank, (100,100), (400,200),(0,150,255), thickness=-1)
cv.rectangle(blank, (100,200), (400,300),(0,255,255), thickness=-1)
cv.rectangle(blank, (100,300), (400,400),(150,255,255), thickness=-1)
cv.rectangle(blank, (200,400), (300,500),(255,255,255), thickness=-1)

# ----------------------------------------
# Dibujar un circulo 
cv.circle(blank, (250,250), 40, (0,0,255), thickness=-1)
cv.circle(blank, (250,250), 60, (0,150,255), thickness=1)
cv.circle(blank, (250,250), 80, (0,255,255), thickness=1)
cv.circle(blank, (250,250), 100, (150,255,255), thickness=1)
cv.circle(blank, (250,250), 120, (255,255,255), thickness=1)

# ----------------------------------------
# Dibujar una línea
cv.line(blank, (0,250),(500,250), (255,255,255), thickness=3)

# ----------------------------------------
# Colocar texto
cv.putText(blank, 'Hola, me llamo Iesus.', (50,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)

cv.imshow('Gato', blank)

cv.waitKey(0)