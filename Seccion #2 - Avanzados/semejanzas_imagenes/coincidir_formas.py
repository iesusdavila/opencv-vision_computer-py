import cv2 as cv
import numpy as np

# Cargar la imagen de referencia
template = cv.imread('../../Resources/Images/4star.jpg',0)
cv.imshow('Template', template)
cv.waitKey()

# Cargar la imagen donde visualizaremos la coincidencia
target = cv.imread('../../Resources/Images/shapestomatch.jpg')
target_gray = cv.cvtColor(target,cv.COLOR_BGR2GRAY)

# Umbral de ambas images usando cv.threshold
ret, thresh1 = cv.threshold(template, 127, 255, 0)
ret, thresh2 = cv.threshold(target_gray, 127, 255, 0)

# Encontrar los contornos de la imagen de referencia
contours, hierarchy = cv.findContours(thresh1, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

# Siempre hacer esto: Ordenar los contornos por area y eliminar el mayor
# Porque toma como contorno la ventana de windows donde se abre la imagen
sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

# Extraer el segundo contorno mas grande
template_contour = contours[1]

# Encontrar los contornos de la imagen donde visualizaremos la coincidencia
contours, hierarchy = cv.findContours(thresh2, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

for c in contours:
    # Iterar atraves de contorno y verificar la coincidencia con la imagen de comparacion 
    match = cv.matchShapes(template_contour, c, 3, 0.0)
    
    # Validar la coincidencia otorgada por la funcion matchShapes
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = [] 
                
cv.drawContours(target, [closest_contour], -1, (0,255,0), 3)
cv.imshow('Output', target)
cv.waitKey()
cv.destroyAllWindows()