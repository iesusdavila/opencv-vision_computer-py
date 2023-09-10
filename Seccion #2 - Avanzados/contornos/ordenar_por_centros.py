import cv2 as cv
import numpy as np

def encontrar_contornos(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    contorno_gray = cv.Canny(gray, 125, 175)

    contornos, hierachies = cv.findContours(contorno_gray.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f'{len(contornos)} encontrados en la imagen de grises!')
    
    return contornos

def centro_figuras(image, c, dibujar_centros = True, dibujar_contornos = False, escribir_texto = False, texto = ""):
    # Centro de los contornos - Centroide
    M = cv.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    if dibujar_centros:
        cv.circle(image,(cx,cy), 10, (0,0,255), -1)
    if dibujar_contornos:
        cv.drawContours(image, [c], -1, (0,0,255), 3)
    if escribir_texto:
        cv.putText(image, texto, (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

def x_cord_contour(contours):
    # Retornar la coordenada x para el centroide del contorno
    if cv.contourArea(contours) > 10:
        M = cv.moments(contours)
        return (int(M['m10']/M['m00']))

image = cv.imread('../../Resources/Images/bunchofshapes.jpg')
orginal_image = image

contornos = encontrar_contornos(image)

for (i, c) in enumerate(contornos):
    orig = centro_figuras(image, c, True, False)

cv.imshow("Centroide de los contornos: ", image)
cv.waitKey(0)

contornos_ordenados = sorted(contornos, key = x_cord_contour, reverse = False)

for (i,c) in enumerate(contornos_ordenados):
    centro_figuras(orginal_image, c, False, True, True, str(i+1))
    
    cv.imshow('Contador de figuras', orginal_image)
    cv.waitKey(0)
    
    # Encontrar las limitadores del rectangulo
    (x, y, w, h) = cv.boundingRect(c)  
    
    # Recortar la imagen para que solo aparezca la figura
    cropped_contour = orginal_image[y:y + h, x:x + w]
    image_name = "figura_salida_" + str(i+1) + ".jpg"
    
    # Guardar imagen
    cv.imwrite(image_name, cropped_contour)
    
cv.destroyAllWindows()