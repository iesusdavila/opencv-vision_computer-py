import cv2 as cv
import numpy as np

def obtener_contornos_areas(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        all_areas.append(area)
    return all_areas

def encontrar_contornos(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    contorno_gray = cv.Canny(gray, 125, 175)

    contornos, hierachies = cv.findContours(contorno_gray.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f'{len(contornos)} encontrados en la imagen de grises!')
    
    return contornos

def mostrar_imagenes(texto, images):
    for c in images:
        cv.drawContours(orginal_image, [c], -1, (255,0,0), 3)
        cv.waitKey(0)
        cv.imshow(texto, orginal_image)

image = cv.imread('../../Resources/Images/bunchofshapes.jpg')
orginal_image = image

contornos = encontrar_contornos(image)

print("Areas antes de ordenarlas: ", obtener_contornos_areas(contornos))

contornos_ordenados = sorted(contornos, key=cv.contourArea, reverse=True)

print("Areas despues de ordenarlas: ", obtener_contornos_areas(contornos_ordenados))

mostrar_imagenes("Contornos ordenados de la imagen", contornos_ordenados)

cv.waitKey(0)
cv.destroyAllWindows()