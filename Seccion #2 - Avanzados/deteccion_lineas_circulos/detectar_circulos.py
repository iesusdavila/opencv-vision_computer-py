import cv2
import numpy as np
import cv2 as cv
 
image = cv.imread('../../Resources/Images/bottlecaps.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Aplicar un suavizado para reducir el ruido
#gray = cv.GaussianBlur(gray, (9, 9), 2, 3)
blur = cv.medianBlur(gray, 5)

# Utilizar la función HoughCircles para detectar círculos
circles = cv.HoughCircles(
    blur,  # Imagen en escala de grises
    cv.HOUGH_GRADIENT,  # Método de detección (otros métodos disponibles)
    dp=1,  # Resolución inversa de acumulador
    minDist=20,  # Distancia mínima entre los centros de los círculos detectados
    param1=50,  # Umbral de detección de bordes (ajustar según la imagen)
    param2=30,  # Umbral de votación (ajustar según la imagen)
    minRadius=35,  # Radio mínimo del círculo
    maxRadius=50   # Radio máximo del círculo
)

# Si se detectan círculos
if circles is not None:
    # Convierte las coordenadas y el radio a enteros
    circles = np.uint16(np.around(circles))

    # Dibujar los círculos detectados
    print(f'Se encontraron un total de {len(circles[0, :])} tapas')

    for circle in circles[0, :]:
        center_x = circle[0]
        center_y = circle[1]
        center = (center_x, center_y)  # Coordenadas del centro
        radius = circle[2]  # Radio
        
        blue = (255,0,0)
        
        cv.circle(image, center, radius, (0, 255, 0), 2)  # Dibuja el círculo
        
        cv.circle(image, center, 2, blue, 5)
        
        cv.line(image, center, (center_x+15, center_y), blue)
        cv.line(image, center, (center_x-15, center_y), blue)
        cv.line(image, center, (center_x, center_y+15), blue)
        cv.line(image, center, (center_x, center_y-15), blue)
        
    
    # Mostrar la imagen con los círculos detectados
    cv.imshow('Circulos Detectados', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No se detectaron circulos en la imagen.")