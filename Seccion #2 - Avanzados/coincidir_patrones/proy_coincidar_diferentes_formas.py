import numpy as np
import cv2 as cv

# Load and then gray scale image

image = cv.imread('../../Resources/Images/someshapes.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

cv.imshow('Identicacion de formas',image)
cv.waitKey(0)

ret, thresh = cv.threshold(gray, 127, 255, 1)

# Encontrar los contornos de la imagen de referencia
contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# Recorrer todos los contornos hallados
for cnt in contours:
    
    # Obtener la aproximacion por poligono
    # arcLength calcula la longitud del contorno
    # Recibe el contorno y si la figura es o no es cerrada
    # 1% de la longitud del contorno como precision
    accuracy = 0.01 * cv.arcLength(cnt, True)
    
    # approxPolyDP recibe 3 parametros
    # contorno, aproximacion de exactitud: 5% es buen datos
    # closed: indica si la figura es cerrada o abierta
    approx = cv.approxPolyDP(cnt, accuracy, True)
    
    if len(approx) == 3:
        shape_name = "Triangulo"
        cv.drawContours(image,[cnt],0,(0,255,0),-1)
        
        # Find contour center to place text at the center
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    
    elif len(approx) == 4:
        x,y,w,h = cv.boundingRect(cnt)
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Check to see if 4-side polygon is square or rectangle
        # cv.boundingRect returns the top left and then width and 
        if abs(w-h) <= 3:
            shape_name = "Cuadrado"
            
            # Find contour center to place text at the center
            cv.drawContours(image, [cnt], 0, (0, 125 ,255), -1)
            cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        else:
            shape_name = "Rectangulo"
            
            # Find contour center to place text at the center
            cv.drawContours(image, [cnt], 0, (0, 0, 255), -1)
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            
    elif len(approx) == 10:
        shape_name = "Estrella"
        cv.drawContours(image, [cnt], 0, (255, 255, 0), -1)
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        
        
    elif len(approx) >= 15:
        shape_name = "Circulo"
        cv.drawContours(image, [cnt], 0, (0, 255, 255), -1)
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv.putText(image, shape_name, (cx-50, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    cv.imshow('Identicacion de formas',image)
    cv.waitKey(0)
    
cv.destroyAllWindows()
