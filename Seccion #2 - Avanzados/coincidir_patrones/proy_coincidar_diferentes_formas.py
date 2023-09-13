import numpy as np
import cv2 as cv

def calcular_centroide(M):
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    return (cx,cy)

def definir_figura(shape_name, image, cnt, color_shape, color_text, cx, cy, desfase_x = 0, desfase_y = 0):    
    cv.drawContours(image,[cnt],0,color_shape,-1)
    cv.putText(image, shape_name, (cx-desfase_x, cy-desfase_y), cv.FONT_HERSHEY_SIMPLEX, 1, color_text, 1)

image = cv.imread('../../Resources/Images/someshapes.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

cv.imshow('Identicacion de formas',image)
cv.waitKey(0)

ret, thresh = cv.threshold(gray, 127, 255, 1)

contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

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
    
    # Encontrar el centroide de la imagen
    M = cv.moments(cnt)
    cx, cy = calcular_centroide(M)
    
    # VALIDACIONES PARA CONOCER QUE TIPO DE FIGURAS TENEMOS
    # Dependiendo del numero de vertices sabemos que figura se tienes
    if len(approx) == 3:
        definir_figura(shape_name = "Triangulo", image=image, cnt=cnt, color_shape=(0,255,0), color_text=(0,0,0), cx=cx, cy=cy, desfase_x=70)
    
    elif len(approx) == 4:
        x,y,w,h = cv.boundingRect(cnt)
        
        if abs(w-h) <= 3:
            definir_figura(shape_name = "Cuadrado", image=image, cnt=cnt, color_shape=(0, 125 ,255), color_text=(0,0,0), cx=cx, cy=cy, desfase_x=70)
        else:
            definir_figura(shape_name = "Rectangulo", image=image, cnt=cnt, color_shape=(0, 0, 255), color_text=(0,0,0), cx=cx, cy=cy, desfase_x=70)

            
    elif len(approx) == 10:
        definir_figura(shape_name = "Estrella", image=image, cnt=cnt, color_shape=(255, 255, 0), color_text=(0,0,0), cx=cx, cy=cy, desfase_x=70)
        
    elif len(approx) >= 15:
        definir_figura(shape_name = "Circulo", image=image, cnt=cnt, color_shape=(0, 255, 255), color_text=(0,0,0), cx=cx, cy=cy, desfase_x=50)


    cv.imshow('Identicacion de formas',image)
    cv.waitKey(0)
    
cv.destroyAllWindows()
