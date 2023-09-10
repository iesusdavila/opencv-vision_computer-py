import cv2
import numpy as np

# Funcion para generar el dibujo animado
def sketch(image):
    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Limpiar la imagen usando Guassian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    
    # Extraer bordes
    canny_edges = cv2.Canny(img_gray_blur, 50, 120)
    
    # Realizar una inversion binaria
    ret, mask = cv2.threshold(canny_edges, 150, 255, cv2.THRESH_BINARY_INV)
    return mask


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Camara animada', sketch(frame))
    if cv2.waitKey(1) == 13: # 13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()  