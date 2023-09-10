import cv2
import numpy as np

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # --> detectar rostros en imagenes
    # recibe 3 parametros: imagen a tratar,
    # factor de escala para la deteccion:  detecta objetos de diferentes tamaños, 1.1 es una reduccion del 10%.
    # numero minimo de vecinos requeridos para que se detecte una region como una cara
    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (i,(x,y,w,h)) in enumerate(faces):
        cv2.putText(frame, "Persona "+str(i+1), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,255,0), thickness=3)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.putText(frame, "Personas encontradas "+str(i+1), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=3)
    
    cv2.imshow('Video en tiempo real',frame)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
    
cap.release()
cv2.destroyAllWindows()