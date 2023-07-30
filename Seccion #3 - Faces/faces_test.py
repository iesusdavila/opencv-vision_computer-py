import numpy as np
import cv2 as cv

# --> cargar el clasificador de imagenes
# el clasificador es de OPENCV, no es creacion propia
haar_cascade = cv.CascadeClassifier('../Proyecto - Reconocimiento Facial/haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# crear un objeto de reconocimiento facial basado en el algoritmo "Local Binary Patterns Histograms" (LBPH)
# entrenar conjunto de imágenes de rostros conocidos para reconocer caras desconocidas basándose en las características extraídas de las imágenes de entrenamiento.
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# leer el modelo entrenado para detectar rostros
face_recognizer.read('face_trained.yml')

img = cv.imread(r'..\Resources\Faces\val\elton_john/1.jpg')

# convertir la imagen a escala de grises
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Persona', gray)

 # --> detectar rostros en imagenes
 # recibe 3 parametros: imagen a tratar,
 # factor de escala para la deteccion:  detecta objetos de diferentes tamaños, 1.1 es una reduccion del 10%.
 # numero minimo de vecinos requeridos para que se detecte una region como una cara
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)


for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]
    
    # predecir a que persona corresponde la imagen pasada
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Etiqueta = {people[label]} con una confianza de {confidence}')
    
    # colocar el nombre de la persona que se detecto
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    # dibujar un rectangulo sobre la imagen
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Imagen detectada', img)

cv.waitKey(0)