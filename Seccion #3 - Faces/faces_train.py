import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'..\Resources\Faces\train'

# --> cargar el clasificador de imagenes
# el clasificador es de OPENCV, no es creacion propia
haar_cascade = cv.CascadeClassifier('../Proyecto - Reconocimiento Facial/haar_face.xml')


# lista de caracteristicas
features = []
# lista de etiquetas
labels = []

# funcion para entrenar el modelo
def create_train():
    # recorrer todas las personas de la lista
    for person in people:
        # añadir a la ruta las personas para conocer
        path = os.path.join(DIR, person)
        label = people.index(person)

        # recorrer todas las imagenes de la carpeta de la persona
        for img in os.listdir(path):
            # obtener la ruta de la imagen
            img_path = os.path.join(path,img)
            
            # leer la imagen por opencv
            img_array = cv.imread(img_path)
            # validar si la imagen es nula
            if img_array is None:
                continue 
            
            # convertir la imagen a escala de grises
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            # --> detectar rostros en imagenes
            # recibe 3 parametros: imagen a tratar,
            # factor de escala para la deteccion:  detecta objetos de diferentes tamaños, 1.1 es una reduccion del 10%.
            # numero minimo de vecinos requeridos para que se detecte una region como una cara
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            # --> recorrer todas las caras encontradas
            # x = coordenada x del vertice superior izquierdo del rectangulo
            # y = coordenada y del vertice superior izquierdo del rectangulo
            # w = ancho del rectangulo, h = altura del rectangulo
            for (x,y,w,h) in faces_rect:
                
                # segmentar el rostro de la persona
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Entrenamiento realizado...!!')

features = np.array(features, dtype='object')
labels = np.array(labels)

# crear un objeto de reconocimiento facial basado en el algoritmo "Local Binary Patterns Histograms" (LBPH)
# entrenar conjunto de imágenes de rostros conocidos para reconocer caras desconocidas basándose en las características extraídas de las imágenes de entrenamiento.
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# entrenar el modelo en base a las caracteristicas y etiquetas
face_recognizer.train(features,labels)

# guardar el modelo en un archivo llamado "face_trained.yml"
face_recognizer.save('face_trained.yml')
# guardar la matriz de caracteristicas y etiquetas
np.save('features.npy', features)
np.save('labels.npy', labels)
