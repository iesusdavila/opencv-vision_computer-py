import cv2 as cv

img = cv.imread('../Resources/Photos/group 2.jpg')
#cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray People', gray)

# --> cargar el clasificador de imagenes
# el clasificador es de OPENCV, no es creacion propia
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# --> detectar rostros en imagenes
# recibe 3 parametros: imagen a tratar,
# factor de escala para la deteccion:  permite detectar objetos de diferentes tamaños, 1.1 es una reduccion del 10%.
# numero minimo de vecinos requeridos para que se detecte una region como una cara
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=2.5, minNeighbors=5, minSize=(60, 60), flags = cv.CASCADE_SCALE_IMAGE)

# --> numero de caras encontradas
print(f'Numero de caras encontradas = {len(faces_rect)}')

# --> recorrer todas las caras encontradas
# x = coordenada x del vertice superior izquierdo del rectangulo
# y = coordenada y del vertice superior izquierdo del rectangulo
# w = ancho del rectangulo, h = altura del rectangulo
for (x,y,w,h) in faces_rect:
    # --> dibujar un rectangulo
    # parametros: imagen a tratar, coordenada donde inicia el rectangulo,
    # coordenada donde termina el rectangulo, color y grosos del rectangulo
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), thickness=3)

cv.imshow('Imagenes encontradas', img)

cv.waitKey(0)