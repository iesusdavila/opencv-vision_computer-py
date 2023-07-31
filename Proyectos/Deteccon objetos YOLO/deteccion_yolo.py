import cv2
import numpy as np

# Cargar el modelo YOLOv3 y las etiquetas
yolo_net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
with open("coco.names", "r") as f:
    labels = f.read().strip().split("\n")

# Obtener las capas de salida del modelo YOLO
output_layers = yolo_net.getUnconnectedOutLayersNames()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # leer cada frame del video
    ret, frame = cap.read()

    # cv2.dnn.blobFromImage: crea un blob (Binary Large Object) del frame. YOLOv3 usa imágenes de entrada blob para su procesamiento.
    # blob: estructura de datos para procesamiento de imágenes y visión por computadora.
    # 1. imagen de entrada capturada desde la cámara.
    # 2. Factor de escala para normalizar los valores de píxeles en el rango [0, 1]
    # 3. Redimensionar la imagen. YOLOv3 utiliza un tamaño fijo de (416, 416)
    # 4. Valor de píxel para rellenar si la imagen se redimensiona a un tamaño diferente al original
    # 5. Si se realiza un cambio de canales BGR a RGB. YOLOv3 trabaja con RGB.
    # 6. Si el blob resultante debe recortarse o no al tamaño especificado
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # blob creado como entrada del modelo YOLOv3
    yolo_net.setInput(blob)
    # forward: cálculo hacia adelante en el modelo, la imagen por la red neuronal y obtiene las salidas de las capas
    outs = yolo_net.forward(output_layers)
    
    # listas con los IDs de las confianzas mas altas
    class_ids = []
    # lista de las confianzas
    confidences = []
    # lista de los recuadros de la deteccion: pos_x0, pos_y0, ancho, alto
    boxes = []
    
    # almacenar el alto, ancho y canales de la imagen
    height, width, channels = frame.shape
    
    # recorrer las salidas del modelo
    for out in outs:
        # recorre cada detección, cada detección tiene info sobre un posible objeto encontrado en la imagen
        for detection in out:
            # guardar el score del modelo
            scores = detection[5:]
            # obtener el indice del score mas alto
            class_id = np.argmax(scores)
            # obtener el nivel de confianza mas alto
            confidence = scores[class_id]
            
            # umbral de confianza para filtrar detecciones débiles
            if confidence > 0.5:
                # detection[0]: Coordenada x del centro del cuadro delimitador normalizada en relación con el ancho de la imagen de entrada
                # detection[1]: Coordenada y del centro del cuadro delimitador normalizada en relación con la altura de la imagen de entrada
                # detection[2]: Ancho del cuadro delimitador normalizado en relación con el ancho de la imagen de entrada
                # detection[3]: Altura del cuadro delimitador normalizada en relación con la altura de la imagen de entrada
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # guardar en la lista el ID de la deteccion mas alta
                class_ids.append(class_id)
                # guardar en la lista la confianza
                confidences.append(float(confidence))
                # guardar en la lista el recuadro de la deteccion
                boxes.append([x, y, w, h])

    # cv2.dnn.NMSBoxes(): supresión de no máximos (Non-Maximum Suppression, NMS) en los objetos encontradas. 
    # La supresión de no máximos es una técnica para eliminar detecciones redundantes y superpuestas
    # 1. lista de las coordenadas y tamaños de los cuadros delimitadores de las detecciones de objetos
    # 2. lista de las confianza de las detecciones de objetos
    # 3. umbral de confianza que se utiliza para filtrar detecciones débiles
    # 4. umbral para determinar si dos cuadros delimitadores se superponen. superposición entre dos cuadros > nms_threshold => elimina el cuadro con la menor confianza
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=0.5, nms_threshold=0.4)

    # Dibujar los resultados de la detección
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    
    # verificar si al menos tenemos un objeto detectado luego de todos los filtros
    if len(indices) > 0:
        # recorrer todos los indices
        for i in indices.flatten():
            # guardar el recuadro delimitador del objeto detectado
            x, y, w, h = boxes[i]
            # obtener el label (nombre del objeto detectado)
            label = str(labels[class_ids[i]])
            # indice de confianza redondeado a 2 decimales
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            
            # dibujar el rectangulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # escribir el nombre del objeto detectado con su nivel de confianza
            cv2.putText(frame, f"{label} {confidence}", (x, y + 30), font, 2, color, 2)

    cv2.imshow("Deteccion de objetos con YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
