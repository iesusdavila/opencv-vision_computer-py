import cv2
import time
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 1024, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

time_pasado = 0

detector = htm.handDetector(minDetectConf=0.7, maxManos=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

area = 0

colorVol = (255, 0, 0)

while True:
    isSuccess, frame = cap.read()

    # Encontrar mano
    frame = detector.encontrarManos(frame)
    ptRefList, bbox = detector.encontrarPosicion(frame, dibujarTrazos=True)
    if len(ptRefList) != 0:
        # Obtener dimensiones de la imagen
        ancho, alto, canales = frame.shape
        
        print(ancho, alto)
        
        # Area del cuadrado de la mano
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        
        # Programa funcionara solo entre un area de 150 y 1000
        if 150 < area < 1000:
            
            # 4 indica la punta del pulgar, 8 indica la punta del indice
            longitud, frame, lineInfo = detector.calcularDistancia(4, 8, frame)

            # Convertidor de volumen
            volBar = np.interp(longitud, [50, 200], [400, 150])
            volPer = np.interp(longitud, [50, 200], [0, 100])

            # El volumen aumentara de 2 en 2
            smoothness = 2
            volPer = smoothness * round(volPer / smoothness)

            # Verificar si tenemos los dedos arriba
            fingers = detector.dedosArriba()

            # Validar si el dedo pulgar esta arriba
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 5, (155, 255, 50), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)
        else:
            cv2.putText(frame, 'Se encuentra fuera del rango.', ((ancho//2)-100, alto-250), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0,0,255), 1)

    # ------- Dibujos -------
    # Dibujo del cuadro de volumen
    cv2.rectangle(frame, (50, 150), (85, 400), (53, 53, 47), 2)
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (76, 76, 65), cv2.FILLED)
    # Numero del nivel del volumen
    cv2.putText(frame, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (53, 53, 47), 1)
    # Volumen seteado
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(frame, f'Vol Set: {int(cVol)}', (570, 30), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 1)

    # Calculo de FPS
    time_actual = time.time()
    fps = 1 / (time_actual - time_pasado)
    time_pasado = time_actual
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 102, 8), 1)

    cv2.imshow("Subir volumen", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()