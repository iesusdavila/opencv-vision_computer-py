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
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
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

    # Find Hand
    frame = detector.encontrarManos(frame)
    ptRefList, bbox = detector.encontrarPosicion(frame, dibujarTrazos=True)
    if len(ptRefList) != 0:

        # Area del cuadrado de la mano
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        print(area)
        
        # Programa funcionara solo entre 150 y 1000
        if 150 < area < 1000:
            
            # 4 indica la punta del pulgar, 8 indica la punta del indice
            longitud, frame, lineInfo = detector.calcularDistancia(4, 8, frame)

            # Convertidor de volumen
            volBar = np.interp(longitud, [50, 200], [400, 150])
            volPer = np.interp(longitud, [50, 200], [0, 100])

            # Reduce Resolution to make it smoother
            smoothness = 10
            volPer = smoothness * round(volPer / smoothness)

            # Check fingers up
            fingers = detector.dedosArriba()

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15, (155, 255, 50), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    # Dibujos
    cv2.rectangle(frame, (50, 150), (85, 400), (53, 53, 47), 2)
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (76, 76, 65), cv2.FILLED)
    cv2.putText(frame, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (53, 53, 47), 2)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(frame, f'Vol Set: {int(cVol)}', (600, 30), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 2)

    # Frame rate
    time_actual = time.time()
    fps = 1 / (time_actual - time_pasado)
    time_pasado = time_actual
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_COMPLEX,
                1, (100, 102, 8), 2)

    cv2.imshow("Subir volumen", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()