import cv2
import mediapipe as mp
import time

video = cv2.VideoCapture(0)

mpManos = mp.solutions.hands
manos = mpManos.Hands()
mpDibujo = mp.solutions.drawing_utils

tiempo_pasado = 0
tiempo_actual = 0

while True:
    isSuccess, frame = video.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resultados = manos.process(frameRGB)
    
    if resultados.multi_hand_landmarks:
        for manosPtsRef in resultados.multi_hand_landmarks:
            for id, ptRef in enumerate(manosPtsRef.landmark):
                h, w, c = frame.shape
                cx, cy = int(ptRef.x*w), int(ptRef.y*h)
                print(id,cx,cy)
                
                cv2.circle(frame, (cx,cy), 15, (255,0,255), cv2.FILLED)
                
            mpDibujo.draw_landmarks(frame, manosPtsRef, mpManos.HAND_CONNECTIONS)
    
    
    tiempo_actual = time.time()
    fps = 1/(tiempo_actual - tiempo_pasado)
    tiempo_pasado = tiempo_actual
    
    cv2.putText(frame, str(int(fps)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0,0,0))
    cv2.imshow("Frame del video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Libera los recursos y cierra las ventanas
video.release()
cv2.destroyAllWindows()    