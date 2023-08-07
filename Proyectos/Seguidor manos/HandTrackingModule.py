import cv2
import mediapipe as mp
import time

class handDetector():
    
    def __init__(self, mode=False, maxManos=2, minDetectConf=0.5, minTrackingConf=0.5):
        self.mode = mode
        self.maxManos = maxManos
        self.minDetectConf = minDetectConf
        self.minTrackingConf = minTrackingConf
        
        self.mpManos = mp.solutions.hands
        self.manos = self.mpManos.Hands(self.mode,self.maxManos,
                                        1,self.minDetectConf,self.minTrackingConf)
        self.mpDibujo = mp.solutions.drawing_utils


    def encontrarManos(self, frame, dibujarTrazos=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.resultados = self.manos.process(frameRGB)
        
        if self.resultados.multi_hand_landmarks:
            for manosPtsRef in self.resultados.multi_hand_landmarks:
                # for id, ptRef in enumerate(manosPtsRef.landmark):
                #     h, w, c = frame.shape
                #     cx, cy = int(ptRef.x*w), int(ptRef.y*h)
                #     print(id,cx,cy)
                    
                #     cv2.circle(frame, (cx,cy), 15, (255,0,255), cv2.FILLED)
                if dibujarTrazos:
                    self.mpDibujo.draw_landmarks(frame, manosPtsRef, self.mpManos.HAND_CONNECTIONS)
        return frame

    
def main():
    video = cv2.VideoCapture(0)
    
    tiempo_pasado = 0
    tiempo_actual = 0
    
    detectorManos = handDetector()
    
    while True:
        isSuccess, frame = video.read()
        
        tiempo_actual = time.time()
        fps = 1/(tiempo_actual - tiempo_pasado)
        tiempo_pasado = tiempo_actual
        
        frame = detectorManos.encontrarManos(frame)
                
        cv2.putText(frame, str(int(fps)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0,0,0))
        cv2.imshow("Frame del video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    
    # Libera los recursos y cierra las ventanas
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()