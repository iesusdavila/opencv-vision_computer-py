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
        
        self.tipIds = [4, 8, 12, 16, 20]

    def encontrarManos(self, frame, dibujarTrazos=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.resultados = self.manos.process(frameRGB)
        
        if self.resultados.multi_hand_landmarks:
            for manosPtsRef in self.resultados.multi_hand_landmarks:
                if dibujarTrazos:
                    self.mpDibujo.draw_landmarks(frame, manosPtsRef, self.mpManos.HAND_CONNECTIONS)
        return frame
    
    def encontrarPosicion(self, frame, numMano=0, dibujarTrazos=True):
        xList = []
        yList = []
        bbox = []
        self.ptRefList = []
        
        manoSeleccionada = ""
        
        if self.resultados.multi_hand_landmarks:
            manoSeleccionada = self.resultados.multi_hand_landmarks[numMano]
        
        if manoSeleccionada != "":
            for id, ptRef in enumerate(manoSeleccionada.landmark):
                h, w, c = frame.shape
                cx, cy = int(ptRef.x * w), int(ptRef.y * h)
                
                xList.append(cx)
                yList.append(cy)
                self.ptRefList.append([id, cx, cy])
                
                if dibujarTrazos:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            
            bbox = xmin, ymin, xmax, ymax
        
            if dibujarTrazos:
                cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 0, 255), 2)
        
        return self.ptRefList, bbox
    
    def dedosArriba(self):
        fingers = []
        
        print(self.ptRefList)
        
        if len(self.ptRefList) != 0:
            if self.ptRefList[self.tipIds[0]][1] > self.ptRefList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if self.ptRefList[self.tipIds[id]][2] < self.ptRefList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers
    
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
        
        ptRefList = detectorManos.encontrarPosicion(frame)
        
        detectorManos.dedosArriba()
                
        cv2.putText(frame, str(int(fps)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0,0,0))
        cv2.imshow("Frame del video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    
    # Libera los recursos y cierra las ventanas
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()