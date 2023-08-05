import cv2 as cv

video = cv.VideoCapture(0)

haar_cascade = cv.CascadeClassifier('../../Proyecto - Reconocimiento Facial/haar_face.xml')

while video.isOpened():
    isTrue, frame = video.read()    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    i=0
    
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
    for (x,y,w,h) in faces_rect:
        i+=1
        cv.putText(frame, "Persona "+str(i), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.1, (0,255,0), thickness=2)
        cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),1)
    
    if isTrue:
        cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

video.release()
cv.destroyAllWindows()