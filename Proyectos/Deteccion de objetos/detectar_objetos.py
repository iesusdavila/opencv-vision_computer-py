import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)

sift = cv.SIFT_create()

# Variables para el seguimiento
old_gray = None
old_keypoints = None
mask = None

while video.isOpened():
    isTrue, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if old_gray is None:
        # Si es el primer fotograma, inicializamos las variables para el seguimiento
        old_gray = gray
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        old_keypoints = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        mask = np.zeros_like(frame)
    else:
        # Si ya se han detectado keypoints en el primer fotograma, continuamos con el seguimiento
        keypoints, st, err = cv.calcOpticalFlowPyrLK(old_gray, gray, old_keypoints, None)
        good_new = keypoints[st == 1]
        good_old = old_keypoints[st == 1]

        # Dibujar las líneas de seguimiento en el fotograma actual
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            #mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv.add(frame, mask)
        cv.imshow('Video', frame)

        # Actualizar el fotograma anterior y keypoints para el siguiente ciclo
        old_gray = gray.copy()
        old_keypoints = good_new.reshape(-1, 1, 2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
