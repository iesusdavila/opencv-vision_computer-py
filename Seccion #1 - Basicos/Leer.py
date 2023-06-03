import cv2 as cv

# Leer imagenes
# img = cv.imread('../Resources/Photos/cat.jpg')

# cv.imshow('Gato', img)

# cv.waitKey(0)

# Leer videos
video = cv.VideoCapture('../Resources/Videos/dog.mp4')

while True:
    isTrue, frame = video.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

video.release()
cv.destroyAllWindows()


