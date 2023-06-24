import cv2 as cv


def redimensionarFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions)


# Redimensionar Imagenes
img = cv.imread('../Resources/Photos/cat.jpg')
img_resized = redimensionarFrame(img, scale=0.3)

cv.imshow('Gato sin redimenzionar', img)
cv.imshow('Gato', img_resized)

cv.waitKey(0)

# Redimensionar Videos
video = cv.VideoCapture('../Resources/Videos/dog.mp4')

while True:
    isTrue, frame = video.read()
    frame_redimen = redimensionarFrame(frame, scale=0.25)

    cv.imshow('Video', frame_redimen)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

video.release()
cv.destroyAllWindows()
