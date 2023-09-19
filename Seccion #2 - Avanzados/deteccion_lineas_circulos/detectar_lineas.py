import cv2 as cv
import numpy as np

image = cv.imread('../../Resources/Images/soduku.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 170, apertureSize = 3)

lines = cv.HoughLinesP(edges, 2.5, np.pi / 180, 200, 5, 10)

print(lines.shape)
print(len(lines))

for line in lines:
    x1,y1,x2,y2 = line[0]
    
    cv.line(image, (x1, y1), (x2, y2),(0, 255, 0), 3)

cv.imshow('Lineas detectadas', image)
cv.waitKey(0)
cv.destroyAllWindows()

