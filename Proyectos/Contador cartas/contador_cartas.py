import cv2
import numpy as np

imagen = cv2.imread('cartas_3.jpg')
imagen = cv2.resize(imagen, (512, 512), interpolation=cv2.INTER_CUBIC)
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(grises, 150, 450, True)


# RETR_TREE: modo de recuperación de contornos
# CHAIN_APPROX_SIMPLE: método de aproximación de contornos
ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

print(ctns[0].shape)

for i in range(len(ctns[0])):
    print(ctns[0][i])

# dibujar los contornos
cv2.drawContours(imagen, ctns, -1, (0,0,255), 2)
print('Número de contornos encontrados: ', len(ctns))


texto = 'Contornos encontrados: '+ str(len(ctns))
cv2.putText(imagen, texto, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(255, 0, 0), 1)

cv2.imshow('Imagen', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()