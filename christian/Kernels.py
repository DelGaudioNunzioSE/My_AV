import cv2
import numpy as np

#verticale 0gradi
kernel = np.array([[-1, 0, 1],[-1,0,1], [-1,0,1]]) #con la matrice prende anche le diagonali
kernel = 1/6 * kernel
image = cv2.imread('./Img/Angles.png')
filtered_image = cv2.filter2D(image, -1, kernel) #-1 indica il tipo di dati (profondità) con cui è rappresentata l'immagine
#cv2.imshow("Original", image)
#cv2.imshow("Filtered", filtered_image)

#orizzontale 90gradi
kernel = np.array([[-1, -1, -1],[0,0,0], [1,0,1]]) #trasposta della precedente
kernel = 1/6 * kernel
image = cv2.imread('./Img/Angles.png')
filtered_image = cv2.filter2D(image, -1, kernel) 
#cv2.imshow("Original", image)
#cv2.imshow("Filtered", filtered_image)

#diagonale sinistra 45 gradi
kernel = np.array([[0, 0, 1],[0,0,0], [-1,0,0]]) #trasposta della precedente
kernel = 1/3 * kernel
image = cv2.imread('./Img/Angles.png')
filtered_image = cv2.filter2D(image, -1, kernel) 
cv2.imshow("Original", image)
cv2.imshow("Filtered", filtered_image)

#30gradi
kernel = np.array([[0, 0, 0], [0,0,1], [-1,0,0]]) #trasposta della precedente
kernel = 1/2*kernel
image = cv2.imread('./Img/Angles.png')
filtered_image = cv2.filter2D(image, -1, kernel) 
cv2.imshow("Original", image)
cv2.imshow("Filtered", filtered_image)

#60gradi
kernel = np.array([[0, 0, 1, 0],[0,0,0,0], [0,0,0,0], [0,0,0,-1]]) #trasposta della precedente
kernel = 1/2*kernel
image = cv2.imread('./Img/Angles.png')
filtered_image = cv2.filter2D(image, -1, kernel) 
#cv2.imshow("Original", image)
#cv2.imshow("Filtered", filtered_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
