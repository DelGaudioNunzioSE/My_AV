import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('./Img/Panels.png', cv2.IMREAD_GRAYSCALE)
brightness_increase = 0.5
bright_image = cv2.convertScaleAbs(image, alpha=2, beta=0)


#Usando otsu non ho bisogno di calcolare la soglia
#Mi sceglie automaticamente quella a minima varianza che in questo caso è 80
threshold, binary_image = cv2.threshold(bright_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(threshold) #80
kernel = np.ones((3, 3), np.uint8)

#open_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
erode_image = cv2.erode(binary_image, kernel, iterations=8)



num_labels, labels = cv2.connectedComponents(erode_image)
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for label in range(1, num_labels):  
    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 3)



cv2.imshow("Original", image)
cv2.imshow("Bright", bright_image)
cv2.imshow("Binary", binary_image)
cv2.imshow("Erode", erode_image)
cv2.imshow("Output", output_image)



cv2.waitKey(0)
cv2.destroyAllWindows()