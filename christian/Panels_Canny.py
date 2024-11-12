import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('./Img/Panels.png', cv2.IMREAD_GRAYSCALE)
threshold, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(threshold)
threshold = 150
edges = cv2.Canny(image, threshold/2, threshold, apertureSize=3, L2gradient = True) #canny algorithm per i bordi, lw threshold + up threshold

struct_elem = np.ones((3, 3), np.uint8)
#opening_image = cv2.morphologyEx(edges, cv2.MORPH_OPEN, struct_elem)
erode_image = cv2.dilate(edges, struct_elem, iterations=1)

num_labels, labels = cv2.connectedComponents(erode_image)
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for label in range(1, num_labels):  
    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 3)





cv2.imshow("Panels", image)
cv2.imshow("Panels edges", edges)
cv2.imshow("Erode_image", erode_image)
cv2.imshow("Output_image", output_image)




cv2.waitKey(0)
cv2.destroyAllWindows()