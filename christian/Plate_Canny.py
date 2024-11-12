import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('./Img/Plate.png', cv2.IMREAD_GRAYSCALE)

#applico un median filter perchè dall'istogramma noto che il rumore è simile a salt and pepper
median_filtered = cv2.medianBlur(image, 9)

threshold, binary_image = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(threshold)

# con questa soglia non c'è rumore
threshold = 200
edges = cv2.Canny(median_filtered, threshold/2, threshold, apertureSize=3, L2gradient = True) #canny algorithm per i bordi, lw threshold + up threshold

num_labels, labels = cv2.connectedComponents(edges)
#rifaccio l'immagine iniziale a colori
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
struct_elem = np.ones((3, 3), np.uint8)
for label in range(1, num_labels):  # 1 per escludere lo sfondo

    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    if h > 150: #per rimuovere la componente interna alla D
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

plt.hist(image.ravel(), bins=256, range=[0, 256])
plt.title("Istogramma dell'Immagine")
plt.xlabel("Valore dei pixel")
plt.ylabel("Frequenza")
plt.show()


cv2.imshow("Median image", median_filtered)
cv2.imshow("Plates edges", edges)
cv2.imshow("Original", image)
cv2.imshow("Original", output_image)



