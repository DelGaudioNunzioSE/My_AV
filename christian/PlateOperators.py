import cv2
import numpy as np

#La devo caricare in scala di grigi altrimenti quando calcola
#le componenti connesse me la considera a più canali.
image = cv2.imread('./Img/Plate.png', cv2.IMREAD_GRAYSCALE)

#tutti i pixel sopra 127 diventano bianchi (255)
#restituisce anche il threshold nel caso usi otsu in cui la sceglie lui
#cv2.THRESH_BINARY_INV inverte i colori
threshold, binary_image = cv2.threshold(image, 126, 255, cv2.THRESH_BINARY_INV)

# Crea uno structuring element 3x3, tutti 1
kernel = np.ones((3, 3), np.uint8)

# Faccio opening, erosion --> dilation
# erosion: se non tutti i pixel sono bianchi allora diventa nero, utile per rimuovere il rumore esterno
opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

#applico 3 volte erosion per rimuovere quel pezzo di rumore esterno
erode_image = cv2.erode(binary_image, kernel, iterations=3)
#data l'erosione, l'immagine è diventata più sottile, la rimetto a grandezza originale con dilate
dilated_image = cv2.dilate(erode_image, kernel, iterations=3)

#calcolo le componenti connesse
num_labels, labels = cv2.connectedComponents(dilated_image)

#rifaccio l'immagine iniziale a colori
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


for label in range(1, num_labels):  # 1 per escludere lo sfondo
    # Ottieni i pixel della componente
    # labels è una matrice in cui ogni pixel ha il tag della componente connessa
    # in questo caso se quel pixel ha il tag della mia label lo metto ad 1 e poi *255 per farlo bianco
    component_mask = (labels == label).astype(np.uint8) * 255

    # Trova i contorni della componente
    # Retr_external prende solo i contorni esterni (e non contorni interni ad altri)
    # chain_approx li rappresenta solo con 4 punti e non tutti per farli piu dettagliati
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    #for contour in contours: solo nel caso di piu contorni, qui ho 1 sola componente connessa
    # Calcola il rettangolo di delimitazione tra tutti i pixel del contorno
    x, y, w, h = cv2.boundingRect(contours[0])
    # rettangolo, (x,y) è il punto in alto a sinistra
    # (x+w,y+h) è quello in basso a destra
    # 2 è quanto è spesso il rettangolo
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 3)


# Mostra i risultati
print("Numero di componenti connesse:", num_labels - 1)  # Esclude lo sfondo
cv2.imshow("Original", image)
cv2.imshow("Binary", binary_image)
cv2.imshow("Opening", opening_image)
cv2.imshow("Eroded then dilated", dilated_image)
cv2.imshow("Componenti connesse", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


