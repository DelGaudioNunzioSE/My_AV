import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Funzione principale per l'elaborazione delle immagini
def main():
    # Percorso alla cartella delle immagini
    image_folder = './Panels/'
    
    # Itera su tutte le immagini nella cartella
    for images in os.listdir(image_folder):

        image = "./Img/Panels/" + images
        # Converte l'immagine in scala di grigi
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # Applica il rilevamento dei bordi con il metodo Canny
        edges = cv2.Canny(gray, 50, 150)

        # Trova i contorni usando i bordi rilevati
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Disegna i contorni in rosso sull'immagine originale
        image_with_contours = gray.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)

        # Salva o mostra il risultato
        cv2.imwrite('/mnt/data/immagine_con_bordi_rossi.png', image_with_contours)
        cv2.imshow('Bordi Rossi', image_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Esegui il main
if __name__ == "__main__":
    main()
