import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# calcola e mostra l'istogramma del immagine 
def histo(image, title = 'histogram'):
    # calcola istogramma di un immagine
        # image -> immagine da considerare
        # 0 -> canale colori (in questo caso bianco e nero)
        # None -> assenza di una maschera che nasconde parte di un immagine
        # 256 -> numero di gruppi (bins) per l'istogramma (in questo caso ogni valore possibile è un gruppo diverso)
        # [0,256] -> range di valori possibili dei pixel 
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Mostra l'istogramma
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Pixel values')
    plt.ylabel('Number of pixel')
    plt.plot(histogram)
    # Limita l'asse X a 256 per visualizzare tutte le intensità
    plt.xlim([0, 256])  
    plt.show()


# CANNY
# fa erosione e dilatazione per migliorare l'immagine
# poi trova i contorni del immagine
def canny(open_image, image):
    open_image = 255-open_image
    kernel = np.ones((14,14), np.uint8)
    #open_image = cv2.morphologyEx(open_image, cv2.MORPH_OPEN, kernel)
    open_image = cv2.erode(open_image, kernel, iterations=1)
    open_image = cv2.dilate(open_image, kernel, iterations=1) #rimuovere i collegamenti tra i pannelli
    
    contours, _ = cv2.findContours(open_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = connComps(open_image, image)
    cv2.imshow("Open", open_image)
    # output_image = image.copy()
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    return output_image

def connComps(open_image, image):
    num_labels, labels = cv2.connectedComponents(open_image)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for label in range(1, num_labels):  
        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20:
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return output_image

for image in os.listdir('./Img/Panels/'):
    image = "./Img/Panels/" + image
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #histo(image)

    #clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(20,20)) #clipLimit è quanto voglio aumentare il contrasto, tile devo metterlo grande se ho tante aree uniformi
    #foto 2 o 3 e 4

    #clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(5,5))

    #clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(7,7))
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(10,10))

    clahe_image = clahe.apply(image)
    clahe_image = cv2.medianBlur(clahe_image, 3) #rumore per rimuovere quello creato da CLAHE
    #clahe_image = cv2.blur(clahe_image, (5, 5))
    #clahe_image = image
    #clahe_image = cv2.GaussianBlur(clahe_image, (9, 9), 0)
    threshold, binary_image = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(threshold)
    edges = cv2.Canny(clahe_image, threshold/2, threshold, apertureSize=3, L2gradient = True) #canny algorithm per i bordi, lw threshold + up threshold
    kernel = np.ones((14, 14), np.uint8)
    open_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel) #rimuovere i collegamenti tra i pannelli
    cv2.imshow("OpenImage", open_image)
    #erode_image = cv2.erode(open_image, kernel2, iterations=2)


    output_comps=connComps(open_image, image)
    output_canny = canny(edges, image)


    images = [binary_image, image, edges, open_image, clahe_image, output_comps, output_canny]
    titles = ["Binary", "Panels", "Panels Edges", "Close", "Clahe", "Box", "CannyBox"]
    # Crea un'unica figura
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(3, 3, i+1)  # 2 righe e 3 colonne
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

    # Mostra tutte le immagini in un'unica finestra
    plt.tight_layout()
    plt.show()


# cv2.imshow("Binary", binary_image)
# cv2.imshow("Panels", image)
# cv2.imshow("Panels edges", edges)
# cv2.imshow("Open_image", open_image)
# cv2.imshow("Output_image", output_image)
# cv2.imshow("Erode_image", erode_image)




cv2.waitKey(0)
cv2.destroyAllWindows()