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


# 
# fa erosione e dilatazione per migliorare l'immagine
# poi trova i contorni del immagine
## open_image -> immagine biancha e nera
## image -> immagine originale
def find_edges(open_image, image):
    # inversione immgaine
    open_image = 255-open_image
    # OPENING
    # forma della struttura morfologica 
    morf_struc = np.ones((14,14), np.uint8)
    open_image = cv2.erode(open_image, morf_struc, iterations=1)
    open_image = cv2.dilate(open_image, morf_struc, iterations=1)
    # cv2.RETR_EXTERNAL -> contorni esterni
    contours, _ = cv2.findContours(open_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = connComps(open_image, image)
    cv2.imshow("Open", open_image)
    return output_image


# Trova componenti connesse e fa i rettangoli
def connComps(open_image, image):
    ## num_labels -> numero totale di componenti connesse
    ## labels -> array che indica la lable di ogni pixel
    num_labels, labels = cv2.connectedComponents(open_image)
    # La rende colorata per fare i rettangoli colorati
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # per ogni componente connessa
    for label in range(1, num_labels):  
        ## labels == label -> mette a true i valori del array che fanno parte di questa label 
        ## 255 -> quelle che fanno parte di questa label vengono messe bianche
        component_mask = (labels == label).astype(np.uint8) * 255
        # Trova i contorni all'interno della maschera componente connessa
        ## cv2.RETR_EXTERNAL -> trova solo i contorni esterni
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # disegna i rettangoli attorno i contorni 
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20:
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return output_image



for image in os.listdir('./Img/Panels/'):
    image = "./Img/Panels/" + image
    # trasformo in bianco e nero
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(10,10))

    clahe_image = clahe.apply(image)
    clahe_image = cv2.medianBlur(clahe_image, 3) #rumore per rimuovere quello creato da CLAHE

    # fa treshould
    threshold, binary_image = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(threshold)

    #rileva i bordi con canny
    edges = cv2.Canny(clahe_image, threshold/2, threshold, apertureSize=3, L2gradient = True) #canny algorithm per i bordi, lw threshold + up threshold
    # operazioni morfologiche apertura
    kernel = np.ones((14, 14), np.uint8)
    open_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel) #rimuovere i collegamenti tra i pannelli
    cv2.imshow("OpenImage", open_image)
    #erode_image = cv2.erode(open_image, kernel2, iterations=2)


    output_comps=connComps(open_image, image)
    output_canny = find_edges(edges, image)


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