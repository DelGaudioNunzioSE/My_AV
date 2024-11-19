import cv2
import numpy as np
import matplotlib.pyplot as plt # usata per mostrare grafici ed immagini
import math



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
    fig=plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('Pixel values')
    plt.ylabel('Number of pixel')
    plt.plot(histogram)
    # Limita l'asse X a 256 per visualizzare tutte le intensità
    plt.xlim([0, 256])  
    plt.grid(False)
    plt.tight_layout()

    # Converte la figura in immagine OpenCV
    fig.canvas.draw()
    histogram_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    histogram_img = histogram_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)  # Chiude la figura per liberare memoria

    return histogram_img





# Funzione per mostrare le immagini in bianco e nero
def show_image(title, img):
        # cmap= gray rappresenta in scala di grigi
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()



def show_images(titles, images, cols=2):
    """
    Mostra più immagini in una griglia.
    
    - param titles: Lista dei titoli delle immagini.
    - param images: Lista delle immagini da mostrare.
    - param cols: Numero di colonne nella griglia.
    """
    assert len(titles) == len(images), "Il numero di titoli deve corrispondere al numero di immagini."
    
    n_images = len(images)
    rows = (n_images + cols - 1) // cols  # Calcola il numero di righe necessario
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Dimensione personalizzabile
    axes = axes.ravel()  # Appiattisce gli assi per iterare facilmente
    
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap="gray")
        axes[i].set_title(titles[i])
        axes[i].axis("off")
    
    # Nascondi eventuali assi extra
    for i in range(len(images), len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()



def show_with_histo(titles,imgs):
    # calcolo numero di colonne
    n_coll=len(imgs)
    if n_coll % 2 != 0:
        n_coll = n_coll + (2-n_coll %2)

    # aggiunta istogrammi
    img_list = []
    title_list = []
    for i in titles:
        title_list.append(i)
        title_list.append("histo")
    for i in imgs:
        img_list.append(i)
        img_list.append(histo(i))
    show_images(title_list, img_list, n_coll)