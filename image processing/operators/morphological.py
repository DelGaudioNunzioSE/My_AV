import cv2
import numpy as np

def dilatation(img, kernel_size=3):
    """
    Funzione che applica l'operazione di dilatazione.
    
    Parametri:
    - img: Immagine in input (in formato binario o in scala di grigi).
    - kernel_size: Dimensione del kernel (default 3x3).
    
    Restituisce:
    - Immagine dilatata.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(img, kernel)
    return dilated_img

def erosion(img, kernel_size=3):
    """
    Funzione che applica l'operazione di erosione.
    
    Parametri:
    - img: Immagine in input (in formato binario o in scala di grigi).
    - kernel_size: Dimensione del kernel (default 3x3).
    
    Restituisce:
    - Immagine erosa.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_img = cv2.erode(img, kernel)
    return eroded_img


# Prima erosion e poi dilatation
## utile per piccoli rumori
def opening(img, kernel_size=3):
    """
    Funzione che applica l'operazione di apertura (erosione seguita da dilatazione).
    
    Parametri:
    - img: Immagine in input (in formato binario o in scala di grigi).
    - kernel_size: Dimensione del kernel (default 3x3).
    
    Restituisce:
    - Immagine dopo apertura.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opened_img


# Prima dilatation e poi erosion
## rende oggetti foreground più 'solidi'
def closing(img, kernel_size=3):
    """
    Funzione che applica l'operazione di chiusura (dilatazione seguita da erosione).
    
    Parametri:
    - img: Immagine in input (in formato binario o in scala di grigi).
    - kernel_size: Dimensione del kernel (default 3x3).
    
    Restituisce:
    - Immagine dopo chiusura.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed_img

# Esempio di utilizzo
if __name__ == "__main__":
    # Carica un'immagine binaria (ad esempio, da un file)
    img = cv2.imread('image.jpg', 0)  # 0 significa che carichiamo l'immagine in scala di grigi

    # Esegui le operazioni morfologiche
    dilated = dilatation(img)
    eroded = erosion(img)
    opened = opening(img)
    closed = closing(img)

    # Visualizza i risultati
    cv2.imshow('Dilated', dilated)
    cv2.imshow('Eroded', eroded)
    cv2.imshow('Opened', opened)
    cv2.imshow('Closed', closed)

    # Attendi la pressione di un tasto per chiudere
    cv2.waitKey(0)
    cv2.destroyAllWindows()
