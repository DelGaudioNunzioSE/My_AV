# sono considerati filtri/operatori locali
# quelli che utilizzano una matrice (kernel)

import cv2
import numpy as np


# Fa effetto blur per togliere rumori
def average_filter(image, ksize=3):
    if ksize % 2 == 0:
        raise ValueError("ksize deve essere un numero dispari.")
    return cv2.blur(image, (ksize, ksize))


# trova valore mediano ed imposta quello al centro della matrice
def median_filter(image, ksize=3):
    if ksize % 2 == 0:
        raise ValueError("ksize deve essere un numero dispari.")
    return cv2.medianBlur(image, ksize)

# Filto gaussiano
    # ksize -> dim matrice
    # st_dev -> deviazione standard
def gaussian_filter(image, ksize=3, st_dev=0):
    if ksize % 2 == 0:
        raise ValueError("ksize deve essere un numero dispari.")
    return cv2.GaussianBlur(image, (ksize, ksize), st_dev)



def prewitt_filter(image,\
                    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), \
                    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) ):
    """
    Applica il filtro Prewitt per rilevare i bordi.
    """
    # -1 -> tipo profondità dell' immagine
    # kernel_x -> matrice orizzontale
    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)
    # combina i bordi rilevati con i 2 filtri 2D
    # 0.5 -> peso contributo
    # 0 -> valore di bias aggiunto ad ogni pixel
    return cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)



def sobel_filter(image, ksize=3):
    """
    Applica il filtro Sobel per rilevare i bordi.
    """
    if ksize % 2 == 0:
        raise ValueError("ksize deve essere un numero dispari.")
    ## cv2.CV_64F -> profondità immagine
    ## 1,0 -> indica che è orizzontale
    ## dimensioni matrice
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    # combina i gradienti
    return cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))



def discrete_gradient(image, ksize=3):
    if ksize % 2 == 0:
        raise ValueError("ksize deve essere un numero dispari.")
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(magnitude)



def laplacian_filter(image, ksize=3):
    """
    Applica il filtro Laplaciano per rilevare i bordi.
    :param image: Immagine di input (RGB o Grayscale).
    :param ksize: Dimensione del kernel (deve essere dispari).
    :return: Immagine filtrata con il filtro Laplaciano.
    """
    if ksize % 2 == 0:
        raise ValueError("ksize deve essere un numero dispari.")
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(laplacian)
