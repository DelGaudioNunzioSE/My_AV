import cv2
import numpy as np



def pre_process(img):
    """
    Valuta se è necessario fare contrast stretching

    Args:
        imag: Immagine bianco e nero.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    if min_val!=0 or max_val!= 0:
        img= ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        print('constrast stretching applicato')
    return img






# 1. Smoothing: ------------------/
    # # Applica il filtro Gaussiano per rimuovere il rumore 
def canny_smothing(img, smoothing_type=None,kernel=3,sigma=1.4):    
    
    if smoothing_type == 'g':
        smoothed_img = cv2.GaussianBlur(img, (kernel,kernel), sigma)  # Kernel 5x5 con sigma=1.4

    elif smoothing_type == 'm':
        print("Applicando MedianBlur...")
        smoothed_img = cv2.medianBlur(img, kernel)
        
    elif smoothing_type == 'b':
        print("Applicando BilateralFilter...")
        smoothed_img = cv2.blur(img, (kernel,kernel))
        
    elif smoothing_type!= None:
        print(f"Tipo di smoothing '{smoothing_type}' non valido. Usando GaussianBlur come predefinito.")
        smoothed_img = cv2.GaussianBlur(img, (kernel,kernel), sigma)
    else:
        print ('Nessun filtro applicato')
        smoothed_img = img
        
    return smoothed_img




# 2. Finding Gradients ------------------/
def canny_finding_gradients(smoothed_img,ksize=3):
    # # cv2.CV_32F -> enorme ingrandimento di rappresentabilità per avere una precisione elevata nel calcolo delle derivate
    # # 1,0 -> derivata orizzontale 
    # # ksize -> dimensioni matrice
    grad_x = cv2.Sobel(smoothed_img, cv2.CV_32F, 1, 0, ksize=3)  # Derivata lungo x
    grad_y = cv2.Sobel(smoothed_img, cv2.CV_32F, 0, 1, ksize=3)  # Derivata lungo y

    # Riadattamento ad 8 bit
    absx_64 = np.absolute(grad_x)
    absy_64 = np.absolute(grad_y)

    sobelx_8u = absx_64 / absx_64.max() * 255
    sobely_8u = absy_64 / absy_64.max() * 255

    sobelx_8u = np.uint8(sobelx_8u)
    sobely_8u = np.uint8(sobely_8u)
    
    # 2.1. Gradient magnitude and orientation
    # Magnitudine
    mag = np.hypot(sobelx_8u, sobely_8u) # hypot = sqrt(a^2+b^2)
    mag = mag / np.max(mag) * 255 # riadatta ad 8 bit la magnitudine
    mag = np.uint8(mag)
    
    # Angolo
    theta = np.arctan2(grad_x, grad_y)
    angle = np.rad2deg(theta)

    return mag, angle





#3. Non-Maxima Suppression ------------------/
def danny_non_maxima(mag, angle, maxima_dif=0):
    M, N = mag.shape
    # matrice che fa da mappa
    Non_max = np.zeros((M, N), dtype=np.uint8)

    # si scorre su tutta l'immagine eccetto che sui bordi (per evitare di uscire)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # Identificazione angolo

            # Horizontal 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180) or (-22.5 <= angle[i, j] < 0) or (-180 <= angle[i, j] < -157.5):
                b = mag[i, j + 1]
                c = mag[i, j - 1]
            # Diagonal 45
            elif (22.5 <= angle[i, j] < 67.5) or (-157.5 <= angle[i, j] < -112.5):
                b = mag[i + 1, j + 1]
                c = mag[i - 1, j - 1]
            # Vertical 90
            elif (67.5 <= angle[i, j] < 112.5) or (-112.5 <= angle[i, j] < -67.5):
                b = mag[i + 1, j]
                c = mag[i - 1, j]
            # Diagonal 135
            elif (112.5 <= angle[i, j] < 157.5) or (-67.5 <= angle[i, j] < -22.5):
                b = mag[i + 1, j - 1]
                c = mag[i - 1, j + 1]
            
            # Operazione effettiva
            # Non-max Suppression
            if (mag[i, j] >= (b-maxima_dif)) and (mag[i, j] >= (c-maxima_dif)):
                Non_max[i, j] = mag[i, j]
            else:
                Non_max[i, j] = 0
    return Non_max




#4. Double Thresholding ------------------/
def canny_double_thesholding(Non_max,high_threshold=None,low_threshold_ratio=0.5):
    M, N = Non_max.shape
    double_thres = np.zeros((M, N), dtype=np.uint8) # matrice dove metteremo tutti i pixel divisi in gruppi

    # Determinazione soglia thresholding
    if high_threshold == None: 
        high_threshold, imtr = cv2.threshold(Non_max, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = high_threshold * low_threshold_ratio

    # Classificazione
    strong_i, strong_j = np.where(Non_max > high_threshold)
    zeros_i, zeros_j = np.where(Non_max < low_threshold)
    weak_i, weak_j = np.where((Non_max <= high_threshold) & (Non_max >= low_threshold))

    # valori fissi ai 3 gruppi di edge
    double_thres[strong_i, strong_j] = 255 # vengono impostati a biacno
    double_thres[zeros_i, zeros_j] = 0 # vengono impostati a nero
    double_thres[weak_i, weak_j] = 75 # vengono impostati a grigio

    return double_thres




# 5. Edge Tracking by Hysteresis ------------------/
def canny_edge_tracking(double_thres):
    M, N = double_thres.shape
    out = double_thres

    for i in range(1, M-1):
        for j in range(1, N-1):
            # valutazione dei weak edges
            if (double_thres[i, j] == 75):
                # controlla i pixel vicini (anche orizzontale)
                if 255 in [ out[i+1, j-1], out[i+1, j], out[i+1, j+1],out[i, j-1],out[i, j+1], out[i-1, j-1], out[i-1, j], out[i-1, j+1] ]:
                    out[i, j] = 255 # se almeno uno è strong edges allora diventa anche lui strong edges
                else:
                    out[i, j] = 0 # altrimenti viene eliminato
    return out





def real_canny(img, high_threshold=150):
    return cv2.Canny(img, threshold1=high_threshold/2, threshold2=150)



# BOXING
def boxing(original_img, canny_image):
    """
    Fa bounding box.
    
    Args:
        original_img= Immagine a colori.
        canny_image = output di canny

    Returns:
        immagine con bounding box
    """

    # se non è a colori la rendo a colori
    if len(original_img.shape) != 3 or original_img.shape[2] != 3:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # Trova i contorni
    contorni, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Disegna i rettangoli rossi sui contorni
    for contorno in contorni:
        # Ottieni il rettangolo di delimitazione (bounding box) per ogni contorno
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Disegna il rettangolo rosso sull'immagine originale
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # (0, 0, 255) è il rosso in BGR

    return original_img