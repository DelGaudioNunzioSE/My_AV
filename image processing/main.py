from operators.puntuali import *
from operators.canny_edge import *
from operators.morphological import *
from support import *
        
MODE=2
IMG_NAME= "img//pannels/pv01.png"


# BORDI
def canny_homemade(img, smoothing_type=None, kernel=3, sigma=1.4, \
                   maxima_dif=0, low_threshold_ratio=0.5, high_threshold=None):
    """
    Canny Homemade.
    
    Args:
        img: Immagine bianco e nero.
        kernel = forma kernel
        s = deviazione standard gaussiano
        low_threshold_ratio (float): Rapporto tra soglia bassa e soglia alta (default: 0.5).
        high_threshold (int): Soglia alta per il double thresholding (default: 100).
        
    Returns:
        edges (numpy.ndarray): Immagine dei bordi rilevati.
    """

    img0=linear_operator(img, 2, -100)
    img0=pre_process(img0)
    
    # 1. Smoothing: ------------------/
    img1=canny_smothing(img0, smoothing_type=smoothing_type,kernel=kernel,sigma=sigma)
    img1=otsu_binarization(img1)
    img1=gray_level_inversion(img1)
    img1=opening(img1, kernel_size=20)

    # 2. Finding Gradients ------------------/
    img2,angle = canny_finding_gradients(img1)
    

    # 3. Non-Maxima Suppression ------------------/
    img3=danny_non_maxima(img2, angle, maxima_dif=0)


    
    # 4. Double Thresholding ------------------/
    img4=canny_double_thesholding(img3,high_threshold=None,low_threshold_ratio=0.5)

    
    # 5. Edge Tracking by Hysteresis ------------------/
    img5=canny_edge_tracking(img4)

    return img5, img4, img3, img2, img1, img0




if __name__ == "__main__":
    # Origian
    original_img= cv2.imread(IMG_NAME)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) # in png i canali colori sono invertiti
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) # conversione in bianco e nero
    # show_with_histo(['Colorful','Origianl'],[original_img,img])

    # Puntual
    if (MODE == 1):
        ## linear operator
        img2=saturated_arithmetic(img, 50)
        img3= multiply_brightness(img, 1.5)
        img4=linear_operator(img, k=1.5, c= -50)
        show_with_histo(['Origianl','Add','Multiply','Linear operator'],[img,img2,img3,img4])
        ## others
        img5=clamping(img, a=50, b=200)
        img6=gray_level_inversion(img)
        show_with_histo(['Origianl','Clampling','Inversion'],[img,img5,img6])
        ## histogram operators
        img7=histogram_equalization(img)
        img8=clahe(img)
        show_with_histo(['Origianl','Equalization','Clahe'],[img,img7,img8])
        ## thresholding
        img9=thresholding(img, thresh_value=127)
        img10=otsu_binarization(img)
        img11=adaptive_thresholding(img)
        show_with_histo(['Origianl','Thresholding','Otsu','Adaptive'],[img,img9,img10,img11])
        
    # Canny
    elif(MODE==2):
        
        img_canny,img_dobule_thresholding, img_Non_maxima, img_gradients, img_smoothing,img_pre = \
        canny_homemade(img, smoothing_type='g',kernel=5,sigma=1.6,maxima_dif=0)
        box=boxing(original_img,img_canny )
        show_images(['Origianl','Pre-process','Smoothing','gradients','NON-Maxima','dobule TresHolding','canny','box'],[img,img_pre,img_smoothing,img_gradients,img_Non_maxima,img_dobule_thresholding,img_canny,box], 3)
        
        # Real Canny
        #plt.imshow(boxing(original_img,real_canny(img)))
        #plt.show()

        # show_with_histo(['Origianl','Pre-process','Smoothing','gradients','NON-Maxima','dobule TresHolding','canny','box'],[img,img_pre,img_smoothing,img_gradients,img_Non_maxima,img_dobule_thresholding,img_canny,box])