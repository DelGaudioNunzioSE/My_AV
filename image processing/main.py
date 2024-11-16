from operators.puntuali import *
from support import *
from algorithm.canny_edge import *
        
MODE=2
IMG_NAME= "img//pannels/pv01.png"

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
        img_pre=linear_operator(img, 2, -50)
        img_pre2=pre_process(img_pre)
        show_with_histo(['Colorful','Origianl','Pre-process','Pre-process2'],[original_img,img,img_pre,img_pre2])
        
        img_canny,img_dobule_thresholding, img_Non_maxima, img_gradients, img_smoothing = \
            canny_homemade(img_pre2, smoothing_type='g',kernel=21,do_maxima=True, closing= 3)
        box=boxing(original_img,img_canny )
        show_images(['Origianl','Pre-process','Pre-process2','Smoothing','gradients','NON-Maxima','dobule TresHolding','canny','box'],[img,img_pre,img_pre2,img_smoothing,img_gradients,img_Non_maxima,img_dobule_thresholding,img_canny,box], 3)
        
        # show_with_histo(['Origianl','Pre-process','Smoothing','gradients','NON-Maxima','dobule TresHolding','canny','box'],[img,img_pre,img_smoothing,img_gradients,img_Non_maxima,img_dobule_thresholding,img_canny,box])