from puntuali import *
from support import *
        

if __name__ == "__main__":
    name= "img/Lenna.png"
    # Origian
    img= read_grayscale(name)
    show(['Origianl'],[img])
    # Puntual
    ## linear operator
    img2=saturated_arithmetic(img, 50)
    img3= multiply_brightness(img, 1.5)
    img4=linear_operator(img, k=1.5, c= -50)
    show(['Origianl','Add','Multiply','Linear operator'],[img,img2,img3,img4])
    ## others
    img5=clamping(img, a=50, b=200)
    img6=gray_level_inversion(img)
    show(['Origianl','Clampling','Inversion'],[img,img5,img6])
    ## histogram operators
    img7=histogram_equalization(img)
    img8=clahe(img)
    show(['Origianl','Equalization','Clahe'],[img,img7,img8])
    ## thresholding
    img9=thresholding(img, thresh_value=127)
    img10=otsu_binarization(img)
    img11=adaptive_thresholding(img)
    show(['Origianl','Thresholding','Otsu','Adaptive'],[img,img9,img10,img11])