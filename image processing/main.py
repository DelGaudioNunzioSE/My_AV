from puntuali import *

if __name__ == "__main__":
    name= "img/Lenna.png"
    img= read_grayscale(name)
    img2=saturated_arithmetic(img, 50)
    img3= multiply_brightness(img, 1.5)
    img4=linear_operator(img, k=1.5, c= -50)
    img5=clamping(img, 50, 200)
    img6=gray_level_inversion(img)
    img7=histogram_equalization(img)
    img8=clahe(img)
    img9=thresholding(img, 127)
    img10=otsu_binarization(img)
    img11=adaptive_thresholding(img)
    show_images(["Original","Saturated","Multiply Brightness","Linear Operator","Clamping","Gray Level Inversion","Histogram Equalization","CLAHE","Thresholding","Otsu's Binarization","Adaptive Thresholding"],[img,img2,img3,img4,img5,img6,img7,img8,img9, img10,img11],4)