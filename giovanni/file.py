import numpy as np
import matplotlib.pyplot as plt
import cv2


# Matrice di rotazione attorno all'asse Y

def R_z(theta_z):
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0,0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1 ,0],
        [0, 0, 0, 1]
    ])
    return Rz


def R_y(theta_y):
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y),0],
        [0, 1, 0,0],
        [-np.sin(theta_y), 0, np.cos(theta_y),0],
        [0, 0, 0, 1]
    ])
    return Ry

def C(x,y,z):
    C = np.array([
        [1,0,0,-x],
        [0,1,0,-y],
        [0,0,1,-z],
        [0,0,0,1]
    ])
    return C

def UVW(x,y,z):
    UVW = np.array([
        [x],
        [y],
        [z],
        [1]
    ])
    return UVW

def real_to_camera(xt, yt, zt, xp, yp, zp, theta_x=0, theta_y=0, theta_z=0):
    translation_matrix = C(xt, yt, zt)
    
    # Poi, applica la rotazione (intorno agli assi Z, Y, e X)
    rotation_matrix = R_z(theta_z) @ R_y(theta_y)  # Assumiamo l'ordine di rotazione Z-Y-X
    
    # Ora combiniamo la trasformazione (rotazione + traslazione)
    transformation_matrix = translation_matrix @ rotation_matrix
    
    # Applichiamo la trasformazione al punto in coordinate UVW
    UVW_point = UVW(xp, yp, zp)
    XYZ = transformation_matrix @ UVW_point
    return XYZ




def focal_length(focal,s_resolution,p_sensor):
    return focal*s_resolution/p_sensor



def camera_to_plane(XYZ,focal,s_resolutionx,s_resolutiony,p_sensorx,p_sensory):

    fx = focal_length(focal, s_resolutionx, p_sensorx)  # Calcolo della focale in x
    fy = focal_length(focal, s_resolutiony, p_sensory)  # Calcolo della focale in y

    ox = s_resolutionx / 2  # Centro dell'immagine in x
    oy = s_resolutiony / 2  # Centro dell'immagine in y

    # Matrice di proiezione
    ff = np.array([
        [fx, 0, ox, 0],
        [0, fy, oy, 0],
        [0, 0, 1, 0]
    ])

    # Moltiplicazione della matrice di proiezione per le coordinate 3D
    xy_homogeneous = ff @ XYZ  # Risultato in coordinate omogenee (3x1)

    # Normalizzazione per ottenere le coordinate in 2D nel piano immagine
    x = xy_homogeneous[0] / xy_homogeneous[2]  # Coordinata x normalizzata
    y = xy_homogeneous[1] / xy_homogeneous[2]  # Coordinata y normalizzata

    return np.array([x, y])





def plane_to_pixel(xy):
    u = xy[0]  # Coordinata u (pixel x)
    v = xy[1]  # Coordinata v (pixel y)
    return u, v





# Dati iniziali
x_real = np.array([-2.5, 0.5, 0.5, 4.6])
y_real = np.array([13.41, 8.00, 13.00, 10.91])
z_real = np.zeros_like(x_real)

xt, yt, zt = 0, 0, 7.20  # Coordinate della camera
thyaw = 0 * np.pi / 180  # Yaw (rotazione attorno a Z)
thpitch = -32 * np.pi / 180  # Pitch (rotazione attorno a Y) (radianti)
throll = 0 * np.pi / 180  # Roll (rotazione attorno a X)
# Parametri immagine
f = 0.003  # Distanza focale (m)
s_w = 0.00498  # Larghezza sensore (m)
s_h = 0.00374  # Altezza sensore (m)
U = 1280  # Larghezza immagine (pixel)
V = 720  # Altezza immagine (pixel)



# Lettura immagine con OpenCV
image_file = "./test20241212.png"
image = cv2.imread(image_file)




# Disegna i punti sull'immagine
i=0
#for i in range(len(x_real)):

XYZ=real_to_camera(xt, zt, yt, x_real[i], z_real[i], y_real[i], theta_x=0, theta_z=thpitch, theta_y=0)
print("XYZ è:")
print(XYZ)

xy=camera_to_plane(XYZ,focal=f,s_resolutionx=U,s_resolutiony=V,p_sensorx=s_w,p_sensory=s_h)
print("xy è:")
print(xy)


u,v= plane_to_pixel(xy)
print("u e v sono:")
print(u,v)



point_x = int(u) 
point_y = int(v)
point = (point_x, point_y)
print("point è:")
print(point)

cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  # Cerchi rossi



# Salva e visualizza il risultato
output_file = ".output_image.png"
cv2.imwrite(output_file, image)

cv2.imshow("Projected Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()