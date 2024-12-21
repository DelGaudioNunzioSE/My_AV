import cv2
import numpy as np

# Parametri della fotocamera
image_file = "test.png"  # Immagine da caricare
f = 0.00325  # lunghezza focale in metri
U = 1920  # larghezza dell'immagine in pixel
V = 1080    # altezza dell'immagine in pixel
thyaw = 12 * np.pi / 180  # rotazione Yaw
throll =  10.5* np.pi / 180  # rotazione Roll
thpitch = -36 * np.pi / 180  # rotazione Pitch
xc, yc, zc = 0, 0, 6.92  # posizione della fotocamera nel sistema mondo

# Punti reali nello spazio 3D
x_real = np.array([0,-3.5, 3.6, -2.8, 2])
y_real = np.array([11.20, 13.11, 10.61, 4.11, 5.81])
# HO AGGIUNTO IL PUNTO (O,11.20) PERCHè HO CERCATO LA COORDINATA REALE DEL CENTRO
z_real = np.zeros_like(x_real)  # i punti sono su un piano z=0 (terra)

# Dimensioni di un pixel in metri
s_w = 0.00498 / U
s_h = 0.00374 / V 


'''
#Seconda configurazione:

image_file = "test20241212.png"  # Immagine da caricare
f = 0.003           # Lunghezza focale in metri
U = 1280            # Larghezza immagine in pixel
V = 720             # Altezza immagine in pixel
thpitch = -32 * np.pi / 180  # Angolo pitch (inclinazione verso il basso)
throll = 0
thyaw = 0
zc = 7.20           # Altezza della fotocamera in metri

# Coordinate reali dei punti in metri
x_real = [0,-2.5, 0.5, 0.5, 4.6]
y_real = [11.20,13.41, 8.00, 13.00, 10.91]
z_real = [0,0, 0, 0, 0]  # Piano a terra
# Centro Immagine (640,360) / (u,v) 
# Dimensioni pixel del sensore
s_w = 0.00498 / U  # Dimensione pixel orizzontale
s_h = 0.00374 / V  # Dimensione pixel verticale

'''

# Matrice intrinseca della fotocamera
fx = f / s_w
fy = f / s_h
cx = U / 2
cy = V / 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Matrice di rotazione basata sugli angoli di Euler
R_pitch = np.array([[1, 0, 0],
                    [0, np.cos(thpitch), -np.sin(thpitch)],
                    [0, np.sin(thpitch), np.cos(thpitch)]])

R_roll = np.array([[np.cos(throll), 0, np.sin(throll)],
                  [0, 1, 0],
                  [-np.sin(throll), 0, np.cos(throll)]])

R_yaw = np.array([[np.cos(thyaw), -np.sin(thyaw), 0],
                   [np.sin(thyaw), np.cos(thyaw), 0],
                   [0, 0, 1]])
R =  R_yaw @ R_pitch @ R_roll #Combinazione delle tre Matrici di rotazione 


# Vettore di traslazione
t = np.array([[0], [0], [zc]])

# Combina le coordinate reali in forma omogenea
points_world = np.vstack((x_real, y_real, z_real, np.ones_like(x_real)))

# Matrice extrinseca (rotazione e traslazione)
extrinsic = np.hstack((R_pitch, -R_pitch @ t))

# Trasforma i punti nel sistema di riferimento della camera
points_camera = extrinsic @ points_world

# Proietta i punti sul piano immagine
points_image = K @ points_camera[:3, :]
points_image /= points_image[2, :]  # Normalizza per la profondità

# Coordinate proiettate originali
u = points_image[0, :]
v = points_image[1, :]

# Ribalto i punti ripsetto l'asse y
u = U - u + 0
v = v + 272.57

# Stampa delle coordinate finali
print("Coordinate finali (u, v):")
for i in range(len(u)):
    print(f"Punto {i+1}: ({u[i]:.2f}, {v[i]:.2f})")

# Rimuovi punti fuori dall'immagine
valid_points = []
for u, v in zip(u, v):
    if 0 <= u < U and 0 <= v < V:
        valid_points.append((int(u), int(v)))

# Carica l'immagine e visualizza i punti proiettati
image = cv2.imread(image_file)
if image is not None:
    for (u_corr, v_corr) in valid_points:
        cv2.circle(image, (u_corr, v_corr), radius=8, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Corrected Rotated and Mirrored Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Errore: Immagine non trovata.")
