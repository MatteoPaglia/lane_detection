""" import cv2
import numpy as np

def compute_homography_matrix(fx, fy, cx, cy, h, pitch=0, output_width=1920, output_height=1080):
    
    Calcola la matrice di Omografia (IPM) basandosi sui parametri intrinseci ed estrinseci della telecamera.
    
    # 1. Definiamo i punti sull'immagine originale (src_points)
    # Assicuriamoci di campionare sotto l'orizzonte (che si trova a cy)
    v_top = int(cy + 117)       # Es. 483 + 117 = 600 (guarda lontano sulla strada)
    v_bottom = output_height    # 1080 (vicino all'auto)
    
    src_points = np.array([
        [800, v_top],   
        [1140, v_top],  
        [300, v_bottom], 
        [1640, v_bottom] 
    ], dtype=np.float32)

    # 2. Calcoliamo le coordinate reali (X, Z) in metri sulla strada
    world_points = []
    for u, v in src_points:
        Z = (h * fy) / (v - cy)           # Profondità
        X = Z * (u - cx) / fx             # Spostamento laterale
        world_points.append([X, Z])
    
    # 3. Mappiamo i punti nella nuova immagine BEV (dst_points)
    pixels_per_meter = 40 
    
    dst_points = []
    for X, Z in world_points:
        u_bev = (X * pixels_per_meter) + (output_width / 2)
        v_bev = output_height - (Z * pixels_per_meter) 
        dst_points.append([u_bev, v_bev])
        
    dst_points = np.array(dst_points, dtype=np.float32)

    # 4. Calcoliamo e restituiamo la matrice
    homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return homography_matrix

def calculate_ipm(image, homography_matrix, output_size=(1920, 1080)):
    
    Applica la trasformazione prospettica (Inverse Perspective Mapping) all'immagine.
    Restituisce l'immagine in Bird's Eye View.
    
    # cv2.warpPerspective deforma l'immagine usando la matrice 3x3
    # flags=cv2.INTER_LINEAR assicura che l'immagine ridimensionata sia fluida
    bev_image = cv2.warpPerspective(image, homography_matrix, output_size, flags=cv2.INTER_LINEAR)
    
    return bev_image """

import cv2
import numpy as np

def compute_homography_matrix(fx, fy, cx, cy, h, pitch=0, 
                              x_min=-1.85, x_max=1.85, 
                              z_min=4.0, z_max=30.0, 
                              bev_width=400, bev_height=800):
    """
    Calcola la matrice di Omografia (IPM) definendo un rettangolo esatto in metri 
    nel mondo reale e mappandolo sulla dimensione finale della BEV.
    
    ROI: 3.7 metri di larghezza (dimensione standard di una corsia americana)
    """
    # 1. Definiamo i 4 angoli della ROI in metri nel mondo reale (X laterale, Z profondità)
    # Origine: centro della telecamera (a terra)
    # IMPORTANTE: Ordine circolare orario per evitare il "farfallino"
    world_points = np.array([
        [x_min, z_max],  # 0: Top-Left (lontano a sinistra)
        [x_max, z_max],  # 1: Top-Right (lontano a destra)
        [x_max, z_min],  # 2: Bottom-Right (vicino a destra) - NON è Bottom-Left!
        [x_min, z_min]   # 3: Bottom-Left (vicino a sinistra)
    ], dtype=np.float32)

    # 2. Proiettiamo questi punti 3D sull'immagine originale 2D usando i parametri della telecamera
    src_points = []
    for X, Z in world_points:
        u = (X * fx / Z) + cx  # Coordinata X pixel
        v = (h * fy / Z) + cy  # Coordinata Y pixel
        src_points.append([u, v])
    src_points = np.array(src_points, dtype=np.float32)

    # 3. Definiamo i punti di destinazione in modo che il rettangolo riempia esattamente la BEV
    # L'ordine DEVE corrispondere a quello di world_points (ordine circolare)
    dst_points = np.array([
        [0, 0],                               # 0: Top-Left
        [bev_width - 1, 0],                   # 1: Top-Right
        [bev_width - 1, bev_height - 1],      # 2: Bottom-Right
        [0, bev_height - 1]                   # 3: Bottom-Left
    ], dtype=np.float32)

    # 4. Calcoliamo la matrice
    homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Restituiamo anche la dimensione e i src_points per disegnare la ROI
    return homography_matrix, (bev_width, bev_height), src_points

def calculate_ipm(image, homography_matrix, output_size):
    """Applica la trasformazione."""
    return cv2.warpPerspective(image, homography_matrix, output_size, flags=cv2.INTER_LINEAR)