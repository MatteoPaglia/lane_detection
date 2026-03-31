import cv2
import numpy as np

def compute_homography_matrix(fx, fy, cx, cy, h, pitch=0, output_width=1920, output_height=1080):
    """
    Calcola la matrice di Omografia (IPM) basandosi sui parametri intrinseci ed estrinseci della telecamera.
    """
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
    """
    Applica la trasformazione prospettica (Inverse Perspective Mapping) all'immagine.
    Restituisce l'immagine in Bird's Eye View.
    """
    # cv2.warpPerspective deforma l'immagine usando la matrice 3x3
    # flags=cv2.INTER_LINEAR assicura che l'immagine ridimensionata sia fluida
    bev_image = cv2.warpPerspective(image, homography_matrix, output_size, flags=cv2.INTER_LINEAR)
    
    return bev_image