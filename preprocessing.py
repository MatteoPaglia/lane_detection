import cv2
import numpy as np

def preprocess_bev_sobel(bev_image):
    """
    Preprocessing con SOBEL-X per trovare i bordi verticali.
    Adatto per linee nette e ben definite.
    
    Returns:
        Immagine binaria con corsie rilevate
    """
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-8))
    
    _, binary_output = cv2.threshold(scaled_sobel, 40, 255, cv2.THRESH_BINARY)
    
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.dilate(binary_output, kernel_dilation, iterations=1)
    
    # Rimuove le strisce pedonali
    row_white_pixel_count = np.sum(cleaned == 255, axis=1)
    white_pixel_threshold = np.percentile(row_white_pixel_count, 70)
    stripe_rows = row_white_pixel_count > white_pixel_threshold
    cleaned[stripe_rows, :] = 0
    
    return cleaned


def preprocess_bev_canny(bev_image):
    """
    Preprocessing con CANNY EDGE DETECTOR robusto contro rumore esterno.
    Utilizza filtri di colore restrittivi (Giallo, Bianco, Grigio) per escludere auto parcheggiate.
    
    Returns:
        Immagine binaria con corsie rilevate
    """
    # 1. Spazio Colore: Converti in HSV
    hsv = cv2.cvtColor(bev_image, cv2.COLOR_BGR2HSV)
    
    # 2. Maschera Giallo (molto allargata)
    lower_yellow = np.array([5, 40, 50])
    upper_yellow = np.array([50, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 3. Maschera Bianco (molto permissiva)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # 4. Fusione: Mantieni solo GIALLO e BIANCO
    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    
    # 5. Applicazione Maschera: Converti in scala di grigi e applica maschera
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=combined_mask)
    
    # 6. Blur: Applica GaussianBlur per ammorbidire la grana dell'asfalto
    blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    
    # 7. Canny Edge Detection: Threshold più alti per essere meno permissivo
    edges = cv2.Canny(blur, 60, 150)
    
    # 8. Dilatazione: Kernel più piccolo per non ingrassare troppo
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    cleaned = cv2.dilate(edges, kernel_dilation, iterations=1)
    
    # 8a. CONNESSIONE SEGMENTI: Dilata orizzontalmente per connettere corsie spezzate
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    cleaned = cv2.dilate(cleaned, kernel_horizontal, iterations=1)
    
    # 8b. FILTRAGGIO SEGNI ISOLATI: Rimuovi colonne con troppi pochi pixel bianchi
    # I segni pedonali/strisce sono isolati verticalmente
    # Le corsie continue hanno molti pixel bianchi per colonna
    white_pixels_per_column = np.sum(cleaned == 255, axis=0)
    min_pixels_threshold = 10  # Threshold più basso ora che corsie sono connesse
    columns_to_remove = white_pixels_per_column < min_pixels_threshold
    cleaned[:, columns_to_remove] = 0
    
    # 9. Ritorna immagine binaria finale
    return cleaned


def preprocess_bev_tophat(bev_image):
    """
    Preprocessing con TOP-HAT TRANSFORM (morphological operation).
    Estrae linee chiare (corsie) su sfondo scuro usando contrast stretching.
    
    Returns:
        Immagine binaria con corsie rilevate
    """
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    
    # Contrast stretching (CLAHE) per migliorare il contrasto
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Top-Hat con kernel verticale (le corsie sono linee verticali in BEV)
    kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))
    tophat_ver = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel_ver)
    
    # Anche un kernel più generale
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    tophat_gen = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    
    # Combina i due approcci
    combined = cv2.bitwise_or(tophat_ver, tophat_gen)
    
    # Normalizza a 0-255 per Canny
    normalized = np.uint8(255 * (combined / (np.max(combined) + 1e-8)))
    
    # Applica Canny come ulteriore filtro (combina morphology + edge detection)
    edges = cv2.Canny(normalized, 20, 50)
    
    # Dilation aggressiva per ricostruire
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.dilate(edges, kernel_dilation, iterations=3)
    
    # Rimuove le strisce pedonali
    row_white_pixel_count = np.sum(cleaned == 255, axis=1)
    white_pixel_threshold = np.percentile(row_white_pixel_count, 70)
    stripe_rows = row_white_pixel_count > white_pixel_threshold
    cleaned[stripe_rows, :] = 0
    
    return cleaned


# Alias per compatibilità (di default usa Sobel)
def preprocess_bev_image(bev_image):
    """
    Funzione di default che usa Sobel.
    Cambia questa riga in run_gold.py per provare altri algoritmi:
    - preprocess_bev_sobel()
    - preprocess_bev_canny()
    - preprocess_bev_tophat()
    """
    return preprocess_bev_sobel(bev_image)
