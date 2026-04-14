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
    Implementazione esatta del filtro Dark-Light-Dark del paper GOLD (1998)
    con l'aggiunta di un filtro geometrico (Connected Components) per 
    rimuovere le strisce pedonali in base alla loro larghezza.
    """
    # Convertiamo in float32 per evitare overflow durante le sottrazioni
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Il parametro 'm' del paper: la distanza dei vicini.
    m = 20 
    
    # Creiamo i vicini shiftando l'intera matrice a sinistra e a destra di 'm' pixel
    left_neighbor = np.roll(gray, -m, axis=1)
    right_neighbor = np.roll(gray, m, axis=1)
    
    # Applichiamo la formula di Bertozzi/Broggi
    enhanced = gray - ((left_neighbor + right_neighbor) / 2.0)
    
    # 1. Rimuoviamo i valori negativi
    # 2. Riportiamo nel range 0-255 uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Binarizzazione finale
    _, binary = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
    
    # Pulizia del rumore (piccoli puntini)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # =================================================================
    # FILTRO STRISCE PEDONALI: "Muro Orizzontale"
    # =================================================================
    # Logica geometrica semplice:
    # - BEV width: 400 pixel
    # - Corsia singola: 10-15 pixel
    # - 4 corsie max: ~60 pixel bianchi per riga
    # - Strisce pedonali: 150-200+ pixel bianchi per riga
    # 
    # Soluzione: Se una riga ha troppi pixel bianchi (> soglia), 
    # è una striscia pedonale e va annerita completamente
    
    row_white_count = np.sum(cleaned == 255, axis=1)  # Conta pixel bianchi per ogni riga
    max_lane_pixels = 30  # Soglia: max pixel bianchi per una riga di corsie normali
    
    # Trova le righe che superano la soglia (strisce pedonali)
    stripe_rows = row_white_count > max_lane_pixels
    
    # Annerisce completamente quelle righe
    cleaned[stripe_rows, :] = 0
    
    # =================================================================
    
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
    return preprocess_bev_tophat(bev_image)
