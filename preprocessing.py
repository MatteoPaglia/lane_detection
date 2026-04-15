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

def preprocess_bev_tophat(bev_image, debug=False):
    """
    Implementazione esatta del filtro Dark-Light-Dark del paper GOLD (1998)
    con l'aggiunta di filtraggio morfologico per unire i tratteggi e 
    il filtro a muro orizzontale per rimuovere le strisce pedonali.
    """
    # Convertiamo in float32 per evitare overflow durante le sottrazioni
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # ===== MASCHERA GIALLA (corsie gialle/arancioni) =====
    hsv = cv2.cvtColor(bev_image, cv2.COLOR_BGR2HSV)
    # Molto più permissiva su Saturazione e Luminosità (Value) per catturare il giallo in ombra!
    lower_yellow = np.array([10, 40, 40])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # ===== MASCHERA BIANCA (Top-Hat sul canale grigio) =====
    # Il parametro 'm' del paper: la distanza dei vicini.
    m = 20 
    
    # Creiamo i vicini shiftando l'intera matrice a sinistra e a destra di 'm' pixel
    left_neighbor = np.roll(gray, -m, axis=1)
    right_neighbor = np.roll(gray, m, axis=1)
    
    # Applichiamo la formula di Bertozzi/Broggi
    enhanced = gray - ((left_neighbor + right_neighbor) / 2.0)
    
    # ===== FILTRO SPAZIALE LATERALE (Oscuramento Estremi) =====
    # Costruiamo un moltiplicatore orizzontale che vale 1.0 al centro e tende a 0.0 (o molto basso)
    # man mano che ci si avvicina ai bordi (sinistro e destro).
    h_img, w_img = enhanced.shape
    x_coords = np.arange(w_img)
    center = w_img / 2.0
    # Usiamo una curva parabolica: 1 - ((x - centro) / centro)^2
    # Al centro (x=center): 1 - 0 = 1.0
    # Ai bordi (x=0 o x=w): 1 - 1 = 0.0
    spatial_falloff = 1.0 - ((x_coords - center) / center)**2
    spatial_falloff = np.clip(spatial_falloff, 0, 1.0).astype(np.float32)
    
    # Moltiplichiamo la mappa 'enhanced' per far decadere la luminosità ai lati
    enhanced = enhanced * spatial_falloff
    
    # Rimuoviamo i valori negativi e riportiamo nel range uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Binarizzazione finale
    _, binary = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
    
    # ===== UNIONE DELLE MASCHERE (BIANCA + GIALLA) =====
    combined = cv2.bitwise_or(binary, mask_yellow)
    
    # ===== CLOSING MORFOLOGICO: Fonde i segmenti sparsi (es. macchie auto) in un sol blocco =====
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined_closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    
    # ===== FILTRO OSTACOLI DENSI (Auto e grossi blocchi) =====
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_closed, connectivity=8)
    
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        height_bb = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Le corsie continue (solide) coprono un'Area elevata e, curvando, hanno una Larghezza (Width) elevata.
        # Se un blocco è larghissimo (es auto) ed ha una elevata "compattezza/area", lo cancelliamo.
        # Un bounding box di un'auto è compatto (>6000). La corsia curva è sparpagliata e lunga.
        if width > 150 and area > 8000:
            combined[labels == i] = 0

    # ===== FILTRO VERTICALITÀ (Azzardo: Strisce contigue in altezza) =====
    # Usiamo un'apertura (Opening) morfologica con un kernel stretto e alto 4 pixel.
    # La logica matematica di questo filtro è identica alla tua richiesta:
    # Sopravvivono SOLO e unicamente i pixel che fanno parte di un blocco bianco
    # ininterrotto di almeno 4 pixel sull'asse Y (cioè y, y+1, y+2, y+3).
    # I puntini orizzontali o isolati si azzerano.
    # kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    # combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_vert)

    # ===== FILTRO COLONNE (Rumore di fondo strutturale orizzontale) =====
    img_height = combined.shape[0]
    
    # Somma pixel bianchi SOLO nella metà inferiore (allineato all'istogramma)
    lower_half = combined[img_height//2:, :]
    col_sums = np.sum(lower_half == 255, axis=0)
    
    # Soglia minima: se in quella colonna, nella metà inferiore, ci sono meno di 5 pixel totali
    # azzeriamo l'intera colonna (elimina rumore microscopico verticale).
    min_pixels_lower = 5
    invalid_cols = col_sums < min_pixels_lower
    combined[:, invalid_cols] = 0
    
    if debug:
        return combined, binary, mask_yellow
    return combined
# Alias per compatibilità (di default usa Sobel)
def preprocess_bev_image(bev_image, debug=False):
    """
    Funzione di default che usa Sobel.
    Cambia questa riga in run_gold.py per provare altri algoritmi:
    - preprocess_bev_sobel()
    - preprocess_bev_canny()
    - preprocess_bev_tophat()
    """
    return preprocess_bev_tophat(bev_image, debug=debug)
