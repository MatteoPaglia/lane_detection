import cv2
import numpy as np

def apply_yellow_mask(bev_image, lower_hsv=np.array([10, 40, 40]), upper_hsv=np.array([40, 255, 255])):
    """
    Estrae i pixel gialli/arancioni convertendo l'immagine in spazio HSV
    e applicando un range di tolleranza, utile per rilevare linee gialle.
    """
    hsv = cv2.cvtColor(bev_image, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask_yellow

def apply_tophat_white(gray_image, m=20):
    """
    Applica il filtro Dark-Light-Dark (Top-Hat di GOLD).
    Sottrae all'immagine originale la media dei vicini spostati di 'm' pixel
    a sinistra e a destra, evidenziando le linee chiare verticali.
    """
    left_neighbor = np.roll(gray_image, -m, axis=1)
    right_neighbor = np.roll(gray_image, m, axis=1)
    enhanced = gray_image - ((left_neighbor + right_neighbor) / 2.0)
    return enhanced

def apply_spatial_falloff(enhanced_image, tracked_left=None, tracked_right=None, confidence=0.0):
    """
    Applica un decadimento parabolico dell'intensità.
    Se le corsie sono tracciate, applica una "doppia parabola" centrata sulle rilevazioni.
    L'oscuramento (vanishing) diventa sempre più forte all'aumentare della confidenza temporale,
    aiutando a sopprimere totalmente il rumore esterno nei frame successivi e garantendo stabilità.
    Altrimenti usa un oscuramento base leggero dal centro.
    """
    h_img, w_img = enhanced_image.shape
    x_coords = np.arange(w_img)
    
    if tracked_left is not None and tracked_right is not None and confidence > 0.0:
        L = float(tracked_left)
        R = float(tracked_right)
        M = (L + R) / 2.0
        mask = np.zeros(w_img, dtype=np.float32)
        
        # Oscura progressivamente verso SX partendo da L
        if L > 0: mask[x_coords < L] = 1.0 - ((L - x_coords[x_coords < L]) / L)**2
        # Oscura progressivamente verso il centro M partendo da L
        if M > L: mask[(x_coords >= L) & (x_coords < M)] = 1.0 - ((x_coords[(x_coords >= L) & (x_coords < M)] - L) / (M - L))**2
        # Oscura progressivamente verso il centro M partendo da R
        if R > M: mask[(x_coords >= M) & (x_coords < R)] = 1.0 - ((R - x_coords[(x_coords >= M) & (x_coords < R)]) / (R - M))**2
        # Oscura progressivamente verso DX partendo da R
        if w_img > R: mask[x_coords >= R] = 1.0 - ((x_coords[x_coords >= R] - R) / (w_img - R))**2
        
        mask = np.clip(mask, 0, 1.0)
        
        # Sfumatura adattiva: 
        # Confidence 0 -> spatial falloff è pari a 1 (luce massima ovunque)
        # Confidence 1 -> spatial falloff è pari alla doppia maschera pura calcolata
        spatial_falloff = 1.0 - (1.0 - mask) * confidence
    else:
        # Comportamento iniziale di default fallback (Parabola standard dal centro)
        center = w_img / 2.0
        spatial_falloff = 1.0 - ((x_coords - center) / center)**2
        spatial_falloff = np.clip(spatial_falloff, 0, 1.0).astype(np.float32)
        
    enhanced_image = enhanced_image * spatial_falloff
    return np.clip(enhanced_image, 0, 255).astype(np.uint8)

def filter_dense_obstacles(binary_image, max_area=6000, min_area=10, max_width=80):
    """
    Utilizza i componenti connessi per rimuovere blocchi troppo grandi (es. auto parcheggiate)
    o troppo piccoli (rumore isolato o macchie).
    """
    # Aumento la forza del closing a 15x15. Questo espande e fonde tra loro 
    # i bordi frammentati della macchina rendendola un unico gigantesco blocco fuso (blob),
    # cosicché la sua area e larghezza esplodano garantendone l'eliminazione!
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    combined_closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_closed, connectivity=8)
    
    cleaned_image = binary_image.copy()
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Elimina se:
        # - È enorme (ostacolo/auto fusa) > 6000
        # - È microscopico < 10
        # - È "tozzo": molto largo (>80), consistente (>1500) e alto, forma tipica di una vettura spiaccicata in BEV
        if area > max_area or area < min_area or (width > max_width and height > 40 and area > 1500):
            cleaned_image[labels == i] = 0
            
    return cleaned_image

def enhance_verticality_and_density(binary_image, dilate_ksize=(3, 15), close_ksize=(5, 15)):
    """
    Ingrassa e allunga le corsie verticalmente usando operazioni morfologiche
    (dilatazione e chiusura). Trasforma i tratteggi deboli in segmenti continui e spessi.
    """
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_ksize)
    enhanced = cv2.dilate(binary_image, kernel_dilate, iterations=1)
    
    kernel_close_final = cv2.getStructuringElement(cv2.MORPH_RECT, close_ksize)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close_final)
    return enhanced

def filter_columns_by_density(binary_image, min_pixels=5):
    """
    Elimina il rumore strutturale orizzontale rimuovendo intere colonne
    che possiedono meno di 'min_pixels' bianchi nella metà inferiore dell'immagine.
    """
    img_height = binary_image.shape[0]
    lower_half = binary_image[img_height//2:, :]
    col_sums = np.sum(lower_half == 255, axis=0)
    
    invalid_cols = col_sums < min_pixels
    cleaned_image = binary_image.copy()
    cleaned_image[:, invalid_cols] = 0
    return cleaned_image

def preprocess_bev_image(bev_image, 
                         use_yellow_mask=True, 
                         use_tophat=True, 
                         use_spatial_falloff=True,
                         use_dense_obstacles_filter=True,
                         use_density_enhancement=True,
                         use_column_filter=True,
                         tracked_left=None,
                         tracked_right=None,
                         confidence=0.0,
                         blind_shade_factor=0.0,
                         debug=False):
    """
    Pipeline di preprocessing dell'immagine BEV. 
    Applica una serie di tecniche attivabili/disattivabili tramite i flag booleani.
    Tutte le funzioni collaborano per isolare le corsie dal rumore.
    """
    # Maschera Gialla Init
    mask_yellow = np.zeros(bev_image.shape[:2], dtype=np.uint8)
    if use_yellow_mask:
        mask_yellow = apply_yellow_mask(bev_image)
        
    binary_white = np.zeros(bev_image.shape[:2], dtype=np.uint8)
    
    if use_tophat:
        gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        enhanced = apply_tophat_white(gray, m=20)
        
        # Oscuramento globale adattivo ("Blind shade")
        if blind_shade_factor > 0.0:
            enhanced = enhanced * (1.0 - blind_shade_factor)
            
        if use_spatial_falloff:
            enhanced = apply_spatial_falloff(enhanced, tracked_left, tracked_right, confidence)
        else:
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
        # Binarizzazione finale del Top-Hat
        _, binary_white = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
    
    # Unione Giallo e Bianco
    combined = cv2.bitwise_or(binary_white, mask_yellow)
    
    if use_dense_obstacles_filter:
        combined = filter_dense_obstacles(combined)
        
    if use_density_enhancement:
        combined = enhance_verticality_and_density(combined)
        
    if use_column_filter:
        combined = filter_columns_by_density(combined)
        
    # Controllo di sicurezza globale: se dopo tutti i filtri l'intera immagine ha 
    # davvero troppi pochi pixel bianchi (es: solo una targa lontana scampata al filtro),
    # consideriamo la BEV completamente vuota per non generare falsi positivi sulle colonne.
    total_white_pixels = cv2.countNonZero(combined)
    if total_white_pixels < 250: # Soglia minima: una corsia di 2px * 125 di altezza
        combined[:] = 0
        
    if debug:
        return combined, binary_white, mask_yellow
    return combined

