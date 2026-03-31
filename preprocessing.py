import cv2
import numpy as np

def preprocess_bev_image(bev_image):
    """
    Prende in input l'immagine Bird's Eye View e restituisce un'immagine binaria
    (bianco e nero) dove i pixel bianchi rappresentano i probabili bordi delle corsie.
    
    Usa Sobel-X per trovare le corsie verticali, filtra le righe troppo chiare (strisce).
    """
    # 1. Convertiamo in scala di grigi
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Applichiamo un Gaussian Blur per ridurre il rumore
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Sobel-X per trovare i bordi verticali (CORSIE)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    
    # 4. Calcola la luminosità media per ogni riga
    # Se una riga è molto chiara (strisce bianche), ha media alta
    brightness_per_row = np.mean(blur, axis=1)
    
    # 5. Identifica le righe troppo chiare (probabilmente strisce pedonali)
    brightness_threshold = np.percentile(brightness_per_row, 50)
    bright_rows = brightness_per_row > brightness_threshold
    
    # 6. Scaliamo Sobel-X tra 0 e 255
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-8))
    
    # 7. Dilation per ricostruire le corsie
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.dilate(scaled_sobel, kernel_dilation, iterations=1)
    
    # 8. Sogliatura (Thresholding)
    thresh_min = 70
    thresh_max = 255
    _, binary_output = cv2.threshold(cleaned, thresh_min, thresh_max, cv2.THRESH_BINARY)
    
    # 9. IDENTIFICA le righe piene di bianco (strisce pedonali)
    # Somma quanti pixel bianchi (255) ci sono in ogni riga
    row_white_pixel_count = np.sum(binary_output == 255, axis=1)
    
    # 10. Se una riga ha troppi pixel bianchi, è una striscia
    # Azzera il top 30% delle righe più piene
    white_pixel_threshold = np.percentile(row_white_pixel_count, 70)
    stripe_rows = row_white_pixel_count > white_pixel_threshold
    
    # 11. Azzera le righe piene di strisce
    binary_output[stripe_rows, :] = 0
    
    return binary_output
