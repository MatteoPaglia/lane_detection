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
    
    # 4. Scaliamo Sobel-X tra 0 e 255
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-8))
    
    # 5. Sogliatura (Thresholding)
    thresh_min = 70
    thresh_max = 255
    _, binary_output = cv2.threshold(scaled_sobel, thresh_min, thresh_max, cv2.THRESH_BINARY)
    
    # 6. Dilation per ricostruire le corsie tratteggiate
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.dilate(binary_output, kernel_dilation, iterations=1)
    
    # 7. IDENTIFICA e AZZERA le righe piene di bianco (strisce pedonali)
    row_white_pixel_count = np.sum(cleaned == 255, axis=1)
    white_pixel_threshold = np.percentile(row_white_pixel_count, 70)
    stripe_rows = row_white_pixel_count > white_pixel_threshold
    cleaned[stripe_rows, :] = 0
    
    return cleaned
