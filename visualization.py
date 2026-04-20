import cv2
import numpy as np

def overlay_results_and_display(original_image, result_bev, binary_bev, debug_data, lanes_bev_for_warping, H, yolo_detector, display_height=480):
    """
    Gestisce tutto l'output visivo. Proietta le linee rilevate sull'immagine originale,
    aggiunge il testo del tipo di linea o messaggi d'errore, ed esegue il pass di YOLO. 
    Infine compone e renderizza la schermata combinata a 3 riquadri.
    """
    h_orig, w_orig = original_image.shape[:2]
    bev_height, bev_width = result_bev.shape[:2]
    
    # Proietta linee curve sull'immagine originale
    if debug_data['lane_found']:
        # Omografia Inversa
        inv_H = np.linalg.inv(H)
        
        # Warp Perspective: mappa le linee verticali complete all'immagine originale
        warped_lanes = cv2.warpPerspective(lanes_bev_for_warping, inv_H, (w_orig, h_orig))
        
        # Estrai i pixel verdi dal warping
        mask_warped_green = cv2.inRange(warped_lanes, (0, 255, 0), (0, 255, 0))
        
        # Trova i contorni delle corsie (linee continue)
        contours, _ = cv2.findContours(mask_warped_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        original_with_lanes = original_image.copy()
        
        # Disegna i contorni riempiendoli di verde continuo (-1 significa riempimento)
        if contours:
            cv2.drawContours(original_with_lanes, contours, -1, (0, 255, 0), -1)
        else:
            # Fallback: se non ci sono contorni, colora i pixel in modo pieno
            original_with_lanes[mask_warped_green == 255] = (0, 255, 0)
    else:
        original_with_lanes = original_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Disegna la scritta rossa al centro dell'immagine se nessuna corsia è trovata
        cv2.putText(original_with_lanes, "No lanes found", 
                    (w_orig//2 - 250, h_orig//2 - 150), 
                    font, 2.5, (0, 0, 255), 4, cv2.LINE_AA)
    
    # Aggiungi tipo di linea sull'immagine originale
    if debug_data['lane_found']:
        if debug_data['left_type'] is not None:
            cv2.putText(original_with_lanes, f"L:{debug_data['left_type']}", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if debug_data['right_type'] is not None:
            cv2.putText(original_with_lanes, f"R:{debug_data['right_type']}", 
                        (w_orig - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Applica YOLO Object Detection sull'immagine con le corsie disegnate
    original_with_lanes = yolo_detector.detect_and_draw(original_with_lanes)
    
    # Original con lanes
    scale_orig = display_height / h_orig
    w_orig_scaled = int(w_orig * scale_orig)
    original_resized = cv2.resize(original_with_lanes, (w_orig_scaled, display_height))
    
    # BEV con lanes (usa result_bev per visualizzazione)
    scale_bev = display_height / bev_height
    w_bev_scaled = int(bev_width * scale_bev)
    bev_resized = cv2.resize(result_bev, (w_bev_scaled, display_height))
    
    # Binary map della BEV (converti in BGR per visualizzazione)
    binary_resized = cv2.resize(binary_bev, (w_bev_scaled, display_height))
    binary_bgr = cv2.cvtColor(binary_resized, cv2.COLOR_GRAY2BGR)
    
    # Crea side-by-side: Original | BEV | Binary Map
    side_by_side = np.hstack([original_resized, bev_resized, binary_bgr])
    
    # Mostra risultato
    cv2.imshow("GOLD Lane Detection - Original | BEV | Binary Map", side_by_side)
    
    return original_with_lanes, side_by_side