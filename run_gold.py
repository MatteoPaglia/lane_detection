"""
Lane Detection Assignment - GOLD Algorithm
PandaSet Dataset Processing
University Project - Final Submission
"""

import cv2
import numpy as np
import sys
import glob
from pathlib import Path
from camera_init import principalPoint, focalLength, height, pitch
from geometry import compute_homography_matrix, calculate_ipm
from preprocessing import preprocess_bev_image
from yolo import ObstacleDetector


def project_lanes_to_original(original_image, src_points, left_x_bev, right_x_bev, bev_size, 
                              left_detected=False, right_detected=False):
    """
    Proietta le coordinate delle corsie rilevate dalla BEV all'immagine originale
    e disegna le linee verdi.
    
    Args:
        original_image: Immagine originale
        src_points: I 4 punti della ROI in coordinate immagine
        left_x_bev: Coordinata X della corsia sinistra nella BEV
        right_x_bev: Coordinata X della corsia destra nella BEV
        bev_size: (bev_width, bev_height) dimensione della BEV
        left_detected: Flag se la linea sinistra è stata rilevata
        right_detected: Flag se la linea destra è stata rilevata
        
    Returns:
        Immagine originale con linee verdi delle corsie (solo quelle rilevate)
    """
    lanes_image = original_image.copy()
    
    bev_width, bev_height = bev_size
    
    pt_top_left = src_points[0]
    pt_top_right = src_points[1]
    pt_bottom_right = src_points[2]
    pt_bottom_left = src_points[3]
    
    ratio_left = left_x_bev / max(bev_width - 1, 1)
    ratio_right = right_x_bev / max(bev_width - 1, 1)
    
    left_point_top = pt_top_left + (pt_top_right - pt_top_left) * ratio_left
    right_point_top = pt_top_left + (pt_top_right - pt_top_left) * ratio_right
    
    left_point_bottom = pt_bottom_left + (pt_bottom_right - pt_bottom_left) * ratio_left
    right_point_bottom = pt_bottom_left + (pt_bottom_right - pt_bottom_left) * ratio_right
    
    # Disegna solo la linea sinistra se è stata rilevata
    if left_detected:
        cv2.line(lanes_image, 
                 tuple(left_point_top.astype(int)), 
                 tuple(left_point_bottom.astype(int)), 
                 (0, 255, 0), 3, cv2.LINE_AA)
    
    # Disegna solo la linea destra se è stata rilevata
    if right_detected:
        cv2.line(lanes_image,
                 tuple(right_point_top.astype(int)),
                 tuple(right_point_bottom.astype(int)),
                 (0, 255, 0), 3, cv2.LINE_AA)
    
    return lanes_image


def classify_lane_type(binary_image, x_center, window_size=5):
    """
    Classifica il tipo di linea analizzando una finestra (strip) centrata attorno alla x.
    Usa il 'fill ratio' (rapporto tra pixel bianchi e lunghezza totale della linea).
    """
    h, w = binary_image.shape
    
    # 1. Estrai una piccola 'striscia' verticale intorno al centro della linea
    x_min = max(0, x_center - window_size)
    x_max = min(w, x_center + window_size + 1)
    lane_strip = binary_image[:, x_min:x_max]
    
    # 2. Crea un profilo 1D: per ogni riga, dimmi se c'è almeno un pixel bianco
    vertical_profile = np.max(lane_strip, axis=1) / 255.0  # Array di 0.0 e 1.0
    
    # Trova dove si trovano i pixel della linea
    non_zero_indices = np.where(vertical_profile == 1)[0]
    
    if len(non_zero_indices) == 0:
        return "Unknown"
        
    # 3. Trova l'inizio (strada lontana) e la fine (strada vicina) effettivi della linea visibile
    top_idx = non_zero_indices[0]
    bottom_idx = non_zero_indices[-1]
    
    line_length = bottom_idx - top_idx + 1
    white_pixels = len(non_zero_indices)
    
    # 4. Calcola il rapporto di riempimento
    # (Ad es. 0.9 = 90% di linea bianca continua)
    fill_ratio = white_pixels / line_length if line_length > 0 else 0
    
    # 5. Restituisci la classificazione basata sul Ratio (indipendente dalla risoluzione)
    if fill_ratio < 0.10: # Meno del 10% di pixel non è una corsia valida (solo rumore)
        return "None"
    elif fill_ratio > 0.65: # Soglia empirica (65% di riempimento)
        return "Solid"
    else:
        return "Dashed"


def find_lanes_and_draw(bev_image, binary_image):
    """
    Calcola l'istogramma sulla metà inferiore dell'immagine binaria, trova i picchi
    e disegna le linee rilevate. Usa una soglia dinamica basata sull'altezza dell'immagine.
    
    Returns:
        tuple: (output_image, debug_data) - immagine e dati di debug
    """
    output_image = bev_image.copy()
    height, width = binary_image.shape
    
    # DEBUG: Salva la binary_image per ogni frame
    cv2.imwrite(f"debug_binary_{len(str(binary_image.sum()))}.png", binary_image)
    
    lower_half = binary_image[height//2:, :]
    histogram = np.sum(lower_half, axis=0)
    num_rows = lower_half.shape[0]
    
    midpoint = int(histogram.shape[0] / 2)
    
    # Margin Cropping: ignora i bordi estremi (10%)
    margin = int(width * 0.1)
    
    left_search_range = histogram[margin:midpoint - margin]
    left_offset = np.argmax(left_search_range) + margin
    left_x_base = left_offset
    
    right_search_range = histogram[midpoint + margin:width - margin]
    right_offset = np.argmax(right_search_range) + midpoint + margin
    right_x_base = right_offset
    
    # Soglia dinamica - minimo valore per rilevare una linea
    min_peak_threshold = int((height / 1080.0) * 5000) if height > 0 else 5000
    
    lane_found = False
    left_detected = False
    right_detected = False
    left_type = None
    right_type = None
    
    # DEBUG: Analizza output
    white_pixels = np.sum(binary_image == 255)
    print(f"\n🔍 DEBUG - White pixels in binary_image: {white_pixels}")
    print(f"🔍 Histogram max: {np.max(histogram)}, Threshold: {min_peak_threshold}")
    
    # Verifichiamo e disegniamo la linea sinistra
    left_value = histogram[left_x_base]
    print(f"🔍 LEFT ({left_x_base}): {left_value} (threshold: {min_peak_threshold}) - {'✅ DETECTED' if left_value > min_peak_threshold else '❌ MISSED'}")
    if left_value > min_peak_threshold:
        left_type = classify_lane_type(binary_image, left_x_base)
        if left_type != "None":
            cv2.line(output_image, (left_x_base, 0), (left_x_base, height), (0, 255, 0), 3)
            cv2.putText(output_image, f"L:{left_type}", (max(0, left_x_base - 60), 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            lane_found = True
            left_detected = True
        
    # Verifichiamo e disegniamo la linea destra
    right_value = histogram[right_x_base]
    print(f"🔍 RIGHT ({right_x_base}): {right_value} (threshold: {min_peak_threshold}) - {'✅ DETECTED' if right_value > min_peak_threshold else '❌ MISSED'}")
    if right_value > min_peak_threshold:
        right_type = classify_lane_type(binary_image, right_x_base)
        if right_type != "None":
            cv2.line(output_image, (right_x_base, 0), (right_x_base, height), (0, 255, 0), 3)
            cv2.putText(output_image, f"R:{right_type}", (min(width - 100, right_x_base + 10), 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            lane_found = True
            right_detected = True
    
    # Gestione del fallimento
    if not lane_found:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output_image, "No lanes found", (width//2 - 150, height//2), 
                    font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    debug_data = {
        'histogram': histogram,
        'left_x_base': left_x_base,
        'right_x_base': right_x_base,
        'left_value': left_value,
        'right_value': right_value,
        'left_ratio': left_value / (num_rows * 255) if num_rows > 0 else 0,
        'right_ratio': right_value / (num_rows * 255) if num_rows > 0 else 0,
        'threshold': min_peak_threshold,
        'left_detected': left_detected,
        'right_detected': right_detected,
        'left_type': left_type,
        'right_type': right_type,
        'lane_found': lane_found,
        'binary_image_white_pixels': np.sum(binary_image) // 255,
        'binary_image_mean': np.mean(binary_image)
    }
    
    return output_image, debug_data


def main():
    """
    Main pipeline: carica immagini da riga di comando, applica GOLD algorithm, mostra risultati.
    """
    # Gestione argomenti da riga di comando
    if len(sys.argv) < 2:
        print("Usage: python3 run_gold.py <path_to_images>")
        print("Example: python3 run_gold.py PandaSetSensorData/archive/044/camera/front_camera/*.jpg")
        sys.exit(1)
    
    # Gestisci sia l'espansione della shell che il passaggio di stringa singola
    image_paths = []
    for arg in sys.argv[1:]:
        if '*' in arg or '?' in arg:
            # Se contiene wildcard, usa glob
            matches = sorted(glob.glob(arg))
            image_paths.extend(matches)
        else:
            # Altrimenti, trattalo come percorso singolo
            image_paths.append(arg)
    
    # Se ancora niente, prova a globbare il primo argomento
    if not image_paths and len(sys.argv) > 1:
        arg = sys.argv[1]
        matches = sorted(glob.glob(arg))
        if matches:
            image_paths = matches
    
    if not image_paths:
        print("Error: No images found at the specified path.")
        sys.exit(1)
    
    print(f"[✓] Found {len(image_paths)} images")
    print("=" * 60)
    
    # Inizializza detector ostacoli
    print("\n[*] Initializing YOLO Object Detector...")
    yolo_detector = ObstacleDetector()
    
    # Inizializza camera e geometria
    fx, fy = focalLength
    cx, cy = principalPoint
    h = height
    
    # Computa omografia (IPM)
    H, (bev_width, bev_height), roi_src_points = compute_homography_matrix(
        fx, fy, cx, cy, h, pitch=pitch, 
        x_min=-2.25, x_max=2.25, 
        z_min=4.0, z_max=30.0,
        bev_width=400, bev_height=800
    )
    
    # Ciclo su tutte le immagini
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {Path(image_path).name}")
        
        # Load immagine
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  ✗ Failed to load image: {image_path}")
            continue
        
        h_orig, w_orig = original_image.shape[:2]
        
        # Applica IPM per ottenere BEV
        bev_image = calculate_ipm(original_image, H, output_size=(bev_width, bev_height))
        
        # Preprocessing binario
        binary_bev = preprocess_bev_image(bev_image)
        
        # Trova e disegna corsie
        result_bev, debug_data = find_lanes_and_draw(bev_image, binary_bev)
        
        # Proietta linee sull'immagine originale
        if debug_data['lane_found']:
            left_x_bev = debug_data['left_x_base']
            right_x_bev = debug_data['right_x_base']
            original_with_lanes = project_lanes_to_original(original_image, roi_src_points, 
                                                             left_x_bev, right_x_bev, 
                                                             (bev_width, bev_height),
                                                             left_detected=debug_data['left_detected'],
                                                             right_detected=debug_data['right_detected'])
        else:
            original_with_lanes = original_image.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Disegna la scritta rossa al centro dell'immagine originale se nessuna corsia è trovata
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
        
        # Ridimensiona per display
        display_height = 720
        
        # Original con lanes
        scale_orig = display_height / h_orig
        w_orig_scaled = int(w_orig * scale_orig)
        original_resized = cv2.resize(original_with_lanes, (w_orig_scaled, display_height))
        
        # BEV con lanes
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
        
        # Stampa risultati
        if debug_data['lane_found']:
            print(f"  ✓ Lanes found!")
            if debug_data['left_detected']:
                print(f"    - Left lane:  {debug_data['left_type']} (value: {debug_data['left_value']:.0f})")
            if debug_data['right_detected']:
                print(f"    - Right lane: {debug_data['right_type']} (value: {debug_data['right_value']:.0f})")
        else:
            print(f"  ✗ No lanes found")
        
        # Attendi input (premi qualsiasi tasto per andare avanti, ESC per uscire)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            print("\n[*] Exiting...")
            break
    
    # Chiudi finestre
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("[✓] Processing complete")


if __name__ == "__main__":
    main()
