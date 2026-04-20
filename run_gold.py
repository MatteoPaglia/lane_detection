"""
Lane Detection Assignment - GOLD Algorithm
PandaSet Dataset Processing
University Project - Final Submission
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from camera_init import principalPoint, focalLength, height, pitch
from geometry import compute_homography_matrix, calculate_ipm
from preprocessing import preprocess_bev_image
from yolo import ObstacleDetector
from utils import resolve_image_paths
from visualization import overlay_results_and_display


def classify_lane_type(binary_image, x_center, window_size=5):
    """
    Classifica il tipo di linea misurando la densità lungo la colonna verticale 
    corrispondente al picco dell'istogramma, senza seguire le curve (no sliding window).
    """
    h, w = binary_image.shape
    
    # Seleziona una stretta ROI verticale fissa attorno al picco
    search_start = max(0, x_center - window_size)
    search_end = min(w, x_center + window_size + 1)
    
    roi_colonna = binary_image[:, search_start:search_end]
    
    # Contiamo quante righe intere hanno almeno un pixel bianco nella colonna
    righe_con_bianco = np.sum(np.max(roi_colonna == 255, axis=1))
    
    fill_ratio = righe_con_bianco / h if h > 0 else 0
    
    if fill_ratio < 0.05: 
        return "None"
    elif fill_ratio > 0.75: # Alzato a 65% per evitare che linee tratteggiate siano viste come continue
        return "Solid"
    else:
        return "Dashed"


def find_lanes_and_draw(bev_image, binary_image):
    """
    Calcola l'istogramma sulla metà inferiore dell'immagine binaria, trova i picchi
    e traccia linee dritte verticali nella visuale BEV.
    
    Returns:
        tuple: (output_image, debug_data, lanes_bev_only, lanes_bev_for_warping)
    """
    output_image = bev_image.copy()
    height_img, width = binary_image.shape
    
    # Immagine BEV nera per disegnare solo i pixel bianchi colorati (per visualizzazione)
    lanes_bev_only = np.zeros((height_img, width, 3), dtype=np.uint8)
    
    # Immagine BEV per il warping: contiene linee verticali complete
    lanes_bev_for_warping = np.zeros((height_img, width, 3), dtype=np.uint8)
    
    # DEBUG: Salva la binary_image per ogni frame in output/debug
    script_dir = Path(__file__).parent.absolute()
    debug_dir = script_dir / "output" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"debug_binary_{len(str(binary_image.sum()))}.png"), binary_image)
    
    lower_half = binary_image[height_img//2:, :]
    histogram = np.sum(lower_half, axis=0)
    
    midpoint = int(histogram.shape[0] / 2)
    
    # Margin Cropping: ignora i bordi estremi (1.5% al posto di 10%)
    margin = int(width * 0.015)
    
    left_search_range = histogram[margin:midpoint - margin]
    left_offset = np.argmax(left_search_range) + margin
    left_x_base = left_offset
    
    right_search_range = histogram[midpoint + margin:width - margin]
    right_offset = np.argmax(right_search_range) + midpoint + margin
    right_x_base = right_offset
    
    # Soglia dinamica - minimo valore per rilevare una linea
    min_peak_threshold = max(int((height_img / 1080.0) * 5000), 40 * 255)
    
    lane_found = False
    left_detected = False
    right_detected = False
    left_type = None
    right_type = None
    
    # Verifichiamo la linea sinistra
    left_value = histogram[left_x_base]
    if left_value > min_peak_threshold:
        left_type = classify_lane_type(binary_image, left_x_base)
        if left_type != "None":
            # Disegna solo i pixel trovati per la visuale BEV
            x_start_vis, x_end_vis = max(0, left_x_base-10), min(width, left_x_base+11)
            roi = binary_image[:, x_start_vis:x_end_vis] == 255
            lanes_bev_only[:, x_start_vis:x_end_vis][roi] = (0, 255, 0)
            
            # Linea intera continua per proiettarla in vista frontale
            x_start, x_end = max(0, left_x_base-2), min(width, left_x_base+3)
            lanes_bev_for_warping[:, x_start:x_end] = (0, 255, 0)
            
            cv2.putText(output_image, f"L:{left_type}", (max(0, left_x_base - 60), 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            lane_found = True
            left_detected = True
        
    # Verifichiamo la linea destra
    right_value = histogram[right_x_base]
    if right_value > min_peak_threshold:
        right_type = classify_lane_type(binary_image, right_x_base)
        if right_type != "None":
            # Disegna solo i pixel trovati per la visuale BEV
            x_start_vis, x_end_vis = max(0, right_x_base-10), min(width, right_x_base+11)
            roi = binary_image[:, x_start_vis:x_end_vis] == 255
            lanes_bev_only[:, x_start_vis:x_end_vis][roi] = (0, 255, 0)
            
            # Linea intera continua per proiettarla in vista frontale
            x_start, x_end = max(0, right_x_base-2), min(width, right_x_base+3)
            lanes_bev_for_warping[:, x_start:x_end] = (0, 255, 0)
            
            cv2.putText(output_image, f"R:{right_type}", (min(width - 100, right_x_base + 10), 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            lane_found = True
            right_detected = True
    
    # Sovrapponi i pixel estratti sulla BEV visuale
    green_mask = cv2.inRange(lanes_bev_only, (0, 255, 0), (0, 255, 0))
    output_image[green_mask == 255] = (0, 255, 0)

    # Gestione del fallimento
    if not lane_found:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output_image, "No lanes found", (width//2 - 150, height_img//2), 
                    font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    debug_data = {
        'left_value': left_value,
        'right_value': right_value,
        'left_detected': left_detected,
        'right_detected': right_detected,
        'left_type': left_type,
        'right_type': right_type,
        'lane_found': lane_found,
        'left_x': left_x_base,
        'right_x': right_x_base,
    }
    
    return output_image, debug_data, lanes_bev_only, lanes_bev_for_warping


def main():
    """
    Main pipeline: carica immagini da riga di comando, applica GOLD algorithm, mostra risultati.
    """
    if len(sys.argv) < 2:
        print("Usage: python3 run_gold.py <path_to_images>")
        sys.exit(1)
    
    # Risolve i path incapsulando tutta la logica sporca
    image_paths = resolve_image_paths(sys.argv[1:])
    
    if not image_paths:
        print("Error: No images found at the specified path.")
        sys.exit(1)
    
    print(f"[✓] Found {len(image_paths)} images")
    print("=" * 60)
    
    # Inizializza detector ostacoli
    yolo_detector = ObstacleDetector()
    
    # Omografia (IPM) parameters
    fx, fy = focalLength
    cx, cy = principalPoint
    h = height
    
    # Computa omografia (IPM)
    H, (bev_width, bev_height), _ = compute_homography_matrix(
        fx, fy, cx, cy, h, pitch=pitch, 
        x_min=-3.0, x_max=3.0, z_min=6.0, z_max=25.0,
        bev_width=400, bev_height=800
    )
    
    # Variabile per il controllo della stampa da terminale
    print_terminal = False
    
    # Variabili per il tracking adattivo dello spatial falloff
    tracked_left = None
    tracked_right = None
    tracking_confidence = 0.0
    
    # Variabili per oscuramento adattivo blind
    found_count = 0
    not_found_count = 0
    blind_shade_factor = 0.0
    
    for idx, image_path in enumerate(image_paths, 1):
        if print_terminal:
            print(f"\n[{idx}/{len(image_paths)}] Processing: {Path(image_path).name}")
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue
            
        bev_image = calculate_ipm(original_image, H, output_size=(bev_width, bev_height))
        binary_bev = preprocess_bev_image(
            bev_image,
            tracked_left=tracked_left,
            tracked_right=tracked_right,
            confidence=tracking_confidence,
            blind_shade_factor=blind_shade_factor
        )
        result_bev, debug_data, lanes_bev_only, lanes_bev_for_warping = find_lanes_and_draw(bev_image, binary_bev)
        
        # Logica di aggiornamento Tracker spaziale temporale
        if debug_data['left_detected'] and debug_data['right_detected']:
            tracked_left = debug_data['left_x']
            tracked_right = debug_data['right_x']
            tracking_confidence = min(1.0, tracking_confidence + 0.15) # Aumenta la fiducia gradualmente
        elif tracking_confidence > 0.0:
            tracking_confidence = max(0.0, tracking_confidence - 0.2) # Perde fiducia più in fretta
            if tracking_confidence == 0.0:
                tracked_left = None
                tracked_right = None
                
        # Adaptive Blind Shading Fallback
        if not debug_data['lane_found']:
            not_found_count += 1
            # Scurisce su tutta la BEV gradualmente fino a un limite del 60%
            blind_shade_factor = min(0.6, blind_shade_factor + 0.05) 
        else:
            found_count += 1
            
        # Se superiamo le volte in cui abbiamo fallito, riprendi luce (dimezza)
        if found_count > not_found_count:
            blind_shade_factor /= 2.0
            found_count = 0
            not_found_count = 0
        
        # 4. Visualizzazione
        overlay_results_and_display(
            original_image, result_bev, binary_bev, debug_data,
            lanes_bev_for_warping, H, yolo_detector
        )
        
        # 5. Stampa resoconto opzionale a terminale
        if print_terminal:
            if debug_data['lane_found']:
                print(f"  ✓ Lanes found!")
                if debug_data['left_detected']:
                    print(f"    - Left lane:  {debug_data['left_type']} (val: {debug_data['left_value']:.0f})")
                if debug_data['right_detected']:
                    print(f"    - Right lane: {debug_data['right_type']} (val: {debug_data['right_value']:.0f})")
            else:
                print(f"  ✗ No lanes found")
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            if print_terminal:
                print("\n[*] Exiting...")
            break
    
    cv2.destroyAllWindows()
    if print_terminal:
        print("\n[✓] Processing complete")


if __name__ == "__main__":
    main()
