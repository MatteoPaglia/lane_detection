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


def classify_lane_type(binary_image, x_center, window_size=15):
    """
    Classifica il tipo di linea seguendo dinamicamente la curva (Sliding Window)
    e calcolando il fill ratio sui pixel effettivamente percorsi dalla linea.
    Questo previene falsi "Dashed" causati dall'uscita della curva dal box dritto.
    """
    h, w = binary_image.shape
    current_x = x_center
    found_y_indices = []
    
    # Scorri dal basso verso l'alto seguendo la curva
    for y in reversed(range(h)):
        search_start = max(0, current_x - window_size)
        search_end = min(w, current_x + window_size + 1)
        row_search = binary_image[y, search_start:search_end]
        
        white_indices = np.where(row_search == 255)[0]
        if len(white_indices) > 0:
            # Aggiorna il centro seguendo l'andamento della corsia
            current_x = search_start + int(np.mean(white_indices))
            found_y_indices.append(y)
    
    if len(found_y_indices) == 0:
        return "Unknown"
        
    # Trova l'inizio (strada lontana = y minore) e fine (strada vicina = y maggiore)
    top_y = min(found_y_indices)
    bottom_y = max(found_y_indices)
    
    # La lunghezza è coperta dai limiti y dove la linea esiste
    line_length = bottom_y - top_y + 1
    white_pixels = len(found_y_indices)
    
    # Rapporto di riempimento lungo *tutta* e *sola* la curva effettiva
    fill_ratio = white_pixels / line_length if line_length > 0 else 0
    
    if fill_ratio < 0.10: 
        return "None"
    elif fill_ratio > 0.85: # Alzato all'85% per evitare falsi positivi su linee tratteggiate molto dense
        return "Solid"
    else:
        return "Dashed"


def find_lanes_and_draw(bev_image, binary_image):
    """
    Calcola l'istogramma sulla metà inferiore dell'immagine binaria, trova i picchi
    ed estrae i pixel adiacenti ai picchi per tracciare le curve reali.
    
    Returns:
        tuple: (output_image, debug_data, lanes_bev_only, lanes_bev_for_warping)
    """
    output_image = bev_image.copy()
    height, width = binary_image.shape
    
    # Immagine BEV nera per disegnare solo i pixel bianchi colorati (per visualizzazione)
    lanes_bev_only = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Immagine BEV per il warping: contiene linee verticali complete
    lanes_bev_for_warping = np.zeros((height, width, 3), dtype=np.uint8)
    
    # DEBUG: Salva la binary_image per ogni frame in output/debug
    debug_dir = Path("output/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"debug_binary_{len(str(binary_image.sum()))}.png"), binary_image)
    
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
    min_peak_threshold = max(int((height / 1080.0) * 5000), 40 * 255)
    
    lane_found = False
    left_detected = False
    right_detected = False
    left_type = None
    right_type = None
    
    # DEBUG: Analizza output
    white_pixels = np.sum(binary_image == 255)
    print(f"\n🔍 DEBUG - White pixels in binary_image: {white_pixels}")
    print(f"🔍 Histogram max: {np.max(histogram)}, Threshold: {min_peak_threshold}")
    
    window_width = 15  # Finestra di ricerca +/- 15 pixel (ridotta per maggior precisione)
    
    # Verifichiamo la linea sinistra
    left_value = histogram[left_x_base]
    print(f"🔍 LEFT ({left_x_base}): {left_value} (threshold: {min_peak_threshold}) - {'✅ DETECTED' if left_value > min_peak_threshold else '❌ MISSED'}")
    if left_value > min_peak_threshold:
        left_type = classify_lane_type(binary_image, left_x_base)
        if left_type != "None":
            current_x = left_x_base
            lane_pixels = []
            
            for y in reversed(range(height)):
                row = binary_image[y, :]
                search_start = max(0, current_x - window_width)
                search_end = min(width, current_x + window_width)
                row_search = row[search_start:search_end]
                
                white_indices = np.where(row_search == 255)[0]
                if len(white_indices) > 0:
                    center_pos = search_start + int(np.mean(white_indices))
                    current_x = center_pos
                    lane_pixels.append((center_pos, y))
                    
            if lane_pixels:
                xs = np.array([p[0] for p in lane_pixels])
                ys = np.array([p[1] for p in lane_pixels])
                
                if left_type == "Solid" and len(ys) >= 3:
                    # Polyfit lineare o quadratico (assicura una linea continua smussata)
                    poly_coeffs = np.polyfit(ys, xs, 2)
                    poly_fn = np.poly1d(poly_coeffs)
                    
                    min_y = int(np.min(ys))
                    max_y = int(np.max(ys))
                    plot_y = np.arange(min_y, max_y + 1)
                    plot_x = np.clip(poly_fn(plot_y).astype(int), 0, width - 1)
                    
                    # Disegna la linea interpolata continua (3x3 su ogni punto)
                    for x, y in zip(plot_x, plot_y):
                        y_start, y_end = max(0, y-1), min(height, y+2)
                        x_start, x_end = max(0, x-1), min(width, x+2)
                        lanes_bev_only[y_start:y_end, x_start:x_end] = (0, 255, 0)
                        lanes_bev_for_warping[y_start:y_end, x_start:x_end] = (0, 255, 0)
                        
                else:
                    # Dashed: colora solo un quadratino 3x3 per ogni pixel bianco rilevato (che fa un quadrato)
                    for x, y in lane_pixels:
                        y_start, y_end = max(0, y-1), min(height, y+2)
                        x_start, x_end = max(0, x-1), min(width, x+2)
                        lanes_bev_only[y_start:y_end, x_start:x_end] = (0, 255, 0)
                        lanes_bev_for_warping[y_start:y_end, x_start:x_end] = (0, 255, 0)
            
            cv2.putText(output_image, f"L:{left_type}", (max(0, left_x_base - 60), 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            lane_found = True
            left_detected = True
        
    # Verifichiamo la linea destra
    right_value = histogram[right_x_base]
    print(f"🔍 RIGHT ({right_x_base}): {right_value} (threshold: {min_peak_threshold}) - {'✅ DETECTED' if right_value > min_peak_threshold else '❌ MISSED'}")
    if right_value > min_peak_threshold:
        right_type = classify_lane_type(binary_image, right_x_base)
        if right_type != "None":
            current_x = right_x_base
            lane_pixels = []
            
            for y in reversed(range(height)):
                row = binary_image[y, :]
                search_start = max(0, current_x - window_width)
                search_end = min(width, current_x + window_width)
                row_search = row[search_start:search_end]
                
                white_indices = np.where(row_search == 255)[0]
                if len(white_indices) > 0:
                    center_pos = search_start + int(np.mean(white_indices))
                    current_x = center_pos
                    lane_pixels.append((center_pos, y))
                    
            if lane_pixels:
                xs = np.array([p[0] for p in lane_pixels])
                ys = np.array([p[1] for p in lane_pixels])
                
                if right_type == "Solid" and len(ys) >= 3:
                    # Polyfit lineare/quadratico (assicura una linea continua smussata)
                    poly_coeffs = np.polyfit(ys, xs, 2)
                    poly_fn = np.poly1d(poly_coeffs)
                    
                    min_y = int(np.min(ys))
                    max_y = int(np.max(ys))
                    plot_y = np.arange(min_y, max_y + 1)
                    plot_x = np.clip(poly_fn(plot_y).astype(int), 0, width - 1)
                    
                    # Disegna la linea interpolata continua (3x3 su ogni intero dal min al max)
                    for x, y in zip(plot_x, plot_y):
                        y_start, y_end = max(0, y-1), min(height, y+2)
                        x_start, x_end = max(0, x-1), min(width, x+2)
                        lanes_bev_only[y_start:y_end, x_start:x_end] = (0, 255, 0)
                        lanes_bev_for_warping[y_start:y_end, x_start:x_end] = (0, 255, 0)
                        
                else:
                    # Dashed: colora solo un quadratino 3x3 per ogni pixel bianco rilevato (che fa un quadrato)
                    for x, y in lane_pixels:
                        y_start, y_end = max(0, y-1), min(height, y+2)
                        x_start, x_end = max(0, x-1), min(width, x+2)
                        lanes_bev_only[y_start:y_end, x_start:x_end] = (0, 255, 0)
                        lanes_bev_for_warping[y_start:y_end, x_start:x_end] = (0, 255, 0)
            
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
    
    return output_image, debug_data, lanes_bev_only, lanes_bev_for_warping


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
        # Prova prima il path come fornito dall'utente
        if '*' in arg or '?' in arg:
            # Se contiene wildcard, usa glob
            matches = sorted(glob.glob(arg))
            image_paths.extend(matches)
            
            # Se non trova niente, prova con "dataset/" come prefix
            if not matches:
                alt_arg = f"dataset/{arg}"
                matches = sorted(glob.glob(alt_arg))
                image_paths.extend(matches)
            
            # Se ancora non trova, prova rimuovendo "archive/" dal path
            if not matches:
                alt_arg2 = arg.replace("archive/", "")
                matches = sorted(glob.glob(alt_arg2))
                image_paths.extend(matches)
            
            # ultima opzione: prova con "dataset/" + path senza "archive/"
            if not matches:
                alt_arg3 = f"dataset/{arg.replace('archive/', '')}"
                matches = sorted(glob.glob(alt_arg3))
                image_paths.extend(matches)
        else:
            # Altrimenti, trattalo come percorso singolo
            if Path(arg).exists():
                image_paths.append(arg)
            else:
                # Prova con "dataset/" come prefix
                alt_path = Path(f"dataset/{arg}")
                if alt_path.exists():
                    image_paths.append(str(alt_path))
                else:
                    # Prova rimuovendo "archive/"
                    alt_path2 = Path(arg.replace("archive/", ""))
                    if alt_path2.exists():
                        image_paths.append(str(alt_path2))
                    else:
                        # ultima opzione: dataset/ + path senza archive/
                        alt_path3 = Path(f"dataset/{arg.replace('archive/', '')}")
                        if alt_path3.exists():
                            image_paths.append(str(alt_path3))
    
    # Se ancora niente, prova a globbare il primo argomento
    if not image_paths and len(sys.argv) > 1:
        arg = sys.argv[1]
        matches = sorted(glob.glob(arg))
        if matches:
            image_paths = matches
        else:
            # Prova con "dataset/" prefix
            alt_arg = f"dataset/{arg}"
            matches = sorted(glob.glob(alt_arg))
            if matches:
                image_paths = matches
            else:
                # Prova rimuovendo "archive/"
                alt_arg2 = arg.replace("archive/", "")
                matches = sorted(glob.glob(alt_arg2))
                if matches:
                    image_paths = matches
                else:
                    # ultima opzione: dataset/ + path senza archive/
                    alt_arg3 = f"dataset/{arg.replace('archive/', '')}"
                    matches = sorted(glob.glob(alt_arg3))
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
    
    # Computa omografia (IPM) - GOLD Paper Style bounds
    H, (bev_width, bev_height), roi_src_points = compute_homography_matrix(
        fx, fy, cx, cy, h, pitch=pitch, 
        x_min=-3.0, x_max=3.0,        # 6 metri totali di larghezza (previene deformazioni esterne estreme)
        z_min=6.0, z_max=25.0,        # Distanza ad alta intensità di pixel (da subito danti al cofano a 25m)
        bev_width=400, bev_height=800 # Canvas stretto e lungo anti-blur
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
        
        # OBIETTIVO 2: Trova e disegna corsie estraendo le curve su 'lanes_bev_only'
        result_bev, debug_data, lanes_bev_only, lanes_bev_for_warping = find_lanes_and_draw(bev_image, binary_bev)
        
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
                
                # Opzionale: aggiungiamo un filo di trasparenza per far sembrare le corsie un evidenziatore (neon)
                # cv2.addWeighted(original_image, 0.6, original_with_lanes, 0.4, 0, original_with_lanes)
            else:
                # Fallback: se non ci sono contorni, colora i pixel in modo pieno
                original_with_lanes[mask_warped_green == 255] = (0, 255, 0)
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
        
        # Ridimensiona per display (480 per finestra compatta)
        display_height = 480
        
        # Original con lanes
        scale_orig = display_height / h_orig
        w_orig_scaled = int(w_orig * scale_orig)
        original_resized = cv2.resize(original_with_lanes, (w_orig_scaled, display_height))
        
        # BEV con lanes (usa lanes_bev_only per visualizzazione)
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
