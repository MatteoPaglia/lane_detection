import cv2
import numpy as np
import sys
import glob
from pathlib import Path
from camera_init import principalPoint, focalLength, height, pitch
from geometry import compute_homography_matrix, calculate_ipm
from preprocessing import preprocess_bev_image
from run_gold import find_lanes_and_draw

def main():
    """
    Strumento di test per isolare e fare il debug ESCLUSIVAMENTE del preprocessing.
    Mostra solo l'immagine BEV e la relativa mappa binaria affiancate.
    """
    # Gestione argomenti da riga di comando
    if len(sys.argv) < 2:
        print("Usage: python3 test_preprocessing.py <path_to_images>")
        print("Example: python3 test_preprocessing.py PandaSetSensorData/archive/001/camera/front_camera/*.jpg")
        sys.exit(1)
    
    # Logica di caricamento immagini
    image_paths = []
    for arg in sys.argv[1:]:
        if '*' in arg or '?' in arg:
            matches = sorted(glob.glob(arg))
            if not matches: matches = sorted(glob.glob(f"dataset/{arg}"))
            if not matches: matches = sorted(glob.glob(arg.replace("archive/", "")))
            if not matches: matches = sorted(glob.glob(f"dataset/{arg.replace('archive/', '')}"))
            image_paths.extend(matches)
        else:
            if Path(arg).exists(): image_paths.append(arg)
            elif Path(f"dataset/{arg}").exists(): image_paths.append(f"dataset/{arg}")
            elif Path(arg.replace("archive/", "")).exists(): image_paths.append(arg.replace("archive/", ""))
    
    if not image_paths and len(sys.argv) > 1:
        arg = sys.argv[1]
        matches = sorted(glob.glob(arg))
        if matches: image_paths = matches
    
    if not image_paths:
        print("Error: No images found at the specified path.")
        sys.exit(1)
    
    print(f"[✓] Found {len(image_paths)} images for preprocessing tests.")
    print("Premi SPACE per andare all'immagine successiva, 'q' per uscire.")
    print("=" * 60)
    
    # Inizializza camera e geometria
    fx, fy = focalLength
    cx, cy = principalPoint
    h = height
    
    # Computa omografia (IPM) - GOLD Paper Style bounds
    H, (bev_width, bev_height), _ = compute_homography_matrix(
        fx, fy, cx, cy, h, pitch=pitch, 
        x_min=-3.0, x_max=3.0,        # 6 metri totali (focus sulle corsie limitrofe)
        z_min=6.0, z_max=25.0,        # Inizia a 6m (dentro lo schermo) fino a 25m (senza interpolazioni distruttive)
        bev_width=400, bev_height=800 # Canvas lungo e stretto che rispecchia la strada e annulla lo stretch orizzontale
    )
    
    for idx, image_path in enumerate(image_paths, 1):
        # Load immagine
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        print(f"[{idx}/{len(image_paths)}] Processing: {Path(image_path).name}")
        
        # 1. Crea la Bird's Eye View (BEV)
        bev_image = calculate_ipm(img, H, (bev_width, bev_height))
        
        # 2. Applica solo il preprocessing per ottenere la Binary Map
        binary_map, white_mask, yellow_mask = preprocess_bev_image(bev_image, debug=True)
        
        # 3. Create Histogram
        lower_half = binary_map[bev_height//2:, :]
        histogram = np.sum(lower_half, axis=0) # Sum of the bottom half
        hist_img = np.zeros((200, bev_width, 3), dtype=np.uint8)
        
        # Disegno l'istogramma con scala ASSOLUTA: 
        # Il valore massimo possibile (se tutta la colonna fosse bianca) è (bev_height / 2) * 255
        max_possible_val = (bev_height / 2) * 255
        
        for x, val in enumerate(histogram):
            # Normalizziamo su un'altezza massima di 200 pixel in modo assoluto
            h = int((val / max_possible_val) * 200)
            if h > 0:
                cv2.line(hist_img, (x, 200), (x, 200 - h), (255, 0, 0), 1)
        
        # 4. Create inverted mask (white where original is NOT lane, black where it IS lane)
        combined_inv = cv2.bitwise_not(binary_map)
        
        # 5. Eseguiamo il tracking della linea sfruttando la logica attuale di 'run_gold.py'
        # Questo sovrapporrà i pixel di corsia "tracciati" in verde brillante sulla BEV
        tracked_bev, debug_data, lanes_bev_only, lanes_bev_for_warping = find_lanes_and_draw(bev_image, binary_map)
        
        # 6. Visualizzazione affiancata per BEV original, Binary map e Tracked Map
        # Convertiamo la binary (1 canale) in 3 canali per poterla affiancare alla BEV a colori
        binary_map_color = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
        
        # Crea una linea separatrice rossa per chiarezza
        separator = np.zeros((bev_height, 10, 3), dtype=np.uint8)
        separator[:] = (0, 0, 255)
        
        # Affianca le immagini: BEV originale | Binary Map | Tracked Lanes
        combined_view = np.hstack((bev_image, separator, binary_map_color, separator, tracked_bev))
        
        # Aggiungi testi descrittivi
        cv2.putText(combined_view, "BEV (Input)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_view, "Binary Map (Output Filter)", (bev_width + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_view, "Lane Tracking (run_gold)", (bev_width * 2 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Preprocessing Debug Sandbox", combined_view)
        cv2.imshow("Histogram", hist_img)
        cv2.imshow("Non-Lane Pixels (Inverted Mask)", combined_inv)
        
        # Controlli utente
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # 27 = ESC
            break

    cv2.destroyAllWindows()
    print("Test terminato.")

if __name__ == "__main__":
    main()
