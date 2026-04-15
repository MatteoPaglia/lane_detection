import cv2
import numpy as np
import glob
import sys
import os
from pathlib import Path

# Importiamo la logica esistente
from run_gold import compute_homography_matrix, find_lanes_and_draw
from preprocessing import preprocess_bev_image
from geometry import calculate_ipm
from camera_init import focalLength, principalPoint, height, pitch
from yolo import ObstacleDetector

def main():
    print("=" * 60)
    print("🚗 PANDASET BATCH PROCESSOR (Video Mode) 🚗")
    print("=" * 60)
    
    # Path di default per tutto il dataset PandaSet
    dataset_path = "PandaSetSensorData/archive/*/camera/front_camera/*.jpg"
    
    # Usa il path fornito dall'utente oppure il default per tutto l'archivio
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        
    print(f"[*] Cerco immagini in: {dataset_path}")
    image_paths = sorted(glob.glob(dataset_path))
    
    if not image_paths:
        print("Error: Nessuna immagine trovata! Verifica il percorso.")
        sys.exit(1)
        
    print(f"[✓] Trovate {len(image_paths)} immagini totali.")
    print("[*] Premi 'q' o 'ESC' in qualsiasi momento per interrompere il video.")
    print("=" * 60)
    
    print("\n[*] Initializing YOLO Object Detector...")
    yolo_detector = ObstacleDetector()
    
    # Inizializza camera e geometria come in run_gold
    fx, fy = focalLength
    cx, cy = principalPoint
    h = height
    
    # Omografia GOLD Paper
    H, (bev_width, bev_height), roi_src_points = compute_homography_matrix(
        fx, fy, cx, cy, h, pitch=pitch, 
        x_min=-3.0, x_max=3.0,        
        z_min=6.0, z_max=25.0,        
        bev_width=400, bev_height=800 
    )
    
    # Ciclo continuo su tutte le immagini (stile video)
    for idx, image_path in enumerate(image_paths, 1):
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue
            
        h_orig, w_orig = original_image.shape[:2]
        
        # 1. BEV
        bev_image = calculate_ipm(original_image, H, output_size=(bev_width, bev_height))
        # 2. Binary Map (Preprocessing)
        binary_bev = preprocess_bev_image(bev_image)
        # 3. Lane Tracking
        result_bev, debug_data, lanes_bev_only, lanes_bev_for_warping = find_lanes_and_draw(bev_image, binary_bev)
        
        # 4. Riallineamento sull'immagine Originale
        if debug_data['lane_found']:
            inv_H = np.linalg.inv(H)
            warped_lanes = cv2.warpPerspective(lanes_bev_for_warping, inv_H, (w_orig, h_orig))
            mask_warped_green = cv2.inRange(warped_lanes, (0, 255, 0), (0, 255, 0))
            contours, _ = cv2.findContours(mask_warped_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            original_with_lanes = original_image.copy()
            if contours:
                cv2.drawContours(original_with_lanes, contours, -1, (0, 255, 0), -1)
            else:
                original_with_lanes[mask_warped_green == 255] = (0, 255, 0)
        else:
            original_with_lanes = original_image.copy()
        
        # YOLO Detection Overlay
        original_with_lanes = yolo_detector.detect_and_draw(original_with_lanes)
        
        # --- UI Compositing ---
        display_height = 480
        
        # Original scaling
        scale_orig = display_height / h_orig
        w_orig_scaled = int(w_orig * scale_orig)
        original_resized = cv2.resize(original_with_lanes, (w_orig_scaled, display_height))
        
        # BEV scaling
        scale_bev = display_height / bev_height
        w_bev_scaled = int(bev_width * scale_bev)
        bev_resized = cv2.resize(result_bev, (w_bev_scaled, display_height))
        
        # Binary map scaling
        binary_resized = cv2.resize(binary_bev, (w_bev_scaled, display_height))
        binary_bgr = cv2.cvtColor(binary_resized, cv2.COLOR_GRAY2BGR)
        
        # Concatenate: Original | BEV | Binary Map
        side_by_side = np.hstack([original_resized, bev_resized, binary_bgr])
        
        # Titolo frame (opzionale per capire in che cartella sei)
        folder_name = Path(image_path).parent.parent.parent.name # Estrae '001', '043' ecc..
        file_name = Path(image_path).name
        cv2.putText(side_by_side, f"Seq: {folder_name} | {file_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostra risultato a video
        cv2.imshow("Pandaset Auto-Play (Original | BEV | Binary)", side_by_side)
        
        # --- DIFFERENZA PRINCIPALE: aspetta 1 ms invece di infinito ---
        # In questo modo i frame scorrono come in un video a 1000 FPS (o il massimo permesso dalla CPU)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC o 'q' per chiudere anticipatamente
            print("\n[*] Riproduzione interrotta dall'utente.")
            break
            
        # Stampiamo solo una micro log della cartella per non intasare il terminale
        if idx % 10 == 0:
            print(f"Progresso: {idx}/{len(image_paths)} (Seq {folder_name})", end='\r')

    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("[✓] Processamento Batch completato!")

if __name__ == "__main__":
    main()
