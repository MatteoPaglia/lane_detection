"""
Lane Detection Assignment - Process All Datasets Automatically
PandaSet Dataset Processing - Batch Mode (No User Interaction)
University Project - Final Submission
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from camera_init import principalPoint, focalLength, height, pitch
from geometry import compute_homography_matrix, calculate_ipm
from preprocessing import preprocess_bev_image
from yolo import ObstacleDetector
from run_gold import find_lanes_and_draw


def process_single_image(original_image, yolo_detector, H, bev_width, bev_height, display_time_ms=100):
    """
    Processa una singola immagine e ritorna il risultato.
    
    Args:
        original_image: Immagine originale caricata
        yolo_detector: Detector YOLO per ostacoli
        H: Matrice di omografia (IPM)
        bev_width, bev_height: Dimensioni della BEV
        display_time_ms: Tempo di visualizzazione in millisecondi
        
    Returns:
        tuple: (original_with_lanes, bev_result, debug_data)
    """
    if original_image is None:
        return None, None, None
    
    h_orig, w_orig = original_image.shape[:2]
    
    # Applica IPM per ottenere BEV
    bev_image = calculate_ipm(original_image, H, output_size=(bev_width, bev_height))
    
    # Preprocessing binario
    binary_bev = preprocess_bev_image(bev_image)
    
    # Trova e disegna corsie
    result_bev, debug_data, lanes_bev_only, lanes_bev_for_warping = find_lanes_and_draw(bev_image, binary_bev)
    
    # Proietta linee curve sull'immagine originale
    if debug_data['lane_found']:
        # Omografia Inversa
        inv_H = np.linalg.inv(H)
        
        # Warp Perspective: mappa le linee verticali complete all'immagine originale
        warped_lanes = cv2.warpPerspective(lanes_bev_for_warping, inv_H, (w_orig, h_orig))
        
        # Estrai i pixel verdi dal warping
        mask_warped_green = cv2.inRange(warped_lanes, (0, 255, 0), (0, 255, 0))
        
        # Trova i contorni delle corsie
        contours, _ = cv2.findContours(mask_warped_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        original_with_lanes = original_image.copy()
        
        # Disegna i contorni
        if contours:
            cv2.drawContours(original_with_lanes, contours, -1, (0, 255, 0), -1)
        else:
            original_with_lanes[mask_warped_green == 255] = (0, 255, 0)
    else:
        original_with_lanes = original_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_with_lanes, "No lanes found", 
                    (w_orig//2 - 250, h_orig//2 - 150), 
                    font, 2.5, (0, 0, 255), 4, cv2.LINE_AA)
    
    # Aggiungi tipo di linea
    if debug_data['lane_found']:
        if debug_data['left_type'] is not None:
            cv2.putText(original_with_lanes, f"L:{debug_data['left_type']}", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if debug_data['right_type'] is not None:
            cv2.putText(original_with_lanes, f"R:{debug_data['right_type']}", 
                        (w_orig - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Applica YOLO
    original_with_lanes = yolo_detector.detect_and_draw(original_with_lanes)
    
    return original_with_lanes, result_bev, debug_data


def main():
    """
    Main pipeline: processa TUTTI i dataset in dataset/ automaticamente.
    Frame by frame, senza interazione utente.
    """
    dataset_dir = Path("dataset")
    
    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Trova tutti i folder numerici (scene folders)
    scene_folders = []
    for item in sorted(dataset_dir.iterdir()):
        if item.is_dir() and item.name != '__MACOSX' and item.name != '__pycache__':
            try:
                int(item.name)  # Verifica che sia un numero
                scene_folders.append(item)
            except ValueError:
                continue
    
    scene_folders.sort(key=lambda x: int(x.name))
    
    if not scene_folders:
        print(f"✗ No scene folders found in {dataset_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print(f"[✓] Found {len(scene_folders)} scene folders")
    print("=" * 70)
    
    # Inizializza detector e geometria
    print("\n[*] Initializing YOLO Object Detector...")
    yolo_detector = ObstacleDetector()
    
    fx, fy = focalLength
    cx, cy = principalPoint
    h = height
    
    print("[*] Computing Homography Matrix (IPM)...")
    H, (bev_width, bev_height), roi_src_points = compute_homography_matrix(
        fx, fy, cx, cy, h, pitch=pitch, 
        x_min=-3.0, x_max=3.0,
        z_min=6.0, z_max=25.0,
        bev_width=400, bev_height=800
    )
    
    # Crea cartelle di output
    output_dir = Path("output/results")
    debug_dir = Path("output/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    total_lanes_found = 0
    
    # Processa ogni scene folder
    for scene_idx, scene_folder in enumerate(scene_folders, 1):
        scene_name = scene_folder.name
        front_camera_dir = scene_folder / "camera" / "front_camera"
        
        if not front_camera_dir.exists():
            print(f"\n[{scene_idx}/{len(scene_folders)}] Scene {scene_name}: ✗ camera/front_camera not found")
            continue
        
        # Raccogli tutte le immagini
        image_files = sorted(front_camera_dir.glob("*.jpg")) + sorted(front_camera_dir.glob("*.png"))
        
        if not image_files:
            print(f"\n[{scene_idx}/{len(scene_folders)}] Scene {scene_name}: ✗ No images found")
            continue
        
        print(f"\n[{scene_idx}/{len(scene_folders)}] Scene {scene_name}: Processing {len(image_files)} images...")
        
        # Crea cartella output per questa scene
        scene_output_dir = output_dir / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processa ogni immagine
        scene_lanes_found = 0
        for img_idx, image_path in enumerate(tqdm(image_files, desc=f"  Scene {scene_name}"), 1):
            total_images += 1
            
            # Load immagine
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                continue
            
            # Processa
            original_with_lanes, result_bev, debug_data = process_single_image(
                original_image, yolo_detector, H, bev_width, bev_height, display_time_ms=50
            )
            
            if original_with_lanes is None:
                continue
            
            # Tracking statistiche
            if debug_data['lane_found']:
                total_lanes_found += 1
                scene_lanes_found += 1
            
            # Salva risultati
            output_filename = image_path.stem + "_result.jpg"
            cv2.imwrite(str(scene_output_dir / output_filename), original_with_lanes)
            
            # Visualizza brevemente
            h_img, w_img = original_with_lanes.shape[:2]
            display_height = 480
            scale = display_height / h_img
            w_scaled = int(w_img * scale)
            display_img = cv2.resize(original_with_lanes, (w_scaled, display_height))
            
            cv2.imshow(f"[{img_idx}/{len(image_files)}] Scene {scene_name} - Lane Detection", display_img)
            cv2.waitKey(50)  # 50ms di visualizzazione
        
        # Statistiche della scene
        print(f"  ✓ Scene {scene_name}: {scene_lanes_found}/{len(image_files)} images with lanes detected")
    
    # Chiudi finestre
    cv2.destroyAllWindows()
    
    # Stampa statistiche finali
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"[✓] Total images processed: {total_images}")
    print(f"[✓] Total lanes found: {total_lanes_found}")
    if total_images > 0:
        print(f"[✓] Success rate: {100*total_lanes_found/total_images:.1f}%")
    print(f"[✓] Results saved to: {output_dir.absolute()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
