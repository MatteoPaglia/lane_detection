"""
Lane Detection Assignment - Main Script
PandaSet Dataset Processing
"""

import cv2
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm
from camera_init import camera, K, imageSize, principalPoint, focalLength, height, pitch
from config import DATASET_DIR, IMAGES_DIR


def draw_roi_on_image(image, src_points):
    """
    Disegna la ROI (Region of Interest) sull'immagine originale come un rettangolo verde.
    
    Args:
        image: Immagine originale
        src_points: Array numpy 4x2 dei punti della ROI in coordinate immagine
        
    Returns:
        Immagine con ROI disegnata in verde
    """
    roi_image = image.copy()
    h, w = roi_image.shape[:2]
    
    # Converti i punti della ROI in formato intero per disegnarli
    src_points_int = src_points.astype(int)
    
    # Assicura che i punti siano dentro i limiti dell'immagine
    src_points_int = np.clip(src_points_int, 0, [w-1, h-1])
    
    # Determina lo spessore della linea in base alle dimensioni dell'immagine
    thickness = max(1, int(h / 180))  # Scala il thickness con l'altezza
    circle_radius = max(2, int(h / 90))  # Scala il raggio dei cerchi
    text_scale = max(0.4, h / 360)  # Scala il testo
    
    # Disegna il poligono della ROI in verde
    cv2.polylines(roi_image, [src_points_int], isClosed=True, color=(0, 255, 0), thickness=thickness)
    
    # Disegna i vertici come cerchi verdi
    for point in src_points_int:
        cv2.circle(roi_image, tuple(point), circle_radius, (0, 255, 0), -1)
    
    # Aggiungi etichetta (ROI) in alto a sinistra
    cv2.putText(roi_image, "ROI", (int(w*0.02), int(h*0.15)), cv2.FONT_HERSHEY_SIMPLEX, 
                text_scale, (0, 255, 0), max(1, int(thickness*0.8)))
    
    return roi_image


def project_lanes_to_original(original_image, src_points, left_x_bev, right_x_bev, bev_size):
    """
    Proietta le coordinate delle corsie rilevate dalla BEV all'immagine originale
    e disegna le linee verdi.
    
    Args:
        original_image: Immagine originale
        src_points: I 4 punti della ROI in coordinate immagine
        left_x_bev: Coordinata X della corsia sinistra nella BEV (0-bev_width)
        right_x_bev: Coordinata X della corsia destra nella BEV (0-bev_width)
        bev_size: (bev_width, bev_height) dimensione della BEV
        
    Returns:
        Immagine originale con linee verdi delle corsie
    """
    lanes_image = original_image.copy()
    
    bev_width, bev_height = bev_size
    
    # I punti della ROI sono in ordine circolare: TL, TR, BR, BL
    # Dobbiamo mappare le coordinate BEV indietro all'immagine originale
    # Usiamo interpolazione lineare tra i punti della ROI
    
    # Punto top sinistra (lontano) e bottom sinistra (vicino)
    pt_top_left = src_points[0]    # TL
    pt_top_right = src_points[1]   # TR
    pt_bottom_right = src_points[2] # BR
    pt_bottom_left = src_points[3]  # BL
    
    # Per ogni coordinata X nella BEV, calcola la posizione Y corrispondente sulla strada
    # La BEV ha: colonna 0 = sinistra (x_min), colonna bev_width-1 = destra (x_max)
    
    # Sinistra: interpola tra TL e BL
    t_left = left_x_bev / (bev_width - 1) if bev_width > 1 else 0.5
    left_top_orig = pt_top_left * (1 - t_left) + pt_bottom_left * t_left
    left_bottom_orig = pt_top_left * (1 - t_left) + pt_bottom_left * t_left
    
    # Destra: interpola tra TR e BR
    t_right = right_x_bev / (bev_width - 1) if bev_width > 1 else 0.5
    right_top_orig = pt_top_right * (1 - t_right) + pt_bottom_right * t_right
    right_bottom_orig = pt_top_right * (1 - t_right) + pt_bottom_right * t_right
    
    # In realtà, le linee nella BEV sono verticali, quindi mappiamo diversamente
    # Una linea verticale nella BEV a x=left_x_bev corrisponde a una linea nell'immagine originale
    # che passa per i due punti interpolati tra i lati sinistro/destro della ROI
    
    # Interpolazione lineare semplice:
    # ratio_x = posizione relativa nella BEV (0 = sinistra, 1 = destra)
    ratio_left = left_x_bev / max(bev_width - 1, 1)
    ratio_right = right_x_bev / max(bev_width - 1, 1)
    
    # Punto sulla linea top della ROI
    left_point_top = pt_top_left + (pt_top_right - pt_top_left) * ratio_left
    right_point_top = pt_top_left + (pt_top_right - pt_top_left) * ratio_right
    
    # Punto sulla linea bottom della ROI
    left_point_bottom = pt_bottom_left + (pt_bottom_right - pt_bottom_left) * ratio_left
    right_point_bottom = pt_bottom_left + (pt_bottom_right - pt_bottom_left) * ratio_right
    
    # Disegna linee verdi per le corsie rilevate
    cv2.line(lanes_image, 
             tuple(left_point_top.astype(int)), 
             tuple(left_point_bottom.astype(int)), 
             (0, 255, 0), 3, cv2.LINE_AA)
    
    cv2.line(lanes_image,
             tuple(right_point_top.astype(int)),
             tuple(right_point_bottom.astype(int)),
             (0, 255, 0), 3, cv2.LINE_AA)
    
    return lanes_image


def check_dataset_exists() -> bool:
    """
    Check if dataset images already exist in multiple possible locations.
    
    Returns:
        bool: True if dataset exists and has images, False otherwise
    """
    # Check format 1: PandaSet archive structure (dataset/archive/008/camera/front_camera/)
    archive_dir = DATASET_DIR / "archive"
    if archive_dir.exists():
        # Search for image files in the archive structure, skipping __MACOSX
        image_files = list(archive_dir.glob("[!_]*/camera/front_camera/*.jpg")) + \
                      list(archive_dir.glob("[!_]*/camera/front_camera/*.png"))
        if len(image_files) > 0:
            print(f"[✓] Found dataset in {archive_dir}")
            return True
    
    # Check format 2: Direct images in dataset/images/
    if IMAGES_DIR.exists():
        image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
        if len(image_files) > 0:
            return True
    
    # Check format 3: Images in dataset/images/front_camera/
    front_camera_dir = IMAGES_DIR / "front_camera"
    if front_camera_dir.exists():
        image_files = list(front_camera_dir.glob("*.jpg")) + list(front_camera_dir.glob("*.png"))
        if len(image_files) > 0:
            return True
    
    return False


def get_image_files() -> list:
    """
    Get all image files from the dataset, checking multiple possible locations.
    
    Returns:
        list: List of image file paths
    """
    images = []
    
    # Check format 1: Direct images in dataset/images/
    if IMAGES_DIR.exists():
        images.extend(list(IMAGES_DIR.glob("*.jpg")))
        images.extend(list(IMAGES_DIR.glob("*.png")))
    
    # Check format 2: Images in dataset/images/front_camera/
    front_camera_dir = IMAGES_DIR / "front_camera"
    if front_camera_dir.exists():
        images.extend(list(front_camera_dir.glob("*.jpg")))
        images.extend(list(front_camera_dir.glob("*.png")))
    
    # Check format 3: PandaSet archive structure
    archive_dir = DATASET_DIR / "archive"
    if archive_dir.exists():
        images.extend(list(archive_dir.glob("*/Camera/front_camera/*.jpg")))
        images.extend(list(archive_dir.glob("*/Camera/front_camera/*.png")))
    
    return sorted(images)


def get_all_folder_images(max_folders=None) -> list:
    """
    Get all image files from all scene folders in dataset/archive/ in order.
    Processes folders like 008, 016, 020, 021, 024, 032, 033, 039, 040, 043, 046
    Each folder contains: camera/front_camera/*.jpg
    Skips __MACOSX folder
    
    Args:
        max_folders: Maximum number of folders to process. None = process all folders
    
    Returns:
        list: List of (image_path, folder_name) tuples sorted by folder name numerically
    """
    images = []
    
    # Primary source: dataset/archive/
    archive_dir = DATASET_DIR / "archive"
    
    if not archive_dir.exists():
        print(f"[ERROR] Dataset folder not found: {archive_dir}")
        return images
    
    # Get all folders that look like scene IDs (numeric), skip __MACOSX
    scene_folders = []
    for item in archive_dir.iterdir():
        if item.is_dir() and item.name != '__MACOSX' and item.name != '__pycache__':
            try:
                # Try to convert folder name to int for sorting
                int(item.name)
                scene_folders.append(item)
            except ValueError:
                continue
    
    # Sort numerically by folder name
    scene_folders.sort(key=lambda x: int(x.name))
    
    # Apply limit if specified
    if max_folders is not None and max_folders > 0:
        scene_folders = scene_folders[:max_folders]
    
    print(f"\n[INFO] Processing {len(scene_folders)} scene folders from archive...")
    
    # Collect images from each scene folder
    for scene_folder in scene_folders:
        front_camera_dir = scene_folder / "camera" / "front_camera"
        if front_camera_dir.exists():
            jpg_files = sorted(front_camera_dir.glob("*.jpg"))
            png_files = sorted(front_camera_dir.glob("*.png"))
            scene_images = jpg_files + png_files
            
            for img_path in scene_images:
                images.append((img_path, scene_folder.name))
            
            print(f"  [✓] Scene {scene_folder.name}: {len(scene_images)} images")
        else:
            print(f"  [!] Scene {scene_folder.name}: camera/front_camera/ not found")
    
    print(f"[INFO] Total images loaded: {len(images)}\n")
    return images


def download_dataset_from_kaggle() -> bool:
    """
    Download PandaSet dataset directly from Kaggle using kagglehub.
    
    Returns:
        bool: True if download successful, False otherwise
    """
    print("\n" + "=" * 70)
    print("Downloading PandaSet Dataset from Kaggle...")
    print("=" * 70)
    print("Note: You need Kaggle API credentials configured")
    print("Setup: https://www.kaggle.com/settings/account")
    print("=" * 70)
    
    try:
        import kagglehub
        
        print("\n[INFO] Downloading from Kaggle (usharengaraju/pandaset-dataset)...")
        print("This may take a while (dataset is large)...\n")
        
        # Download latest version
        path = kagglehub.dataset_download("usharengaraju/pandaset-dataset")
        
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"Path: {path}")
        
        # Move files to our dataset directory
        from shutil import copytree, ignore_patterns
        
        # Check if the downloaded path contains 'archive' or 'data'
        download_path = Path(path)
        
        # Copy contents to our dataset directory
        if (download_path / "archive").exists():
            # Already has the correct structure
            archive_src = download_path / "archive"
            archive_dst = DATASET_DIR / "archive"
            
            if not archive_dst.exists():
                print(f"\n[INFO] Copying dataset to: {archive_dst}")
                copytree(str(archive_src), str(archive_dst))
                print("✓ Dataset copied successfully")
            
            return True
        else:
            # Dataset structure might be different, search for images
            print(f"\n[INFO] Organizing dataset files...")
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            
            # Copy all jpg/png files
            for img_file in download_path.glob("**/*.jpg"):
                IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(img_file, IMAGES_DIR / img_file.name)
            
            for img_file in download_path.glob("**/*.png"):
                IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(img_file, IMAGES_DIR / img_file.name)
            
            return True
        
    except ImportError:
        print("✗ kagglehub not installed. Installing...")
        os.system("pip install kagglehub")
        return download_dataset_from_kaggle()  # Retry
    
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual setup:")
        print("1. Go to: https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset")
        print("2. Download the dataset")
        print("3. Extract to: dataset/archive/")
        return False


def download_dataset(url: str = None, output_path: Path = None) -> bool:
    """
    Download PandaSet dataset from Dropbox (fallback).
    
    Args:
        url: Dropbox URL for the dataset (17.26 GB)
        output_path: Path to save the zip file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if url is None:
        url = "https://www.dropbox.com/scl/fi/sch1t7ns9vfpa22setcfw/archive.zip?rlkey=0sglcvm9l9xbzb81zoerkhx0b&dl=1"
    
    if output_path is None:
        output_path = DATASET_DIR / "archive.zip"
    
    # Create dataset directory if it doesn't exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Downloading PandaSet Dataset from Dropbox...")
    print("=" * 70)
    print(f"This may take a while (file size: 17.26 GB)")
    print(f"Destination: {output_path}")
    print("=" * 70)
    
    try:
        # Start download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc="Downloading", ncols=80) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"\n✓ Download completed successfully!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Download failed: {e}")
        return False


def extract_dataset(zip_path: Path = None, extract_to: Path = None) -> bool:
    """
    Extract the downloaded dataset archive.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    if zip_path is None:
        zip_path = DATASET_DIR / "archive.zip"
    
    if extract_to is None:
        extract_to = DATASET_DIR
    
    if not zip_path.exists():
        print(f"✗ Archive not found: {zip_path}")
        return False
    
    print("\n" + "=" * 70)
    print("Extracting Dataset...")
    print("=" * 70)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total files for progress
            file_list = zip_ref.namelist()
            
            with tqdm(total=len(file_list), desc="Extracting", ncols=80) as pbar:
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info, extract_to)
                    pbar.update(1)
        
        print(f"\n✓ Extraction completed successfully!")
        
        # Clean up zip file
        zip_path.unlink()
        print(f"✓ Cleaned up: {zip_path}")
        
        return True
        
    except zipfile.BadZipFile:
        print(f"✗ Invalid or corrupted zip file: {zip_path}")
        return False
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def ensure_dataset_available() -> bool:
    """
    Ensure dataset is available. If not, automatically download from Kaggle.
    
    Returns:
        bool: True if dataset is available, False otherwise
    """
    # Check if dataset already exists
    if check_dataset_exists():
        image_count = len(get_image_files())
        print(f"✓ Dataset found locally! ({image_count} images)")
        return True
    
    print("✗ Dataset not found locally")
    # print("\n[INFO] Attempting automatic download from Kaggle...")
    # 
    # # Try to download from Kaggle
    # if download_dataset_from_kaggle():
    #     # Verify download was successful
    #     if check_dataset_exists():
    #         image_count = len(get_image_files())
    #         print(f"✓ Dataset ready! ({image_count} images)")
    #         return True
    
    # Fallback: inform user
    print("\n[WARNING] Could not download dataset automatically")
    print("\nManual setup options:")
    print("1. Kaggle: https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset")
    print("   - Download and extract to: dataset/archive/")
    print("2. Dropbox: https://www.dropbox.com/scl/fi/sch1t7ns9vfpa22setcfw/archive.zip")
    print("   - Download and extract to: dataset/archive/")
    
    return False


class LaneDetectionPipeline:
    """
    Lane detection pipeline for PandaSet dataset.
    """
    
    def __init__(self, camera_obj=None):
        """
        Initialize the lane detection pipeline.
        
        Args:
            camera_obj: Camera object with calibration parameters
        """
        self.camera = camera_obj if camera_obj else camera
        self.K = self.camera.intrinsic_matrix
        
        print("[INFO] Lane Detection Pipeline initialized")
        print(f"[INFO] Image size: {self.camera.image_size}")
        print(f"[INFO] Camera height: {self.camera.position[2]} m")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Image in BGR format
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return img
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for lane detection.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_edges(self, image: np.ndarray, low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detection.
        
        Args:
            image: Input image
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny
            
        Returns:
            np.ndarray: Edge detection result
        """
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges
    
    def process_image(self, image_path: str) -> dict:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Dictionary containing processed results
        """
        # Load image
        image = self.load_image(image_path)
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(preprocessed)
        
        results = {
            'original': image,
            'preprocessed': preprocessed,
            'edges': edges
        }
        
        return results


def main():
    """
    Main function to demonstrate lane detection pipeline with IPM.
    Implements GOLD algorithm as per assignment requirements.
    """
    print("=" * 60)
    print("Lane Detection Assignment - PandaSet Dataset")
    print("GOLD Algorithm Implementation")
    print("=" * 60)
    
    # Setup Kaggle credentials if kaggle.json exists locally
    project_kaggle_json = Path(__file__).parent / "kaggle.json"
    if project_kaggle_json.exists():
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_dest = kaggle_dir / "kaggle.json"
        
        import shutil
        shutil.copy2(str(project_kaggle_json), str(kaggle_dest))
        
        # Set proper permissions (chmod 600 on Unix-like systems)
        import stat
        kaggle_dest.chmod(stat.S_IRUSR | stat.S_IWUSR)
        
        print("[INFO] Kaggle credentials configured from kaggle.json")
    else:
        print("[WARNING] kaggle.json not found in project directory")
    
    # Ensure dataset is available (download from Kaggle if necessary)
    print("\n[INFO] Checking for dataset...")
    if not ensure_dataset_available():
        print("\n[ERROR] Could not retrieve dataset")
        return
    
    # Initialize camera parameters
    print("\n[INFO] Camera Parameters:")
    print(f"  - Focal Length: {focalLength}")
    print(f"  - Principal Point: {principalPoint}")
    print(f"  - Image Size: {imageSize}")
    print(f"  - Camera Height: {height} m")
    print(f"  - Camera Pitch: {pitch}°")
    
    # Initialize pipeline
    pipeline = LaneDetectionPipeline(camera)
    
    # Import external modules
    from geometry import compute_homography_matrix, calculate_ipm
    from preprocessing import preprocess_bev_image
    from config import OUTPUT_DIR
    from visualizzation import close_windows
    
    # Compute homography matrix for IPM
    print("\n[INFO] Computing IPM (Inverse Perspective Mapping) matrix...")
    fx, fy = focalLength
    cx, cy = principalPoint
    
    # compute_homography_matrix now returns the matrix, BEV size, and ROI src_points
    homography_matrix, (bev_width, bev_height), roi_src_points = compute_homography_matrix(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        h=height,
        pitch=pitch
    )
    
    print(f"[INFO] IPM matrix computed successfully - BEV size: {bev_width}x{bev_height}")
    
    # Create output directories
    output_lanes_dir = OUTPUT_DIR / "lanes_detected"
    output_bev_dir = OUTPUT_DIR / "bev_debug"
    
    output_lanes_dir.mkdir(parents=True, exist_ok=True)
    output_bev_dir.mkdir(parents=True, exist_ok=True)
    
    # Ask user how many folders to analyze
    print("\n" + "=" * 60)
    print("FOLDER SELECTION")
    print("=" * 60)
    try:
        max_folders_input = input("Quante cartelle vuoi analizzare? (premi INVIO per tutte): ").strip()
        if max_folders_input == "":
            max_folders = None
            print("[INFO] Analizzerai TUTTE le cartelle disponibili")
        else:
            max_folders = int(max_folders_input)
            if max_folders <= 0:
                print("[WARNING] Numero non valido, analizzerai tutte le cartelle")
                max_folders = None
            else:
                print(f"[INFO] Analizzerai le prime {max_folders} cartelle")
    except ValueError:
        print("[WARNING] Input non valido, analizzerai tutte le cartelle")
        max_folders = None
    
    # Get image files from specified number of scene folders
    print("\n[INFO] Scanning scene folders for images...")
    image_files = get_all_folder_images(max_folders=max_folders)
    
    total_images = len(image_files)
    print(f"\n[INFO] Found {total_images} images to process across all scenes")
    
    if total_images == 0:
        print("[ERROR] No images found!")
        return
    
    print("=" * 60 + "\n")
    
    # Store results for video playback
    result_images = {}
    result_binary_images = {}  # Store binary images separately
    result_secondary_images = {}  # Store analysis window (histogram only)
    result_binary_display_images = {}  # Store binary map for display
    
    # Track statistics by scene
    stats_by_scene = {}
    
    # Process each image
    processed_count = 0
    detected_count = 0
    current_scene = None
    
    for idx, (image_path, scene_folder) in enumerate(image_files):  # Process ALL images from ALL scenes
        # Track scene changes for better progress display
        if current_scene != scene_folder:
            current_scene = scene_folder
            print(f"\n[SCENE {scene_folder}]")
            stats_by_scene[scene_folder] = {'processed': 0, 'detected': 0}
        
        print(f"  [{idx+1}/{total_images}] {image_path.name}", end=" ... ")
        
        try:
            # 1. Load original image
            original_image = pipeline.load_image(str(image_path))
            original_h, original_w = original_image.shape[:2]
            
            # 2. Apply IPM - transform to Bird's Eye View
            bev_image = calculate_ipm(original_image, homography_matrix, 
                                      output_size=(bev_width, bev_height))
            
            # 3. Preprocess BEV image - get binary image
            binary_bev = preprocess_bev_image(bev_image)
            
            # Store binary image for separate playback
            frame_key = f"{idx+1:05d}_{scene_folder}_{image_path.stem}"
            result_binary_images[frame_key] = binary_bev
            
            # 4. Find lanes using histogram method
            result_bev, debug_data = find_lanes_and_draw(bev_image, binary_bev)
            
            # Check if lanes were detected
            lanes_detected = debug_data['lane_found']
            
            # 5. Save BEV result (debug)
            output_path_bev = output_bev_dir / f"{scene_folder}_{image_path.stem}_bev.jpg"
            cv2.imwrite(str(output_path_bev), result_bev)
            
            # 6. Create WINDOW 1: Original with lanes (sx) | BEV with lanes (dx)
            h_orig, w_orig = original_image.shape[:2]
            h_bev, w_bev = result_bev.shape[:2]
            
            # Resize all images to a manageable size for display (360px height)
            display_height = 360
            
            # Original image
            scale_orig = display_height / h_orig
            w_orig_scaled = int(w_orig * scale_orig)
            original_resized = cv2.resize(original_image, (w_orig_scaled, display_height))
            
            # Project lanes to original image (full size first, then scale)
            if debug_data['lane_found']:
                left_x_bev = debug_data['left_x_base']
                right_x_bev = debug_data['right_x_base']
                original_with_lanes = project_lanes_to_original(original_image, roi_src_points, 
                                                                 left_x_bev, right_x_bev, 
                                                                 (bev_width, bev_height))
            else:
                original_with_lanes = original_image.copy()
            
            # Resize to display size
            original_with_lanes_resized = cv2.resize(original_with_lanes, (w_orig_scaled, display_height))
            
            # BEV with lanes
            scale_bev = display_height / h_bev
            w_bev_scaled = int(w_bev * scale_bev)
            result_bev_resized = cv2.resize(result_bev, (w_bev_scaled, display_height))
            
            # Create side-by-side: Original with lanes | BEV with lanes (WINDOW 1)
            side_by_side = np.hstack([original_with_lanes_resized, result_bev_resized])
            
            # Save side-by-side result
            output_path_side = output_lanes_dir / f"{scene_folder}_{image_path.stem}_comparison.jpg"
            cv2.imwrite(str(output_path_side), side_by_side)
            
            # Store for main video playback (WINDOW 1)
            result_images[frame_key] = side_by_side
            
            # 7. Create WINDOW 2: Histogram only
            # Histogram visualization
            histogram_binary_viz = visualize_histogram_and_binary(binary_bev, min_peak_threshold=debug_data['threshold'])
            # Extract left part (histogram) - histogram_viz is 400 wide
            histogram_only = histogram_binary_viz[:, :400]
            # Extract right part (binary map) - binary_viz is 400 wide
            binary_map_only = histogram_binary_viz[:, 400:]
            
            # Resize histogram viz to match display height
            h_hist, w_hist = histogram_only.shape[:2]
            scale_hist = display_height / h_hist
            histogram_resized = cv2.resize(histogram_only, 
                                          (int(w_hist * scale_hist), display_height))
            # Resize binary map to match display height
            binary_map_resized = cv2.resize(binary_map_only, 
                                           (int(w_hist * scale_hist), display_height))
            
            # Store for secondary display (WINDOW 2)
            frame_key_secondary = f"{frame_key}_analysis"
            result_secondary_images[frame_key_secondary] = histogram_resized
            
            # Store for tertiary display (WINDOW 3)
            frame_key_tertiary = f"{frame_key}_binary"
            result_binary_display_images[frame_key_tertiary] = binary_map_resized
            
            print(f"✓ {'LANES FOUND' if lanes_detected else 'no lanes'}")
            
            processed_count += 1
            stats_by_scene[scene_folder]['processed'] += 1
            
            if lanes_detected:
                detected_count += 1
                stats_by_scene[scene_folder]['detected'] += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Close all windows
    close_windows()
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY - ALL SCENES")
    print("=" * 80)
    print(f"Total images processed: {processed_count}/{total_images}")
    print(f"Overall lanes detected: {detected_count}/{processed_count}")
    print(f"Overall success rate: {(detected_count/processed_count*100):.1f}%" if processed_count > 0 else "N/A")
    
    print("\nPer-Scene Statistics:")
    print("-" * 80)
    for scene in sorted(stats_by_scene.keys(), key=lambda x: int(x)):
        stats = stats_by_scene[scene]
        success = (stats['detected'] / stats['processed'] * 100) if stats['processed'] > 0 else 0
        print(f"  Scene {scene}: {stats['processed']} images, {stats['detected']} detected ({success:.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - BEV with lanes: {output_bev_dir}")
    print(f"  - Comparisons: {output_lanes_dir}")
    print("=" * 80)
    
    # Automatic video playback of results
    if result_images:
        print("\n" + "=" * 60)
        print("Riproduzione Video dei Risultati")
        print("=" * 60)
        print("Mostrando: Original vs BEV with Detected Lanes")
        print("")
        print("Premi ESC per interrompere il playback")
        print("=" * 60 + "\n")
        
        # Get sorted list of results (by frame number)
        sorted_results = sorted(result_images.items())
        
        # Playback loop
        playing = True
        frame_count = 0
        for frame_name, frame_image in sorted_results:
            if '_binary' not in frame_name:
                frame_count += 1
        
        while playing:
            for frame_name, frame_image in sorted_results:
                # Skip binary images in main loop
                if '_binary' in frame_name:
                    continue
                    
                # Display with frame counter
                frame_num = frame_name.split('_')[0]
                display_image = frame_image.copy()
                
                # Add frame number overlay
                cv2.putText(display_image, f"Frame: {frame_num}/{frame_count}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show main window (Original vs BEV with lanes)
                cv2.imshow("Lane Detection - Original vs BEV with Detected Lanes", display_image)
                
                # Show secondary analysis window (Histogram)
                frame_key_secondary = f"{frame_name}_analysis"
                if frame_key_secondary in result_secondary_images:
                    secondary_image = result_secondary_images[frame_key_secondary]
                    cv2.imshow("Analysis - Histogram", secondary_image)
                
                # Show tertiary window (Binary Map)
                frame_key_tertiary = f"{frame_name}_binary"
                if frame_key_tertiary in result_binary_display_images:
                    binary_display = result_binary_display_images[frame_key_tertiary]
                    cv2.imshow("Binary Map", binary_display)
                
                # Delay between frames (100ms = 10 fps)
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC to exit
                    playing = False
                    break
            
            if playing:
                # Loop again if not interrupted
                continue
        
        close_windows()
        print("\n✓ Video playback completed")


def visualize_histogram_and_binary(binary_image, min_peak_threshold=7000):
    """
    Crea una visualizzazione con:
    - SINISTRA: Istogramma with detected peaks in green
    - DESTRA: Binary lane map in white on black
    
    Args:
        binary_image: Immagine binaria del rilevamento corsie
        min_peak_threshold: Soglia dinamica per considerare un picco valido
        
    Returns:
        ndarray: Immagine combinata istogramma + binary
    """
    height, width = binary_image.shape[:2]
    
    # Calcola istogramma dalla metà inferiore
    lower_half = binary_image[height//2:, :]
    histogram = np.sum(lower_half, axis=0)
    midpoint = width // 2
    
    # Trova picchi
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    left_value = histogram[left_x_base]
    right_value = histogram[right_x_base]
    threshold = min_peak_threshold  # Usa il parametro passato
    
    # Crea figura per istogramma (sinistra)
    hist_height, hist_width = 300, 400
    histogram_viz = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    
    # Normalizza istogramma per visualizzazione (scala a 0-hist_height)
    hist_normalized = histogram / (np.max(histogram) + 1) * (hist_height - 50)
    
    # Disegna asse
    cv2.line(histogram_viz, (30, hist_height-30), (hist_width-10, hist_height-30), (200, 200, 200), 1)
    cv2.line(histogram_viz, (30, hist_height-30), (30, 10), (200, 200, 200), 1)
    
    # Disegna linea verticale al midpoint  
    mid_x = int(30 + (midpoint / width) * (hist_width - 50))
    cv2.line(histogram_viz, (mid_x, 10), (mid_x, hist_height-30), (100, 100, 100), 1)
    
    # Disegna istogramma come barre
    bar_width = max(1, (hist_width - 50) // width)
    for col in range(0, width, 5):  # Ogni 5 colonne per visibilità
        x = int(30 + (col / width) * (hist_width - 50))
        h = int(hist_normalized[col])
        cv2.line(histogram_viz, (x, hist_height-30), (x, hist_height-30-h), (100, 150, 200), 1)
    
    # Disegna linee verdi per i picchi rilevati
    # Sinistra
    left_x_viz = int(30 + (left_x_base / width) * (hist_width - 50))
    cv2.line(histogram_viz, (left_x_viz, 10), (left_x_viz, hist_height-30), (0, 255, 0), 2)
    
    # Destra
    right_x_viz = int(30 + (right_x_base / width) * (hist_width - 50))
    cv2.line(histogram_viz, (right_x_viz, 10), (right_x_viz, hist_height-30), (0, 255, 0), 2)
    
    # Aggiungi testo con valori
    cv2.putText(histogram_viz, f"L:{left_value} {'OK' if left_value>threshold else 'X'}", 
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if left_value>threshold else (0, 0, 255), 1)
    cv2.putText(histogram_viz, f"R:{right_value} {'OK' if right_value>threshold else 'X'}", 
                (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if right_value>threshold else (0, 0, 255), 1)
    cv2.putText(histogram_viz, f"Threshold: {threshold}", 
                (5, hist_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Prepara binary image per visualizzazione (destra)
    binary_display = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    binary_resized = cv2.resize(binary_display, (400, hist_height))
    cv2.putText(binary_resized, "Binary Lane Map", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Combina: istogramma (sx) | binary (dx)
    combined = np.hstack([histogram_viz, binary_resized])
    
    return combined


def find_lanes_and_draw(bev_image, binary_image):
    """
    Calcola l'istogramma sulla metà inferiore dell'immagine binaria, trova i picchi
    e disegna le linee rilevate. Usa una soglia dinamica basata sull'altezza dell'immagine.
    
    Returns:
        tuple: (output_image, debug_data) - immagine e dati di debug
    """
    # Creiamo una copia dell'immagine BEV a colori per poterci disegnare sopra
    output_image = bev_image.copy()
    height, width = binary_image.shape
    
    # Calcoliamo l'istogramma sulla metà inferiore dell'immagine
    # axis=0 somma i valori lungo le colonne
    lower_half = binary_image[height//2:, :]
    histogram = np.sum(lower_half, axis=0)
    num_rows = lower_half.shape[0]  # Numero di righe nella metà inferiore
    
    # Dividiamo l'istogramma a metà per cercare la corsia di sinistra e di destra
    midpoint = int(histogram.shape[0] / 2)
    
    # Margin Cropping: Ignora i bordi estremi (10% sui margini) 
    # per evitare di rilevare auto parcheggiate e marciapiedi
    margin = int(width * 0.1)  # 10% dei margini
    
    # Per la corsia sinistra: cerca nel range [margin : midpoint - margin]
    left_search_range = histogram[margin:midpoint - margin]
    left_offset = np.argmax(left_search_range) + margin
    left_x_base = left_offset
    
    # Per la corsia destra: cerca nel range [midpoint + margin : width - margin]
    right_search_range = histogram[midpoint + margin:width - margin]
    right_offset = np.argmax(right_search_range) + midpoint + margin
    right_x_base = right_offset
    
    # Soglia minima dinamica basata sull'altezza effettiva dell'immagine
    # Se l'altezza è 400 (anzichè 1080), la soglia è scalata proporzionalmente
    # Baseline: 7000 per altezza 1080, scala lineare per altre altezze
    min_peak_threshold = int((height / 1080.0) * 7000) if height > 0 else 7000
    
    lane_found = False
    left_detected = False
    right_detected = False
    
    # Verifichiamo e disegniamo la linea sinistra
    left_value = histogram[left_x_base]
    if left_value > min_peak_threshold:
        cv2.line(output_image, (left_x_base, 0), (left_x_base, height), (0, 255, 0), 3)
        lane_found = True
        left_detected = True
        
    # Verifichiamo e disegniamo la linea destra
    right_value = histogram[right_x_base]
    if right_value > min_peak_threshold:
        cv2.line(output_image, (right_x_base, 0), (right_x_base, height), (0, 255, 0), 3)
        lane_found = True
        right_detected = True
    
    # Gestione del fallimento
    if not lane_found:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output_image, "No lanes found", (width//2 - 150, height//2), 
                    font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Prepare debug data
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
        'lane_found': lane_found,
        'binary_image_white_pixels': np.sum(binary_image) // 255,
        'binary_image_mean': np.mean(binary_image)
    }
    
    return output_image, debug_data


if __name__ == "__main__":
    main()
