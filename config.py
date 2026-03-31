"""
Lane Detection Assignment - Configuration
Project configuration and paths
"""

import os
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Lane detection configuration
LANE_DETECTION_CONFIG = {
    # Edge detection parameters
    'canny_low_threshold': 50,
    'canny_high_threshold': 150,
    
    # Hough line detection parameters
    'hough_rho': 1,
    'hough_theta': 1,
    'hough_threshold': 50,
    'hough_min_line_length': 50,
    'hough_max_line_gap': 10,
    
    # Image preprocessing
    'blur_kernel_size': (5, 5),
    'histogram_equalization': False,
}

# Camera calibration (from camera_init.py)
CAMERA_PARAMS = {
    'ImageSize': (1920, 1080),
    'PrincipalPoint': (970, 483),
    'FocalLength': (1970, 1970),
    'Position': (1.8750, 0, 1.6600),
    'Rotation': (0, 0, 0),
}


def create_project_structure():
    """Create necessary project directories if they don't exist."""
    directories = [
        OUTPUT_DIR,
        IMAGES_DIR,
        ANNOTATIONS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created/verified directory: {directory}")


def print_configuration():
    """Print project configuration."""
    print("\n" + "=" * 60)
    print("Project Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Dir: {DATASET_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print("\nLane Detection Parameters:")
    for key, value in LANE_DETECTION_CONFIG.items():
        print(f"  - {key}: {value}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    create_project_structure()
    print_configuration()
