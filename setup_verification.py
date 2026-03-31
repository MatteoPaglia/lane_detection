"""
Lane Detection Assignment - Setup Verification
Test that all dependencies and configuration are working
"""

import sys
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    dependencies = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'requests': 'Requests',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_camera_setup():
    """Check camera initialization."""
    print("\n" + "=" * 60)
    print("Checking Camera Setup")
    print("=" * 60)
    
    try:
        from camera_init import camera, K
        print("✓ Camera module imported successfully")
        print(f"\n  Image Size: {camera.image_size}")
        print(f"  Focal Length: {camera.focal_length}")
        print(f"  Principal Point: {camera.principal_point}")
        print(f"  Camera Height: {camera.position[2]} m")
        print(f"  Camera Pitch: {camera.rotation[1]}°")
        
        print(f"\n  Intrinsic Matrix K shape: {K.shape}")
        print(f"  Matrix K:\n{K}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading camera: {e}")
        return False


def check_config():
    """Check configuration setup."""
    print("\n" + "=" * 60)
    print("Checking Configuration")
    print("=" * 60)
    
    try:
        from config import (
            PROJECT_ROOT, 
            DATASET_DIR, 
            OUTPUT_DIR,
            LANE_DETECTION_CONFIG,
            create_project_structure
        )
        
        print(f"✓ Config module imported successfully")
        print(f"\n  Project Root: {PROJECT_ROOT}")
        print(f"  Dataset Dir: {DATASET_DIR}")
        print(f"  Output Dir: {OUTPUT_DIR}")
        
        # Create project structure
        create_project_structure()
        
        print(f"\n  Lane Detection Config:")
        for key, value in list(LANE_DETECTION_CONFIG.items())[:3]:
            print(f"    - {key}: {value}")
        print(f"    ... ({len(LANE_DETECTION_CONFIG)} total parameters)")
        
        return True
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def check_pipeline():
    """Check lane detection pipeline."""
    print("\n" + "=" * 60)
    print("Checking Lane Detection Pipeline")
    print("=" * 60)
    
    try:
        from lane_detection import LaneDetectionPipeline
        from camera_init import camera
        
        pipeline = LaneDetectionPipeline(camera)
        print("✓ Pipeline initialized successfully")
        print(f"✓ Pipeline is ready to process images")
        
        return True
    except Exception as e:
        print(f"✗ Error initializing pipeline: {e}")
        return False


def main():
    """Run all setup verification checks."""
    print("\n" + "=" * 70)
    print(" " * 15 + "Lane Detection Assignment - Setup Verification")
    print("=" * 70)
    
    results = {
        'Dependencies': check_dependencies(),
        'Camera Setup': check_camera_setup(),
        'Configuration': check_config(),
        'Pipeline': check_pipeline(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_ok = True
    for check_name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check_name:30s}: {'PASS' if status else 'FAIL'}")
        if not status:
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("\n✓ All checks passed! Assignment is ready to proceed.")
        print("\nNext steps:")
        print("  1. Download the PandaSet dataset from Kaggle or Dropbox")
        print("  2. Extract images to: dataset/images/")
        print("  3. Run: python lane_detection.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    
    print()


if __name__ == "__main__":
    main()
