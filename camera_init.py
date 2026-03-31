"""
Lane Detection Assignment - Camera Initialization
Camera parameters for PandaSet dataset
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Camera:
    """
    Camera calibration and configuration parameters
    for the PandaSet lane detection dataset.
    """
    
    # Image dimensions
    image_size: Tuple[int, int] = (1920, 1080)  # [width, height]
    
    # Principal point (optical center) in pixels
    principal_point: Tuple[float, float] = (970, 483)  # [x, y]
    
    # Focal length in pixels
    focal_length: Tuple[float, float] = (1970, 1970)  # [fx, fy]
    
    # Camera position in world coordinates [x, y, z] in meters
    position: Tuple[float, float, float] = (1.8750, 0, 1.6600)
    
    # Camera rotation [roll, pitch, yaw] in degrees
    rotation: Tuple[float, float, float] = (0, 0, 0)
    
    def __post_init__(self):
        """Convert to numpy arrays for easier computation"""
        self._image_size = np.array(self.image_size, dtype=np.int32)
        self._principal_point = np.array(self.principal_point, dtype=np.float32)
        self._focal_length = np.array(self.focal_length, dtype=np.float32)
        self._position = np.array(self.position, dtype=np.float32)
        self._rotation = np.array(self.rotation, dtype=np.float32)
    
    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        Get camera intrinsic matrix K.
        
        K = [fx  0  cx]
            [ 0 fy  cy]
            [ 0  0   1]
        """
        fx, fy = self._focal_length
        cx, cy = self._principal_point
        
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)
        
        return K
    
    def get_params(self) -> dict:
        """Get all camera parameters as a dictionary"""
        return {
            'image_size': self.image_size,
            'principal_point': self.principal_point,
            'focal_length': self.focal_length,
            'position': self.position,
            'rotation': self.rotation,
            'height': self.position[2],  # Camera height above ground
            'pitch': self.rotation[1]     # Camera pitch angle
        }


def initialize_camera() -> Camera:
    """
    Initialize camera with PandaSet dataset parameters.
    
    Returns:
        Camera: Camera object with calibration parameters
    """
    camera = Camera(
        image_size=(1920, 1080),
        principal_point=(970, 483),
        focal_length=(1970, 1970),
        position=(1.8750, 0, 1.6600),
        rotation=(0, 0, 0)
    )
    
    return camera


# Initialize camera on module import
camera = initialize_camera()

# Extract commonly used parameters
focalLength = camera.focal_length
principalPoint = camera.principal_point
imageSize = camera.image_size
height = camera.position[2]
pitch = camera.rotation[1]

# Camera intrinsic matrix
K = camera.intrinsic_matrix


if __name__ == "__main__":
    # Print camera parameters for verification
    print("=" * 60)
    print("Camera Calibration Parameters - PandaSet Dataset")
    print("=" * 60)
    
    params = camera.get_params()
    for key, value in params.items():
        print(f"{key:20s}: {value}")
    
    print("\nCamera Intrinsic Matrix K:")
    print(K)
    print("=" * 60)
