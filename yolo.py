import cv2
import numpy as np
from geometry import estimate_distance
from camera_init import focalLength, principalPoint, height

try:
    from ultralytics import YOLO
except ImportError:
    print("WARNING: ultralytics is not installed. Please run: pip install ultralytics")
    YOLO = None

class ObstacleDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_path (str): Path to the YOLO model file (default: yolov8n.pt)
            conf_threshold (float): Minimum confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        
        if YOLO is not None:
            self.model = YOLO(model_path)
            # COCO classes to detect: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
            self.target_classes = [0, 1, 2, 3, 5, 7]
            self.class_names = {
                0: 'Person', 1: 'Bicycle', 2: 'Car', 
                3: 'Motorcycle', 5: 'Bus', 7: 'Truck'
            }
        else:
            self.model = None

    def detect_and_draw(self, image):
        """
        Run YOLO detection on the image and draw bounding boxes for targets.
        
        Args:
            image (numpy.ndarray): The input image (BGR)
            
        Returns:
            numpy.ndarray: Image with bounding boxes drawn
        """
        if self.model is None:
            return image
            
        # Run inference
        results = self.model(image, verbose=False)[0]
        output_img = image.copy()
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Check if it's a target class and confidence is high enough
            if cls_id in self.target_classes and conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Usa la geometria del Pinhole Camera Model per calcolare la distanza
                # Assumiamo che la base del bounding box (y2) tocchi la superficie stradale
                fy = focalLength[1]
                cy = principalPoint[1]
                v_bottom = y2
                
                distance = estimate_distance(v_bottom, fy, cy, height)
                
                # Formatta il testo della label
                label_name = self.class_names.get(cls_id, 'Object')
                if distance != float('inf') and distance > 0:
                    label = f"{label_name} {distance:.1f}m"
                else:
                    label = f"{label_name}"
                    
                color = (0, 0, 255) # Red for obstacles
                
                # Draw bounding box
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                (w, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_img, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(output_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        return output_img
