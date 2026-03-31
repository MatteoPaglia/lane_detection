"""
Lane Detection Assignment - Visualization Module
Display functions for lane detection results
"""

import cv2
import numpy as np


def display_bev_with_lanes(bev_image, image_name="BEV with Lanes Detected", wait_key=0):
    """
    Display Bird's Eye View image in a window.
    
    Args:
        bev_image: The BEV image to display (with lanes drawn)
        image_name: Title of the window
        wait_key: Time to wait in milliseconds (0 = wait for keypress)
    """
    cv2.imshow(image_name, bev_image)
    cv2.waitKey(wait_key)


def display_side_by_side(original_image, bev_image, title="Lane Detection: Original vs BEV"):
    """
    Display original image and BEV side by side.
    
    Args:
        original_image: Original image
        bev_image: Bird's Eye View image
        title: Window title
    """
    # Resize to same height for side-by-side display
    h_orig, w_orig = original_image.shape[:2]
    h_bev, w_bev = bev_image.shape[:2]
    
    # Scale BEV to match original height
    if h_orig != h_bev:
        scale = h_orig / h_bev
        w_bev_scaled = int(w_bev * scale)
        bev_scaled = cv2.resize(bev_image, (w_bev_scaled, h_orig))
    else:
        bev_scaled = bev_image
    
    # Concatenate horizontally
    combined = np.hstack([original_image, bev_scaled])
    
    cv2.imshow(title, combined)
    cv2.waitKey(0)


def display_results(original_image, bev_image, bev_with_lanes, image_name="", display_mode="bev"):
    """
    Display lane detection results in various modes.
    
    Args:
        original_image: Original input image
        bev_image: Bird's Eye View transformed image
        bev_with_lanes: BEV image with lanes drawn
        image_name: Name of the processed image
        display_mode: 'bev' (only BEV), 'side-by-side', or 'full'
    """
    window_title = f"Lane Detection - {image_name}" if image_name else "Lane Detection Results"
    
    if display_mode == "bev":
        # Display only BEV with lanes
        display_bev_with_lanes(bev_with_lanes, window_title)
    
    elif display_mode == "side-by-side":
        # Display original and BEV side by side
        display_side_by_side(original_image, bev_with_lanes, window_title)
    
    elif display_mode == "full":
        # Display both original and BEV separately then combined
        cv2.imshow(f"{window_title} - BEV", bev_with_lanes)
        cv2.imshow(f"{window_title} - Original", original_image)
        cv2.waitKey(0)


def create_lane_summary(bev_image, lane_info=None):
    """
    Create a summary view with information about detected lanes.
    
    Args:
        bev_image: BEV image with lanes drawn
        lane_info: Dictionary with lane information (optional)
        
    Returns:
        np.ndarray: Image with lane information displayed
    """
    result = bev_image.copy()
    h, w = result.shape[:2]
    
    # Add information overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text
    
    # Background rectangle for text
    y_offset = 20
    cv2.rectangle(result, (10, 10), (400, y_offset + 120), (0, 0, 0), -1)
    
    # Display information
    y_pos = y_offset
    cv2.putText(result, "Lane Detection Results", (20, y_pos), font, font_scale, color, thickness)
    
    y_pos += 30
    if lane_info:
        if 'left_lane' in lane_info:
            cv2.putText(result, f"Left Lane: x={lane_info['left_lane']}", (20, y_pos), font, font_scale, color, thickness)
            y_pos += 30
        
        if 'right_lane' in lane_info:
            cv2.putText(result, f"Right Lane: x={lane_info['right_lane']}", (20, y_pos), font, font_scale, color, thickness)
            y_pos += 30
        
        if 'confidence' in lane_info:
            cv2.putText(result, f"Confidence: {lane_info['confidence']}", (20, y_pos), font, font_scale, color, thickness)
    
    return result


def interactive_display(image_dict, max_images=None):
    """
    Display results in interactive mode.
    Allows user to navigate through results with keyboard.
    
    Args:
        image_dict: Dictionary with image_name -> image data
        max_images: Maximum number of images to display
    """
    images_list = list(image_dict.items())
    
    if max_images:
        images_list = images_list[:max_images]
    
    idx = 0
    
    while idx < len(images_list):
        name, img = images_list[idx]
        
        cv2.imshow(f"Lane Detection [{idx+1}/{len(images_list)}]: {name}", img)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC to quit
            break
        elif key == ord('n') or key == 83:  # n or right arrow for next
            idx += 1
        elif key == ord('p') or key == 81:  # p or left arrow for previous
            idx = max(0, idx - 1)
    
    cv2.destroyAllWindows()


def display_stats(image_path, detected_lanes=None):
    """
    Display statistics about lane detection.
    
    Args:
        image_path: Path to the processed image
        detected_lanes: Boolean indicating if lanes were detected
    """
    status = "✓ Lanes Detected" if detected_lanes else "✗ No Lanes Found"
    print(f"{image_path.name}: {status}")


def save_and_display(output_image, output_path, display=True, wait_ms=100):
    """
    Save image to disk and optionally display it.
    
    Args:
        output_image: Image to save and display
        output_path: Path where to save the image
        display: Whether to display the image
        wait_ms: Time to display in milliseconds
    """
    # Save to disk
    cv2.imwrite(str(output_path), output_image)
    
    # Display if requested
    if display:
        cv2.imshow(f"Processing: {output_path.name}", output_image)
        cv2.waitKey(wait_ms)


def close_windows():
    """Close all OpenCV windows."""
    cv2.destroyAllWindows()


def create_comparison_grid(images_dict, grid_size=(2, 2)):
    """
    Create a grid of images for comparison.
    
    Args:
        images_dict: Dictionary of {name: image}
        grid_size: (rows, cols) for grid layout
        
    Returns:
        np.ndarray: Combined grid image
    """
    images = list(images_dict.values())
    n_images = min(len(images), grid_size[0] * grid_size[1])
    
    if n_images == 0:
        return None
    
    # Resize images to same size
    h, w = images[0].shape[:2]
    
    grid = np.zeros((h * grid_size[0], w * grid_size[1], 3), dtype=np.uint8)
    
    for idx, img in enumerate(images[:n_images]):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        # Resize to grid cell size if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        # Convert to BGR if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Place in grid
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    return grid
