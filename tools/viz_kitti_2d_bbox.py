#!/usr/bin/python3

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import ROOT_PATH

# KITTI object type to color mapping (BGR format for OpenCV)
KITTI_COLORS_BGR = {
    'Car': (255, 0, 0),           # Blue
    'Pedestrian': (0, 0, 255),    # Red
    'Cyclist': (0, 255, 0),       # Green
    'Van': (255, 128, 0),         # Light Blue
    'Truck': (128, 0, 128),       # Purple
    'Person_sitting': (0, 128, 255),  # Orange
    'Tram': (128, 128, 128),      # Gray
    'Misc': (0, 128, 128),        # Olive
    'DontCare': (76, 76, 76),     # Dark Gray
}

# Default color for unknown types
DEFAULT_COLOR_BGR = (0, 255, 255)  # Yellow

# Anomaly detection thresholds
MIN_BBOX_WIDTH = 1
MIN_BBOX_HEIGHT = 1
MIN_BBOX_AREA = 1


def load_kitti_label(label_path: str) -> List[Dict]:
    """Load KITTI label file containing 3D bounding box annotations.

    KITTI label format (per line):
    type truncated occluded alpha bbox_2d(4) dimensions(3) location(3) rotation_y [score]

    Args:
        label_path: Path to label .txt file

    Returns:
        List of dictionaries containing parsed label information
    """
    if not os.path.exists(label_path):
        return []

    labels = []
    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                label = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': [float(x) for x in parts[4:8]],
                    'dimensions': [float(x) for x in parts[8:11]],  # h, w, l
                    'location': [float(x) for x in parts[11:14]],   # x, y, z in camera coords
                    'rotation_y': float(parts[14]),
                    'line_num': line_num
                }
                labels.append(label)
        return labels
    except Exception as e:
        print(f"Warning: Failed to parse label file {label_path}: {e}")
        return []


def validate_bbox(bbox_2d: List[float], image_width: int, image_height: int) -> Dict:
    """Validate 2D bounding box and detect anomalies.

    Args:
        bbox_2d: [x_min, y_min, x_max, y_max]
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Dictionary containing validation results and anomaly flags
    """
    x_min, y_min, x_max, y_max = bbox_2d

    width = x_max - x_min
    height = y_max - y_min
    area = width * height

    anomalies = []

    # Check for invalid dimensions
    if width < MIN_BBOX_WIDTH:
        anomalies.append(f"Width too small: {width:.1f}")
    if height < MIN_BBOX_HEIGHT:
        anomalies.append(f"Height too small: {height:.1f}")
    if area < MIN_BBOX_AREA:
        anomalies.append(f"Area too small: {area:.1f}")

    # Check for negative dimensions
    if width <= 0:
        anomalies.append(f"Negative/zero width: {width:.1f}")
    if height <= 0:
        anomalies.append(f"Negative/zero height: {height:.1f}")

    # Check if bbox is out of image bounds
    if x_min < 0:
        anomalies.append(f"x_min out of bounds: {x_min:.1f}")
    if y_min < 0:
        anomalies.append(f"y_min out of bounds: {y_min:.1f}")
    if x_max > image_width:
        anomalies.append(f"x_max out of bounds: {x_max:.1f} > {image_width}")
    if y_max > image_height:
        anomalies.append(f"y_max out of bounds: {y_max:.1f} > {image_height}")

    # Check for inverted coordinates
    if x_min > x_max:
        anomalies.append(f"Inverted x: {x_min:.1f} > {x_max:.1f}")
    if y_min > y_max:
        anomalies.append(f"Inverted y: {y_min:.1f} > {y_max:.1f}")

    return {
        'width': width,
        'height': height,
        'area': area,
        'is_valid': len(anomalies) == 0,
        'anomalies': anomalies
    }


def draw_bbox_on_image(image: np.ndarray, labels: List[Dict],
                       show_details: bool = True,
                       filter_types: Optional[List[str]] = None,
                       highlight_anomalies: bool = True) -> Tuple[np.ndarray, Dict]:
    """Draw 2D bounding boxes on image.

    Args:
        image: Input image (BGR format)
        labels: List of label dictionaries
        show_details: Whether to show detailed information
        filter_types: List of object types to display (None = all)
        highlight_anomalies: Whether to highlight anomalous bboxes

    Returns:
        Tuple of (annotated image, statistics dictionary)
    """
    annotated = image.copy()
    img_height, img_width = image.shape[:2]

    stats = {
        'total_objects': 0,
        'valid_objects': 0,
        'anomalous_objects': 0,
        'by_type': {}
    }

    for label in labels:
        obj_type = label['type']

        # Apply type filter
        if filter_types and obj_type not in filter_types:
            continue

        stats['total_objects'] += 1

        # Track by type
        if obj_type not in stats['by_type']:
            stats['by_type'][obj_type] = 0
        stats['by_type'][obj_type] += 1

        bbox_2d = label['bbox_2d']
        x_min, y_min, x_max, y_max = [int(x) for x in bbox_2d]

        # Validate bbox
        validation = validate_bbox(bbox_2d, img_width, img_height)

        if validation['is_valid']:
            stats['valid_objects'] += 1
            color = KITTI_COLORS_BGR.get(obj_type, DEFAULT_COLOR_BGR)
            thickness = 2
        else:
            stats['anomalous_objects'] += 1
            if highlight_anomalies:
                color = (0, 0, 255)  # Red for anomalies
                thickness = 3
            else:
                color = KITTI_COLORS_BGR.get(obj_type, DEFAULT_COLOR_BGR)
                thickness = 2

        # Draw rectangle
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, thickness)

        # Prepare label text
        if show_details:
            truncated = label['truncated']
            occluded = label['occluded']
            text_parts = [obj_type]

            if truncated > 0:
                text_parts.append(f"T:{truncated:.2f}")
            if occluded > 0:
                text_parts.append(f"O:{occluded}")
            if not validation['is_valid']:
                text_parts.append("⚠")

            text = " ".join(text_parts)
        else:
            text = obj_type

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        label_y = max(y_min - 5, text_height + 5)
        cv2.rectangle(annotated,
                     (x_min, label_y - text_height - baseline),
                     (x_min + text_width, label_y + baseline),
                     color, -1)

        # Draw text
        cv2.putText(annotated, text, (x_min, label_y - baseline),
                   font, font_scale, (255, 255, 255), font_thickness)

        # Draw anomaly details if needed
        if not validation['is_valid'] and show_details:
            for i, anomaly in enumerate(validation['anomalies'][:3]):  # Show max 3 anomalies
                anomaly_text = f"  {anomaly}"
                cv2.putText(annotated, anomaly_text,
                           (x_min, y_max + 15 + i * 15),
                           font, 0.4, (0, 0, 255), 1)

    return annotated, stats


def visualize_single_image(image_path: str, label_path: str,
                          show_details: bool = True,
                          filter_types: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
    """Visualize 2D bboxes for a single image.

    Args:
        image_path: Path to image file
        label_path: Path to label file
        show_details: Whether to show detailed information
        filter_types: List of object types to display
        save_path: Optional path to save annotated image
    """
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return

    # Load labels
    labels = load_kitti_label(label_path)
    if not labels:
        print(f"Warning: No labels found in {label_path}")

    # Draw bboxes
    annotated, stats = draw_bbox_on_image(image, labels, show_details, filter_types)

    # Add statistics overlay
    img_height, img_width = image.shape[:2]
    info_text = [
        f"Image: {os.path.basename(image_path)}",
        f"Objects: {stats['total_objects']} (Valid: {stats['valid_objects']}, Anomalous: {stats['anomalous_objects']})",
    ]

    if stats['by_type']:
        type_summary = ", ".join([f"{k}:{v}" for k, v in sorted(stats['by_type'].items())])
        info_text.append(f"Types: {type_summary}")

    # Draw info panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20

    for i, text in enumerate(info_text):
        y_pos = 20 + i * line_height
        # Background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(annotated, (5, y_pos - text_height - baseline),
                     (15 + text_width, y_pos + baseline), (0, 0, 0), -1)
        # Text
        cv2.putText(annotated, text, (10, y_pos), font, font_scale, (255, 255, 255), font_thickness)

    # Save or display
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved annotated image to: {save_path}")
    else:
        cv2.imshow('KITTI 2D BBox Visualization', annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_directory(image_dir: str, label_dir: str,
                       show_details: bool = True,
                       filter_types: Optional[List[str]] = None,
                       save_dir: Optional[str] = None,
                       fps: int = 10) -> None:
    """Visualize 2D bboxes for a directory of images.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing label files
        show_details: Whether to show detailed information
        filter_types: List of object types to display
        save_dir: Optional directory to save annotated images
        fps: Frames per second for animation
    """
    # Get sorted list of images
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not image_files:
        print(f"Error: No PNG images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    delay = int(1000 / fps)  # Delay in milliseconds

    for idx, image_path in enumerate(image_files):
        # Get corresponding label file
        frame_id = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, f"{frame_id}.txt")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            continue

        # Load labels
        labels = load_kitti_label(label_path)

        # Draw bboxes
        annotated, stats = draw_bbox_on_image(image, labels, show_details, filter_types)

        # Add frame info
        frame_info = f"Frame {idx + 1}/{len(image_files)} - {frame_id}"
        cv2.putText(annotated, frame_info, (10, annotated.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save or display
        if save_dir:
            save_path = os.path.join(save_dir, f"{frame_id}_annotated.png")
            cv2.imwrite(save_path, annotated)
            print(f"Saved {idx + 1}/{len(image_files)}: {save_path}")
        else:
            cv2.imshow('KITTI 2D BBox Visualization', annotated)

            # Wait for key press
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("Visualization interrupted by user")
                break
            elif key == ord(' '):  # Space - pause
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)

    if not save_dir:
        cv2.destroyAllWindows()


def main():
    """Main entry point for KITTI 2D bbox visualization tool."""
    parser = argparse.ArgumentParser(
        description='Visualize KITTI 2D bounding boxes on images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single image
  python viz_kitti_2d_bbox.py --image dataset/kitti_object/training/image_2/000001.png \\
                               --label dataset/kitti_object/training/label_2/000001.txt

  # Visualize directory (animation)
  python viz_kitti_2d_bbox.py --image-dir dataset/kitti_object/training/image_2 \\
                               --label-dir dataset/kitti_object/training/label_2

  # Filter specific object types
  python viz_kitti_2d_bbox.py --image-dir ... --label-dir ... --filter Car,Pedestrian

  # Save annotated images
  python viz_kitti_2d_bbox.py --image-dir ... --label-dir ... --save output_dir

  # Adjust animation speed
  python viz_kitti_2d_bbox.py --image-dir ... --label-dir ... --fps 30
        """
    )

    # Input arguments
    parser.add_argument('--image', type=str, help='Single image file path')
    parser.add_argument('--label', type=str, help='Single label file path')
    parser.add_argument('--kitti-dir', type=str, help='Directory containing image_2 and label_2 subdirectories')

    # Visualization options
    parser.add_argument('--no-details', action='store_true',
                       help='Hide detailed information (truncated, occluded, etc.)')
    parser.add_argument('--filter', type=str,
                       help='Comma-separated list of object types to display (e.g., Car,Pedestrian)')
    parser.add_argument('--save', type=str,
                       help='Directory to save annotated images')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for directory animation (default: 10)')

    args = parser.parse_args()

    # Validate arguments
    if args.image and args.label:
        # Single image mode
        visualize_single_image(
            args.image, args.label,
            show_details=not args.no_details,
            filter_types=args.filter.split(',') if args.filter else None,
            save_path=args.save
        )
    elif args.kitti_dir:
        # Directory mode
        visualize_directory(
            os.path.join(args.kitti_dir, "image_2"), os.path.join(args.kitti_dir, "label_2"),
            show_details=not args.no_details,
            filter_types=args.filter.split(',') if args.filter else None,
            save_dir=args.save,
            fps=args.fps
        )
    else:
        print("Error: Please provide either:")
        print("  1. --image and --label for single image mode")
        print("  2. --kitti-dir for directory mode")
        print("\nRun with --help for more information")
        sys.exit(1)


if __name__ == "__main__":
    main()
