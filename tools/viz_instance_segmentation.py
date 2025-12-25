#!/usr/bin/env python3
"""
CARLA Instance Segmentation Image Analysis Tool
Extracts different instance IDs and tags from CityScapesPalette encoded instance segmentation images,
and draws 2D bounding boxes with text annotations.

CARLA Instance Segmentation Encoding Format:
- R channel: Semantic tag ID (semantic tag)
- G channel: Instance ID high byte (instance_id >> 8)
- B channel: Instance ID low byte (instance_id & 0xFF)
- A channel: Transparency

Instance ID calculation: instance_id = G * 256 + B

Semantic tags reference: https://carla.readthedocs.io/en/0.9.16/ref_sensors/#semantic-segmentation-camera
"""
import argparse
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# CARLA 0.9.16 Complete semantic labels definition
CARLA_LABELS = {
    0: {'name': 'Unlabeled', 'color': (0, 0, 0), 'description': 'Elements not categorized'},
    1: {'name': 'Road', 'color': (128, 64, 128), 'description': 'Ground where cars drive'},
    2: {'name': 'SideWalk', 'color': (244, 35, 232), 'description': 'Pedestrian/cyclist areas'},
    3: {'name': 'Building', 'color': (70, 70, 70), 'description': 'Houses, skyscrapers, etc.'},
    4: {'name': 'Wall', 'color': (102, 102, 156), 'description': 'Individual standing walls'},
    5: {'name': 'Fence', 'color': (190, 153, 153), 'description': 'Barriers, railings'},
    6: {'name': 'Pole', 'color': (153, 153, 153), 'description': 'Small vertical poles'},
    7: {'name': 'TrafficLight', 'color': (250, 170, 30), 'description': 'Traffic light boxes'},
    8: {'name': 'TrafficSign', 'color': (220, 220, 0), 'description': 'Regulatory signs'},
    9: {'name': 'Vegetation', 'color': (107, 142, 35), 'description': 'Trees, hedges'},
    10: {'name': 'Terrain', 'color': (152, 251, 152), 'description': 'Ground-level vegetation'},
    11: {'name': 'Sky', 'color': (70, 130, 180), 'description': 'Open sky'},
    12: {'name': 'Pedestrian', 'color': (220, 20, 60), 'description': 'Walking humans'},
    13: {'name': 'Rider', 'color': (255, 0, 0), 'description': 'Humans riding vehicles'},
    14: {'name': 'Car', 'color': (0, 0, 142), 'description': 'Cars, vans'},
    15: {'name': 'Truck', 'color': (0, 0, 70), 'description': 'Trucks'},
    16: {'name': 'Bus', 'color': (0, 60, 100), 'description': 'Buses'},
    17: {'name': 'Train', 'color': (0, 80, 100), 'description': 'Trains'},
    18: {'name': 'Motorcycle', 'color': (0, 0, 230), 'description': 'Motorcycles'},
    19: {'name': 'Bicycle', 'color': (119, 11, 32), 'description': 'Bicycles'},
    20: {'name': 'Static', 'color': (110, 190, 160), 'description': 'Immovable elements'},
    21: {'name': 'Dynamic', 'color': (170, 120, 50), 'description': 'Changeable positions'},
    22: {'name': 'Other', 'color': (55, 90, 80), 'description': 'Unspecified elements'},
    23: {'name': 'Water', 'color': (45, 60, 150), 'description': 'Horizontal water'},
    24: {'name': 'RoadLine', 'color': (157, 234, 50), 'description': 'Road markings'},
    25: {'name': 'Ground', 'color': (81, 0, 81), 'description': 'Ground-level structures'},
    26: {'name': 'Bridge', 'color': (150, 100, 100), 'description': 'Bridge structures'},
    27: {'name': 'RailTrack', 'color': (230, 150, 140), 'description': 'Train/subway tracks'},
    28: {'name': 'GuardRail', 'color': (180, 165, 180), 'description': 'Crash barriers'}
}

def get_label_info(label_id):
    """Get semantic label information"""
    label_info = CARLA_LABELS.get(label_id, {
        'name': f'Unknown_{label_id}',
        'color': (128, 128, 128),
        'description': 'Unknown semantic tag'
    })
    return label_info

def extract_instances(image, filter_background=True):
    """
    Extract all instance information from instance segmentation image

    Args:
        image: Instance segmentation image in BGRA format (H, W, 4)
        filter_background: Whether to filter background objects (roads, sidewalks, etc.)

    Returns:
        instances: List of dictionaries, each containing instance_id, tag_id, mask, bbox information
    """
    height, width = image.shape[:2]

    # Extract channels (Note: OpenCV reads in BGR order)
    b_channel = image[:, :, 0]  # Blue channel (instance ID low byte)
    g_channel = image[:, :, 1]  # Green channel (instance ID high byte)
    r_channel = image[:, :, 2]  # Red channel (semantic tag ID)
    a_channel = image[:, :, 3]  # Alpha channel

    # Calculate instance ID: instance_id = G * 256 + B
    instance_ids = g_channel.astype(np.uint16) * 256 + b_channel.astype(np.uint16)
    tag_ids = r_channel  # Semantic tag ID

    # Get all unique instance IDs
    unique_instances = np.unique(instance_ids)

    # Define background labels to filter (roads, buildings, sky, etc.)
    BACKGROUND_LABELS = {
        0: 'Unlabeled',
        1: 'Road',
        2: 'SideWalk',
        3: 'Building',
        4: 'Wall',
        5: 'Fence',
        9: 'Vegetation',
        10: 'Terrain',
        11: 'Sky',
        22: 'Other',
        23: 'Water',
        24: 'RoadLine',
        25: 'Ground',
        26: 'Bridge',
        27: 'RailTrack',
        28: 'GuardRail'
    }

    # Foreground labels to keep (vehicles, pedestrians, traffic facilities, etc.)
    FOREGROUND_LABELS = {
        6: 'Pole',
        7: 'TrafficLight',
        8: 'TrafficSign',
        12: 'Pedestrian',
        13: 'Rider',
        14: 'Car',
        15: 'Truck',
        16: 'Bus',
        17: 'Train',
        18: 'Motorcycle',
        19: 'Bicycle',
        20: 'Static',
        21: 'Dynamic',
    }

    instances = []
    for instance_id in unique_instances:
        # Skip background and invalid instances (ID=0 is usually background)
        if instance_id == 0:
            continue

        # Create instance mask
        mask = (instance_ids == instance_id)

        # Skip if mask is too small (likely noise)
        if np.sum(mask) < 50:  # Increase minimum pixel threshold to filter noise
            continue

        # Get corresponding semantic tag
        tag_ids_in_mask = tag_ids[mask]
        tag_id = np.unique(tag_ids_in_mask)[0] if len(tag_ids_in_mask) > 0 else 0

        # Decide whether to include this instance based on filter settings
        if filter_background and tag_id in BACKGROUND_LABELS:
            continue

        # Calculate bounding box
        y_indices, x_indices = np.where(mask)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Calculate center point
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        instances.append({
            'instance_id': int(instance_id),
            'tag_id': int(tag_id),
            'tag_info': get_label_info(tag_id),
            'mask': mask,
            'bbox': {
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max),
                'center': (center_x, center_y),
                'width': int(x_max - x_min),
                'height': int(y_max - y_min)
            },
            'pixel_count': int(np.sum(mask))
        })

    return instances

def draw_instance_annotations(image, instances, output_path=None, show_legend=True):
    """
    Draw instance annotations on image

    Args:
        image: Original BGRA image
        instances: List of instance information
        output_path: Output file path (optional)
        show_legend: Whether to show legend

    Returns:
        annotated_image: Annotated image
    """
    # Convert to RGB format for display
    rgb_image = cv.cvtColor(image, cv.COLOR_BGRA2RGB)

    # Create image copy for drawing
    annotated_image = rgb_image.copy()

    # Set font
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Group statistics by semantic tag
    tag_stats = {}
    for instance in instances:
        tag_id = instance['tag_id']
        if tag_id not in tag_stats:
            tag_stats[tag_id] = 0
        tag_stats[tag_id] += 1

    filter_mode = "background filtered" if len(instances) > 0 else "no foreground objects detected"
    print(f"\nFound {len(instances)} foreground instances, {len(tag_stats)} semantic tag types ({filter_mode}):")
    print("=" * 80)

    for i, instance in enumerate(instances):
        bbox = instance['bbox']
        tag_info = instance['tag_info']

        # Use standard color for the tag
        color_rgb = tag_info['color']
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert RGB to BGR

        # Draw bounding box
        cv.rectangle(annotated_image,
                    (bbox['x_min'], bbox['y_min']),
                    (bbox['x_max'], bbox['y_max']),
                    color_bgr, 2)

        # Prepare annotation text
        text_lines = [
            f"ID:{instance['instance_id']}",
            f"{tag_info['name']}({instance['tag_id']})"
            # f"{bbox['width']}x{bbox['height']}"
        ]

        # Draw background box and text
        text_y = bbox['y_min'] - 10
        for j, text in enumerate(text_lines):
            # Calculate text size
            (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, font_thickness)

            # Ensure text box is within image bounds
            text_y = max(text_y, text_height + 5)
            text_x = max(bbox['x_min'], 5)
            text_x = min(text_x, annotated_image.shape[1] - text_width - 10)

            # Draw background
            cv.rectangle(annotated_image,
                        (text_x, text_y - text_height - 3),
                        (text_x + text_width + 6, text_y + 3),
                        (0, 0, 0), -1)  # Black background

            # Draw text
            cv.putText(annotated_image, text,
                      (text_x + 3, text_y),
                      font, font_scale, (255, 255, 255), font_thickness)

            text_y -= (text_height + 8)

        # Print instance information (show details for every 5th instance)
        if i % 5 == 0 or i == len(instances) - 1:
            print(f"Instance {i+1}:")
            print(f"  Instance ID: {instance['instance_id']}")
            print(f"  Semantic tag: {tag_info['name']} (ID: {instance['tag_id']}) - {tag_info['description']}")
            print(f"  Bounding box: ({bbox['x_min']}, {bbox['y_min']}) -> ({bbox['x_max']}, {bbox['y_max']})")
            print(f"  Size: {bbox['width']} x {bbox['height']} pixels")
            print(f"  Center: {bbox['center']}")
            print(f"  Pixel count: {instance['pixel_count']}")
            print("-" * 40)

    # Print tag statistics
    print("\nForeground semantic tag statistics:")
    print("-" * 40)
    for tag_id, count in tag_stats.items():
        tag_info = get_label_info(tag_id)
        print(f"{tag_info['name']} (ID:{tag_id}): {count} instances")

    print("\nFilter description:")
    print("Default filtered background objects include:")
    print("- Roads, sidewalks, buildings, sky, vegetation and other static backgrounds")
    print("- Kept: Vehicles, pedestrians, traffic lights, traffic signs and other dynamic or interactive objects")

    # Save results
    if output_path:
        cv.imwrite(output_path, cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
        print(f"\nAnnotated image saved to: {output_path}")

    return annotated_image

def analyze_instance_image(image_path, output_dir=None, show_plot=True, filter_background=True):
    """
    Analyze a single instance segmentation image

    Args:
        image_path: Input image path
        output_dir: Output directory (optional)
        show_plot: Whether to show matplotlib plot
        filter_background: Whether to filter background objects (roads, sidewalks, etc.)
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist: {image_path}")
        return

    print(f"Analyzing image: {image_path}")

    # Read image (BGRA format)
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to read image file: {image_path}")
        return

    if len(image.shape) != 3 or image.shape[2] != 4:
        print(f"Error: Incorrect image format, expected BGRA format, actual: {image.shape}")
        return

    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")

    if filter_background:
        print("Background filtering enabled - ignoring roads, buildings, sky and other background objects")

    # Extract instance information
    instances = extract_instances(image, filter_background)

    if not instances:
        print("No instances detected")
        return

    # Generate output file paths
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_annotated.png")

        # Create instance statistics report
        report_path = os.path.join(output_dir, f"{image_name}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"CARLA Instance Segmentation Analysis Report\n")
            f.write(f"{'='*60}\n")
            f.write(f"Image file: {image_path}\n")
            f.write(f"Image size: {width}x{height}\n")
            f.write(f"Number of instances detected: {len(instances)}\n\n")

            # Group by tag
            tag_stats = {}
            for instance in instances:
                tag_id = instance['tag_id']
                if tag_id not in tag_stats:
                    tag_stats[tag_id] = []
                tag_stats[tag_id].append(instance)

            f.write("Instances grouped by semantic tag:\n")
            f.write("-" * 40 + "\n")
            for tag_id, tag_instances in tag_stats.items():
                tag_info = get_label_info(tag_id)
                f.write(f"{tag_info['name']} (ID: {tag_id}): {len(tag_instances)}  instances\n")
                for instance in tag_instances:
                    bbox = instance['bbox']
                    f.write(f"  Instance ID: {instance['instance_id']}, ")
                    f.write(f"Position:({bbox['x_min']},{bbox['y_min']})-({bbox['x_max']},{bbox['y_max']}), ")
                    f.write(f"Size: {bbox['width']}x{bbox['height']}\n")
                f.write("\n")

        print(f"Analysis report saved to: {report_path}")
    else:
        output_path = None

    # Draw annotations
    annotated_image = draw_instance_annotations(image, instances, output_path)

    # Display results (if supportedmatplotlib)
    if show_plot:
        try:
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.title('Raw Instance Segmentation Image (Raw Encoding)', fontsize=12)
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGRA2RGB))
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Annotated Instance Segmentation Image', fontsize=12)
            plt.imshow(annotated_image)
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error displaying image: {e}")

def main():
    parser = argparse.ArgumentParser(description='CARLA Instance Segmentation Image Analysis Tool')
    parser.add_argument('input', help='Input instance segmentation image path')
    parser.add_argument('-o', '--output', help='Output directory path')
    parser.add_argument('--no-plot', action='store_true', help='do not showmatplotlibimage window')
    parser.add_argument('--include-background', action='store_true', help='include background objects (roads, buildings, etc.), filter background by default')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

    args = parser.parse_args()

    print("CARLA Instance Segmentation Image Analysis Tool")
    print("Supports CityScapesPalette encoded instance segmentation images")
    print("=" * 60)
    print(f"Input file: {args.input}")
    if args.output:
        print(f"Output directory: {args.output}")
    print(f"Background filtering: {'disabled' if args.include_background else 'enabled (default)'}")
    print()

    filter_background = not args.include_background
    analyze_instance_image(args.input, args.output, not args.no_plot, filter_background)

if __name__ == '__main__':
    main()