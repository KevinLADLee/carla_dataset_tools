#!/usr/bin/python3
"""
CARLA Dataset to MCAP Converter

This tool converts CARLA raw_data (PNG/NPY/CSV) to MCAP format for visualization
in Foxglove Studio. It uses the Foxglove SDK to create MCAP files with proper
message schemas.

Usage:
    python3 convert_to_mcap.py --input raw_data/record_2024_1109_1430
    python3 convert_to_mcap.py --input raw_data/record_2024_1109_1430 --actor vehicle_1st
    python3 convert_to_mcap.py --input raw_data/record_2024_1109_1430 --sensors rgb_front,lidar
"""

import argparse
import base64
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH

try:
    from mcap.writer import Writer
except ImportError:
    print("Error: mcap library not installed. Please install:")
    print("  pip install mcap")
    sys.exit(1)


class SensorType:
    """Sensor type identifiers based on CARLA sensor types"""
    RGB_CAMERA = "sensor.camera.rgb"
    SEMANTIC_CAMERA = "sensor.camera.semantic_segmentation"
    DEPTH_CAMERA = "sensor.camera.depth"
    LIDAR = "sensor.lidar.ray_cast"
    SEMANTIC_LIDAR = "sensor.lidar.ray_cast_semantic"
    RADAR = "sensor.other.radar"


class MCAPConverter:
    """Main converter class for CARLA raw_data to MCAP"""

    def __init__(self, input_dir: str, output_file: Optional[str] = None,
                 actor_filter: Optional[str] = None, sensor_filter: Optional[List[str]] = None):
        """
        Initialize the MCAP converter

        Args:
            input_dir: Path to the record directory (e.g., raw_data/record_2024_1109_1430)
            output_file: Output MCAP file path (default: input_dir.mcap)
            actor_filter: Optional actor name filter
            sensor_filter: Optional list of sensor names to include
        """
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Set output file
        if output_file is None:
            self.output_file = self.input_dir.parent / f"{self.input_dir.name}.mcap"
        else:
            self.output_file = Path(output_file)

        self.actor_filter = actor_filter
        self.sensor_filter = sensor_filter

        # Storage for channels and schemas
        self.channels = {}
        self.schemas = {}

        print(f"Input directory: {self.input_dir}")
        print(f"Output file: {self.output_file}")

    def scan_directory(self) -> Dict[str, List[str]]:
        """
        Scan the input directory to find all actors and their sensors

        Returns:
            Dictionary mapping actor names to list of sensor directories
        """
        actors = {}

        # Iterate through directories in the record folder
        for item in self.input_dir.iterdir():
            if not item.is_dir():
                continue

            actor_name = item.name

            # Apply actor filter if specified
            if self.actor_filter and actor_name != self.actor_filter:
                continue

            # Find sensor directories
            sensors = []
            for sensor_dir in item.iterdir():
                if not sensor_dir.is_dir():
                    continue

                sensor_name = sensor_dir.name

                # Apply sensor filter if specified
                if self.sensor_filter and sensor_name not in self.sensor_filter:
                    continue

                # Check if it has poses.csv (all sensors should have this)
                if (sensor_dir / "poses.csv").exists():
                    sensors.append(sensor_name)

            if sensors:
                actors[actor_name] = sensors

        return actors

    def detect_sensor_type(self, sensor_dir: Path) -> Optional[str]:
        """
        Detect sensor type based on directory contents

        Args:
            sensor_dir: Path to sensor directory

        Returns:
            Sensor type string or None if unknown
        """
        # Check for PNG files (cameras)
        png_files = list(sensor_dir.glob("*.png"))
        if png_files:
            sensor_name = sensor_dir.name.lower()
            if "semantic" in sensor_name or "segmentation" in sensor_name:
                return SensorType.SEMANTIC_CAMERA
            elif "depth" in sensor_name:
                return SensorType.DEPTH_CAMERA
            else:
                return SensorType.RGB_CAMERA

        # Check for NPY files (lidar/radar)
        npy_files = list(sensor_dir.glob("*.npy"))
        if npy_files:
            # Load first file to check structure
            data = np.load(npy_files[0])
            if data.dtype.names:  # Semantic LiDAR has named fields
                return SensorType.SEMANTIC_LIDAR
            elif data.shape[1] == 4:  # Regular LiDAR: x, y, z, intensity
                return SensorType.LIDAR
            elif data.shape[1] == 7:  # Radar: x, y, z, depth, velocity, azimuth, altitude
                return SensorType.RADAR

        return None

    def read_poses_csv(self, csv_path: Path) -> List[Dict]:
        """
        Read poses from CSV file

        Args:
            csv_path: Path to poses.csv

        Returns:
            List of pose dictionaries with frame, timestamp, and transform data
        """
        poses = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose = {
                    'frame': int(row['frame']),
                    'timestamp': float(row['timestamp']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'roll': float(row['roll']),
                    'pitch': float(row['pitch']),
                    'yaw': float(row['yaw'])
                }
                poses.append(pose)
        return poses

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """
        Convert Euler angles (degrees) to quaternion

        Args:
            roll, pitch, yaw: Euler angles in degrees

        Returns:
            Quaternion as (x, y, z, w)
        """
        # Convert to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        # Calculate quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (x, y, z, w)

    def create_foxglove_schemas(self, writer: Writer):
        """
        Create Foxglove message schemas for MCAP

        This uses JSON schemas compatible with Foxglove Studio
        """
        # CompressedImage schema (for cameras)
        compressed_image_schema = """
        {
          "type": "object",
          "properties": {
            "timestamp": {
              "type": "object",
              "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"}
              }
            },
            "frame_id": {"type": "string"},
            "data": {
              "type": "string",
              "contentEncoding": "base64"
            },
            "format": {"type": "string"}
          }
        }
        """
        self.schemas['CompressedImage'] = writer.register_schema(
            name="foxglove.CompressedImage",
            encoding="jsonschema",
            data=compressed_image_schema.encode()
        )

        # CameraCalibration schema
        camera_calibration_schema = """
        {
          "type": "object",
          "properties": {
            "timestamp": {
              "type": "object",
              "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"}
              }
            },
            "frame_id": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "distortion_model": {"type": "string"},
            "D": {
              "type": "array",
              "items": {"type": "number"}
            },
            "K": {
              "type": "array",
              "items": {"type": "number"}
            },
            "R": {
              "type": "array",
              "items": {"type": "number"}
            },
            "P": {
              "type": "array",
              "items": {"type": "number"}
            }
          }
        }
        """
        self.schemas['CameraCalibration'] = writer.register_schema(
            name="foxglove.CameraCalibration",
            encoding="jsonschema",
            data=camera_calibration_schema.encode()
        )

        # PointCloud schema (for LiDAR/Radar)
        # Using Foxglove PointCloud schema definition
        pointcloud_schema = """
        {
          "type": "object",
          "properties": {
            "timestamp": {
              "type": "object",
              "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"}
              }
            },
            "frame_id": {"type": "string"},
            "pose": {
              "type": "object",
              "properties": {
                "position": {
                  "type": "object",
                  "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"}
                  }
                },
                "orientation": {
                  "type": "object",
                  "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "w": {"type": "number"}
                  }
                }
              }
            },
            "point_stride": {"type": "integer"},
            "fields": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "offset": {"type": "integer"},
                  "type": {"type": "integer"}
                }
              }
            },
            "data": {
              "type": "string",
              "contentEncoding": "base64"
            }
          }
        }
        """
        self.schemas['PointCloud'] = writer.register_schema(
            name="foxglove.PointCloud",
            encoding="jsonschema",
            data=pointcloud_schema.encode()
        )

        # FrameTransform schema (for TF tree)
        frametransform_schema = """
        {
          "type": "object",
          "properties": {
            "timestamp": {
              "type": "object",
              "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"}
              }
            },
            "parent_frame_id": {"type": "string"},
            "child_frame_id": {"type": "string"},
            "translation": {
              "type": "object",
              "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"}
              }
            },
            "rotation": {
              "type": "object",
              "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"}
              }
            }
          }
        }
        """
        self.schemas['FrameTransform'] = writer.register_schema(
            name="foxglove.FrameTransform",
            encoding="jsonschema",
            data=frametransform_schema.encode()
        )

    def convert(self):
        """Main conversion method"""
        print("\nScanning directory structure...")
        actors = self.scan_directory()

        if not actors:
            print("No actors found matching the filters.")
            return

        print(f"Found {len(actors)} actor(s):")
        for actor_name, sensors in actors.items():
            print(f"  - {actor_name}: {len(sensors)} sensor(s)")
            for sensor in sensors:
                print(f"    - {sensor}")

        # Create MCAP file
        print(f"\nCreating MCAP file: {self.output_file}")
        with open(self.output_file, 'wb') as f:
            writer = Writer(f)
            writer.start()

            # Register schemas
            self.create_foxglove_schemas(writer)

            # Process each actor and sensor
            for actor_name, sensors in actors.items():
                print(f"\nProcessing actor: {actor_name}")
                self.process_actor(writer, actor_name, sensors)

            writer.finish()

        print(f"\n✓ Conversion complete!")
        print(f"  Output: {self.output_file}")
        print(f"  Size: {self.output_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"\nYou can now open this file in Foxglove Studio:")
        print(f"  https://foxglove.dev/download")

    def process_actor(self, writer: Writer, actor_name: str, sensors: List[str]):
        """Process all sensors for a given actor"""
        actor_dir = self.input_dir / actor_name

        for sensor_name in sensors:
            sensor_dir = actor_dir / sensor_name
            sensor_type = self.detect_sensor_type(sensor_dir)

            if sensor_type is None:
                print(f"  ⚠ Skipping {sensor_name}: unknown sensor type")
                continue

            print(f"  Processing {sensor_name} ({sensor_type})...")

            # Read poses
            poses = self.read_poses_csv(sensor_dir / "poses.csv")

            # Process based on sensor type
            if sensor_type in [SensorType.RGB_CAMERA, SensorType.SEMANTIC_CAMERA, SensorType.DEPTH_CAMERA]:
                self.process_camera(writer, actor_name, sensor_name, sensor_dir, poses)
            elif sensor_type == SensorType.LIDAR:
                self.process_lidar(writer, actor_name, sensor_name, sensor_dir, poses)
            elif sensor_type == SensorType.SEMANTIC_LIDAR:
                self.process_semantic_lidar(writer, actor_name, sensor_name, sensor_dir, poses)
            elif sensor_type == SensorType.RADAR:
                self.process_radar(writer, actor_name, sensor_name, sensor_dir, poses)

            # Always write transforms
            self.process_transforms(writer, actor_name, sensor_name, poses)

    def read_camera_info(self, sensor_dir: Path) -> Optional[Dict]:
        """Read camera info from CSV file"""
        camera_info_path = sensor_dir / "camera_info.csv"
        if not camera_info_path.exists():
            return None

        with open(camera_info_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                return {
                    'width': int(float(row['width'])),
                    'height': int(float(row['height'])),
                    'fx': float(row['fx']),
                    'fy': float(row['fy']),
                    'cx': float(row['cx']),
                    'cy': float(row['cy'])
                }
        return None

    def process_camera(self, writer: Writer, actor_name: str, sensor_name: str,
                      sensor_dir: Path, poses: List[Dict]):
        """Convert camera images to CompressedImage messages"""
        import json
        import cv2

        # Register image channel
        topic = f"/{actor_name}/{sensor_name}/image"
        if topic not in self.channels:
            self.channels[topic] = writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=self.schemas['CompressedImage']
            )
        image_channel_id = self.channels[topic]

        # Register camera_info channel
        calib_topic = f"/{actor_name}/{sensor_name}/camera_info"
        if calib_topic not in self.channels:
            self.channels[calib_topic] = writer.register_channel(
                topic=calib_topic,
                message_encoding="json",
                schema_id=self.schemas['CameraCalibration']
            )
        calib_channel_id = self.channels[calib_topic]

        # Read camera calibration
        camera_info = self.read_camera_info(sensor_dir)

        # Process each frame
        png_files = sorted(sensor_dir.glob("*.png"))
        for png_file in tqdm(png_files, desc=f"    {sensor_name}", leave=False):
            frame_id = int(png_file.stem)

            # Find corresponding pose
            pose_data = next((p for p in poses if p['frame'] == frame_id), None)
            if pose_data is None:
                continue

            timestamp_ns = int(pose_data['timestamp'] * 1e9)
            timestamp_msg = {
                "sec": int(pose_data['timestamp']),
                "nsec": int((pose_data['timestamp'] % 1) * 1e9)
            }

            # Read image with OpenCV (handles BGRA format from CARLA)
            img = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            # Convert BGRA to BGR if needed (remove alpha channel)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Encode as JPEG for better compatibility and smaller size
            success, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            if not success:
                continue

            # Encode as base64 string for JSON
            jpeg_data = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

            # Create image message
            image_message = {
                "timestamp": timestamp_msg,
                "frame_id": f"{actor_name}/{sensor_name}",
                "data": jpeg_data,
                "format": "jpeg"
            }

            # Write image to MCAP
            writer.add_message(
                channel_id=image_channel_id,
                log_time=timestamp_ns,
                data=json.dumps(image_message).encode('utf-8'),
                publish_time=timestamp_ns
            )

            # Write camera calibration (once or per frame based on preference)
            if camera_info:
                # Create calibration matrix K = [fx, 0, cx; 0, fy, cy; 0, 0, 1]
                K = [
                    camera_info['fx'], 0.0, camera_info['cx'],
                    0.0, camera_info['fy'], camera_info['cy'],
                    0.0, 0.0, 1.0
                ]

                # Projection matrix P = [fx, 0, cx, 0; 0, fy, cy, 0; 0, 0, 1, 0]
                P = [
                    camera_info['fx'], 0.0, camera_info['cx'], 0.0,
                    0.0, camera_info['fy'], camera_info['cy'], 0.0,
                    0.0, 0.0, 1.0, 0.0
                ]

                calib_message = {
                    "timestamp": timestamp_msg,
                    "frame_id": f"{actor_name}/{sensor_name}",
                    "width": camera_info['width'],
                    "height": camera_info['height'],
                    "distortion_model": "plumb_bob",
                    "D": [0.0, 0.0, 0.0, 0.0, 0.0],  # No distortion in CARLA
                    "K": K,
                    "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Identity
                    "P": P
                }

                writer.add_message(
                    channel_id=calib_channel_id,
                    log_time=timestamp_ns,
                    data=json.dumps(calib_message).encode('utf-8'),
                    publish_time=timestamp_ns
                )

    def process_lidar(self, writer: Writer, actor_name: str, sensor_name: str,
                     sensor_dir: Path, poses: List[Dict]):
        """Convert LiDAR point clouds to PointCloud messages"""
        import json

        # Register channel if not exists
        topic = f"/{actor_name}/{sensor_name}/points"
        if topic not in self.channels:
            self.channels[topic] = writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=self.schemas['PointCloud']
            )

        channel_id = self.channels[topic]

        # Process each frame
        npy_files = sorted(sensor_dir.glob("*.npy"))
        for npy_file in tqdm(npy_files, desc=f"    {sensor_name}", leave=False):
            frame_id = int(npy_file.stem)

            # Find corresponding pose
            pose_data = next((p for p in poses if p['frame'] == frame_id), None)
            if pose_data is None:
                continue

            # Load point cloud
            points = np.load(npy_file)  # Nx4: x, y, z, intensity

            if points.size == 0:
                continue

            # Encode as base64 string for JSON
            points_data = base64.b64encode(points.astype(np.float32).tobytes()).decode('utf-8')

            # Create message
            message = {
                "timestamp": {
                    "sec": int(pose_data['timestamp']),
                    "nsec": int((pose_data['timestamp'] % 1) * 1e9)
                },
                "frame_id": f"{actor_name}/{sensor_name}",
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                },
                "point_stride": 16,  # 4 floats * 4 bytes
                "fields": [
                    {"name": "x", "offset": 0, "type": 7},  # FLOAT32 = 7
                    {"name": "y", "offset": 4, "type": 7},
                    {"name": "z", "offset": 8, "type": 7},
                    {"name": "intensity", "offset": 12, "type": 7}
                ],
                "data": points_data
            }

            # Write to MCAP
            timestamp_ns = int(pose_data['timestamp'] * 1e9)
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_ns,
                data=json.dumps(message).encode('utf-8'),
                publish_time=timestamp_ns
            )

    def process_semantic_lidar(self, writer: Writer, actor_name: str, sensor_name: str,
                               sensor_dir: Path, poses: List[Dict]):
        """Convert Semantic LiDAR to PointCloud messages with labels"""
        # Similar to regular LiDAR but with semantic labels
        # For now, treat as regular point cloud (can be enhanced later)
        self.process_lidar(writer, actor_name, sensor_name, sensor_dir, poses)

    def process_radar(self, writer: Writer, actor_name: str, sensor_name: str,
                     sensor_dir: Path, poses: List[Dict]):
        """Convert Radar data to PointCloud messages"""
        import json

        # Register channel if not exists
        topic = f"/{actor_name}/{sensor_name}/detections"
        if topic not in self.channels:
            self.channels[topic] = writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=self.schemas['PointCloud']
            )

        channel_id = self.channels[topic]

        # Process each frame
        npy_files = sorted(sensor_dir.glob("*.npy"))
        for npy_file in tqdm(npy_files, desc=f"    {sensor_name}", leave=False):
            frame_id = int(npy_file.stem)

            # Find corresponding pose
            pose_data = next((p for p in poses if p['frame'] == frame_id), None)
            if pose_data is None:
                continue

            # Load radar data
            detections = np.load(npy_file)  # Nx7: x, y, z, depth, velocity, azimuth, altitude

            if detections.size == 0:
                continue

            # Extract x, y, z, velocity for visualization
            points = np.column_stack([
                detections[:, 0],  # x
                detections[:, 1],  # y
                detections[:, 2],  # z
                detections[:, 4]   # velocity
            ]).astype(np.float32)

            # Encode as base64 string for JSON
            points_data = base64.b64encode(points.tobytes()).decode('utf-8')

            message = {
                "timestamp": {
                    "sec": int(pose_data['timestamp']),
                    "nsec": int((pose_data['timestamp'] % 1) * 1e9)
                },
                "frame_id": f"{actor_name}/{sensor_name}",
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                },
                "point_stride": 16,
                "fields": [
                    {"name": "x", "offset": 0, "type": 7},
                    {"name": "y", "offset": 4, "type": 7},
                    {"name": "z", "offset": 8, "type": 7},
                    {"name": "velocity", "offset": 12, "type": 7}
                ],
                "data": points_data
            }

            # Write to MCAP
            timestamp_ns = int(pose_data['timestamp'] * 1e9)
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_ns,
                data=json.dumps(message).encode('utf-8'),
                publish_time=timestamp_ns
            )

    def process_transforms(self, writer: Writer, actor_name: str, sensor_name: str,
                          poses: List[Dict]):
        """Write TF transforms for sensor poses"""
        # Register channel if not exists
        topic = "/tf"
        if topic not in self.channels:
            self.channels[topic] = writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=self.schemas['FrameTransform']
            )

        channel_id = self.channels[topic]

        # Write transform for each pose
        for pose_data in poses:
            qx, qy, qz, qw = self.euler_to_quaternion(
                pose_data['roll'],
                pose_data['pitch'],
                pose_data['yaw']
            )

            import json
            message = {
                "timestamp": {
                    "sec": int(pose_data['timestamp']),
                    "nsec": int((pose_data['timestamp'] % 1) * 1e9)
                },
                "parent_frame_id": "map",
                "child_frame_id": f"{actor_name}/{sensor_name}",
                "translation": {
                    "x": pose_data['x'],
                    "y": pose_data['y'],
                    "z": pose_data['z']
                },
                "rotation": {
                    "x": qx,
                    "y": qy,
                    "z": qz,
                    "w": qw
                }
            }

            # Write to MCAP
            timestamp_ns = int(pose_data['timestamp'] * 1e9)
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_ns,
                data=json.dumps(message).encode('utf-8'),
                publish_time=timestamp_ns
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert CARLA raw_data to MCAP format for Foxglove visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire recording
  python3 convert_to_mcap.py --input raw_data/record_2024_1109_1430

  # Convert specific actor
  python3 convert_to_mcap.py --input raw_data/record_2024_1109_1430 --actor vehicle_1st

  # Convert specific sensors
  python3 convert_to_mcap.py --input raw_data/record_2024_1109_1430 --sensors rgb_front,lidar

  # Custom output file
  python3 convert_to_mcap.py --input raw_data/record_2024_1109_1430 --output my_data.mcap
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        type=str,
        help='Input directory (e.g., raw_data/record_2024_1109_1430)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output MCAP file path (default: <input_dir>.mcap)'
    )

    parser.add_argument(
        '--actor', '-a',
        type=str,
        help='Filter by actor name (e.g., vehicle_1st)'
    )

    parser.add_argument(
        '--sensors', '-s',
        type=str,
        help='Comma-separated list of sensor names to include (e.g., rgb_front,lidar)'
    )

    args = parser.parse_args()

    # Parse sensor filter
    sensor_filter = None
    if args.sensors:
        sensor_filter = [s.strip() for s in args.sensors.split(',')]

    # Create converter and run
    try:
        converter = MCAPConverter(
            input_dir=args.input,
            output_file=args.output,
            actor_filter=args.actor,
            sensor_filter=sensor_filter
        )
        converter.convert()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
