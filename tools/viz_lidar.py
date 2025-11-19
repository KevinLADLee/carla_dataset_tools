#!/usr/bin/python3

import argparse
import glob
import os.path
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from plyfile import PlyData

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import ROOT_PATH

# Constants
VIRIDIS = np.array(plt.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
INTENSITY_MIN = 0.01  # Minimum intensity to avoid log(0)
INTENSITY_SCALE = -0.004 * 100  # Scale factor for intensity normalization

LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

# KITTI object type to color mapping
KITTI_COLORS = {
    'Car': (0.0, 0.0, 1.0),           # Blue
    'Pedestrian': (1.0, 0.0, 0.0),    # Red
    'Cyclist': (0.0, 1.0, 0.0),       # Green
    'Van': (0.0, 0.5, 1.0),           # Light Blue
    'Truck': (0.5, 0.0, 0.5),         # Purple
    'Person_sitting': (1.0, 0.5, 0.0), # Orange
    'Tram': (0.5, 0.5, 0.5),          # Gray
    'Misc': (0.5, 0.5, 0.0),          # Olive
    'DontCare': (0.3, 0.3, 0.3),      # Dark Gray
}


class PointcloudType(Enum):
    LIDAR = 0
    SEMANTIC_LIDAR = 1
    RADAR = 2
    KITTI = 3  # KITTI format with labels


def load_kitti_bin(bin_path: str) -> np.ndarray:
    """Load KITTI binary point cloud file (.bin format).

    Args:
        bin_path: Path to .bin file

    Returns:
        Numpy array of shape (N, 4) with columns [x, y, z, intensity]
    """
    try:
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points
    except Exception as e:
        raise RuntimeError(f"Failed to load KITTI bin file {bin_path}: {e}")


def load_ply_lidar(ply_path: str) -> np.ndarray:
    """Load PLY format LIDAR point cloud.

    Args:
        ply_path: Path to .ply file

    Returns:
        Numpy array compatible with existing code.
        For regular lidar: (N, 4) with columns [x, y, z, intensity]
        For semantic lidar: structured array with fields [x, y, z, CosAngle, ObjIdx, ObjTag]
    """
    try:
        ply_data = PlyData.read(ply_path)
        vertex = ply_data['vertex']
        vertex_props = [prop.name for prop in vertex.properties]

        if 'intensity' in vertex_props:
            # Regular LiDAR format
            return np.column_stack([
                vertex['x'],
                vertex['y'],
                vertex['z'],
                vertex['intensity']
            ]).astype(np.float32)
        elif 'cos_angle' in vertex_props:
            # Semantic LiDAR format - return structured array
            n_points = len(vertex)
            dtype = [
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('CosAngle', np.float32),
                ('ObjIdx', np.uint32),
                ('ObjTag', np.uint32)
            ]
            structured = np.zeros(n_points, dtype=dtype)
            structured['x'] = vertex['x']
            structured['y'] = vertex['y']
            structured['z'] = vertex['z']
            structured['CosAngle'] = vertex['cos_angle']
            structured['ObjIdx'] = vertex['obj_idx']
            structured['ObjTag'] = vertex['obj_tag']
            return structured
        else:
            # Fallback: return only xyz if no known format
            return np.column_stack([
                vertex['x'],
                vertex['y'],
                vertex['z']
            ]).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load PLY file {ply_path}: {e}")


def load_kitti_calib(calib_path: str) -> dict:
    """Load KITTI calibration file.

    Args:
        calib_path: Path to calibration .txt file

    Returns:
        Dictionary containing calibration matrices
    """
    if not os.path.exists(calib_path):
        return {}

    calib = {}
    try:
        with open(calib_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                key = parts[0].rstrip(':')
                values = [float(x) for x in parts[1:]]

                if key == 'Tr_velo_to_cam':
                    # Tr_velo_to_cam is a 3x4 matrix
                    calib['Tr_velo_to_cam'] = np.array(values).reshape(3, 4)
                elif key.startswith('P'):
                    # P0-P3 are projection matrices (3x4)
                    calib[key] = np.array(values).reshape(3, 4)
                elif key == 'R0_rect':
                    # R0_rect is a 3x3 rectification matrix
                    calib['R0_rect'] = np.array(values).reshape(3, 3)

        return calib
    except Exception as e:
        print(f"Warning: Failed to parse calibration file {calib_path}: {e}")
        return {}


def load_kitti_label(label_path: str) -> list:
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
            for line in f:
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
                    'rotation_y': float(parts[14])
                }
                labels.append(label)
        return labels
    except Exception as e:
        print(f"Warning: Failed to parse label file {label_path}: {e}")
        return []


def create_3d_bbox(label: dict, calib: Optional[dict] = None) -> o3d.geometry.OrientedBoundingBox:
    """Create Open3D oriented bounding box from KITTI label.

    Args:
        label: Dictionary containing KITTI label information
        calib: Optional calibration dictionary containing Tr_velo_to_cam matrix

    Returns:
        Open3D OrientedBoundingBox object in LiDAR coordinate system
    """
    h, w, l = label['dimensions']  # height, width, length
    x, y, z = label['location']    # center location in camera coords
    ry = label['rotation_y']       # rotation around Y-axis in camera coords

    # Create bounding box in camera coordinate system
    # KITTI camera: X right, Y down, Z forward
    center_cam = np.array([x, y, z])
    extent = np.array([l, w, h])  # Open3D uses [length, width, height]

    # Create rotation matrix (rotation around Y-axis in camera coords)
    R_cam = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Transform from camera coordinates to LiDAR coordinates
    if calib and 'Tr_velo_to_cam' in calib:
        # Get the transformation matrix from velodyne to camera
        Tr_velo_to_cam = calib['Tr_velo_to_cam']  # 3x4 matrix

        # Create 4x4 homogeneous matrix
        T_velo_to_cam = np.eye(4)
        T_velo_to_cam[0:3, :] = Tr_velo_to_cam

        # Compute inverse transformation (camera to velodyne)
        T_cam_to_velo = np.linalg.inv(T_velo_to_cam)

        # Transform center from camera to LiDAR
        center_cam_h = np.append(center_cam, 1.0)  # Homogeneous coordinates
        center_lidar_h = T_cam_to_velo @ center_cam_h
        center_lidar = center_lidar_h[0:3]

        # Transform rotation from camera to LiDAR
        # R_lidar = R_cam_to_velo @ R_cam @ R_velo_to_cam
        R_cam_to_velo = T_cam_to_velo[0:3, 0:3]
        R_lidar = R_cam_to_velo @ R_cam
    else:
        # No calibration provided, use camera coordinates directly
        center_lidar = center_cam
        R_lidar = R_cam
        print("Warning: No calibration data, bbox might be in wrong coordinate system")

    # Create oriented bounding box
    bbox = o3d.geometry.OrientedBoundingBox(center_lidar, R_lidar, extent)

    # Set color based on object type
    obj_type = label['type']
    bbox.color = KITTI_COLORS.get(obj_type, (1.0, 1.0, 0.0))  # Default yellow

    return bbox


class LidarVisualizer:
    def __init__(self, pointcloud_type: PointcloudType, source: str):
        self.pointcloud_type = pointcloud_type
        self.source = source
        self.pcd = o3d.geometry.PointCloud()

    def visualize(self):
        """Visualize point cloud data with optional 3D bounding boxes."""
        # Single file mode
        if self.source.endswith('.ply') or self.source.endswith('.bin'):
            try:
                if self.source.endswith('.ply'):
                    raw_pcd = load_ply_lidar(self.source)
                else:  # .bin file
                    raw_pcd = load_kitti_bin(self.source)

                self.numpy_to_o3d(raw_pcd)

                # Load labels if KITTI type
                geometries = [self.pcd]
                if self.pointcloud_type == PointcloudType.KITTI:
                    bboxes = self._load_kitti_labels_for_file(self.source)
                    geometries.extend(bboxes)

                o3d.visualization.draw_geometries(geometries)
            except Exception as e:
                print(f"Error loading file {self.source}: {e}")
                raise

        # Directory mode - animate through frames
        else:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Carla Lidar')
            vis.get_render_option().point_size = 1
            vis.get_render_option().show_coordinate_frame = True
            self.add_open3d_axis(vis)

            # Get file list based on type
            if self.pointcloud_type == PointcloudType.KITTI:
                files = sorted(glob.glob(f"{self.source}/*.bin"))
            else:
                files = sorted(glob.glob(f"{self.source}/*.ply"))

            if not files:
                print(f"No point cloud files found in {self.source}")
                vis.destroy_window()
                return

            print(f"Found {len(files)} files to visualize")

            # Track geometries for updates
            bbox_geometries = []
            frame = 0

            for file in files:
                try:
                    # Load point cloud
                    if file.endswith('.ply'):
                        raw_pcd = load_ply_lidar(file)
                    else:  # .bin file
                        raw_pcd = load_kitti_bin(file)

                    self.numpy_to_o3d(raw_pcd)

                    # Add or update point cloud
                    if frame == 0:
                        vis.add_geometry(self.pcd)
                    vis.update_geometry(self.pcd)

                    # Handle bounding boxes for KITTI
                    if self.pointcloud_type == PointcloudType.KITTI:
                        # Remove old bboxes
                        for bbox in bbox_geometries:
                            vis.remove_geometry(bbox, reset_bounding_box=False)
                        bbox_geometries.clear()

                        # Add new bboxes
                        bboxes = self._load_kitti_labels_for_file(file)
                        for bbox in bboxes:
                            vis.add_geometry(bbox, reset_bounding_box=False)
                            bbox_geometries.append(bbox)

                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.1)
                    frame += 1

                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue

            vis.destroy_window()

    def _load_kitti_labels_for_file(self, point_cloud_path: str) -> list:
        """Load KITTI labels corresponding to a point cloud file.

        Args:
            point_cloud_path: Path to point cloud file (e.g., .../velodyne/000001.bin)

        Returns:
            List of Open3D bounding box geometries
        """
        # Determine label file path
        # Typical structure: .../training/velodyne/000001.bin -> .../training/label_2/000001.txt
        path_obj = Path(point_cloud_path)
        frame_id = path_obj.stem  # Get filename without extension

        # Try to find label_2 and calib directories
        parent_dir = path_obj.parent.parent  # Go up from velodyne to training
        label_path = parent_dir / "label_2" / f"{frame_id}.txt"
        calib_path = parent_dir / "calib" / f"{frame_id}.txt"

        if not label_path.exists():
            return []

        # Load calibration data
        calib = None
        if calib_path.exists():
            calib = load_kitti_calib(str(calib_path))
            if not calib:
                print(f"Warning: Failed to load calibration from {calib_path}")

        # Load and parse labels
        labels = load_kitti_label(str(label_path))

        # Create bounding boxes with proper coordinate transformation
        bboxes = []
        for label in labels:
            if label['type'] != 'DontCare':  # Skip DontCare objects
                try:
                    bbox = create_3d_bbox(label, calib)
                    bboxes.append(bbox)
                except Exception as e:
                    print(f"Warning: Failed to create bbox for {label['type']}: {e}")
                    continue

        return bboxes

    def numpy_to_o3d(self, numpy_cloud):
        """Convert numpy point cloud to Open3D format with appropriate coloring.

        Args:
            numpy_cloud: Numpy array containing point cloud data

        Returns:
            True if successful, False otherwise
        """
        if numpy_cloud is None or len(numpy_cloud) == 0:
            print("Warning: Empty point cloud data")
            return False

        try:
            if self.pointcloud_type == PointcloudType.LIDAR or self.pointcloud_type == PointcloudType.KITTI:
                # Isolate the intensity and compute a color for it
                intensity = numpy_cloud[:, -1]

                # Ensure intensity is positive and non-zero to avoid log errors
                intensity = np.maximum(intensity, INTENSITY_MIN)

                # Compute color based on intensity
                intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(INTENSITY_SCALE))
                intensity_col = np.clip(intensity_col, 0.0, 1.0)  # Clamp to valid range

                int_color = np.c_[
                    np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                    np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                    np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

                # Isolate the 3D data
                points = numpy_cloud[:, 0:3]
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(int_color)
                return True

            elif self.pointcloud_type == PointcloudType.SEMANTIC_LIDAR:
                # Read points
                points = np.array([numpy_cloud['x'], numpy_cloud['y'], numpy_cloud['z']]).T

                # Colorize the pointcloud based on the CityScapes color palette
                labels = np.array(numpy_cloud['ObjTag'])

                # Validate label indices to prevent out-of-bounds errors
                labels = np.clip(labels, 0, len(LABEL_COLORS) - 1)

                int_color = LABEL_COLORS[labels]

                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(int_color)
                return True

            elif self.pointcloud_type == PointcloudType.RADAR:
                self.pcd.points = o3d.utility.Vector3dVector(numpy_cloud[:, 0:3])
                return True

            else:
                print(f"Unknown pointcloud type: {self.pointcloud_type}")
                return False

        except Exception as e:
            print(f"Error converting numpy cloud to Open3D: {e}")
            return False

    def add_open3d_axis(self, vis):
        """Add a small 3D axis on Open3D Visualizer"""
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        axis.lines = o3d.utility.Vector2iVector(np.array([
            [0, 1],
            [0, 2],
            [0, 3]]))
        axis.colors = o3d.utility.Vector3dVector(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        vis.add_geometry(axis)

def main():
    """Main entry point for LiDAR visualization tool."""
    argparser = argparse.ArgumentParser(
        description='Visualize LiDAR point clouds with optional 3D bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize raw CARLA lidar data (single file)
  python visualize_lidar.py --type lidar --source raw_data/record_XXX/vehicle/000001_lidar.ply

  # Visualize KITTI format with 3D bboxes (single file)
  python visualize_lidar.py --type kitti --source dataset/record_XXX/vehicle/kitti_object/training/velodyne/000001.bin

  # Visualize KITTI format directory (animation with bboxes)
  python visualize_lidar.py --type kitti --source dataset/record_XXX/vehicle/kitti_object/training/velodyne
        """
    )
    argparser.add_argument(
        '--type',
        default='lidar',
        choices=['lidar', 'semantic_lidar', 'radar', 'kitti'],
        help='Type of point cloud: lidar (raw CARLA), semantic_lidar, radar, or kitti (with 3D labels)'
    )
    argparser.add_argument(
        '--source',
        type=str,
        required=True,
        help='File (.ply/.bin) or folder containing point clouds to visualize'
    )

    args = argparser.parse_args()

    # Map type string to enum
    type_mapping = {
        'lidar': PointcloudType.LIDAR,
        'semantic_lidar': PointcloudType.SEMANTIC_LIDAR,
        'radar': PointcloudType.RADAR,
        'kitti': PointcloudType.KITTI
    }

    pointcloud_type = type_mapping.get(args.type)
    if pointcloud_type is None:
        print(f"Error: Invalid point cloud type '{args.type}'")
        print(f"Valid types: {', '.join(type_mapping.keys())}")
        sys.exit(1)

    # Resolve source path
    source = args.source
    if not os.path.exists(source):
        # Try relative to ROOT_PATH
        alternative_source = os.path.join(ROOT_PATH, source)
        if os.path.exists(alternative_source):
            source = alternative_source
        else:
            print(f"Error: File or folder not found: {source}")
            sys.exit(1)

    print(f"Point cloud type: {args.type}")
    print(f"Reading data from: {source}")

    try:
        lidar_visualizer = LidarVisualizer(pointcloud_type, source)
        lidar_visualizer.visualize()
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # execute only if run as a script
    main()
