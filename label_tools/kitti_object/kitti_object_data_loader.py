#!/usr/bin/python3
import csv
import glob
import os.path
import sys

import cv2
import numpy as np
from pathlib import Path
from plyfile import PlyData

sys.path.append(Path(__file__).parent.parent.as_posix())
from core.transform import *


def load_lidar_data(path: str):
    """Load lidar data and poses, return as a list of dictionaries."""
    lidar_rawdata_path_list = sorted(glob.glob(f"{path}/*.ply"))
    lidar_rawdata_list = []

    # Read poses from CSV
    lidar_poses = {}
    with open(f"{path}/poses.csv", 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            frame = int(row['frame'])
            lidar_poses[frame] = {
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row['z']),
                'yaw': float(row['yaw']),
                'roll': float(row['roll']),
                'pitch': float(row['pitch'])
            }

    for lidar_rawdata_path in lidar_rawdata_path_list:
        frame = get_frame_from_fullpath(lidar_rawdata_path)
        pose_data = lidar_poses.get(frame)
        if pose_data is None:
            continue

        lidar_pose = Transform(
            Location(pose_data['x'], pose_data['y'], pose_data['z']),
            Rotation(yaw=pose_data['yaw'], roll=pose_data['roll'], pitch=pose_data['pitch'])
        )

        lidar_rawdata_list.append({
            'frame': frame,
            'lidar_rawdata_path': lidar_rawdata_path,
            'lidar_pose': lidar_pose
        })

    return lidar_rawdata_list


def load_camera_data(path: str):
    """Load camera data, poses and camera matrix, return as a list of dictionaries."""
    camera_rawdata_list = sorted(glob.glob(f"{path}/*.png"))
    camera_rawdata_result = []

    # Read camera poses from CSV
    camera_poses = {}
    with open(f"{path}/poses.csv", 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            frame = int(row['frame'])
            camera_poses[frame] = {
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row['z']),
                'yaw': float(row['yaw']),
                'roll': float(row['roll']),
                'pitch': float(row['pitch'])
            }

    # Read camera info
    with open(f"{path}/camera_info.csv", 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        camera_info = next(reader)
        camera_matrix = np.array([[float(camera_info["fx"]), 0.0, float(camera_info["cx"])],
                                  [0.0, float(camera_info["fy"]), float(camera_info["cy"])],
                                  [0.0, 0.0, 1.0]])

    for camera_rawdata_path in camera_rawdata_list:
        frame = get_frame_from_fullpath(camera_rawdata_path)
        pose_data = camera_poses.get(frame)
        if pose_data is None:
            continue

        camera_pose = Transform(
            Location(pose_data['x'], pose_data['y'], pose_data['z']),
            Rotation(yaw=pose_data['yaw'], roll=pose_data['roll'], pitch=pose_data['pitch'])
        )

        camera_rawdata_result.append({
            'frame': frame,
            'camera_pose': camera_pose,
            'camera_matrix': camera_matrix,
            'camera_rawdata_path': camera_rawdata_path
        })

    return camera_rawdata_result


def get_frame_from_fullpath(path: str) -> int:
    return int(os.path.splitext(os.path.split(path)[-1])[0])


def load_object_labels(path: str):
    """Load object label paths, return as a list of dictionaries."""
    object_labels_path_list = sorted(glob.glob("{}/*.pkl".format(path)))
    object_labels_list = []

    for objects_labels_rawdata_path in object_labels_path_list:
        frame = get_frame_from_fullpath(objects_labels_rawdata_path)
        object_labels_list.append({
            'frame': frame,
            'object_labels_path': objects_labels_rawdata_path
        })

    return object_labels_list


def load_vehicle_pose(path: str) -> list:
    """Load vehicle poses from CSV, return as a list of dictionaries."""
    vehicle_status_list = []

    with open("{}/vehicle_status.csv".format(path), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            pose = Transform(
                Location(float(row['x']), float(row['y']), float(row['z'])),
                Rotation(roll=float(row['roll']), yaw=float(row['yaw']), pitch=float(row['pitch']))
            )
            vehicle_status_list.append({
                'frame': int(row['frame']),
                'timestamp': float(row['timestamp']),
                'vehicle_pose': pose
            })

    return vehicle_status_list


def read_pointcloud(path: str) -> np.array:
    """Read point cloud data from PLY file.

    Args:
        path: Path to .ply file

    Returns:
        Numpy array with point cloud data.
        For regular lidar: (N, 4) with columns [x, y, z, intensity]
        For semantic lidar: structured array with fields [x, y, z, CosAngle, ObjIdx, ObjTag]
    """
    ply_data = PlyData.read(path)
    vertex = ply_data['vertex']
    vertex_props = [prop.name for prop in vertex.properties]

    if 'intensity' in vertex_props:
        # Regular LiDAR format
        pointcloud = np.column_stack([
            vertex['x'],
            vertex['y'],
            vertex['z'],
            vertex['intensity']
        ]).astype(np.float32)
    elif 'cos_angle' in vertex_props:
        # Semantic LiDAR format
        n_points = len(vertex)
        dtype = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('CosAngle', np.float32),
            ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)
        ]
        pointcloud = np.zeros(n_points, dtype=dtype)
        pointcloud['x'] = vertex['x']
        pointcloud['y'] = vertex['y']
        pointcloud['z'] = vertex['z']
        pointcloud['CosAngle'] = vertex['cos_angle']
        pointcloud['ObjIdx'] = vertex['obj_idx']
        pointcloud['ObjTag'] = vertex['obj_tag']
    else:
        raise ValueError(f"Unknown PLY format. Properties: {vertex_props}")

    return pointcloud


def read_image(path: str) -> np.array:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image