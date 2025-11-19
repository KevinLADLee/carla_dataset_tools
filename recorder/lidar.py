#!/usr/bin/python3

import math
import carla
import numpy as np
import transforms3d
from plyfile import PlyData, PlyElement

from recorder.sensor import Sensor
from core.geometry import Rotation
from core.transform import Transform, carla_transform_to_transform


class Lidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save lidar point cloud to disk

        Args:
            save_dir: Directory to save data
            sensor_data: Lidar sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str, 'points_count': int}
        """
        # Save as a Nx4 numpy array. Each row is a point (x, y, z, intensity)
        lidar_data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))

        # Convert point cloud to right-hand coordinate system
        # Negate y-axis to match your original coordinate convention
        lidar_data[:, 1] *= -1

        # Generate filename using absolute frame ID from sensor_data
        filename = "{:0>10d}.ply".format(sensor_data.frame)
        filepath = "{}/{}".format(save_dir, filename)

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].ply

        # Create structured array for PLY format
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
        structured_array = np.zeros(len(lidar_data), dtype=dtype)
        structured_array['x'] = lidar_data[:, 0]
        structured_array['y'] = lidar_data[:, 1]
        structured_array['z'] = lidar_data[:, 2]
        structured_array['intensity'] = lidar_data[:, 3]

        # Save PLY file
        vertex = PlyElement.describe(structured_array, 'vertex')
        PlyData([vertex]).write(filepath)

        # Save metadata on first frame
        if self.is_first_frame():
            self.save_lidar_metadata(self.save_dir)

        return {
            'success': True,
            'file': filename,
            'points_count': lidar_data.shape[0]
        }

    def save_lidar_metadata(self, save_dir):
        """Save lidar sensor metadata"""
        additional_metadata = {
            'data_format': {
                'file_format': 'ply',  # PLY format
                'point_format': 'xyz_intensity',
                'data_type': 'float32',
                'coordinate_system': 'right_handed_y_negated'
            },
            'lidar_config': {
                'channels': int(self.carla_actor.attributes.get('channels', 64)),
                'range': float(self.carla_actor.attributes.get('range', 100.0)),
                'points_per_second': int(self.carla_actor.attributes.get('points_per_second', 130000)),
                'rotation_frequency': float(self.carla_actor.attributes.get('rotation_frequency', 10.0)),
                'upper_fov': float(self.carla_actor.attributes.get('upper_fov', 10.0)),
                'lower_fov': float(self.carla_actor.attributes.get('lower_fov', -30.0))
            },
            'sensor_category': 'lidar'
        }

        self.save_sensor_metadata(save_dir, additional_metadata)

    def get_transform(self) -> Transform:
        """
        Override get_transform to ensure pose angles are in degrees.
        This ensures consistency with camera sensors and maintains
        compatibility with the KITTI label generation pipeline.

        Returns:
            Transform: Sensor transform with angles in degrees
        """
        c_trans = self.carla_actor.get_transform()
        trans = carla_transform_to_transform(c_trans)

        # Get quaternion and convert back to euler angles (in radians)
        quat = trans.rotation.get_quaternion()
        roll, pitch, yaw = transforms3d.euler.quat2euler(quat)

        # Explicitly convert to degrees to ensure consistency
        return Transform(trans.location, Rotation(
            roll=math.degrees(roll),
            pitch=math.degrees(pitch),
            yaw=math.degrees(yaw)
        ))


class SemanticLidar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save semantic lidar point cloud to disk

        Args:
            save_dir: Directory to save data
            sensor_data: Semantic lidar sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str, 'points_count': int}
        """
        # Save data as a Nx6 numpy array.
        lidar_data = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)])))

        # Convert point cloud to right-hand coordinate system
        # Negate y-axis to match Open3D/ROS conventions (see open3d_lidar.py line 98)
        lidar_data['y'] *= -1

        # Generate filename using absolute frame ID from sensor_data
        filename = "{:0>10d}.ply".format(sensor_data.frame)
        filepath = "{}/{}".format(save_dir, filename)

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].ply

        # Create structured array for PLY format (preserve all semantic fields)
        dtype = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('cos_angle', 'f4'), ('obj_idx', 'i4'), ('obj_tag', 'i4')
        ]
        structured_array = np.zeros(len(lidar_data), dtype=dtype)
        structured_array['x'] = lidar_data['x']
        structured_array['y'] = lidar_data['y']
        structured_array['z'] = lidar_data['z']
        structured_array['cos_angle'] = lidar_data['CosAngle']
        structured_array['obj_idx'] = lidar_data['ObjIdx']
        structured_array['obj_tag'] = lidar_data['ObjTag']

        # Save PLY file
        vertex = PlyElement.describe(structured_array, 'vertex')
        PlyData([vertex]).write(filepath)

        return {
            'success': True,
            'file': filename,
            'points_count': lidar_data.shape[0]
        }

    def get_transform(self) -> Transform:
        """
        Override get_transform to ensure pose angles are in degrees.
        This ensures consistency with camera sensors and maintains
        compatibility with the KITTI label generation pipeline.

        Returns:
            Transform: Sensor transform with angles in degrees
        """
        c_trans = self.carla_actor.get_transform()
        trans = carla_transform_to_transform(c_trans)

        # Get quaternion and convert back to euler angles (in radians)
        quat = trans.rotation.get_quaternion()
        roll, pitch, yaw = transforms3d.euler.quat2euler(quat)

        # Explicitly convert to degrees to ensure consistency
        return Transform(trans.location, Rotation(
            roll=math.degrees(roll),
            pitch=math.degrees(pitch),
            yaw=math.degrees(yaw)
        ))
