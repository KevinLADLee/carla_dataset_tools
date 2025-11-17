#!/usr/bin/python3

import math
import carla
import numpy as np
import transforms3d

from recorder.sensor import Sensor
from core.geometry import Rotation, Location
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
        filename = "{:0>10d}.npy".format(sensor_data.frame)
        filepath = "{}/{}".format(save_dir, filename)

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy
        np.save(filepath, lidar_data)

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
        filename = "{:0>10d}.npy".format(sensor_data.frame)
        filepath = "{}/{}".format(save_dir, filename)

        # Save point cloud to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].npy
        np.save(filepath, lidar_data)

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
