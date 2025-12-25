#!/usr/bin/python3

import carla

from recorder.sensor import Sensor
from core.csv_utils import safe_append_to_csv


class IMU(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        # IMU uses separate CSV for 6-axis data with frame and timestamp
        self.imu_fieldnames = ['frame', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'compass']
        self._first_frame = True

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save IMU data to disk using single CSV file with append mode
        Similar to poses.csv, each frame adds one row to the same file

        Args:
            save_dir: Directory to save data
            sensor_data: IMU sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str, 'data': dict}
        """
        # Extract 6-axis IMU data with frame and timestamp: accelerometer + gyroscope + compass
        data = {
            'frame': sensor_data.frame,
            'timestamp': sensor_data.timestamp,
            'acc_x': sensor_data.accelerometer.x,
            'acc_y': sensor_data.accelerometer.y,
            'acc_z': sensor_data.accelerometer.z,
            'gyro_x': sensor_data.gyroscope.x,
            'gyro_y': sensor_data.gyroscope.y,
            'gyro_z': sensor_data.gyroscope.z,
            'compass': sensor_data.compass
        }

        # Append to single CSV file using unified CSV utility
        csv_path = '{}/imu_data.csv'.format(save_dir)

        result = safe_append_to_csv(
            csv_path=csv_path,
            fieldnames=self.imu_fieldnames,
            data=data,
            is_first_write=self._first_frame,
            logger_name=f"{self.__class__.__name__}_{self.uid}"
        )

        # Save metadata on first frame
        if result['success'] and self._first_frame:
            self.save_imu_metadata(save_dir)
            self._first_frame = False

        # Include sensor data in result
        result['data'] = data

        return result

    def save_imu_metadata(self, save_dir):
        """Save IMU sensor metadata"""
        import json

        metadata = {
            'sensor_type': 'sensor.other.imu',
            'attributes': dict(self.carla_actor.attributes),
            'data_format': 'csv_unified',
            'data_fields': {
                'frame': 'CARLA frame number',
                'timestamp': 'Simulation time in seconds',
                'acc_x': 'm/s^2 - Linear acceleration (X axis)',
                'acc_y': 'm/s^2 - Linear acceleration (Y axis)',
                'acc_z': 'm/s^2 - Linear acceleration (Z axis)',
                'gyro_x': 'rad/s - Angular velocity (X axis)',
                'gyro_y': 'rad/s - Angular velocity (Y axis)',
                'gyro_z': 'rad/s - Angular velocity (Z axis)',
                'compass': 'rad - Orientation (North=0, East=π/2)'
            },
            'data_structure': 'single_file_append_mode',
            'file_name': 'imu_data.csv',
            'coordinate_system': 'CARLA_coordinate_system',
            'units': 'SI_units'
        }

        # Add noise model attributes if available
        noise_attributes = [
            'noise_accel_stddev_x', 'noise_accel_stddev_y', 'noise_accel_stddev_z',
            'noise_gyro_bias_x', 'noise_gyro_bias_y', 'noise_gyro_bias_z',
            'noise_gyro_stddev_x', 'noise_gyro_stddev_y', 'noise_gyro_stddev_z',
            'noise_seed'
        ]

        for attr in noise_attributes:
            if attr in self.carla_actor.attributes:
                metadata['attributes'][attr] = self.carla_actor.attributes[attr]

        with open('{}/sensor_metadata.json'.format(save_dir), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)