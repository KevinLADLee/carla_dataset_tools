#!/usr/bin/python3

import csv
import carla
import numpy as np

from recorder.sensor import Sensor


class Radar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self.csv_fieldnames = ['frame', 'timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
        self._first_frame = True

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save radar data to disk using single CSV format

        Args:
            save_dir: Directory to save data
            sensor_data: Radar sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str, 'detection_count': int}
        """
        # Process radar detection points
        radar_points = []
        for detection in sensor_data:
            # Convert spherical to Cartesian coordinates
            x = detection.depth * np.cos(detection.azimuth) * np.cos(-detection.altitude)
            y = detection.depth * np.sin(-detection.azimuth) * np.cos(detection.altitude)
            z = detection.depth * np.sin(detection.altitude)

            radar_points.append([
                x, y, z,  # Cartesian coordinates
                detection.depth,    # Original depth
                detection.velocity, # Radial velocity
                detection.azimuth,  # Azimuth angle
                detection.altitude  # Altitude angle
            ])

        # Save as CSV format only (structured radar data)
        filename = "{:0>10d}.csv".format(sensor_data.frame)
        filepath = "{}/{}".format(save_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'z', 'depth', 'velocity', 'azimuth', 'altitude'])
            writer.writerows(radar_points)

        # Save metadata on first frame
        if self._first_frame:
            self.save_radar_metadata(save_dir)
            self._first_frame = False

        return {
            'success': True,
            'file': filename,
            'detection_count': len(radar_points)
        }

    def save_radar_metadata(self, save_dir):
        """Save radar sensor metadata"""
        # Get CARLA actor ID
        carla_actor_id = self.get_actor_id()
        
        metadata = {
            'sensor_type': 'sensor.other.radar',
            'sensor_id': self.name,
            'carla_actor_id': carla_actor_id,
            'attributes': dict(self.carla_actor.attributes),
            'horizontal_fov': float(self.carla_actor.attributes.get('horizontal_fov', 30.0)),
            'vertical_fov': float(self.carla_actor.attributes.get('vertical_fov', 30.0)),
            'points_per_second': int(self.carla_actor.attributes.get('points_per_second', 1500)),
            'range': float(self.carla_actor.attributes.get('range', 100.0)),
            'data_format': 'csv'  # Single CSV format for structured radar data
        }

        import json
        with open('{}/sensor_metadata.json'.format(save_dir), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

