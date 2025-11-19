#!/usr/bin/python3

import csv
import os
import carla

from recorder.sensor import Sensor


class GNSS(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        # GNSS uses separate CSV for data, so we don't override poses
        self.gnss_fieldnames = ['frame', 'timestamp', 'latitude', 'longitude', 'altitude']
        self._first_frame = True

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save GNSS data to disk using single CSV file with append mode
        Similar to poses.csv, each frame adds one row to the same file

        Args:
            save_dir: Directory to save data
            sensor_data: GNSS sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str, 'data': dict}
        """
        # Extract GPS coordinates with frame and timestamp
        data = {
            'frame': sensor_data.frame,
            'timestamp': sensor_data.timestamp,
            'latitude': sensor_data.latitude,
            'longitude': sensor_data.longitude,
            'altitude': sensor_data.altitude
        }

        # Append to single CSV file (similar to poses.csv approach)
        csv_path = '{}/gnss_data.csv'.format(save_dir)

        if self._first_frame:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.gnss_fieldnames)
                writer.writeheader()
                writer.writerow(data)
            self.save_gnss_metadata(save_dir)
            self._first_frame = False
        else:
            with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.gnss_fieldnames)
                writer.writerow(data)

        # Save metadata on first frame (moved to be called with initialization)

        return {
            'success': True,
            'file': 'gnss_data.csv',
            'data': data
        }

    def save_gnss_metadata(self, save_dir):
        """Save GNSS sensor metadata"""
        import json

        metadata = {
            'sensor_type': 'sensor.other.gnss',
            'attributes': dict(self.carla_actor.attributes),
            'data_format': 'csv_unified',
            'data_fields': {
                'frame': 'CARLA frame number',
                'timestamp': 'Simulation time in seconds',
                'latitude': 'degrees - Geographic latitude',
                'longitude': 'degrees - Geographic longitude',
                'altitude': 'meters - Altitude above sea level'
            },
            'data_structure': 'single_file_append_mode',
            'file_name': 'gnss_data.csv',
            'coordinate_system': 'WGS84_geographic',
            'reference': 'OpenDRIVE_map_georeference'
        }

        # Add noise model attributes if available
        noise_attributes = [
            'noise_alt_bias', 'noise_alt_stddev',
            'noise_lat_bias', 'noise_lat_stddev',
            'noise_lon_bias', 'noise_lon_stddev',
            'noise_seed'
        ]

        for attr in noise_attributes:
            if attr in self.carla_actor.attributes:
                metadata['attributes'][attr] = self.carla_actor.attributes[attr]

        with open('{}/sensor_metadata.json'.format(save_dir), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)