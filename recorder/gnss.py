#!/usr/bin/python3

import carla

from recorder.sensor import Sensor
from core.csv_utils import safe_append_to_csv


class GNSS(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        # GNSS uses separate CSV for data, with fieldnames for GPS coordinates
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

        # Append to single CSV file using unified CSV utility
        csv_path = '{}/gnss_data.csv'.format(save_dir)

        result = safe_append_to_csv(
            csv_path=csv_path,
            fieldnames=self.gnss_fieldnames,
            data=data,
            is_first_write=self._first_frame,
            logger_name=f"{self.__class__.__name__}_{self.uid}"
        )

        # Save metadata on first frame
        if result['success'] and self._first_frame:
            self.save_gnss_metadata(save_dir)
            self._first_frame = False

        # Include sensor data in result
        result['data'] = data

        return result

    def save_gnss_metadata(self, save_dir):
        """Save GNSS sensor metadata"""
        import json

        # Get CARLA actor ID
        carla_actor_id = self.get_actor_id()

        metadata = {
            'sensor_type': 'sensor.other.gnss',
            'sensor_id': self.name,
            'carla_actor_id': carla_actor_id,
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