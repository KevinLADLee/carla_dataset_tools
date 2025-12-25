#!/usr/bin/python3

import logging
import numpy as np
import carla

from recorder.sensor import Sensor

# Get logger instance
logger = logging.getLogger(__name__)


class DVSCamera(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save DVS (Dynamic Vision Sensor) camera data to disk using NPY format

        Args:
            save_dir: Directory to save data
            sensor_data: DVS camera sensor data from CARLA

        Returns:
            dict: {'success': bool, 'file': str, 'event_count': int}
        """
        # Parse events using the same method as manual_control.py
        events_array = self._parse_dvs_events(sensor_data)

        if len(events_array) == 0:
            return {
                'success': True,
                'file': None,
                'event_count': 0,
                'message': 'No events detected in this frame'
            }

        # Save as NPY format (direct numpy array save)
        filename = "{:0>10d}.npy".format(sensor_data.frame)
        filepath = "{}/{}".format(save_dir, filename)

        try:
            np.save(filepath, events_array)
        except Exception as e:
            logger.error(f"Failed to save NPY file: {e}")
            return {
                'success': False,
                'file': None,
                'event_count': 0,
                'error': str(e)
            }

        # Save metadata on first frame
        if self.is_first_frame():
            self._save_metadata(save_dir, sensor_data)

        return {
            'success': True,
            'file': filename,
            'event_count': len(events_array),
            'event_density': len(events_array) / (sensor_data.width * sensor_data.height)
        }

    def _parse_dvs_events(self, sensor_data):
        """
        Parse DVS events using the exact same method as manual_control.py
        Based on CARLA 0.9.16 official implementation
        """
        try:
            # CARLA manual_control.py line 1194-1195 implementation
            events_array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', bool)]))
            return events_array
        except Exception as e:
            logger.error(f"Failed to parse DVS events: {e}")
            return np.array([])

    def _save_metadata(self, save_dir, sensor_data):
        """Save DVS camera metadata"""
        import json

        metadata = {
            'sensor_type': 'sensor.camera.dvs',
            'attributes': dict(self.carla_actor.attributes),
            'data_format': 'npy',
            'image_size': [sensor_data.width, sensor_data.height],
            'array_structure': {
                'description': 'NumPy structured array with DVS events',
                'dtype': [('x', 'uint16'), ('y', 'uint16'), ('t', 'int64'), ('pol', 'bool')],
                'coordinates': 'pixel coordinates (0-based)',
                'timestamp': 'simulation time (int64 nanoseconds)',
                'polarity': 'False=OFF event, True=ON event'
            },
            'processing_notes': [
                'DVS cameras detect brightness changes asynchronously',
                'Events are sparse and event-driven',
                'High temporal resolution with low bandwidth',
                'Data saved in NumPy binary format for optimal performance',
                'Parsing based on CARLA manual_control.py official implementation'
            ]
        }

        self.save_sensor_metadata(save_dir, additional_metadata=metadata)