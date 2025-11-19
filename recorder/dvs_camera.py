#!/usr/bin/python3

import csv
import struct
import logging
import numpy as np
import carla

from recorder.sensor import Sensor

# Get logger instance
logger = logging.getLogger(__name__)


class DVSCamera(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self._first_frame = True

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save DVS (Dynamic Vision Sensor) camera data to disk using single CSV format

        Args:
            save_dir: Directory to save data
            sensor_data: DVS camera sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str, 'event_count': int}
        """
        # Parse DVS events from CARLA sensor data
        events = self._parse_dvs_events(sensor_data)

        if len(events) == 0:
            # No events in this frame
            return {
                'success': True,
                'file': None,
                'event_count': 0,
                'message': 'No events detected in this frame'
            }

        # Save events as CSV format only (x, y, timestamp, polarity)
        filename = "{:0>10d}.csv".format(sensor_data.frame)
        filepath = "{}/{}".format(save_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'timestamp', 'polarity'])
            writer.writerows(events)

        # Save metadata on first frame
        if self._first_frame:
            self.save_dvs_metadata(save_dir)
            self._first_frame = False

        return {
            'success': True,
            'file': filename,
            'event_count': len(events),
            'event_density': len(events) / (sensor_data.width * sensor_data.height)
        }

    def _parse_dvs_events(self, sensor_data):
        """
        Parse DVS events from CARLA sensor data using proper carla.DVSEventArray

        CARLA DVS sensor provides events as carla.DVSEventArray object
        Each event contains: x, y, timestamp, polarity
        """
        try:
            # Access events directly from CARLA's DVSEventArray
            events = []

            # CARLA provides iterator over events in DVSEventArray
            for event in sensor_data:
                # Extract event information
                x = event.x
                y = event.y
                timestamp = event.t
                polarity = event.polarity  # 1 for positive (ON), 0 for negative (OFF)

                # Validate event coordinates
                if 0 <= x < sensor_data.width and 0 <= y < sensor_data.height:
                    events.append([x, y, timestamp, polarity])

            return events

        except AttributeError:
            # Fallback: Try to parse from raw_data if event iteration fails
            logger.warning("DVS event iteration failed, attempting raw data parsing")
            return self._parse_dvs_events_from_raw(sensor_data)
        except Exception as e:
            # If parsing fails, return empty events
            logger.error(f"Failed to parse DVS events: {e}")
            return []

    def _parse_dvs_events_from_raw(self, sensor_data):
        """
        Fallback method to parse DVS events from raw data buffer
        This method attempts to decode events directly from raw_data
        """
        try:
            # CARLA DVS raw_data structure for DVSEventArray
            # Format: [num_events][event1_x][event1_y][event1_t][event1_pol][event2_x][event2_y][event2_t][event2_pol]...
            # Each event: x (uint16, 2 bytes), y (uint16, 2 bytes), t (float32, 4 bytes), pol (uint8, 1 byte)
            # Total: 9 bytes per event + 4 bytes for num_events header

            raw_data = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)

            # Check minimum size (header + at least one event)
            if len(raw_data) < 13:  # 4 bytes header + 9 bytes for one event
                return []

            # Read number of events (first 4 bytes, little endian)
            num_events = int.from_bytes(raw_data[0:4], byteorder='little', signed=False)

            # Validate event count against buffer size
            expected_size = 4 + num_events * 9
            if len(raw_data) < expected_size:
                logger.warning(f"DVS data size mismatch: expected {expected_size} bytes, got {len(raw_data)}")
                num_events = min(num_events, (len(raw_data) - 4) // 9)

            events = []
            offset = 4  # Skip header

            for i in range(num_events):
                if offset + 9 <= len(raw_data):
                    # Extract x coordinate (uint16, little endian)
                    x = int.from_bytes(raw_data[offset:offset+2], byteorder='little', signed=False)

                    # Extract y coordinate (uint16, little endian)
                    y = int.from_bytes(raw_data[offset+2:offset+4], byteorder='little', signed=False)

                    # Extract timestamp (float32, little endian)
                    timestamp_bytes = raw_data[offset+4:offset+8]
                    timestamp = struct.unpack('<f', timestamp_bytes)[0]

                    # Extract polarity (uint8)
                    polarity = raw_data[offset+8]

                    # Validate event coordinates
                    if 0 <= x < sensor_data.width and 0 <= y < sensor_data.height:
                        events.append([x, y, float(timestamp), int(polarity)])

                    offset += 9
                else:
                    break

            return events

        except Exception as e:
            logger.error(f"Failed to parse DVS events from raw data: {e}")
            return []

  
    def save_dvs_metadata(self, save_dir):
        """Save DVS camera metadata"""
        import json

        metadata = {
            'sensor_type': 'sensor.camera.dvs',
            'attributes': dict(self.carla_actor.attributes),
            'data_format': 'csv_numpy',  # CSV primary, numpy for processing
            'event_structure': {
                'description': 'DVS events with x, y coordinates, timestamp, and polarity',
                'coordinates': 'pixel coordinates (0-based)',
                'timestamp': 'simulation time in seconds',
                'polarity': '0=OFF event, 1=ON event'
            },
            'encoding': 'CARLA_DVS_native',
            'processing_notes': [
                'DVS cameras detect brightness changes',
                'Events are asynchronous and sparse',
                'High temporal resolution with low bandwidth'
            ]
        }

        # Add DVS-specific attributes if available
        dvs_attributes = ['refractory_period', 'threshold_low', 'threshold_high', 'sigma_threshold']
        for attr in dvs_attributes:
            if attr in self.carla_actor.attributes:
                metadata['attributes'][attr] = self.carla_actor.attributes[attr]

        with open('{}/sensor_metadata.json'.format(save_dir), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)