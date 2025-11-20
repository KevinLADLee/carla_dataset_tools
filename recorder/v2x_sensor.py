#!/usr/bin/python3
"""
V2X Sensor implementation for CARLA Dataset Tools

Supports two V2X sensor types:
- sensor.other.v2x: CAM (Cooperative Awareness Message) sensor following ETSI standard
- sensor.other.v2x_custom: Custom V2X message sensor for arbitrary string messages

V2X sensors simulate vehicle-to-everything (V2X) communication with realistic
wireless channel models including path loss, fading, and noise parameters.
"""

import json
import os
import queue
import carla

from recorder.sensor import Sensor
from core.csv_utils import safe_append_to_csv


class V2XSensor(Sensor):
    """
    V2X CAM (Cooperative Awareness Message) Sensor

    Receives CAM messages from other vehicles in the simulation.
    CAM messages contain standardized vehicle state information following ETSI standard:
    - Position (latitude, longitude, altitude)
    - Speed, heading, acceleration, yaw rate
    - Vehicle dimensions

    Messages are automatically triggered when:
    - Heading change > 4 degrees
    - Position change > 4 meters
    - Speed change > 5 m/s
    - Time interval reached (gen_cam_min ~ gen_cam_max)
    """

    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        # Summary CSV for quick statistics
        self.summary_fieldnames = ['frame', 'timestamp', 'message_count', 'avg_power', 'min_power', 'max_power']
        self._first_frame = True

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save V2X sensor data to disk (event-driven, non-blocking)

        V2X sensors are event-driven and only produce data when messages are received.
        Unlike other sensors, they don't produce data every frame.

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Timestamp
            debug: Whether to print debug info

        Returns:
            dict: Save information
        """
        # Ensure target path exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Collect all available messages from queue (non-blocking)
        sensor_data_list = []
        while True:
            try:
                sensor_data = self.queue.get(block=False)
                sensor_data_list.append(sensor_data)
            except queue.Empty:
                break

        # Use the latest sensor_data if available, otherwise create empty event
        if sensor_data_list:
            sensor_data = sensor_data_list[-1]  # Use latest
        else:
            # No messages received - create empty result
            sensor_data = None

        # Save data
        if sensor_data is not None:
            save_result = self.save_to_disk_impl(self.save_dir, sensor_data)
        else:
            # No V2X messages received this frame - save empty result
            save_result = self._save_empty_frame(self.save_dir, frame_id, timestamp)

        if not isinstance(save_result, dict):
            success = save_result
            save_info = {}
        else:
            success = save_result.get('success', False)
            save_info = save_result

        if not success:
            raise IOError(f"Sensor {self.name} failed to save frame {frame_id}")

        # Save sensor pose
        self.save_pose(frame_id, timestamp)
        self._first_frame = False

        # Prepare return info
        pose = self.get_transform()
        result = {
            'type': 'sensor',
            'name': self.name,
            'sensor_type': self.sensor_type,
            'pose': pose.to_dict(),
            'timestamp': timestamp
        }
        result.update(save_info)

        return result

    def _save_empty_frame(self, save_dir, frame_id, timestamp):
        """
        Save empty V2X frame when no messages are received

        Args:
            save_dir: Directory to save data
            frame_id: Frame ID
            timestamp: Timestamp

        Returns:
            dict: Save result
        """
        # Prepare empty JSON output
        output_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'messages': [],
            'message_count': 0
        }

        # Save JSON file
        filename = '{:0>10d}.json'.format(frame_id)
        filepath = '{}/{}'.format(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Save summary statistics
        summary_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'message_count': 0,
            'avg_power': 0.0,
            'min_power': 0.0,
            'max_power': 0.0
        }

        csv_path = '{}/v2x_summary.csv'.format(save_dir)
        safe_append_to_csv(
            csv_path=csv_path,
            fieldnames=self.summary_fieldnames,
            data=summary_data,
            is_first_write=self._first_frame,
            logger_name=f"{self.__class__.__name__}_{self.uid}"
        )

        # Save metadata on first frame
        if self._first_frame:
            self.save_v2x_metadata(save_dir)

        return {
            'success': True,
            'file': filename,
            'message_count': 0
        }

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save V2X CAM data to disk

        Args:
            save_dir: Directory to save data
            sensor_data: CAMEvent from CARLA containing multiple CAMData messages

        Returns:
            dict: {'success': bool, 'file': str, 'message_count': int}
        """
        frame_id = sensor_data.frame
        timestamp = sensor_data.timestamp

        # Extract all CAM messages from the event
        messages = []
        powers = []

        message_count = sensor_data.get_message_count()

        for cam_data in sensor_data:
            power = cam_data.power
            powers.append(power)

            # Get CAM message content (Header + Message dict)
            cam_content = cam_data.get()

            messages.append({
                'power': power,
                'header': cam_content.get('Header', {}),
                'message': cam_content.get('Message', {})
            })

        # Prepare JSON output
        output_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'messages': messages,
            'message_count': message_count
        }

        # Save JSON file for this frame
        filename = '{:0>10d}.json'.format(frame_id)
        filepath = '{}/{}'.format(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Save summary statistics to CSV
        if message_count > 0:
            avg_power = sum(powers) / len(powers)
            min_power = min(powers)
            max_power = max(powers)
        else:
            avg_power = min_power = max_power = 0.0

        summary_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'message_count': message_count,
            'avg_power': avg_power,
            'min_power': min_power,
            'max_power': max_power
        }

        csv_path = '{}/v2x_summary.csv'.format(save_dir)
        safe_append_to_csv(
            csv_path=csv_path,
            fieldnames=self.summary_fieldnames,
            data=summary_data,
            is_first_write=self._first_frame,
            logger_name=f"{self.__class__.__name__}_{self.uid}"
        )

        # Save metadata on first frame
        if self._first_frame:
            self.save_v2x_metadata(save_dir)

        return {
            'success': True,
            'file': filename,
            'message_count': message_count
        }

    def save_v2x_metadata(self, save_dir):
        """Save V2X CAM sensor metadata"""
        metadata = {
            'sensor_type': 'sensor.other.v2x',
            'description': 'V2X CAM (Cooperative Awareness Message) Sensor',
            'attributes': dict(self.carla_actor.attributes),
            'data_format': {
                'file_extension': 'json',
                'structure': 'CAM messages per frame',
                'summary_file': 'v2x_summary.csv'
            },
            'message_format': {
                'header': {
                    'protocolVersion': 'int - Protocol version',
                    'messageID': 'int - Message ID',
                    'stationID': 'int - Sender station ID'
                },
                'message': {
                    'generationDeltaTime': 'int - Time since last message',
                    'basicContainer': {
                        'stationType': 'int - Vehicle type',
                        'referencePosition': {
                            'latitude': 'int - 1/10 micro degrees',
                            'longitude': 'int - 1/10 micro degrees',
                            'altitude': 'int - centimeters'
                        }
                    },
                    'highFrequencyContainer': {
                        'heading': 'int - 0.1 degrees',
                        'speed': 'int - 0.01 m/s',
                        'driveDirection': 'int - 0=forward, 1=backward',
                        'vehicleLength': 'int - 0.1 meters',
                        'vehicleWidth': 'int - 0.1 meters',
                        'longitudinalAcceleration': 'int - 0.1 m/s^2',
                        'yawRate': 'int - 0.01 degrees/s'
                    }
                }
            },
            'channel_parameters': {
                'transmit_power': 'dBm - Transmission power',
                'receiver_sensitivity': 'dBm - Minimum receivable power',
                'frequency_ghz': 'GHz - Operating frequency',
                'filter_distance': 'meters - Maximum reception range',
                'path_loss_model': 'Path loss calculation model',
                'scenario': 'Environment type (urban/rural/highway)'
            },
            'coordinate_system': 'WGS84_for_position_CARLA_for_dynamics'
        }

        with open('{}/sensor_metadata.json'.format(save_dir), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)


class CustomV2XSensor(Sensor):
    """
    V2X Custom Message Sensor

    Receives custom string messages from other V2X transmitters.
    Unlike CAM sensor, custom V2X requires explicit send() calls to transmit.

    This sensor can be used for:
    - V2I (Vehicle-to-Infrastructure) communication
    - Custom protocol implementation
    - Sensor data sharing between vehicles
    """

    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self.summary_fieldnames = ['frame', 'timestamp', 'message_count', 'avg_power', 'min_power', 'max_power']
        self._first_frame = True

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save Custom V2X sensor data to disk (event-driven, non-blocking)

        Custom V2X sensors are event-driven and only produce data when messages are received.

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Timestamp
            debug: Whether to print debug info

        Returns:
            dict: Save information
        """
        # Ensure target path exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Collect all available messages from queue (non-blocking)
        sensor_data_list = []
        while True:
            try:
                sensor_data = self.queue.get(block=False)
                sensor_data_list.append(sensor_data)
            except queue.Empty:
                break

        # Use the latest sensor_data if available
        if sensor_data_list:
            sensor_data = sensor_data_list[-1]
        else:
            sensor_data = None

        # Save data
        if sensor_data is not None:
            save_result = self.save_to_disk_impl(self.save_dir, sensor_data)
        else:
            # No V2X messages received this frame - save empty result
            save_result = self._save_empty_frame(self.save_dir, frame_id, timestamp)

        if not isinstance(save_result, dict):
            success = save_result
            save_info = {}
        else:
            success = save_result.get('success', False)
            save_info = save_result

        if not success:
            raise IOError(f"Sensor {self.name} failed to save frame {frame_id}")

        # Save sensor pose
        self.save_pose(frame_id, timestamp)
        self._first_frame = False

        # Prepare return info
        pose = self.get_transform()
        result = {
            'type': 'sensor',
            'name': self.name,
            'sensor_type': self.sensor_type,
            'pose': pose.to_dict(),
            'timestamp': timestamp
        }
        result.update(save_info)

        return result

    def _save_empty_frame(self, save_dir, frame_id, timestamp):
        """
        Save empty Custom V2X frame when no messages are received

        Args:
            save_dir: Directory to save data
            frame_id: Frame ID
            timestamp: Timestamp

        Returns:
            dict: Save result
        """
        # Prepare empty JSON output
        output_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'messages': [],
            'message_count': 0
        }

        # Save JSON file
        filename = '{:0>10d}.json'.format(frame_id)
        filepath = '{}/{}'.format(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Save summary statistics
        summary_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'message_count': 0,
            'avg_power': 0.0,
            'min_power': 0.0,
            'max_power': 0.0
        }

        csv_path = '{}/v2x_custom_summary.csv'.format(save_dir)
        safe_append_to_csv(
            csv_path=csv_path,
            fieldnames=self.summary_fieldnames,
            data=summary_data,
            is_first_write=self._first_frame,
            logger_name=f"{self.__class__.__name__}_{self.uid}"
        )

        # Save metadata on first frame
        if self._first_frame:
            self.save_custom_v2x_metadata(save_dir)

        return {
            'success': True,
            'file': filename,
            'message_count': 0
        }

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save Custom V2X data to disk

        Args:
            save_dir: Directory to save data
            sensor_data: CustomV2XEvent from CARLA containing CustomV2XData messages

        Returns:
            dict: {'success': bool, 'file': str, 'message_count': int}
        """
        frame_id = sensor_data.frame
        timestamp = sensor_data.timestamp

        # Extract all custom messages from the event
        messages = []
        powers = []

        message_count = sensor_data.get_message_count()

        for custom_data in sensor_data:
            power = custom_data.power
            powers.append(power)

            # Get custom message content (Header + string Message)
            custom_content = custom_data.get()

            messages.append({
                'power': power,
                'header': custom_content.get('Header', {}),
                'message': custom_content.get('Message', '')
            })

        # Prepare JSON output
        output_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'messages': messages,
            'message_count': message_count
        }

        # Save JSON file for this frame
        filename = '{:0>10d}.json'.format(frame_id)
        filepath = '{}/{}'.format(save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Save summary statistics to CSV
        if message_count > 0:
            avg_power = sum(powers) / len(powers)
            min_power = min(powers)
            max_power = max(powers)
        else:
            avg_power = min_power = max_power = 0.0

        summary_data = {
            'frame': frame_id,
            'timestamp': timestamp,
            'message_count': message_count,
            'avg_power': avg_power,
            'min_power': min_power,
            'max_power': max_power
        }

        csv_path = '{}/v2x_custom_summary.csv'.format(save_dir)
        safe_append_to_csv(
            csv_path=csv_path,
            fieldnames=self.summary_fieldnames,
            data=summary_data,
            is_first_write=self._first_frame,
            logger_name=f"{self.__class__.__name__}_{self.uid}"
        )

        # Save metadata on first frame
        if self._first_frame:
            self.save_custom_v2x_metadata(save_dir)

        return {
            'success': True,
            'file': filename,
            'message_count': message_count
        }

    def save_custom_v2x_metadata(self, save_dir):
        """Save Custom V2X sensor metadata"""
        metadata = {
            'sensor_type': 'sensor.other.v2x_custom',
            'description': 'V2X Custom Message Sensor',
            'attributes': dict(self.carla_actor.attributes),
            'data_format': {
                'file_extension': 'json',
                'structure': 'Custom V2X messages per frame',
                'summary_file': 'v2x_custom_summary.csv'
            },
            'message_format': {
                'header': {
                    'stationID': 'int - Sender station ID'
                },
                'message': 'string - Custom payload'
            },
            'channel_parameters': {
                'transmit_power': 'dBm - Transmission power',
                'receiver_sensitivity': 'dBm - Minimum receivable power',
                'frequency_ghz': 'GHz - Operating frequency',
                'filter_distance': 'meters - Maximum reception range',
                'path_loss_model': 'Path loss calculation model',
                'scenario': 'Environment type (urban/rural/highway)'
            },
            'notes': 'Custom V2X requires explicit send() calls to transmit messages'
        }

        with open('{}/sensor_metadata.json'.format(save_dir), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
