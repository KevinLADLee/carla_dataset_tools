#!/usr/bin/python3
import copy
import csv
import os
import weakref
import logging
import queue

import carla

from recorder.actor import Actor
from core.csv_utils import safe_append_to_csv

# Get logger instance
logger = logging.getLogger(__name__)

# Sensor queue timeout in seconds
SENSOR_QUEUE_TIMEOUT = 10.0


class Sensor(Actor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 parent,
                 carla_actor: carla.Sensor):
        super(Sensor, self).__init__(uid=uid, name=name, parent=parent, carla_actor=carla_actor)
        self.sensor_type = copy.deepcopy(self.get_type_id())
        self.save_dir = base_save_dir + '/{}'.format(name)
        self.queue = queue.Queue()
        weak_self = weakref.ref(self)
        self.carla_actor.listen(lambda sensor_data: Sensor.data_callback(weak_self,
                                                                         sensor_data,
                                                                         self.queue))

        self.csv_fieldnames = ['frame',
                               'timestamp',
                               'x', 'y', 'z',
                               'roll', 'pitch', 'yaw']
        self._first_frame = True

    @staticmethod
    def data_callback(weak_self, sensor_data, data_queue: queue.Queue):
        data_queue.put(sensor_data)

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save sensor data to disk with timeout protection

        Args:
            frame_id: Absolute CARLA frame ID (for synchronization and file naming)
            timestamp: Timestamp
            debug: Whether to print debug info

        Returns:
            dict: Save information including file path, pose, and sensor-specific data
                  Format: {
                      'type': 'sensor',
                      'name': str,
                      'sensor_type': str,
                      'pose': dict,
                      'timestamp': float,
                      'file': str,  # relative path
                      ... # sensor-specific fields
                  }

        Raises:
            TimeoutError: If waiting for sensor data times out
            IOError: If data save fails
        """
        sensor_frame_id = 0

        while sensor_frame_id < frame_id:
            try:
                # Get from queue with timeout to prevent permanent blocking
                sensor_data = self.queue.get(block=True, timeout=SENSOR_QUEUE_TIMEOUT)
                sensor_frame_id = sensor_data.frame

                # Drop old frames
                if sensor_frame_id < frame_id:
                    logger.debug(
                        f"Sensor {self.name}: dropping old frame {sensor_frame_id}, "
                        f"waiting for frame {frame_id}"
                    )
                    continue

                # Ensure target path exists
                os.makedirs(self.save_dir, exist_ok=True)

                # Save data and get additional info (pass sensor_data for frame naming)
                save_result = self.save_to_disk_impl(self.save_dir, sensor_data)

                if not isinstance(save_result, dict):
                    # Backward compatibility: if save_to_disk_impl returns bool
                    success = save_result
                    save_info = {}
                else:
                    success = save_result.get('success', False)
                    save_info = save_result

                if not success:
                    error_msg = f"Sensor {self.name} failed to save frame {frame_id}"
                    logger.error(error_msg)
                    raise IOError(error_msg)

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

                # Add sensor-specific info from save_to_disk_impl
                result.update(save_info)

                if debug:
                    self.print_debug_info(sensor_data.frame, sensor_data)

                return result

            except queue.Empty:
                # Queue timeout
                error_msg = (
                    f"Sensor {self.name} timeout waiting for frame {frame_id} ({SENSOR_QUEUE_TIMEOUT}s). "
                    f"Last received frame: {sensor_frame_id}. "
                    f"Possible causes: CARLA running too slow, sensor stopped responding, or system I/O bottleneck."
                )
                logger.error(error_msg)
                raise TimeoutError(error_msg)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        """
        Save sensor data implementation (to be overridden by subclasses)

        Args:
            save_dir: Directory to save data
            sensor_data: Sensor data from CARLA (contains sensor_data.frame for file naming)

        Returns:
            dict or bool: Save result
        """
        raise NotImplementedError

    def print_debug_info(self, data_frame_id, sensor_data):
        logger.debug(
            f"Parent uid: {self.parent.uid}, Frame: {data_frame_id}, "
            f"uid={self.uid}, data: {sensor_data}"
        )

    def get_save_dir(self):
        return self.save_dir

    def is_first_frame(self):
        return self._first_frame

    def save_pose(self, frame_id, timestamp):
        """
        Save sensor pose data to CSV file using unified CSV utility.

        Args:
            frame_id: CARLA frame identifier
            timestamp: Simulation timestamp
        """
        trans = self.get_transform()
        pose_dict = trans.to_dict()
        pose_dict.update({'frame': frame_id,
                          'timestamp': timestamp})

        csv_path = '{}/poses.csv'.format(self.save_dir)

        # Use unified CSV utility with error handling
        result = safe_append_to_csv(
            csv_path=csv_path,
            fieldnames=self.csv_fieldnames,
            data=pose_dict,
            is_first_write=self.is_first_frame(),
            logger_name=f"{self.__class__.__name__}_{self.uid}"
        )

        # Log any errors but continue execution
        if not result['success']:
            logger.warning(f"Failed to save pose data: {result.get('error', 'Unknown error')}")

    def save_sensor_metadata(self, save_dir, additional_metadata=None):
        """
        Save generic sensor metadata file

        Args:
            save_dir: Directory to save metadata
            additional_metadata: Dictionary with additional sensor-specific metadata
        """
        import json

        # Get parent transform for relative positioning
        parent_transform = None
        if hasattr(self.parent, 'get_transform'):
            parent_transform = self.parent.get_transform().to_dict()

        # Base metadata structure
        metadata = {
            'sensor_type': self.sensor_type,
            'sensor_id': self.name,
            'parent_actor': getattr(self.parent, 'name', 'unknown'),
            'carla_blueprint': {
                'type': self.sensor_type,
                'attributes': dict(self.carla_actor.attributes)
            },
            'transform': {
                'relative_to_parent': self.get_transform().to_dict()
            },
            'recording_info': {
                'first_frame': None,  # To be filled by subclasses
                'total_frames': None   # To be filled by subclasses
            },
            'data_structure': {
                'directory': save_dir,
                'poses_file': 'poses.csv',
                'metadata_file': 'sensor_metadata.json'
            }
        }

        # Add parent transform if available
        if parent_transform:
            metadata['parent_transform'] = parent_transform

        # Add sensor-specific metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        # Save metadata file
        metadata_path = '{}/sensor_metadata.json'.format(save_dir)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
