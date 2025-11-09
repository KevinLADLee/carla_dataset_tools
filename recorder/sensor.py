#!/usr/bin/python3
import copy
import csv
import os
import weakref
import logging
import queue

import carla

from recorder.actor import Actor

# 获取logger实例
logger = logging.getLogger(__name__)

# 传感器队列超时时间（秒）
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
            frame_id: Target frame ID
            timestamp: Timestamp
            debug: Whether to print debug info

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

                # Save data
                success = self.save_to_disk_impl(self.save_dir, sensor_data)

                if not success:
                    error_msg = f"Sensor {self.name} failed to save frame {frame_id}"
                    logger.error(error_msg)
                    raise IOError(error_msg)

                # Save sensor pose
                self.save_pose(frame_id, timestamp)
                self._first_frame = False

                if debug:
                    self.print_debug_info(sensor_data.frame, sensor_data)

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
        trans = self.get_transform()
        pose_dict = trans.to_dict()
        pose_dict.update({'frame': frame_id,
                          'timestamp': timestamp})

        csv_path = '{}/poses.csv'.format(self.save_dir)
        if self.is_first_frame():
            with open(csv_path, 'w', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
                writer.writeheader()

        with open(csv_path, 'a', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
            writer.writerow(pose_dict)
