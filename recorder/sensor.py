#!/usr/bin/python3
import copy
import carla
from queue import Queue
import weakref
import os

from .actor import Actor


class Sensor(Actor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 carla_actor: carla.Sensor,
                 parent_actor: carla.Actor):
        super(Sensor, self).__init__(uid, name, carla_actor, parent_actor)
        self.sensor_type = copy.deepcopy(carla_actor.type_id)
        self.save_dir = base_save_dir + '/{}_{}'.format(self.get_id(), self.sensor_type)
        self.queue = Queue()
        weak_self = weakref.ref(self)
        self.carla_actor.listen(lambda sensor_data: Sensor.data_callback(weak_self,
                                                                         sensor_data,
                                                                         self.queue))

    @staticmethod
    def data_callback(weak_self, sensor_data, data_queue: Queue):
        data_queue.put(sensor_data)

    def save_to_disk(self, frame_id):
        sensor_frame_id = 0
        while sensor_frame_id < frame_id:
            sensor_data = self.queue.get(True, 1.0)
            sensor_frame_id = sensor_data.frame

            # Drop previous data
            if sensor_frame_id < frame_id:
                continue

            # make sure target path exist
            os.makedirs(self.save_dir, exist_ok=True)

            success = self.save_to_disk_impl(self.save_dir, sensor_data)
            if not success:
                print("Save to disk failed!")
                raise IOError

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        raise NotImplementedError
