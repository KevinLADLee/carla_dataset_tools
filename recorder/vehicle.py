#!/usr/bin/python3
import carla
from agents.navigation.behavior_agent import BehaviorAgent
from queue import Queue
from queue import Empty
import weakref


class VehicleAgent(BehaviorAgent):
    def __init__(self, vehicle_actor: carla.Vehicle):
        self.vehicle_actor = vehicle_actor
        self.sensor_actor_list = []
        self.queue = Queue()
        BehaviorAgent.__init__(self, vehicle_actor, 'normal')

    def destroy(self):
        for s in self.sensor_actor_list:
            s[1].destroy()
        self.vehicle_actor.destroy()

    def add_sensor(self, sensor_name, sensor_actor: carla.Sensor):
        self.sensor_actor_list.append((sensor_name, sensor_actor))
        sensor_type = sensor_actor.type_id
        weak_self = weakref.ref(self)
        sensor_actor.listen(lambda sensor_data: VehicleAgent.sensor_callback(weak_self, sensor_name, sensor_type, sensor_data, self.queue))

    @staticmethod
    def sensor_callback(weal_self, sensor_name, sensor_type, sensor_data, sensor_queue):
        sensor_queue.put((sensor_name, sensor_type, sensor_data))

    def save_to_disk(self):
        try:
            for _ in range(len(self.sensor_actor_list)):
                sensor_frame = self.queue.get(True, 1.0)
                sensor_name = sensor_frame[0]
                sensor_type = sensor_frame[1]
                sensor_data = sensor_frame[2]
                if sensor_type == 'sensor.camera.rgb':
                    sensor_data.save_to_disk("_out/{}_{}.png".format(sensor_data.frame, sensor_name))
                elif sensor_type == 'sensor.camera.semantic_segmentation':
                    cc = carla.ColorConverter.CityScapesPalette
                    sensor_data.save_to_disk("_out/seg/{}_{}.png".format(sensor_data.frame, sensor_name), cc)
                print("saved: frame: {} sensor: {}".format(sensor_data.frame, sensor_name))
        except Empty:
            print("    Some of the sensor information is missed")

