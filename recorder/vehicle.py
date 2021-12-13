#!/usr/bin/python3
import copy
import csv
import os
import carla

from .actor import Actor


class Vehicle(Actor):
    def __init__(self,
                 uid,
                 name,
                 base_save_dir: str,
                 carla_actor: carla.Actor,
                 parent_actor=None):
        super().__init__(uid, name, carla_actor, None)
        self.vehicle_type = copy.deepcopy(carla_actor.type_id)
        self.save_dir = '{}/{}_{}'.format(base_save_dir, self.get_id(), self.vehicle_type)
        self.first_tick = True

    def get_save_dir(self):
        return self.save_dir

    def save_to_disk(self, frame_id):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.first_tick:
            self.save_vehicle_info()

        # Save vehicle status to csv file
        # frame_id x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az
        fieldnames = ['frame', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
        with open('{}/vehicle_status.csv'.format(self.save_dir), 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if self.first_tick:
                writer.writeheader()
                self.first_tick = False
            csv_line = {'frame': frame_id}
            csv_line.update(self.get_transform().to_dict())
            csv_line.update(self.get_acceleration().to_dict())
            writer.writerow(csv_line)

    def save_vehicle_info(self):
        # TODO: Save vehicle physics info here
        pass
