import carla
from utils.transform import *


class Actor(object):
    def __init__(self, uid, name, carla_actor: carla.Actor, parent_actor):
        self.uid = uid
        self.name = name
        self.carla_actor = carla_actor
        self.parent_actor = parent_actor

    def get_transform(self) -> Transform:
        trans = self.carla_actor.get_transform()
        return carla_transform_to_transform(trans)

    def set_transform(self, transform: Transform):
        trans = transform_to_carla_transform(transform)
        self.carla_actor.set_transform(trans)

    def get_acceleration(self) -> Vec3d:
        acc = self.carla_actor.get_acceleration()
        return carla_vec3d_to_vec3d(acc)

    def get_id(self):
        return self.carla_actor.id



