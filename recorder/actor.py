import carla
from utils.transform import *


class PseudoActor(object):
    def __init__(self, uid, name, parent):
        self.uid = uid
        self.name = name
        self.parent = parent

    def get_type_id(self):
        raise NotImplementedError

    def get_carla_actor(self):
        return None

    def get_save_dir(self):
        raise NotImplementedError


class Actor(PseudoActor):
    def __init__(self, uid, name, parent, carla_actor: carla.Actor):
        super(Actor, self).__init__(uid=uid,
                                    name=name,
                                    parent=parent)
        self.carla_actor = carla_actor

    def get_transform(self) -> Transform:
        trans = self.carla_actor.get_transform()
        return carla_transform_to_transform(trans)

    def set_transform(self, transform: Transform):
        trans = transform_to_carla_transform(transform)
        self.carla_actor.set_transform(trans)

    def get_acceleration(self) -> Vec3d:
        acc = self.carla_actor.get_acceleration()
        return carla_vec3d_to_vec3d(acc)

    def get_type_id(self):
        return self.carla_actor.type_id

    def get_uid(self):
        return self.uid

    def get_actor_id(self):
        return self.carla_actor.id

    def get_carla_actor(self):
        return self.carla_actor

    def get_save_dir(self):
        raise NotImplementedError



