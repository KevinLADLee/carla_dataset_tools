#!/usr/bin/python3
import carla
from utils.transform import *


class PseudoActor(object):
    def __init__(self, uid, name, parent):
        self.uid = uid
        self.name = name
        self.parent = parent

    def destroy(self):
        return True

    def get_type_id(self):
        raise NotImplementedError

    def get_carla_actor(self):
        return None

    def get_save_dir(self):
        raise NotImplementedError

    def get_uid(self):
        return self.uid


class Actor(PseudoActor):
    def __init__(self, uid, name, parent, carla_actor: carla.Actor):
        super(Actor, self).__init__(uid=uid,
                                    name=name,
                                    parent=parent)
        self.carla_actor = carla_actor

    def destroy(self):
        print("Destroying: uid={} name={} carla_id={}".format(self.uid, self.name, self.carla_actor.id))
        if self.carla_actor is not None:
            try:
                status = self.carla_actor.destroy()
                # time.sleep(1)
                if status:
                    print("-> success")
                return status
            except RuntimeError:
                print("-> failed")
                return False

    def get_transform(self) -> Transform:
        trans = self.carla_actor.get_transform()
        return carla_transform_to_transform(trans)

    def set_transform(self, transform: Transform):
        trans = transform_to_carla_transform(transform)
        self.carla_actor.set_transform(trans)

    def get_acceleration(self):
        acc = self.carla_actor.get_acceleration()
        return math.sqrt(acc.x * acc.x
                         + acc.y * acc.y
                         + acc.z * acc.z)

    def get_speed(self):
        v = self.carla_actor.get_velocity()
        return math.sqrt(v.x * v.x
                         + v.y * v.y
                         + v.z * v.z)

    def get_type_id(self):
        return self.carla_actor.type_id

    def get_actor_id(self):
        return self.carla_actor.id

    def get_carla_actor(self):
        return self.carla_actor

    def get_save_dir(self):
        raise NotImplementedError



