#!/usr/bin/python3
import math
import numpy
from transforms3d.euler import euler2mat


class Vec3d(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vec3d = numpy.array([[
            self.x,
            self.y,
            self.z
        ]]).reshape(3, 1)

    def get_vector(self):
        return self.vec3d

    def to_dict(self) -> dict:
        return {'x': self.x,
                'y': self.y,
                'z': self.z}

    def __eq__(self, other):
        return numpy.array_equal(self.vec3d, other.vec3d)


class Location(Vec3d):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)


class Rotation:
    def __init__(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.rotation_mat = euler2mat(math.radians(roll),
                                      math.radians(pitch),
                                      math.radians(yaw))
        self.rotation_mat = self.rotation_mat[:3, :3]

    def get_matrix(self):
        return self.rotation_mat

    def to_dict(self) -> dict:
        return {'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw}

    def __eq__(self, other):
        return numpy.array_equal(self.rotation_mat, other.rotation_mat)


class Transform:
    def __init__(self, location: Location, rotation: Rotation):
        self.location = location
        self.rotation = rotation
        t_vec = self.location.get_vector()
        r_mat = self.rotation.get_matrix()
        tmp = numpy.concatenate((r_mat, t_vec), axis=1)
        self.trans_mat = numpy.concatenate((tmp, numpy.array([[0, 0, 0, 1]])),
                                           axis=0)

    def to_dict(self):
        location_dict = self.location.to_dict()
        rotation_dict = self.rotation.to_dict()
        return location_dict.update(rotation_dict)

    def get_matrix(self):
        return self.trans_mat

    def get_inverse_matrix(self):
        return numpy.linalg.inv(self.trans_mat)

    def transform(self, point: Location):
        t_vec = point.get_vector()
        t_vec_out = numpy.dot(self.trans_mat, t_vec)
        return Location(t_vec_out[0], t_vec_out[1], t_vec_out[2])

    def __eq__(self, other):
        return (self.location == other.location) and (self.rotation == other.rotation)


class Pose(Transform):
    def __init__(self, location: Location, rotation: Rotation):
        super().__init__(location, rotation)