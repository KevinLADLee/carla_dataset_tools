#!/usr/bin/python3
import math
import numpy
import transforms3d as tf3d


class Vector3d(object):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def get_vector(self):
        return numpy.array([[
            self.x,
            self.y,
            self.z,
        ]], dtype=numpy.float32).reshape(3, 1)

    def to_dict(self) -> dict:
        return {'x': self.x,
                'y': self.y,
                'z': self.z}

    def __eq__(self, other):
        return numpy.array_equal(self.get_vector(),
                                 other.get_vector())

    def __ne__(self, other):
        return not numpy.array_equal(self.get_vector(),
                                     other.get_vector())

    def __str__(self):
        return "Vector3d(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Location(Vector3d):
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)

    def __str__(self):
        return "Location(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Rotation:
    def __init__(self, *, pitch=0.0, yaw=0.0, roll=0.0):
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def get_rotation_matrix(self):
        return tf3d.euler.euler2mat(math.radians(self.roll),
                                    math.radians(self.pitch),
                                    math.radians(self.yaw))

    def to_dict(self) -> dict:
        return {'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw}

    def __eq__(self, other):
        return (self.roll == other.roll) \
               and (self.pitch == other.pitch) \
               and (self.yaw == other.yaw)

    def __ne__(self, other):
        return (self.roll != other.roll) \
               or (self.pitch != other.pitch) \
               or (self.yaw != other.yaw)

    def __str__(self):
        return "Rotation(pitch={}, yaw={}, roll={})".format(self.pitch, self.yaw, self.roll)


class Transform:
    def __init__(self, location: Location, rotation: Rotation):
        self.location = location
        self.rotation = rotation

    def to_dict(self):
        location_dict = self.location.to_dict()
        rotation_dict = self.rotation.to_dict()
        location_dict.update(rotation_dict)
        return location_dict

    def get_matrix(self):
        t_vec = self.location.get_vector()
        r_mat = self.rotation.get_rotation_matrix()
        t_mat = numpy.concatenate((r_mat, t_vec), axis=1)
        t_mat_homo = numpy.concatenate((t_mat,
                                        numpy.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)
        return t_mat_homo

    def get_inverse_matrix(self):
        trans_mat = self.get_matrix()
        return numpy.linalg.inv(trans_mat)

    def transform(self, point: Location):
        trans_mat = self.get_matrix()
        p = numpy.concatenate((point.get_vector(), numpy.array([[1]])), axis=0)
        p = numpy.matmul(trans_mat, p, dtype=numpy.float)
        numpy.set_printoptions(precision=8)
        return Location(p[0, 0], p[1, 0], p[2, 0])

    def __eq__(self, other):
        return (self.location == other.location) \
               and (self.rotation == other.rotation)

    def __ne__(self, other):
        return (self.location != other.location) \
               or (self.rotation != other.rotation)

    def __str__(self):
        return "Transform({}, {})".format(self.location, self.rotation)
