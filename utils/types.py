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

    def __ne__(self, other):
        return not numpy.array_equal(self.vec3d, other.vec3d)

    def __str__(self):
        return "Vec3d(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Location(Vec3d):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)

    def __str__(self):
        return "Location(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Rotation:
    def __init__(self, pitch, yaw, roll):
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

    def __ne__(self, other):
        return not numpy.array_equal(self.rotation_mat, other.rotation_mat)

    def __str__(self):
        return "Rotation(pitch={}, yaw={}, roll={})".format(self.pitch, self.yaw, self.roll)


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
        location_dict.update(rotation_dict)
        return location_dict

    def get_matrix(self):
        return self.trans_mat

    def get_inverse_matrix(self):
        return numpy.linalg.inv(self.trans_mat)

    def transform(self, point: Location):
        t_vec = point.get_vector()
        t_vec = numpy.concatenate((t_vec, [[1]]), axis=0)
        t_vec_out = numpy.dot(self.trans_mat, t_vec)
        return Location(t_vec_out[0, 0], t_vec_out[1, 0], t_vec_out[2, 0])

    def __eq__(self, other):
        return (self.location == other.location) and (self.rotation == other.rotation)

    def __ne__(self, other):
        return (self.location != other.location) or (self.rotation != other.rotation)

    def __str__(self):
        return "Transform({}, {})".format(self.location, self.rotation)


class Pose(Transform):
    def __init__(self, location: Location, rotation: Rotation):
        super().__init__(location, rotation)