#!/usr/bin/python3

import carla
import cv2 as cv
import numpy as np
import transforms3d

from recorder.sensor import Sensor
from utils.types import Transform, Rotation
from utils.transform import carla_transform_to_transform


class CameraBase(Sensor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 parent,
                 carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)
        self.color_converter = color_converter

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        # Convert to target color template
        if self.color_converter is not None:
            sensor_data.convert(self.color_converter)

        # Convert raw data to numpy array, image type is 'bgra8'
        carla_image_data_array = np.ndarray(shape=(sensor_data.height,
                                                   sensor_data.width,
                                                   4),
                                            dtype=np.uint8,
                                            buffer=sensor_data.raw_data)

        # Save image to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].png
        success = cv.imwrite("{}/{:0>10d}.png".format(save_dir,
                                                      sensor_data.frame),
                             carla_image_data_array)
        return success

    def get_transform(self) -> Transform:
        c_trans = self.carla_actor.get_transform()
        trans = carla_transform_to_transform(c_trans)
        quat = transforms3d.quaternions.mat2quat(trans.rotation.get_rotation_matrix())
        quat_swap = transforms3d.quaternions.mat2quat(np.matrix(
                      [[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]]))
        quat_camera = transforms3d.quaternions.qmult(quat, quat_swap)
        roll, pitch, yaw = transforms3d.euler.quat2euler(quat_camera)
        return Transform(trans.location, Rotation(roll=roll, pitch=pitch, yaw=yaw))


class RgbCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)


class SemanticSegmentationCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        color_converter = carla.ColorConverter.CityScapesPalette
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)


class DepthCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        color_converter = carla.ColorConverter.Raw
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)