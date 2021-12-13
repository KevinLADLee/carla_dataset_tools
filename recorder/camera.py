#!/usr/bin/python3

import carla
import cv2 as cv
import numpy as np

from .sensor import Sensor


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