#!/usr/bin/python3
import sys
import csv
import carla
import cv2 as cv
import numpy as np
import transforms3d
import math

from recorder.sensor import Sensor
from core.geometry import Transform, Rotation
from core.transform import carla_transform_to_transform

# Camera constants
CARLA_IMAGE_CHANNELS = 4  # BGRA format: Blue, Green, Red, Alpha
CARLA_IMAGE_DTYPE = 'uint8'  # Standard CARLA image data type
IMAGE_FRAME_ID_FORMAT = '{:0>10d}'  # Format for frame ID in filenames
DEGREES_TO_RADIANS = math.pi / 180.0  # Conversion factor from degrees to radians


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

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save camera image to disk

        Args:
            save_dir: Directory to save data
            sensor_data: Camera sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str, 'camera_info': dict}
        """
        # Convert to target color template
        if self.color_converter is not None:
            sensor_data.convert(self.color_converter)

        # Convert raw data to numpy array, image type is 'bgra8'
        carla_image_data_array = np.ndarray(shape=(sensor_data.height,
                                                   sensor_data.width,
                                                   CARLA_IMAGE_CHANNELS),
                                            dtype=CARLA_IMAGE_DTYPE,
                                            buffer=sensor_data.raw_data)

        # Generate filename using absolute frame ID from sensor_data
        filename = "{}.png".format(IMAGE_FRAME_ID_FORMAT.format(sensor_data.frame))
        filepath = "{}/{}".format(save_dir, filename)

        # Save image to [RAW_DATA_PATH]/.../[ID]_[SENSOR_TYPE]/[FRAME_ID].png
        success = cv.imwrite(filepath, carla_image_data_array)

        if success and self.is_first_frame():
            self.save_camera_info(save_dir)

        # Prepare return info
        result = {
            'success': success,
            'file': filename
        }

        # Add camera info on first frame
        if self.is_first_frame():
            result['camera_info'] = self.get_camera_info()

        return result

    def save_camera_info(self, save_dir):
        with open('{}/camera_info.csv'.format(save_dir), 'w', encoding='utf-8') as csv_file:
            fieldnames = {'width',
                          'height',
                          'fx',
                          'fy',
                          'cx',
                          'cy'}
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            camera_info = self.get_camera_info()
            writer.writerow(camera_info)

        # Also save as metadata JSON
        additional_metadata = {
            'data_format': {
                'file_format': 'png',
                'encoding': 'bgra8',
                'compression': 'lossless',
                'bit_depth': 32
            },
            'camera_info': camera_info,
            'sensor_category': 'camera'
        }

        # Add camera-specific metadata
        if hasattr(self, 'color_converter') and self.color_converter:
            additional_metadata['color_conversion'] = str(self.color_converter)

        self.save_sensor_metadata(save_dir, additional_metadata)

    def get_camera_info(self):
        camera_width = int(self.carla_actor.attributes['image_size_x'])
        camera_height = int(self.carla_actor.attributes['image_size_y'])

        # CARLA camera calibration formula
        # Reference: https://carla.readthedocs.io/en/0.9.16/tuto_G_bounding_boxes/
        # focal = w / (2.0 * tan(fov / 2))
        # Note: FOV is the FULL field of view angle, we need the HALF-angle for tan()
        fov_deg = float(self.carla_actor.attributes['fov'])
        fov_half_rad = math.radians(fov_deg / 2.0)  # Convert half FOV to radians
        fx = camera_width / (2.0 * math.tan(fov_half_rad))

        return {
            'width': camera_width,
            'height': camera_height,
            'cx': camera_width / 2.0,
            'cy': camera_height / 2.0,
            'fx': fx,
            'fy': fx
        }

    def get_transform(self) -> Transform:
        c_trans = self.carla_actor.get_transform()
        trans = carla_transform_to_transform(c_trans)
        quat = trans.rotation.get_quaternion()
        quat_swap = transforms3d.quaternions.mat2quat(np.matrix(
                      [[0, 0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]]))
        quat_camera = transforms3d.quaternions.qmult(quat, quat_swap)
        roll, pitch, yaw = transforms3d.euler.quat2euler(quat_camera)
        return Transform(trans.location, Rotation(roll=math.degrees(roll),
                                                  pitch=math.degrees(pitch),
                                                  yaw=math.degrees(yaw)))


class RgbCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)


class SemanticSegmentationCamera(CameraBase):
    """
    Semantic segmentation camera using CityScapesPalette color converter.
    CityScapesPalette encoding provides complete semantic classification with readable colors.
    """
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        # Use CityScapesPalette for semantic segmentation with proper color mapping
        color_converter = carla.ColorConverter.CityScapesPalette
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)


class DepthCamera(CameraBase):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        # Use CARLA Raw converter which encodes depth in BGRA format
        color_converter = carla.ColorConverter.Raw
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save depth camera data to disk using CARLA Raw encoding in PNG format

        Args:
            save_dir: Directory to save data
            sensor_data: Depth camera sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str}
        """
        # CARLA Raw converter already encodes depth information in BGRA format
        # Save directly as PNG - this preserves the full depth encoding from CARLA
        carla_image_data_array = np.ndarray(shape=(sensor_data.height,
                                                   sensor_data.width,
                                                   CARLA_IMAGE_CHANNELS),
                                            dtype=CARLA_IMAGE_DTYPE,
                                            buffer=sensor_data.raw_data)

        # Generate filename using absolute frame ID from sensor_data
        filename = "{}.png".format(IMAGE_FRAME_ID_FORMAT.format(sensor_data.frame))
        filepath = "{}/{}".format(save_dir, filename)

        # Save PNG with CARLA's depth encoding preserved
        import cv2 as cv
        success = cv.imwrite(filepath, carla_image_data_array)

        if success and self.is_first_frame():
            self.save_camera_info(save_dir)

        return {
            'success': success,
            'file': filename,
            'encoding': 'CARLA_Raw_depth_32bit_BGRA'
        }


class InstanceSegmentationCamera(CameraBase):
    """
    Instance segmentation camera using CityScapesPalette color converter.
    CityScapesPalette encoding provides both semantic and instance information with readable colors.
    """
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        # Use CityScapesPalette for instance segmentation with proper color mapping
        color_converter = carla.ColorConverter.Raw
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)


class OpticalFlowCamera(CameraBase):
    # Optical flow constants
    MIN_RANGE = -2.0  # Minimum optical flow value
    MAX_RANGE = 2.0  # Maximum optical flow value
    RANGE_SPAN = MAX_RANGE - MIN_RANGE  # Total range span (4.0)
    UINT16_MAX = 65535.0  # Maximum value for 16-bit unsigned integer

    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor,
                 color_converter: carla.ColorConverter = None):
        super().__init__(uid, name, base_save_dir, parent, carla_actor, color_converter)

    def save_to_disk_impl(self, save_dir, sensor_data) -> dict:
        """
        Save optical flow camera data to disk using PNG format

        Args:
            save_dir: Directory to save data
            sensor_data: Optical flow camera sensor data from CARLA (contains sensor_data.frame)

        Returns:
            dict: {'success': bool, 'file': str}
        """
        # Optical flow data is encoded as 2D vectors (u,v) in BGRA format
        # Each pixel contains flow vector in range [-2, 2]
        flow_data = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
        flow_data = flow_data.reshape((sensor_data.height, sensor_data.width, 2))

        # Convert flow range [-2, 2] to 16-bit PNG for precision preservation
        # Using formula: uint16_value = ((flow_value + 2) / 4) * 65535
        flow_16bit = ((flow_data - self.MIN_RANGE) / self.RANGE_SPAN *
                      self.UINT16_MAX).astype(np.uint16)

        # Create BGRA image (2 channels of 16-bit flow data stored in 4 channels of 8-bit)
        # Pack u and v components into BGRA format
        flow_bgra = np.zeros((sensor_data.height, sensor_data.width, 4), dtype=np.uint8)

        # Pack 16-bit u,v into 8-bit BGRA channels
        # u component: high byte in B, low byte in G
        # v component: high byte in R, low byte in A
        flow_bgra[:, :, 0] = (flow_16bit[:, :, 0] >> 8) & 0xFF  # B - u high
        flow_bgra[:, :, 1] = flow_16bit[:, :, 0] & 0xFF         # G - u low
        flow_bgra[:, :, 2] = (flow_16bit[:, :, 1] >> 8) & 0xFF  # R - v high
        flow_bgra[:, :, 3] = flow_16bit[:, :, 1] & 0xFF         # A - v low

        # Generate filename using absolute frame ID from sensor_data
        filename = "{}.png".format(IMAGE_FRAME_ID_FORMAT.format(sensor_data.frame))
        filepath = "{}/{}".format(save_dir, filename)

        # Save PNG with packed 16-bit optical flow data
        import cv2 as cv
        success = cv.imwrite(filepath, flow_bgra)

        if success and self.is_first_frame():
            self.save_camera_info(save_dir)
            self.save_optical_flow_metadata(save_dir, flow_data)

        return {
            'success': success,
            'file': filename,
            'encoding': '16-bit_optical_flow_packed_BGRA',
            'shape': flow_data.shape,
            'range': '[-2.0, 2.0]'
        }

    def save_optical_flow_metadata(self, save_dir, flow_data):
        """Save optical flow camera metadata"""
        import json

        # Get camera intrinsics
        camera_info = self.get_camera_info()

        metadata = {
            'sensor_type': 'sensor.camera.optical_flow',
            'attributes': dict(self.carla_actor.attributes),
            'data_format': 'png_16bit_packed',
            'encoding': {
                'format': '16-bit_flow_vectors_packed_in_8bit_BGRA',
                'u_channel': 'B(high_byte),G(low_byte)',
                'v_channel': 'R(high_byte),A(low_byte)',
                'range': '[-2.0, 2.0]',
                'units': 'pixels_per_frame'
            },
            'flow_statistics': {
                'u_range': [float(flow_data[:, :, 0].min()), float(flow_data[:, :, 0].max())],
                'v_range': [float(flow_data[:, :, 1].min()), float(flow_data[:, :, 1].max())],
                'magnitude_mean': float(np.sqrt(flow_data[:, :, 0]**2 + flow_data[:, :, 1]**2).mean()),
                'shape': flow_data.shape
            },
            'camera_info': camera_info
        }

        with open('{}/sensor_metadata.json'.format(save_dir), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

