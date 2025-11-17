#!/usr/bin/python3

import carla
import numpy as np

from recorder.sensor import Sensor


class Radar(Sensor):
    def __init__(self, uid, name: str, base_save_dir: str, parent, carla_actor: carla.Sensor):
        super().__init__(uid, name, base_save_dir, parent, carla_actor)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        """
        Save radar data to disk

        Args:
            save_dir: Directory to save data
            sensor_data: Radar sensor data from CARLA (contains sensor_data.frame)

        Returns:
            bool: Success status
        """
        # Save as a Nx7 numpy array. Each row is a point (x, y, z, depth, velocity, azimuth, altitude)
        # radar_raw_data = np.frombuffer(sensor_data.raw_data,
        #                                dtype=np.float32)
        # radar_raw_data = np.reshape(
        #     radar_raw_data, (int(radar_raw_data.shape[0] / 4), 4))

        radar_points = []
        for detection in sensor_data:
            radar_points.append([detection.depth * np.cos(detection.azimuth) * np.cos(-detection.altitude),
                                 detection.depth * np.sin(-detection.azimuth) * np.cos(detection.altitude),
                                 detection.depth * np.sin(detection.altitude),
                                 detection.depth,
                                 detection.velocity,
                                 detection.azimuth,
                                 detection.altitude])
        radar_points = np.asarray(radar_points)
        radar_points.reshape(-1, 7)

        # Save point cloud using absolute frame ID from sensor_data
        np.save("{}/{:0>10d}".format(save_dir, sensor_data.frame),
                radar_points)
        return True

