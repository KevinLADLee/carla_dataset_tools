#!/usr/bin/python3
import glob
import pickle
import time

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH
from utils.transform import *


def load_lidar_data(path: str):
    lidar_rawdata_list = sorted(glob.glob("{}/*.npy".format(path)))
    return lidar_rawdata_list


def parse_data(lidar_rawdata: str):
    return numpy.load(lidar_rawdata)


def load_poses_csv(path: str) -> pd.DataFrame:
    poses_df = pd.read_csv(path)
    return poses_df


def load_world_data(path: str):
    world_data_list = sorted(glob.glob("{}/*.pkl".format(path)))
    return world_data_list


if __name__ == '__main__':
    record_name = "record_2021_1228_1858"
    vehicle_name = "1_vehicle.tesla.model3"
    sensor_name = "4_sensor.lidar.ray_cast"

    objects_label_df = load_poses_csv("{}/{}/{}/{}/poses.csv".format(RAW_DATA_PATH,
                                                             record_name,
                                                             vehicle_name,
                                                             sensor_name))

    lidar_data_list = load_lidar_data("{}/{}/{}/{}".format(RAW_DATA_PATH,
                                                           record_name,
                                                           vehicle_name,
                                                           sensor_name))

    world_data_list = load_world_data("{}/{}/0_others.world_actor".format(RAW_DATA_PATH, record_name))

    objects_label_df.insert(len(objects_label_df.columns), 'lidar_rawdata_path', lidar_data_list)
    objects_label_df.insert(len(objects_label_df.columns), 'world_data_path', world_data_list)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Kitti Objects Label')
    # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True

    frame = 0
    for index, row in objects_label_df.iterrows():
        vis.clear_geometries()

        # Parse point cloud data
        lidar_trans = Transform(Location(row['x'],
                                         row['y'],
                                         row['z']),
                                Rotation(roll=row['roll'],
                                         yaw=row['yaw'],
                                         pitch=row['pitch']))
        lidar_data = numpy.load(row['lidar_rawdata_path'])
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(lidar_data[:, 0:3])

        # Load object labels from pickle data
        with open(row['world_data_path'], 'rb') as pkl_file:
            objects_labels = pickle.load(pkl_file)

        # Convert all bbox to o3d bbox
        bbox_list = []
        for label in objects_labels:
            # Check Distance
            dist = np.linalg.norm(lidar_trans.location.get_vector()-label.transform.location.get_vector())
            if dist > 150:
                continue

            # Transform label bbox to lidar coordinate
            world_to_lidar = lidar_trans.get_inverse_matrix()
            label_to_world = label.transform.get_matrix()
            label_in_lidar = np.matmul(world_to_lidar, label_to_world)
            t_vec = label_in_lidar[0:3, -1]
            r_mat = label_in_lidar[0:3, 0:3]
            o3d_bbox = bbox_to_o3d_bbox(label.bounding_box)
            o3d_bbox.rotate(r_mat)
            o3d_bbox.translate(t_vec)
            o3d_bbox.color = np.array([1.0, 0, 0])

            # Check points in bbox
            p_num_in_bbox = o3d_bbox.get_point_indices_within_bounding_box(o3d_pcd.points)
            if len(p_num_in_bbox) < 10:
                continue

            bbox_list.append(o3d_bbox)
            vis.add_geometry(o3d_bbox)

        vis.add_geometry(o3d_pcd)
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.1)
        frame += 1

    vis.destroy_window()
