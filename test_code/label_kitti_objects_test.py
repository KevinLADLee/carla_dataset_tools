#!/usr/bin/python3
import glob
import pickle
import time

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool

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

# frame_id, timestamp, lidar_pose, camera_pose, world_object_label, lidar_rawdata_path, image_rawdata_path


if __name__ == '__main__':
    record_name = "record_2021_1228_1920"
    vehicle_path_list = glob.glob("{}/{}/vehicle.*".format(RAW_DATA_PATH, record_name))

    vehicle_df = pd.DataFrame()
    for vehicle_path in vehicle_path_list:
        camera_path = glob.glob("{}/sensor.camera.rgb*".format(vehicle_path))[0]
        lidar_path = glob.glob("{}/sensor.lidar.ray_cast*".format(vehicle_path))[0]
        vehicle_df = vehicle_df.append({"vehicle_path": vehicle_path, "camera_path": camera_path, "lidar_path": lidar_path}, ignore_index=True)

    for index, row in vehicle_df.iterrows():
        vehicle_path = row['vehicle_path']
        lidar_path = row['lidar_path']

        lidar_poses_df = load_poses_csv("{}/poses.csv".format(lidar_path))
        lidar_data_list = load_lidar_data(lidar_path)
        world_data_list = load_world_data("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name))

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Kitti Objects Label')
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        frame = 0
        thread = ThreadPool()
        thread.starmap()
        for i, pose in lidar_poses_df.iterrows():
            lidar_trans = Transform(Location(pose['x'],
                                             pose['y'],
                                             pose['z']),
                                    Rotation(roll=pose['roll'],
                                             yaw=pose['yaw'],
                                             pitch=pose['pitch']))
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

