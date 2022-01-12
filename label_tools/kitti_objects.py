#!/usr/bin/python3
import pickle
import sys
import time

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH
from utils.transform import *
from utils.label_types import ObjectLabel
from label_tools.data_loader import *


def gather_rawdata_to_dataframe(record_name: str, vehicle_path: str, lidar_path: str, camera_path: str):
    rawdata_frames_df = pd.DataFrame()
    vehicle_poses_df = load_vehicle_pose("{}/{}/{}".format(RAW_DATA_PATH, record_name, vehicle_path))
    rawdata_frames_df = vehicle_poses_df

    object_labels_path_df = load_object_labels("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name))
    rawdata_frames_df = pd.merge(rawdata_frames_df, object_labels_path_df, how='outer', on='frame')

    lidar_rawdata_df = load_lidar_rawdata(f"{RAW_DATA_PATH}/{record_name}/{vehicle_path}/{lidar_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, lidar_rawdata_df, how='outer', on='frame')

    camera_rawdata_path_df = load_camera_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_path}/{camera_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, camera_rawdata_path_df, how='outer', on='frame')

    return rawdata_frames_df


class KittiObjectLabel:
    def __init__(self, args, rawdata_df: pd.DataFrame):
        self.rawdata_df = rawdata_df
        self.range_max = 100.0
        self.range_min = 1.0
        self.points_min = 20
        # self.label_df = pd.DataFrame(columns='type')
        # self.process_pool = ThreadPool()

    def process(self):
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name='Kitti Objects Label')
        # vis.get_render_option().point_size = 1
        # vis.get_render_option().show_coordinate_frame = True
        # vis_ctl = vis.get_view_control()
        # vis_ctl.set_zoom(0.9)
        for index, frame in self.rawdata_df.iterrows():
            # vis.clear_geometries()
            lidar_trans: Transform = frame['lidar_pose']
            cam_trans: Transform = frame['camera_pose']
            lidar_data = numpy.load(frame['lidar_rawdata_path'])
            cam_mat = np.asarray(frame['camera_matrix'])

            image = cv2.imread(frame['camera_rawdata_path'], cv2.IMREAD_UNCHANGED)

            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(lidar_data[:, 0:3])

            # Load object labels from pickle data
            with open(frame['object_labels_path'], 'rb') as pkl_file:
                objects_labels = pickle.load(pkl_file)

            # Convert all bbox to o3d bbox
            bbox_list = [o3d.geometry.OrientedBoundingBox]
            for label in objects_labels:

                if not self.is_valid_distance(lidar_trans.location, label.transform.location):
                    continue

                o3d_bbox = self.get_o3d_bbox_in_target_coordinate(lidar_trans, label)

                if o3d_bbox.center[0] < 0:
                    continue

                # Check points in bbox
                p_num_in_bbox = o3d_bbox.get_point_indices_within_bounding_box(o3d_pcd.points)
                if len(p_num_in_bbox) < self.points_min:
                    continue

                # Transform bbox vertices to camera coordinate
                vertex_points = np.asarray(o3d_bbox.get_box_points())
                bbox_points_2d_x = []
                bbox_points_2d_y = []
                for p in vertex_points:
                    p = np.append(p, [1.0])
                    T_wl = lidar_trans.get_matrix()
                    # p_w = np.matmul(T_wl, p)
                    T_cw = cam_trans.get_inverse_matrix()
                    T_cl = np.matmul(T_cw, T_wl)
                    p_c = np.matmul(T_cl, p)
                    p_c = p_c[0:3] / p_c[2]
                    p_uv = np.matmul(cam_mat, p_c)
                    p_uv = p_uv[0:2].astype(int)
                    # cv2.circle(image, p_uv, 1, (0, 0, 255), 2)
                    bbox_points_2d_x.append(p_uv[0])
                    bbox_points_2d_y.append(p_uv[1])

                # Generate 2d bbox by left-top point and right-bottom point
                min_x = min(bbox_points_2d_x)
                max_x = max(bbox_points_2d_x)
                min_y = min(bbox_points_2d_y)
                max_y = max(bbox_points_2d_y)

                # For Debug
                # Draw 2d bbox
                cv2.rectangle(image, [min_x, min_y], [max_x, max_y], color=(0, 0, 255), thickness=1)
                cv2.imshow('preview', image)
                cv2.waitKey()
                #
                # Draw camera coordinate
                # p_cam = np.append(cam_trans.location.get_vector(), [[1.0]])
                # print(p_cam)
                # p_cam_in_lidar = numpy.matmul(lidar_trans.get_inverse_matrix(), p_cam)[0:3]
                # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[p_cam_in_lidar[0],
                #                                                                              p_cam_in_lidar[1],
                #                                                                              p_cam_in_lidar[2]])
                # coord.rotate(cam_trans.rotation.get_rotation_matrix())
                # print(p_cam_in_lidar)
                # o3d.visualization.draw_geometries([o3d_pcd, o3d_bbox, coord])

                bbox_list.append(o3d_bbox)
                # vis.add_geometry(o3d_bbox)

            # time.sleep(0.05)
            # vis.add_geometry(o3d_pcd)
            # vis.poll_events()
            # vis.update_renderer()

    def is_valid_distance(self, source_location: Location, target_location: Location):
        dist = np.linalg.norm(source_location.get_vector() - target_location.get_vector())
        if self.range_min < dist < self.range_max:
            return True
        else:
            return False

    def get_o3d_bbox_in_target_coordinate(self, target_transform: Transform, label: ObjectLabel):
        world_to_target = target_transform.get_inverse_matrix()
        label_to_world = label.transform.get_matrix()
        label_in_target = np.matmul(world_to_target, label_to_world)
        t_vec = label_in_target[0:3, -1]
        r_mat = label_in_target[0:3, 0:3]
        o3d_bbox = bbox_to_o3d_bbox(label.bounding_box)
        o3d_bbox.translate(t_vec)
        o3d_bbox.rotate(r_mat)
        o3d_bbox.color = np.array([1.0, 0, 0])
        return o3d_bbox



def main():
    rawdata_df = gather_rawdata_to_dataframe("record_2022_0112_2340",
                                             "vehicle.tesla.model3_1",
                                             "sensor.lidar.ray_cast_4",
                                             "sensor.camera.rgb_2")
    kitti_obj_labels = KittiObjectLabel(None, rawdata_df)
    kitti_obj_labels.process()


if __name__ == '__main__':
    main()
