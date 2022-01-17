#!/usr/bin/python3
import copy
import math
import pickle
import sys
import time

import cv2
import numpy as np
import open3d.cpu.pybind.visualization
import pandas as pd
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool

from typing import List

import transforms3d.euler

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
        # cv2.namedWindow('preview_image', cv2.WINDOW_AUTOSIZE)
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
            bbox_list_3d = [o3d.geometry.OrientedBoundingBox]
            bbox_list_2d = []
            for label in objects_labels:
                # Kitti Object - Type
                if label.label_type == 'vehicle':
                    label_type = 'Car'
                elif label.label_type:
                    label_type = 'Pedestrian'
                else:
                    label_type = 'DontCare'

                if not self.is_valid_distance(lidar_trans.location, label.transform.location):
                    continue

                o3d_bbox = self.get_o3d_bbox_in_target_coordinate(lidar_trans, label)

                # Ignore backward vehicles
                if o3d_bbox.center[0] < 0:
                    continue

                # Check lidar points in bbox
                occlusion = self.cal_occlusion(o3d_pcd, o3d_bbox)
                if occlusion < 0:
                    continue

                # Transform bbox vertices to camera coordinate
                vertex_points = np.asarray(o3d_bbox.get_box_points())
                bbox_points_2d_x = []
                bbox_points_2d_y = []
                # bbox_points_3d = []
                for p in vertex_points:
                    p_c = self.lidar_point_to_cam(p, lidar_trans, cam_trans)
                    # bbox_points_3d.append(p_c)
                    p_uv = self.project_point_to_image(p_c, cam_mat)
                    bbox_points_2d_x.append(p_uv[0])
                    bbox_points_2d_y.append(p_uv[1])

                # bbox_points_3d = o3d.utility.Vector3dVector(np.asarray(bbox_points_3d)[:, 0:3])
                # bbox_3d_in_cam = o3d.geometry.OrientedBoundingBox.create_from_points(bbox_points_3d)
                # Generate 2d bbox by left-top point and right-bottom point
                # p_array = np.asarray([sorted(bbox_points_2d_x), sorted(bbox_points_2d_y)])
                # p_array = p_array.transpose()

                # print(p_array)

                x_min = min(bbox_points_2d_x)
                x_max = max(bbox_points_2d_x)
                y_min = min(bbox_points_2d_y)
                y_max = max(bbox_points_2d_y)
                # cv2.circle(image, (x_max, y_min), radius=1, color=(255, 0, 0), thickness=2)

                truncated = self.cal_truncated(image.shape[0], image.shape[1], [x_min, y_min, x_max, y_max])

                # For Debug
                # Draw 2d bbox
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=1)

                # T = np.matmul(lidar_trans.get_inverse_matrix(), cam_trans.get_matrix())
                # cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
                # cam_coord.rotate(T[0:3, 0:3])
                # cam_coord.translate(T[0:3, 3])

                # Transform 3d bbox to camera coordinate
                T_lc = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())
                o3d_pcd.rotate(T_lc[0:3, 0:3], np.array([0, 0, 0]))
                o3d_pcd.translate(T_lc[0:3, 3])

                o3d_bbox.rotate(T_lc[0:3, 0:3], np.array([0, 0, 0]))
                o3d_bbox.translate(T_lc[0:3, 3])

                cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)

                _, rotation_y, _ = transforms3d.euler.mat2euler(o3d_bbox.R)
                # print(math.degrees(rotation_y))

                bbox_center = np.asarray(o3d_bbox.center)
                theta = math.atan2(-bbox_center[0], bbox_center[2])
                alpha = rotation_y - theta
                alpha = math.atan2(math.sin(alpha), math.cos(alpha))

                o3d.visualization.draw_geometries([o3d_pcd, o3d_bbox, cam_coord])

                bbox_list_3d.append(o3d_bbox)
                bbox_list_2d.append([x_min, y_min, x_max, y_max])

            cv2.imshow('preview_image', image)
            # vis.add_geometry(o3d_pcd)
            # vis.poll_events()
            # vis.update_renderer()
            cv2.waitKey(1)
            time.sleep(0.05)

    # def generate_kitti_labels(self,
    #                           type: str,
    #                           bbox_3d_list: o3d.geometry.OrientedBoundingBox,
    #                           bbox_2d_list: List,
    #                           lidar_trans: Transform,
    #                           cam_trans: Transform):
    #
    #     truncation = self.cal_truncated()
    #     label_str = "{} {} {}".format(type, truncation, occlusion, alpha, xmin, ymin, xmax, ymax, height, witdth, length, location, ry)
    #     return

    def lidar_point_to_cam(self, point, lidar_trans, cam_trans):
        p = np.append(point, [1.0])
        T_wl = lidar_trans.get_matrix()
        T_cw = cam_trans.get_inverse_matrix()
        T_cl = np.matmul(T_cw, T_wl)
        p_c = np.matmul(T_cl, p)
        return p_c

    def project_point_to_image(self,
                            point_in_cam,
                            cam_mat: np.array):
        p_c = point_in_cam
        p_c = p_c[0:3] / p_c[2]
        p_uv = np.matmul(cam_mat, p_c)
        p_uv = p_uv[0:2].astype(int)
        return p_uv

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

    def cal_truncated(self, image_length, image_width, bbox_2d: list):
        # x_min y_min x_max y_max
        bbox_2d_in_img = copy.deepcopy(bbox_2d)
        bbox_2d_in_img[0] = max(bbox_2d[0], 0)
        bbox_2d_in_img[1] = max(bbox_2d[1], 0)
        bbox_2d_in_img[2] = min(bbox_2d[2], image_width)
        bbox_2d_in_img[3] = min(bbox_2d[3], image_length)

        size1 = (bbox_2d_in_img[2] - bbox_2d_in_img[0]) * (bbox_2d_in_img[3] - bbox_2d_in_img[1])
        size2 = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
        truncated = size1 / size2
        truncated = max(truncated, 0.0)
        truncated = min(truncated, 1.0)
        return truncated

    def cal_occlusion(self, pcd: o3d.geometry.PointCloud, bbox_3d: o3d.geometry.OrientedBoundingBox):
        occlusion = -1
        p_in_bbox = bbox_3d.get_point_indices_within_bounding_box(pcd.points)
        p_num = len(p_in_bbox)
        if p_num < self.points_min:
            return occlusion
        elif p_num > self.points_min:
            occlusion = 0
        if p_num + bbox_3d.center[0] < 250:
            occlusion = 1
        if p_num + bbox_3d.center[0] < 125:
            occlusion = 2
        return occlusion


def main():
    rawdata_df = gather_rawdata_to_dataframe("record_2022_0113_1337",
                                             "vehicle.tesla.model3_1",
                                             "sensor.lidar.ray_cast_4",
                                             "sensor.camera.rgb_2")
    kitti_obj_labels = KittiObjectLabel(None, rawdata_df)
    kitti_obj_labels.process()


if __name__ == '__main__':
    main()
