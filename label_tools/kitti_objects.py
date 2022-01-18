#!/usr/bin/python3
import argparse
import copy
import glob
import math
import os
import pickle
import sys
import time

import cv2
import numpy as np
import open3d.cuda.pybind.geometry
import pandas as pd
from pathlib import Path
from multiprocessing.pool import Pool as ThreadPool

from typing import List

import transforms3d.euler

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH, DATASET_PATH
from utils.transform import *
from utils.label_types import ObjectLabel
from label_tools.data_loader import *


def gather_rawdata_to_dataframe(record_name: str, vehicle_name: str, lidar_path: str, camera_path: str):
    rawdata_frames_df = pd.DataFrame()
    vehicle_poses_df = load_vehicle_pose("{}/{}/{}".format(RAW_DATA_PATH, record_name, vehicle_name))
    rawdata_frames_df = vehicle_poses_df

    object_labels_path_df = load_object_labels("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name))
    rawdata_frames_df = pd.merge(rawdata_frames_df, object_labels_path_df, how='outer', on='frame')

    lidar_rawdata_df = load_lidar_rawdata(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{lidar_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, lidar_rawdata_df, how='outer', on='frame')

    camera_rawdata_path_df = load_camera_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{camera_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, camera_rawdata_path_df, how='outer', on='frame')

    return rawdata_frames_df


def write_pointcloud(output_dir: str, frame_id: str, lidar_data: np.array):
    lidar_dir = f"{output_dir}/velodyne"
    os.makedirs(lidar_dir, exist_ok=True)
    file_path = "{}/{}.bin".format(lidar_dir, frame_id)
    lidar_data.tofile(file_path)


def write_image(output_dir: str, frame_id: str, image: np.array):
    image_dir = f"{output_dir}/image_2"
    os.makedirs(image_dir, exist_ok=True)
    file_path = "{}/{}.png".format(image_dir, frame_id)
    cv2.imwrite(file_path, image)


def write_label(output_dir, frame_id, kitti_labels):
    label_dir = f"{output_dir}/label_2"
    os.makedirs(label_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(label_dir, frame_id)

    if len(kitti_labels) < 1:
        kitti_labels.append('DontCare -1 -1 -10 522.25 202.35 547.77 219.71 -1 -1 -1 -1000 -1000 -1000 -10 -10 \n')

    with open(file_path, 'w') as label_file:
        label_file.writelines(kitti_labels)


def write_calib(output_dir, frame_id, lidar_trans: Transform, cam_trans: Transform, camera_mat: np.array):
    """ Saves the calibration matrices to a file.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame.
    """
    calib_dir = f"{output_dir}/calib"
    os.makedirs(calib_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(calib_dir, frame_id)

    camera_mat = np.concatenate((camera_mat, np.array([[0.0], [0.0], [0.0]])), axis=1)
    camera_mat = camera_mat.reshape(1, 12)
    calib_str = "P2: "
    for x in camera_mat[0]:
        calib_str += str(x)
        calib_str += ' '
    calib_str += '\n'

    calib_str += "R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 \n"

    velo_to_cam = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())
    velo_to_cam = velo_to_cam[0:3, :]
    velo_to_cam = velo_to_cam.reshape(1, 12).tolist()
    calib_str += "Tr_velo_to_cam: "
    for x in velo_to_cam[0]:
        calib_str += str(x)
        calib_str += ' '

    with open(file_path, 'w') as calib_file:
        calib_file.write(calib_str)
        calib_file.close()


def generate_kitti_labels(label_type: str,
                          truncated: float,
                          occlusion: float,
                          alpha: float,
                          bbox_2d: List,
                          bbox_3d: o3d.geometry.OrientedBoundingBox,
                          rotation_y: float):

    # Note: Kitti Object 3d bbox location is top-plane-center, not the bbox center
    label_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n".format(label_type, truncated, occlusion, alpha,
                                                                         bbox_2d[0], bbox_2d[1],
                                                                         bbox_2d[2], bbox_2d[3],
                                                                         bbox_3d.extent[2],
                                                                         bbox_3d.extent[1],
                                                                         bbox_3d.extent[0],
                                                                         bbox_3d.center[0],
                                                                         bbox_3d.center[1] + (bbox_3d.extent[2] / 2.0),
                                                                         bbox_3d.center[2],
                                                                         rotation_y)
    return label_str


def bbox_to_o3d_bbox_in_target_coordinate(label: ObjectLabel, target_transform: Transform):
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


def cal_truncated(image_length, image_width, bbox_2d: list):
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


class KittiObjectLabelTool:
    def __init__(self, record_name, vehicle_name, rawdata_df: pd.DataFrame):
        self.record_name = record_name
        self.vehicle_name = vehicle_name
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
        # cv2.namedWindow('preview_image', cv2.WINDOW_AUTOSIZE)

        # start = time.time()
        # thread_pool = ThreadPool()
        # thread_pool.starmap(self.process_frame, self.rawdata_df.iterrows())
        # thread_pool.close()
        # thread_pool.join()
        # print("Cost: {:0<10f}s".format(time.time() - start))

        # start = time.time()
        for index, frame in self.rawdata_df.iterrows():
        # vis.clear_geometries()
            self.process_frame(index, frame)
        # cv2.imshow('preview_image', image)
        # vis.add_geometry(o3d_pcd)
        # vis.poll_events()
        # vis.update_renderer()
        # cv2.waitKey(1)
        # time.sleep(0.05)
        # print("Cost: {:0<10f}s".format(time.time()-start))

    def process_frame(self, index, frame):
        frame_id = "{:0>6d}".format(frame['frame'])
        lidar_trans: Transform = frame['lidar_pose']
        cam_trans: Transform = frame['camera_pose']
        cam_mat = np.asarray(frame['camera_matrix'])

        image = cv2.imread(frame['camera_rawdata_path'], cv2.IMREAD_UNCHANGED)

        lidar_data = numpy.load(frame['lidar_rawdata_path'])
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(lidar_data[:, 0:3])

        # Load object labels from pickle data
        with open(frame['object_labels_path'], 'rb') as pkl_file:
            objects_labels = pickle.load(pkl_file)

        # Convert all bbox to o3d bbox
        bbox_list_3d = [o3d.geometry.OrientedBoundingBox]
        bbox_list_2d = []
        kitti_labels = []
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

            o3d_bbox = bbox_to_o3d_bbox_in_target_coordinate(label, lidar_trans)

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
            bbox_2d = [x_min, y_min, x_max, y_max]
            truncated = cal_truncated(image.shape[0], image.shape[1], bbox_2d)

            # For Debug
            # Draw 2d bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=1)

            # Transform 3d bbox to camera coordinate
            T_lc = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())
            o3d_pcd.rotate(T_lc[0:3, 0:3], np.array([0, 0, 0]))
            o3d_pcd.translate(T_lc[0:3, 3])

            o3d_bbox.rotate(T_lc[0:3, 0:3], np.array([0, 0, 0]))
            o3d_bbox.translate(T_lc[0:3, 3])

            _, rotation_y, _ = transforms3d.euler.mat2euler(o3d_bbox.R)
            # print(math.degrees(rotation_y))

            bbox_center = np.asarray(o3d_bbox.center)
            theta = math.atan2(-bbox_center[0], bbox_center[2])
            alpha = rotation_y - theta
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            #
            # if frame['frame'] == 98:
            #     cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
            #     box_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=bbox_center)
            #     o3d.visualization.draw_geometries([o3d_pcd, o3d_bbox, cam_coord, box_coord])

            kitti_label = generate_kitti_labels(label_type, truncated, occlusion, alpha,
                                                bbox_2d, o3d_bbox, rotation_y)

            kitti_labels.append(kitti_label)
            bbox_list_3d.append(o3d_bbox)
            bbox_list_2d.append(bbox_2d)

        output_dir = "{}/{}/{}/training".format(DATASET_PATH, self.record_name, self.vehicle_name)

        write_calib(output_dir, frame_id, lidar_trans, cam_trans, cam_mat)
        write_label(output_dir, frame_id, kitti_labels)
        write_image(output_dir, frame_id, image)
        write_pointcloud(output_dir, frame_id, lidar_data)

    def priview_label_result(self, pcd: o3d.geometry.PointCloud, img: np.array, kitti_labels):
        return

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
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record', '-r',
        required=True,
        help='Rawdata Record ID. e.g. record_2022_0113_1337'
    )
    argparser.add_argument(
        '--vehicle', '-v',
        default='all',
        help='Vehicle name. e.g. `vehicle.tesla.model3_1`. Default to all vehicles. '
    )
    argparser.add_argument(
        '--lidar', '-l',
        default='velodyne',
        help='Lidar name. e.g. sensor.lidar.ray_cast_4'
    )
    argparser.add_argument(
        '--camera', '-c',
        default='image_2',
        help='Camera name. e.g. sensor.camera.rgb_2'
    )

    args = argparser.parse_args()

    record_name = args.record
    if args.vehicle is 'all':
        vehicle_name_list = [os.path.basename(x) for x in glob.glob('{}/{}/vehicle.*'.format(RAW_DATA_PATH, record_name))]
    else:
        vehicle_name_list = [args.vehicle]

    for vehicle_name in vehicle_name_list:
        rawdata_df = gather_rawdata_to_dataframe(args.record,
                                                 vehicle_name,
                                                 args.lidar,
                                                 args.camera)
        print("Process {} - {}".format(record_name, vehicle_name))
        kitti_obj_label_tool = KittiObjectLabelTool(record_name, vehicle_name, rawdata_df)
        kitti_obj_label_tool.process()


if __name__ == '__main__':
    main()
