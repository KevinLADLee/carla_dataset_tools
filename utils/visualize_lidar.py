#!/usr/bin/python3

import argparse
import glob
import os.path
import sys
import time
from enum import Enum
from pathlib import Path

import numpy
import numpy as np
import open3d as o3d

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import ROOT_PATH
from utils.semantic_helper import color_map


class PointcloudType(Enum):
        LIDAR = 0
        SEMANTIC_LIDAR = 1
        RADAR = 2


class LidarVisualizer:
    def __init__(self, pointcloud_type: PointcloudType, source: str):
        self.pointcloud_type = pointcloud_type
        self.source = source

    def visualize(self):
        if self.source.endswith('.npy'):
            raw_pcd = np.load(self.source)
            o3d_pcd = self.numpy_to_o3d(raw_pcd)
            o3d.visualization.draw_geometries([o3d_pcd])
        else:
            o3d_pcd = o3d.geometry.PointCloud()
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            files = sorted(glob.glob("{}/*.npy".format(self.source)))
            for file in files:
                vis.clear_geometries()
                raw_pcd = np.load(file)
                o3d_pcd = self.numpy_to_o3d(raw_pcd)
                print(o3d_pcd)
                vis.add_geometry(o3d_pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)
            vis.destroy_window()

    def numpy_to_o3d(self, numpy_cloud):
        pcd = o3d.geometry.PointCloud()
        if self.pointcloud_type == PointcloudType.LIDAR:
            # Nx4 -> Nx6
            numpy_cloud = np.insert(numpy_cloud, 4, numpy_cloud[:, 3], axis=1)
            numpy_cloud = np.insert(numpy_cloud, 4, numpy_cloud[:, 3], axis=1)
            # numpy cloud XYZIII -> o3d cloud XYZRGB
            # print("Poincloud: {}".format(numpy_cloud.shape))
            pcd.points = o3d.utility.Vector3dVector(numpy_cloud[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(numpy_cloud[:, 3:6])
            return pcd
        elif self.pointcloud_type == PointcloudType.SEMANTIC_LIDAR:
            # Read points
            x = numpy.asarray(numpy_cloud['x'], dtype=numpy.float32).reshape(-1, 1)
            y = numpy.asarray(numpy_cloud['y'], dtype=numpy.float32).reshape(-1, 1)
            z = numpy.asarray(numpy_cloud['z'], dtype=numpy.float32).reshape(-1, 1)
            points = numpy.concatenate((x, y, z), axis=1)
            points.reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(points[:, 0:3].reshape(-1, 3))

            # Read semantic tags, mapping color
            rgb = numpy.asarray(numpy_cloud['ObjTag'])
            rgb = color_map[rgb] * 1.0 / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            return pcd
        elif self.pointcloud_type == PointcloudType.RADAR:
            pcd.points = o3d.utility.Vector3dVector(numpy_cloud[:, 0:3])
            return pcd
        else:
            return pcd



def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--type',
        default='lidar',
        help='Type of point cloud (lidar / semantic_lidar / radar)'
    )
    argparser.add_argument(
        '--source',
        type=str,
        help='File or folder source to visualize'
    )

    args = argparser.parse_args()
    pointcloud_type = PointcloudType.LIDAR
    if args.type == 'lidar':
        pointcloud_type = PointcloudType.LIDAR
    elif args.type == 'semantic_lidar':
        pointcloud_type = PointcloudType.SEMANTIC_LIDAR
    elif args.type == 'radar':
        pointcloud_type = PointcloudType.RADAR
    else:
        print("Not valid point cloud type")
        raise RuntimeError

    source = args.source
    if not os.path.exists(source):
        source = "{}/{}".format(ROOT_PATH, source)
        if not os.path.exists(source):
            print("File or folder not exist: {}".format(source))
            raise RuntimeError
    print("Read data from: {}".format(source))

    lidar_visualizer = LidarVisualizer(pointcloud_type, source)
    lidar_visualizer.visualize()

if __name__ == "__main__":
    # execute only if run as a script
    main()