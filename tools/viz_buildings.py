import pickle

import carla
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.transform import carla_bbox_to_bbox

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '-m', '--map',
        default='Town02',
        help='Load a new map to visualize'
    )

    args = argparser.parse_args()
    carla_client = carla.Client(args.host, args.port, worker_threads=1)

    carla_client.set_timeout(10.0)
    carla_client.load_world(args.map)


    carla_world = carla_client.get_world()
    carla_map = carla_world.get_map()

    buildings_carla_bbox = carla_world.get_level_bbs(carla.CityObjectLabel.Buildings)
    # conver carla bbox to bbox
    buildings_bbox = []
    for cb in buildings_carla_bbox:
        bbox = carla_bbox_to_bbox(cb)
        buildings_bbox.append(bbox)

    # visualize buildings in Open3D
    o3d_bboxes = []
    for bbox in buildings_bbox:
        o3d_bbox = bbox.to_open3d(color=[0.7, 0.5, 0.5])
        o3d_bboxes.append(o3d_bbox)


    o3d.visualization.draw_geometries(o3d_bboxes)

if __name__ == "__main__":
    # execute only if run as a script
    main()
