#!/usr/bin/python3
import sys
from pathlib import Path
import carla
sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.transform import *


def main():
    print("CarlaTypes: ")
    carla_location = carla.Location(1, 2, 3)
    carla_rotation = carla.Rotation(30, 60, 90)
    carla_transform = carla.Transform(carla_location, carla_rotation)
    print("{}\n{}\n{}".format(carla_location, carla_rotation, carla_transform))
    print("---------------------------")
    print("CustomTypes: ")
    location = carla_location_to_location(carla_location)
    rotation = carla_rotation_to_rotation(carla_rotation)
    transform = carla_transform_to_transform(carla_transform)
    print(location)
    print(rotation)
    print(transform)
    print("---------------------------")
    carla_location_1 = location_to_carla_location(location)
    carla_rotation_1 = rotation_to_carla_rotation(rotation)
    carla_transform_1 = transform_to_carla_transform(transform)
    print(carla_location_1 == carla_location,
          carla_rotation_1 == carla_rotation,
          carla_transform_1 == carla_transform)

    carla_point = carla.Location(3, 2, 1)
    carla_trans_point = carla_transform.transform(carla_point)
    point = carla_location_to_location(carla_point)
    trans_point = transform.transform(point)
    print(trans_point,
          carla_location_to_location(carla_trans_point))


if __name__ == "__main__":
    main()
