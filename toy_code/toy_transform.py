import carla
from utils.transform import carla_transform_to_numpy_mat


def main():
    trans = carla.Transform(carla.Location(1, 2, 3), carla.Rotation(30, 60, 90))
    carla_transform_to_numpy_mat(trans)


if __name__ == "__main__":
    main()
