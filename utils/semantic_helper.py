#!/usr/bin/python3

# Reference page: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
import numpy


def obj_tag_to_rgb_color(obj_tag):
    obj_tag = int(obj_tag)
    rgb = {
        '0': (0, 0, 0),
        '1': (70, 70, 70),
        '2': (100, 40, 40),
        '3': (55, 90, 80),
        '4': (220, 20, 60),
        '5': (153, 153, 153),
        '6': (157, 234, 50),
        '7': (128, 64, 128),
        '8': (244, 35, 232),
        '9': (107, 142, 35),
        '10': (0, 0, 142),
        '11': (102, 102, 156),
        '12': (220, 220, 0),
        '13': (70, 130, 180),
        # Ground
        '14': (81, 0, 81),
        '15': (150, 100, 100),
        '16': (230, 150, 140),
        '17': (180, 165, 180),
        # TrafficLight
        '18': (250, 170, 30),
        # Static
        '19': (110, 190, 160),
        # Dynamic
        '20': (170, 120, 50),
        '21': (45, 60, 150),
        '22': (145, 170, 100)
    }.get(str(obj_tag))
    rgb = numpy.asarray(rgb, dtype=numpy.float64)
    rgb *= 1.0 / 255.0
    return rgb
