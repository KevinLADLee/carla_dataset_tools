#!/usr/bin/python3
import os
import json
from enum import Enum

import carla

from param import RAW_DATA_PATH, ROOT_PATH
from utils.types import *
from utils.transform import transform_to_carla_transform

from recorder.actor import Actor, PseudoActor
from recorder.camera import RgbCamera, DepthCamera, SemanticSegmentationCamera
from recorder.lidar import Lidar, SemanticLidar
from recorder.radar import Radar
from recorder.vehicle import Vehicle
from recorder.infrastructure import Infrastructure


class NodeType(Enum):
    DEFAULT = 0
    VEHICLE = 1
    INFRASTRUCTURE = 2
    SENSOR = 3


class Node(object):
    def __init__(self, actor=None, node_type=NodeType.DEFAULT):
        self._actor = actor
        self._node_type = node_type
        self._children_nodes = []

    def add_child(self, node):
        self._children_nodes.append(node)

    def get_actor(self):
        return self._actor

    def get_children(self):
        return self._children_nodes

    def destroy(self):
        for node in self._children_nodes:
            node.destroy()
        if self._actor is not None:
            self._actor.destroy()

    # Tick for control step, running before world.tick()
    def tick_controller(self):
        if self._node_type == NodeType.VEHICLE:
            self._actor.control_step()

    # Depth-First-Search for data collection
    def tick_data_saving(self, frame_id, world_snapshot: carla.WorldSnapshot):
        if self._node_type == NodeType.SENSOR \
                or NodeType.VEHICLE:
            self._actor.save_to_disk(frame_id, world_snapshot, True)


class ActorFactory(object):
    def __init__(self, world: carla.World, base_save_dir=None):
        self._uid_count = 0
        self.world = world
        self.blueprint_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.base_save_dir = base_save_dir

    def create_actor_tree(self, actor_config_file):
        assert (self.base_save_dir is not None)
        if not actor_config_file or not os.path.exists(actor_config_file):
            raise RuntimeError(
                "Could not read actor config file from {}".format(actor_config_file))
        with open(actor_config_file) as handle:
            json_actors = json.loads(handle.read())

        root = Node()
        for actor_info in json_actors["actors"]:
            actor_type = str(actor_info["type"])
            node = Node()
            if actor_type.startswith("vehicle"):
                node = self.create_vehicle_node(actor_info)
                root.add_child(node)
            elif actor_type.startswith("infrastructure"):
                node = self.create_infrastructure_node(actor_info)
                root.add_child(node)
            if node is not None:
                # If it has sensor setting, then create subtree
                if actor_info["sensors_setting"] is not None:
                    sensor_info_file = actor_info["sensors_setting"]
                    with open("{}/config/{}".format(ROOT_PATH, sensor_info_file)) as sensor_handle:
                        sensors_setting = json.loads(sensor_handle.read())
                        for sensor_info in sensors_setting["sensors"]:
                            sensor_node = self.create_sensor_node(sensor_info, node.get_actor())
                            node.add_child(sensor_node)

        return root

    def create_vehicle_node(self, actor_info):
        vehicle_type = actor_info["type"]
        vehicle_name = actor_info["name"]
        spawn_point = actor_info["spawn_point"]
        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = self.create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                spawn_point.pop("roll", 0.0),
                spawn_point.pop("pitch", 0.0),
                spawn_point.pop("yaw", 0.0))
        blueprint = self.blueprint_lib.find(vehicle_type)
        carla_actor = self.world.spawn_actor(blueprint, transform)
        vehicle_object = Vehicle(uid=self.get_uid_count(),
                                 name=vehicle_name,
                                 base_save_dir=self.base_save_dir,
                                 carla_actor=carla_actor)
        vehicle_node = Node(vehicle_object, NodeType.VEHICLE)
        self._uid_count += 1
        return vehicle_node

    def create_infrastructure_node(self, actor_info):
        infrastructure_name = actor_info["name"]
        spawn_point = actor_info["spawn_point"]
        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = self.create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                spawn_point.pop("roll", 0.0),
                spawn_point.pop("pitch", 0.0),
                spawn_point.pop("yaw", 0.0))
        infrastructure_object = Infrastructure(uid=self.get_uid_count(),
                                               name=infrastructure_name,
                                               base_save_dir=self.base_save_dir,
                                               transform=transform)
        infrastructure_node = Node(infrastructure_object, NodeType.INFRASTRUCTURE)
        self._uid_count += 1
        return infrastructure_node

    def create_sensor_node(self, sensor_info: dict, parent_actor: PseudoActor):
        sensor_type = str(sensor_info.pop("type"))
        sensor_name = str(sensor_info.pop("name"))
        spawn_point = sensor_info.pop("spawn_point")
        sensor_transform = self.create_spawn_point(
            spawn_point.pop("x", 0.0),
            spawn_point.pop("y", 0.0),
            spawn_point.pop("z", 0.0),
            spawn_point.pop("roll", 0.0),
            spawn_point.pop("pitch", 0.0),
            spawn_point.pop("yaw", 0.0))
        blueprint = self.blueprint_lib.find(sensor_type)
        for attribute, value in sensor_info.items():
            blueprint.set_attribute(attribute, str(value))
        carla_actor = self.world.spawn_actor(blueprint, sensor_transform, parent_actor.get_carla_actor())

        sensor_actor = None
        if sensor_type == 'sensor.camera.rgb':
            sensor_actor = RgbCamera(uid=self.get_uid_count(),
                                     name=sensor_name,
                                     base_save_dir=parent_actor.get_save_dir(),
                                     carla_actor=carla_actor,
                                     parent=parent_actor)
        elif sensor_type == 'sensor.camera.depth':
            sensor_actor = DepthCamera(uid=self.get_uid_count(),
                                       name=sensor_name,
                                       base_save_dir=parent_actor.get_save_dir(),
                                       carla_actor=carla_actor,
                                       parent=parent_actor)
        elif sensor_type == 'sensor.camera.semantic_segmentation':
            sensor_actor = SemanticSegmentationCamera(uid=self.get_uid_count(),
                                                      name=sensor_name,
                                                      base_save_dir=parent_actor.get_save_dir(),
                                                      carla_actor=carla_actor,
                                                      parent=parent_actor)
        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_actor = Lidar(uid=self.get_uid_count(),
                                 name=sensor_name,
                                 base_save_dir=parent_actor.get_save_dir(),
                                 carla_actor=carla_actor,
                                 parent=parent_actor)
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            sensor_actor = SemanticLidar(uid=self.get_uid_count(),
                                         name=sensor_name,
                                         base_save_dir=parent_actor.get_save_dir(),
                                         carla_actor=carla_actor,
                                         parent=parent_actor)
        elif sensor_type == 'sensor.other.radar':
            sensor_actor = Radar(uid=self.get_uid_count(),
                                 name=sensor_name,
                                 base_save_dir=parent_actor.get_save_dir(),
                                 carla_actor=carla_actor,
                                 parent=parent_actor)
        else:
            print("Unsupported sensor type: {}".format(sensor_type))
            raise AttributeError
        sensor_node = Node(sensor_actor, NodeType.SENSOR)
        self._uid_count += 1
        return sensor_node

    def get_uid_count(self):
        return self._uid_count

    def create_spawn_point(self, x, y, z, roll, pitch, yaw):
        return transform_to_carla_transform(Transform(Location(x, y, z), Rotation(roll=roll, pitch=pitch, yaw=yaw)))
