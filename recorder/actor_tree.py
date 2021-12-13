import json

from actor import Actor, PseudoActor
from camera import RgbCamera, DepthCamera, SemanticSegmentationCamera
from lidar import Lidar, SemanticLidar
from param import RAW_DATA_PATH
from utils.types import *
from vehicle import Vehicle


class Node(object):
    def __init__(self, actor: PseudoActor):
        self._actor = actor
        self._children_nodes = []

    def add_child(self, node):
        self._children_nodes.append(node)

    def get_actor(self):
        return self._actor


class ActorTree(object):
    def __init__(self, world: carla.World, actor_config_file):

        self.world = world
        self.map = self.world.get_map()
        self.actor_config_file = actor_config_file
        self.spawn_points = self.map.get_spawn_points()
        self.actor_factory = ActorFactory()
        self.root = Node(None)
        self.current_node = None

    def create_actor_tree(self):
        if not self.actor_config_file or not os.path.exists(self.actor_config_file):
            raise RuntimeError(
                "Could not read actor config file from {}".format(self.actor_config_file))
        with open(self.actor_config_file) as handle:
            json_actors = json.loads(handle.read())

        for actor_info in json_actors["actors"]:
            actor_type = str(actor_info["type"])
            if actor_type.startswith("vehicle"):
                node = self.actor_factory.create_vehicle_node(actor_info)
                self.add_node(node)
            elif actor_type.startswith("infrastructure"):
                # TODO
                pass
            # TODO: Create Sensor Node

    def add_node(self, node):
        self.root.add_child(node)


class ActorFactory(object):
    def __init__(self, world: carla.World):
        self._uid_count = 0
        self.world = world
        self.blueprint_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

    def create_carla_actor(self):
        pass

    def create_vehicle_node(self, actor_info):
        vehicle_type = actor_info["type"]
        vehicle_name = actor_info["name"]
        spawn_point = actor_info["spawn_point"]
        transform = self.spawn_points[spawn_point]
        blueprint = self.blueprint_lib.find(vehicle_type)
        carla_actor = self.world.spawn_actor(blueprint, transform)
        vehicle_object = Vehicle(uid=self.get_uid_count(),
                                 name=vehicle_name,
                                 base_save_dir=RAW_DATA_PATH.as_posix(),
                                 carla_actor=carla_actor)
        vehicle_node = Node(vehicle_object)
        self._uid_count += 1
        return vehicle_node

    def create_sensor_node(self, sensor_info: dict, parent_node: Node):
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
            blueprint.set_attribute(attribute, value)
        carla_actor = self.world.spawn_actor(blueprint, sensor_transform, parent_node.get_actor().get_carla_actor())

        sensor_actor = None
        if sensor_type == 'sensor.camera.rgb':
            sensor_actor = RgbCamera(uid=self.get_uid_count(),
                                     name=sensor_name,
                                     base_save_dir=parent_node.get_actor().get_save_dir(),
                                     carla_actor=carla_actor,
                                     parent=parent_node.get_actor())
        elif sensor_type == 'sensor.camera.depth':
            sensor_actor = DepthCamera(uid=self.get_uid_count(),
                                       name=sensor_name,
                                       base_save_dir=parent_node.get_actor().get_save_dir(),
                                       carla_actor=carla_actor,
                                       parent=parent_node.get_actor())
        elif sensor_type == 'sensor.camera.semantic_segmentation':
            sensor_actor = SemanticSegmentationCamera(uid=self.get_uid_count(),
                                                      name=sensor_name,
                                                      base_save_dir=parent_node.get_actor().get_save_dir(),
                                                      carla_actor=carla_actor,
                                                      parent=parent_node.get_actor())
        elif sensor_type == 'sensor.camera.ray_cast':
            sensor_actor = Lidar(uid=self.get_uid_count(),
                                 name=sensor_name,
                                 base_save_dir=parent_node.get_actor().get_save_dir(),
                                 carla_actor=carla_actor,
                                 parent=parent_node.get_actor())
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            sensor_actor = SemanticLidar(uid=self.get_uid_count(),
                                         name=sensor_name,
                                         base_save_dir=parent_node.get_actor().get_save_dir(),
                                         carla_actor=carla_actor,
                                         parent=parent_node.get_actor())
        sensor_node = Node(sensor_actor)
        self._uid_count += 1
        return sensor_node

    def get_uid_count(self):
        return self._uid_count

    def create_spawn_point(self, x, y, z, roll, pitch, yaw):
        return Transform(Location(x, y, z), Rotation(roll, pitch, yaw))

