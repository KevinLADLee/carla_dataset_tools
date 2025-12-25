#!/usr/bin/python3
import os
import random
import warnings
import logging
from enum import Enum
from pathlib import Path
import yaml

import carla

from param import RAW_DATA_PATH, ROOT_PATH
from core.geometry import *
from core.transform import transform_to_carla_transform

from recorder.actor import Actor, PseudoActor
from recorder.camera import RgbCamera, DepthCamera, SemanticSegmentationCamera
from recorder.lidar import Lidar, SemanticLidar
from recorder.radar import Radar
from recorder.vehicle import Vehicle, OtherVehicle
from recorder.infrastructure import Infrastructure
from recorder.world import WorldActor

# Get logger instance
logger = logging.getLogger(__name__)


class NodeType(Enum):
    DEFAULT = 0
    WORLD = 1
    VEHICLE = 2
    INFRASTRUCTURE = 3
    SENSOR = 4
    OTHER_VEHICLE = 5


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

    def get_node_type(self):
        return self._node_type

    def destroy(self):
        for node in self._children_nodes:
            node.destroy()
        if self._actor is not None:
            self._actor.destroy()

    # Tick for control step, running before world.tick()
    def tick_controller(self):
        if self._node_type == NodeType.VEHICLE or \
                self._node_type == NodeType.OTHER_VEHICLE:
            self._actor.control_step()

    def tick_data_saving(self, frame_id, timestamp):
        """
        Save data for this node

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Timestamp

        Returns:
            Save information from the actor
        """
        # Fixed condition: check if node type is one of SENSOR, VEHICLE, or WORLD
        if self.get_node_type() in (NodeType.SENSOR, NodeType.VEHICLE, NodeType.WORLD):
            return self._actor.save_to_disk(frame_id, timestamp, True)


def get_name_from_json(json_info, name_set: set):
    # Get actor name from json, default to ''.
    # Actor will generate unique name "[TYPE_ID]_[UID]" later for saving data.
    try:
        name = str(json_info.pop("name"))
    except (KeyError, AttributeError):
        name = ''

    # If there is a same name in the name set, fallback to default
    if name != '':
        if name in name_set:
            warnings.warn(f"Invalid duplicated name {name}, fallback to default.")
            name = ''
        else:
            name_set.add(name)

    return name


def create_spawn_point(x, y, z, roll, pitch, yaw):
    return transform_to_carla_transform(Transform(Location(x, y, z), Rotation(roll=roll, pitch=pitch, yaw=yaw)))


class ActorFactory(object):
    def __init__(self, world: carla.World, base_save_dir=None):
        self._uid_count = 0
        self.world = world
        self.blueprint_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.base_save_dir = base_save_dir
        self.v2x_layer_name_set = set()
        self.sensor_layer_name_set = set()

    def create_actor_tree(self, config):
        """
        Create actor tree from unified configuration

        Args:
            config: Configuration dictionary from ConfigManager

        Returns:
            Root node of the actor tree
        """
        assert (self.base_save_dir is not None)

        if not config or 'actors' not in config:
            raise RuntimeError("Invalid configuration: missing 'actors' section")

        logger.info("Creating world node...")
        root = self.create_world_node()

        # Create actors from config
        logger.info(f"Creating {len(config['actors'])} actors...")
        for idx, actor_info in enumerate(config["actors"]):
            actor_type = str(actor_info["type"])
            actor_name = actor_info.get("name", f"actor_{idx}")
            logger.info(f"  [{idx+1}/{len(config['actors'])}] Creating {actor_type} '{actor_name}'...")
            node = Node()

            if actor_type.startswith("vehicle"):
                node = self.create_vehicle_node(actor_info)
                root.add_child(node)
            elif actor_type.startswith("infrastructure"):
                node = self.create_infrastructure_node(actor_info)
                root.add_child(node)

            if node is not None:
                # If actor has sensors, create sensor nodes
                if "sensors" in actor_info and actor_info["sensors"]:
                    sensor_count = len(actor_info["sensors"])
                    logger.info(f"    Creating {sensor_count} sensors for '{actor_name}'...")
                    sensor_name_set = set()
                    for sensor_idx, sensor_info in enumerate(actor_info["sensors"]):
                        sensor_type = sensor_info.get("type", "unknown")
                        sensor_name = sensor_info.get("name", f"sensor_{sensor_idx}")
                        logger.debug(f"      [{sensor_idx+1}/{sensor_count}] Creating {sensor_type} '{sensor_name}'...")
                        sensor_node = self.create_sensor_node(
                            sensor_info, node.get_actor(), sensor_name_set
                        )
                        node.add_child(sensor_node)

        # Create other/background vehicles
        other_vehicle_info = config.get("other_vehicles", {})
        logger.info("Creating background traffic vehicles...")
        ov_nodes = self.create_other_vehicles(other_vehicle_info)
        root.get_children().extend(ov_nodes)
        logger.info(f"✓ Created {len(ov_nodes)} background vehicles")

        return root

    def create_world_node(self):
        world_actor = WorldActor(uid=self.generate_uid(),
                                 carla_world=self.world,
                                 base_save_dir=self.base_save_dir)
        world_node = Node(world_actor, NodeType.WORLD)
        return world_node

    def create_vehicle_node(self, actor_info):
        vehicle_type = actor_info["type"]
        vehicle_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
        spawn_point = actor_info["spawn_point"]
        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                spawn_point.pop("roll", 0.0),
                spawn_point.pop("pitch", 0.0),
                spawn_point.pop("yaw", 0.0))
        blueprint = self.blueprint_lib.find(vehicle_type)
        carla_actor = self.world.spawn_actor(blueprint, transform)
        logger.debug(f"Created vehicle: {vehicle_name}")

        # Parse route configuration if present
        route_config = None
        if "route" in actor_info:
            route_config = self._parse_route_config(actor_info["route"])

        vehicle_object = Vehicle(uid=self.generate_uid(),
                                 name=vehicle_name,
                                 base_save_dir=self.base_save_dir,
                                 carla_actor=carla_actor,
                                 route_config=route_config)

        # Note: In synchronous mode, actors need a world tick before set_autopilot()
        # Autopilot will be set after initialization in DataRecorder

        vehicle_node = Node(vehicle_object, NodeType.VEHICLE)
        return vehicle_node

    def create_other_vehicles(self, other_vehicles_info):
        """
        Create background traffic vehicles

        Args:
            other_vehicles_info: Dictionary with 'count' and optional 'spawn_points'

        Returns:
            List of other vehicle nodes
        """
        blueprints = self.blueprint_lib.filter('vehicle.*')
        other_vehicle_nodes = []

        # Get spawn points list if specified
        try:
            spawn_points = other_vehicles_info.get('spawn_points', [])
        except (AttributeError, ValueError):
            spawn_points = []

        # Spawn vehicles at specific points
        if spawn_points:
            for spawn_point in spawn_points:
                bp = random.choice(blueprints)
                transform = self.spawn_points[spawn_point]
                carla_actor = self.world.spawn_actor(bp, transform)
                other_vehicle_object = OtherVehicle(uid=self.generate_uid(),
                                                    name='',
                                                    base_save_dir="/tmp",
                                                    carla_actor=carla_actor)
                other_vehicle_node = Node(other_vehicle_object, NodeType.OTHER_VEHICLE)
                other_vehicle_nodes.append(other_vehicle_node)

        # Spawn random vehicles
        try:
            vehicle_count = other_vehicles_info.get('count', 0)
        except (AttributeError, ValueError):
            vehicle_count = 0

        if vehicle_count:
            for i in range(vehicle_count):
                bp = random.choice(blueprints)
                all_spawn_points = self.world.get_map().get_spawn_points()
                try:
                    carla_actor = self.world.spawn_actor(bp, random.choice(all_spawn_points))
                except RuntimeError:
                    i -= 1
                    continue

                other_vehicle_object = OtherVehicle(uid=self.generate_uid(),
                                                    name='',
                                                    base_save_dir="/tmp",
                                                    carla_actor=carla_actor)
                other_vehicle_node = Node(other_vehicle_object, NodeType.OTHER_VEHICLE)
                other_vehicle_nodes.append(other_vehicle_node)

        return other_vehicle_nodes

    def create_infrastructure_node(self, actor_info, carla_actor: carla.Actor = None):
        """
        Create infrastructure node (legacy mode - for create_actor_tree)
        
        Args:
            actor_info: Infrastructure configuration dictionary
            carla_actor: Optional pre-spawned CARLA actor (for batch mode)
        """
        infrastructure_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
        spawn_point = actor_info["spawn_point"]
        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                0,
                0,
                0,
            )
        # If carla_actor is provided (from batch spawn), use it; otherwise spawn directly (legacy)
        if carla_actor is None:
            # Legacy mode: spawn directly (for backward compatibility)
            blueprint = self._get_infrastructure_attachment_blueprint()
            if blueprint is not None:
                try:
                    carla_actor = self.world.spawn_actor(blueprint, transform)
                    carla_actor.set_simulate_physics(False)
                    carla_actor.set_enable_gravity(False)
                except Exception as e:
                    logger.warning(f"Failed to spawn attachment actor for infrastructure '{infrastructure_name}': {e}")
                    carla_actor = None
        
        infrastructure_object = Infrastructure(uid=self.generate_uid(),
                                               name=infrastructure_name,
                                               base_save_dir=self.base_save_dir,
                                               transform=transform,
                                               carla_actor=carla_actor)
        infrastructure_node = Node(infrastructure_object, NodeType.INFRASTRUCTURE)
        return infrastructure_node
    
    def _get_infrastructure_attachment_blueprint(self):
        """
        Get blueprint for infrastructure attachment actor.
        Returns the smallest available static prop.
        
        Returns:
            carla.ActorBlueprint or None
        """
        # Try to find the smallest static prop (colacan - cola can)
        blueprint = self.blueprint_lib.find('static.prop.colacan')
        if blueprint is None:
            # Fallback to other small props
            blueprint = self.blueprint_lib.find('static.prop.mobile')
            if blueprint is None:
                blueprint = self.blueprint_lib.find('static.prop.briefcase')
        return blueprint
    
    def create_infrastructure_spawn_command(self, actor_info):
        """
        Create infrastructure spawn command (不立即执行spawn) - for batch mode
        
        Args:
            actor_info: Infrastructure configuration dictionary
            
        Returns:
            dict: Spawn command containing all necessary information
            {
                'type': 'infrastructure',
                'blueprint': carla.ActorBlueprint,
                'transform': carla.Transform,
                'actor_info': dict,
                'infrastructure_name': str,
                'sensors': [sensor_cmd1, sensor_cmd2, ...]
            }
        """
        infrastructure_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
        spawn_point = actor_info["spawn_point"]
        
        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                0,
                0,
                0,
            )
        
        # Get blueprint for attachment actor
        blueprint = self._get_infrastructure_attachment_blueprint()
        if blueprint is None:
            logger.warning(f"Could not find suitable static prop blueprint for infrastructure '{infrastructure_name}'")
        
        # Create sensor commands
        sensor_commands = []
        if "sensors" in actor_info and actor_info["sensors"]:
            for sensor_info in actor_info["sensors"]:
                sensor_cmd = self.create_sensor_spawn_command(sensor_info, infrastructure_name)
                sensor_commands.append(sensor_cmd)
        
        return {
            'type': 'infrastructure',
            'blueprint': blueprint,
            'transform': transform,
            'actor_info': actor_info,
            'infrastructure_name': infrastructure_name,
            'sensors': sensor_commands
        }

    def _parse_route_config(self, route_info):
        """
        Parse route configuration from actor info

        Args:
            route_info: Route configuration dictionary

        Returns:
            Parsed route configuration dictionary
        """
        route_config = {}

        # Check if loading from file
        if "from_file" in route_info:
            route_file = Path(route_info["from_file"])
            # If relative path, make it relative to project root
            if not route_file.is_absolute():
                route_file = Path(ROOT_PATH) / route_file

            try:
                with open(route_file, 'r', encoding='utf-8') as f:
                    route_data = yaml.safe_load(f)
                    # Use waypoints and mode from file
                    route_config['waypoints'] = route_data.get('waypoints', [])
                    route_config['mode'] = route_data.get('mode', 'strict')
                    logger.info(f"Loaded route from file: {route_file} ({len(route_config['waypoints'])} waypoints)")
            except Exception as e:
                warnings.warn(f"Failed to load route from {route_file}: {e}")
                return None
        else:
            # Use directly specified waypoints
            route_config['waypoints'] = route_info.get('waypoints', [])
            route_config['mode'] = route_info.get('mode', 'strict')

        # Validate minimum requirements
        if len(route_config.get('waypoints', [])) < 2:
            warnings.warn("Route must have at least 2 waypoints. Ignoring route.")
            return None

        return route_config

    def create_sensor_node(self, sensor_info: dict, parent_actor: PseudoActor, sensor_name_set: set):
        sensor_type = str(sensor_info.pop("type"))
        sensor_name = get_name_from_json(sensor_info, sensor_name_set)
        spawn_point = sensor_info.pop("spawn_point")
        sensor_transform = create_spawn_point(
            spawn_point.pop("x", 0.0),
            spawn_point.pop("y", 0.0),
            spawn_point.pop("z", 0.0),
            spawn_point.pop("roll", 0.0),
            spawn_point.pop("pitch", 0.0),
            spawn_point.pop("yaw", 0.0))
        blueprint = self.blueprint_lib.find(sensor_type)
        for attribute, value in sensor_info.items():
            blueprint.set_attribute(attribute, str(value))
        if parent_actor.get_carla_actor() is not None:
            carla_actor = self.world.spawn_actor(blueprint, sensor_transform, parent_actor.get_carla_actor())
        else:
            sensor_location = parent_actor.get_carla_transform().transform(sensor_transform.location)
            sensor_transform = carla.Transform(sensor_location, sensor_transform.rotation)
            carla_actor = self.world.spawn_actor(blueprint, sensor_transform)

        sensor_actor = None
        if sensor_type == 'sensor.camera.rgb':
            sensor_actor = RgbCamera(uid=self.generate_uid(),
                                     name=sensor_name,
                                     base_save_dir=parent_actor.get_save_dir(),
                                     carla_actor=carla_actor,
                                     parent=parent_actor)
        elif sensor_type == 'sensor.camera.depth':
            sensor_actor = DepthCamera(uid=self.generate_uid(),
                                       name=sensor_name,
                                       base_save_dir=parent_actor.get_save_dir(),
                                       carla_actor=carla_actor,
                                       parent=parent_actor)
        elif sensor_type == 'sensor.camera.semantic_segmentation':
            sensor_actor = SemanticSegmentationCamera(uid=self.generate_uid(),
                                                      name=sensor_name,
                                                      base_save_dir=parent_actor.get_save_dir(),
                                                      carla_actor=carla_actor,
                                                      parent=parent_actor)
        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_actor = Lidar(uid=self.generate_uid(),
                                 name=sensor_name,
                                 base_save_dir=parent_actor.get_save_dir(),
                                 carla_actor=carla_actor,
                                 parent=parent_actor)
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            sensor_actor = SemanticLidar(uid=self.generate_uid(),
                                         name=sensor_name,
                                         base_save_dir=parent_actor.get_save_dir(),
                                         carla_actor=carla_actor,
                                         parent=parent_actor)
        elif sensor_type == 'sensor.other.radar':
            sensor_actor = Radar(uid=self.generate_uid(),
                                 name=sensor_name,
                                 base_save_dir=parent_actor.get_save_dir(),
                                 carla_actor=carla_actor,
                                 parent=parent_actor)
        else:
            logger.error(f"Unsupported sensor type: {sensor_type}")
            raise AttributeError(f"Unsupported sensor type: {sensor_type}")
        sensor_node = Node(sensor_actor, NodeType.SENSOR)
        return sensor_node

    def generate_uid(self):
        uid = self._uid_count
        self._uid_count += 1
        return uid

    # ========================================================================
    # Command Batching Methods (for synchronous initialization)
    # ========================================================================

    def create_vehicle_spawn_command(self, actor_info):
        """
        Create vehicle spawn command (不立即执行spawn)

        Args:
            actor_info: Vehicle configuration dictionary

        Returns:
            dict: Spawn command containing all necessary information
            {
                'type': 'vehicle',
                'blueprint': carla.ActorBlueprint,
                'transform': carla.Transform,
                'actor_info': dict,
                'vehicle_name': str,
                'route_config': dict or None,
                'use_autopilot': bool,
                'sensors': [sensor_cmd1, sensor_cmd2, ...]
            }
        """
        vehicle_type = actor_info["type"]
        vehicle_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
        spawn_point = actor_info["spawn_point"]

        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                spawn_point.pop("roll", 0.0),
                spawn_point.pop("pitch", 0.0),
                spawn_point.pop("yaw", 0.0)
            )

        blueprint = self.blueprint_lib.find(vehicle_type)

        # Parse route configuration
        route_config = None
        if "route" in actor_info:
            route_config = self._parse_route_config(actor_info["route"])

        use_autopilot = (route_config is None)

        # Create sensor commands
        sensor_commands = []
        if "sensors" in actor_info and actor_info["sensors"]:
            for sensor_info in actor_info["sensors"]:
                sensor_cmd = self.create_sensor_spawn_command(sensor_info, vehicle_name)
                sensor_commands.append(sensor_cmd)

        return {
            'type': 'vehicle',
            'blueprint': blueprint,
            'transform': transform,
            'actor_info': actor_info,
            'vehicle_name': vehicle_name,
            'route_config': route_config,
            'use_autopilot': use_autopilot,
            'sensors': sensor_commands
        }

    def create_sensor_spawn_command(self, sensor_info, parent_vehicle_name):
        """
        Create sensor spawn command

        Args:
            sensor_info: Sensor configuration dictionary
            parent_vehicle_name: Parent vehicle name (for later association)

        Returns:
            dict: Sensor spawn command
        """
        sensor_type = str(sensor_info.get("type"))
        sensor_name = sensor_info.get("name", "")

        # Get blueprint
        blueprint = self.blueprint_lib.find(sensor_type)

        # Set attributes
        for attr_key, attr_value in sensor_info.items():
            if attr_key not in ['type', 'name', 'spawn_point']:
                if blueprint.has_attribute(attr_key):
                    blueprint.set_attribute(attr_key, str(attr_value))

        # Get relative transform
        spawn_point = sensor_info["spawn_point"]
        transform = create_spawn_point(
            spawn_point.get("x", 0.0),
            spawn_point.get("y", 0.0),
            spawn_point.get("z", 0.0),
            spawn_point.get("roll", 0.0),
            spawn_point.get("pitch", 0.0),
            spawn_point.get("yaw", 0.0)
        )

        return {
            'type': 'sensor',
            'sensor_type': sensor_type,
            'sensor_name': sensor_name,
            'blueprint': blueprint,
            'transform': transform,
            'parent_vehicle_name': parent_vehicle_name,
            'sensor_info': sensor_info
        }

    def create_other_vehicle_spawn_commands(self, other_vehicles_info):
        """
        Create background vehicle spawn commands (avoid duplicate spawn points)

        Args:
            other_vehicles_info: Dictionary with 'count' and optional 'spawn_points'

        Returns:
            list: List of spawn command dictionaries
        """
        commands = []
        blueprints = self.blueprint_lib.filter('vehicle.*')

        # Get spawn points list if specified
        try:
            spawn_points = other_vehicles_info.get('spawn_points', [])
        except (AttributeError, ValueError):
            spawn_points = []

        # Create commands for vehicles at specific points
        for spawn_point in spawn_points:
            bp = random.choice(blueprints)
            transform = self.spawn_points[spawn_point]
            commands.append({
                'type': 'other_vehicle',
                'blueprint': bp,
                'transform': transform,
                'use_autopilot': True
            })

        # Create commands for random vehicles
        try:
            vehicle_count = other_vehicles_info.get('count', 0)
        except (AttributeError, ValueError):
            vehicle_count = 0

        if vehicle_count:
            # Get all available spawn points
            all_spawn_points = self.world.get_map().get_spawn_points()

            # Limit count to available spawn points (CARLA official pattern)
            if vehicle_count > len(all_spawn_points):
                logger.warning(
                    f"Requested {vehicle_count} other_vehicles, but only "
                    f"{len(all_spawn_points)} spawn points available. "
                    f"Limiting to {len(all_spawn_points)}."
                )
                vehicle_count = len(all_spawn_points)

            # Shuffle to avoid duplicates (CARLA official pattern)
            random.shuffle(all_spawn_points)

            # Create commands using shuffled spawn points
            for i in range(vehicle_count):
                bp = random.choice(blueprints)
                transform = all_spawn_points[i]  # No duplicates!
                commands.append({
                    'type': 'other_vehicle',
                    'blueprint': bp,
                    'transform': transform,
                    'use_autopilot': True
                })

        return commands

    def create_sensor_object(self, sensor_cmd, carla_sensor, parent_vehicle):
        """
        Create sensor object from command and spawned actor

        Args:
            sensor_cmd: Sensor spawn command dictionary
            carla_sensor: Spawned CARLA sensor actor
            parent_vehicle: Parent vehicle object

        Returns:
            Sensor object (RgbCamera, Lidar, etc.)
        """
        sensor_type = sensor_cmd['sensor_type']
        sensor_name = sensor_cmd['sensor_name']
        save_dir = f"{parent_vehicle.save_dir}/{sensor_name}"

        # Create sensor object based on type
        if sensor_type == "sensor.camera.rgb":
            sensor_object = RgbCamera(
                uid=self.generate_uid(),
                name=sensor_name,
                base_save_dir=parent_vehicle.save_dir,
                carla_actor=carla_sensor,
                parent=parent_vehicle
            )
        elif sensor_type == "sensor.camera.semantic_segmentation":
            sensor_object = SemanticSegmentationCamera(
                uid=self.generate_uid(),
                name=sensor_name,
                base_save_dir=parent_vehicle.save_dir,
                carla_actor=carla_sensor,
                parent=parent_vehicle
            )
        elif sensor_type == "sensor.camera.depth":
            sensor_object = DepthCamera(
                uid=self.generate_uid(),
                name=sensor_name,
                base_save_dir=parent_vehicle.save_dir,
                carla_actor=carla_sensor,
                parent=parent_vehicle
            )
        elif sensor_type == "sensor.lidar.ray_cast":
            sensor_object = Lidar(
                uid=self.generate_uid(),
                name=sensor_name,
                base_save_dir=parent_vehicle.save_dir,
                carla_actor=carla_sensor,
                parent=parent_vehicle
            )
        elif sensor_type == "sensor.lidar.ray_cast_semantic":
            sensor_object = SemanticLidar(
                uid=self.generate_uid(),
                name=sensor_name,
                base_save_dir=parent_vehicle.save_dir,
                carla_actor=carla_sensor,
                parent=parent_vehicle
            )
        elif sensor_type == "sensor.other.radar":
            sensor_object = Radar(
                uid=self.generate_uid(),
                name=sensor_name,
                base_save_dir=parent_vehicle.save_dir,
                carla_actor=carla_sensor,
                parent=parent_vehicle
            )
        else:
            logger.error(f"Unsupported sensor type: {sensor_type}")
            raise AttributeError(f"Unsupported sensor type: {sensor_type}")

        return sensor_object
