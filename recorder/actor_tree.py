#!/usr/bin/python3
"""
Actor Tree management for CARLA Dataset Tools

Manages the hierarchical structure of actors (vehicles, infrastructure, sensors)
in the CARLA simulation. Supports both legacy sequential spawning and modern
batch spawning for improved performance.

The actor tree follows a hierarchical structure:
- Root (WorldActor)
  - Vehicle nodes
    - Sensor nodes (attached to vehicles)
  - Infrastructure nodes
    - Sensor nodes (attached to infrastructure)
  - Other vehicle nodes (background traffic)
"""

import os
import logging

import carla
from recorder.actor_factory import ActorFactory, Node
from multiprocessing.dummy import Pool as ThreadPool

# Get logger instance
logger = logging.getLogger(__name__)


class ActorTree(object):
    """
    Manages hierarchical actor tree structure for CARLA simulation.

    Handles initialization, spawning, and lifecycle management of all actors
    (vehicles, infrastructure, sensors) in the simulation. Supports efficient
    batch spawning for improved performance.

    Attributes:
        world: CARLA world instance
        config: Configuration dictionary
        actor_factory: Factory for creating actors
        root: Root node (WorldActor)
        node_list: Flat list of all nodes in the tree
        thread_pool: Persistent thread pool for parallel data saving
        spawn_commands: List of spawn commands for batch operations
        vehicle_nodes_map: Map of vehicle nodes for autopilot management
    """

    def __init__(self, world: carla.World, config=None, base_save_dir=None):
        """
        Initialize ActorTree.

        Args:
            world: CARLA world instance
            config: Configuration dictionary with actors and sensors
            base_save_dir: Base directory for saving recorded data
        """
        self.world = world
        self.config = config
        self.actor_factory = ActorFactory(self.world, base_save_dir)
        self.root = Node(None)
        self.node_list = []
        # Create persistent thread pool for data saving (reused across frames)
        # Using 4 workers for parallel sensor data saving
        self.thread_pool = ThreadPool(processes=4)

        # Store spawn commands for batch spawning
        self.spawn_commands = []

        # Store vehicle nodes map for later autopilot enabling
        self.vehicle_nodes_map = {}

    def init_legacy(self):
        """Legacy initialization method (kept for reference)"""
        logger.info("Creating actor tree from configuration...")
        self.root = self.actor_factory.create_actor_tree(self.config)
        logger.debug("Building node list...")
        self.node_list.append(self.root)
        for node in self.root.get_children():
            self.node_list.append(node)
            for sensor_node in node.get_children():
                self.node_list.append(sensor_node)
        logger.info(f"✓ Actor tree built: {len(self.node_list)} nodes total")

    def init(self, client, tm_port):
        """
        Initialize actor tree using optimized batch spawning.

        This method uses a 5-phase initialization process for efficient actor
        and sensor spawning. Autopilot is NOT enabled during initialization;
        it should be enabled separately by the caller after stabilization.

        Initialization Phases:
            Phase 1: Prepare spawn commands
                - Collect all vehicle and infrastructure spawn commands
                - Create world node
                - Prepare sensor commands for each actor

            Phase 2: Batch spawn actors (without autopilot)
                - Execute batch spawn for all vehicles
                - Infrastructure actors are virtual (no spawn needed)
                - Vehicles are spawned but autopilot is NOT enabled

            Phase 3: Build actor nodes
                - Create Vehicle/Infrastructure objects from spawn responses
                - Build node hierarchy and add to tree
                - Map command indices to nodes for sensor attachment

            Phase 4: Batch spawn sensors
                - Attach sensors to vehicles (relative transform)
                - Spawn sensors for infrastructure (absolute transform)
                - Execute batch spawn for all sensors

            Phase 5: Build sensor nodes and attach
                - Create sensor objects from spawn responses
                - Attach sensor nodes to parent actor nodes
                - Register V2X Custom sensors for message broadcasting

        NOTE: After initialization, caller should:
            1. Wait for stabilization (world.tick() multiple times)
            2. Call enable_autopilot_batch() to enable vehicle autopilot

        Args:
            client: carla.Client instance for batch operations
            tm_port: Traffic Manager port (for autopilot, used later)
        """
        # Phase 1: Prepare spawn commands
        self.prepare_spawn_commands()

        # Phase 2: Batch spawn actors (without autopilot)
        actor_responses = self.spawn_actors_batch(client, tm_port)

        # Phase 3: Build actor nodes (vehicles)
        self.vehicle_nodes_map = self.build_actor_nodes(actor_responses)

        # Phase 4: Batch spawn sensors (attached to vehicles)
        sensor_responses = self.spawn_sensors_batch(client, self.vehicle_nodes_map)

        # Phase 5: Build sensor nodes and attach
        self.build_sensor_nodes(sensor_responses, self.vehicle_nodes_map)

    def prepare_spawn_commands(self):
        """
        Phase 1: Prepare all spawn commands without executing spawn.

        Collects spawn commands for all actors (vehicles, infrastructure, other
        vehicles) and their sensors. Commands are stored for later batch execution.
        This phase does NOT spawn any actors in CARLA.

        Process:
            1. Create world node (root of actor tree)
            2. Collect vehicle spawn commands with sensor configurations
            3. Collect infrastructure spawn commands (virtual actors)
            4. Collect other_vehicles spawn commands (background traffic)

        After this phase, self.spawn_commands contains all commands ready for
        batch execution in Phase 2.
        """
        from recorder.world import WorldActor
        from recorder.actor_factory import NodeType

        logger.info("Phase 1: Preparing spawn commands...")

        # Create world node
        world_actor = WorldActor(
            uid=self.actor_factory.generate_uid(),
            carla_world=self.world,
            base_save_dir=self.actor_factory.base_save_dir
        )
        self.root = Node(world_actor, NodeType.WORLD)

        # Collect vehicle spawn commands
        for actor_info in self.config["actors"]:
            actor_type = str(actor_info["type"])

            if actor_type.startswith("vehicle"):
                # Create vehicle spawn command with sensors
                cmd = self.actor_factory.create_vehicle_spawn_command(actor_info)
                self.spawn_commands.append(cmd)
            elif actor_type.startswith("infrastructure"):
                # Infrastructure is a virtual actor (no CARLA spawn needed)
                # Create command to track info for node building and sensor spawning
                cmd = self._create_infrastructure_command(actor_info)
                self.spawn_commands.append(cmd)

        # Collect other_vehicles spawn commands
        other_vehicle_info = self.config.get("other_vehicles", {})
        if other_vehicle_info:
            other_cmds = self.actor_factory.create_other_vehicle_spawn_commands(other_vehicle_info)
            self.spawn_commands.extend(other_cmds)

        logger.info(f"✓ Prepared {len(self.spawn_commands)} spawn commands")

    def _create_infrastructure_command(self, actor_info):
        """
        Create infrastructure spawn command (virtual actor, no CARLA spawn).

        Infrastructure actors are virtual - they don't exist as CARLA actors
        but can have sensors attached. This method creates a command structure
        that tracks infrastructure information for node building and sensor spawning.

        Validates that infrastructure does NOT use V2X CAM sensors (which require
        vehicle dynamics). Only V2X Custom sensors are supported for infrastructure.

        Args:
            actor_info: Infrastructure configuration dictionary containing:
                - type: "infrastructure"
                - name: Infrastructure name
                - spawn_point: Position (int index or dict with x,y,z)
                - sensors: List of sensor configurations

        Returns:
            dict: Infrastructure command containing:
                - type: 'infrastructure'
                - infrastructure_name: Name of infrastructure
                - transform: CARLA transform for position
                - sensors: List of sensor spawn commands

        Raises:
            RuntimeError: If infrastructure tries to use V2X CAM sensor
        """
        from recorder.actor_factory import get_name_from_json, create_spawn_point

        infrastructure_name = get_name_from_json(actor_info, self.actor_factory.v2x_layer_name_set)
        spawn_point = actor_info["spawn_point"]

        if type(spawn_point) is int:
            transform = self.actor_factory.spawn_points[spawn_point]
        else:
            transform = create_spawn_point(
                spawn_point.get("x", 0.0),
                spawn_point.get("y", 0.0),
                spawn_point.get("z", 0.0),
                0, 0, 0
            )

        # Create sensor commands
        sensor_commands = []
        if "sensors" in actor_info and actor_info["sensors"]:
            for sensor_info in actor_info["sensors"]:
                # Validate sensor type: Infrastructure does not support V2X CAM
                sensor_type = sensor_info.get("type", "")

                if sensor_type == "sensor.other.v2x":
                    raise RuntimeError(
                        f"Infrastructure '{infrastructure_name}' cannot use 'sensor.other.v2x' (V2X CAM).\n\n"
                        f"Reason: V2X CAM sensors require vehicle dynamics data (speed, acceleration, yaw rate) "
                        f"which static Infrastructure cannot provide.\n\n"
                        f"Solution: Use 'sensor.other.v2x_custom' instead for Infrastructure.\n\n"
                        f"Note: Infrastructure V2X Custom sensors support one-way broadcast only (send messages). "
                        f"For bi-directional V2X communication, use Vehicle actors."
                    )

                sensor_cmd = self.actor_factory.create_sensor_spawn_command(
                    sensor_info, infrastructure_name
                )
                sensor_commands.append(sensor_cmd)

        return {
            'type': 'infrastructure',
            'infrastructure_name': infrastructure_name,
            'transform': transform,
            'sensors': sensor_commands
        }

    def spawn_actors_batch(self, client, tm_port):
        """
        Phase 2: Batch spawn actors WITHOUT autopilot

        Args:
            client: carla.Client instance
            tm_port: Traffic Manager port (kept for interface compatibility)

        Returns:
            dict: Spawn result containing responses and command indices
        """
        from carla import command

        logger.info("Phase 2: Batch spawning actors (without autopilot)...")

        batch = []
        vehicle_cmd_indices = []  # List of (cmd_index, batch_index)

        # Build batch commands for vehicles (without autopilot)
        for cmd_idx, cmd in enumerate(self.spawn_commands):
            if cmd['type'] in ['vehicle', 'other_vehicle']:
                batch_index = len(batch)
                vehicle_cmd_indices.append((cmd_idx, batch_index))

                # Only spawn, DO NOT enable autopilot yet
                spawn_cmd = command.SpawnActor(cmd['blueprint'], cmd['transform'])
                batch.append(spawn_cmd)

        # Execute batch
        logger.info(f"Spawning {len(batch)} vehicles...")
        responses = client.apply_batch_sync(batch, True)  # True = auto tick

        # Check responses
        success_count = sum(1 for r in responses if not r.error)
        logger.info(f"✓ Spawned {success_count}/{len(responses)} vehicles")

        # Log failures
        for i, response in enumerate(responses):
            if response.error:
                logger.error(f"Failed to spawn vehicle at batch index {i}: {response.error}")

        return {
            'all_responses': responses,
            'vehicle_cmd_indices': vehicle_cmd_indices
        }

    def build_actor_nodes(self, spawn_result):
        """
        Phase 3: Build actor nodes from spawn responses (vehicles and infrastructures)

        Args:
            spawn_result: Result from spawn_actors_batch

        Returns:
            dict: actor_nodes_map {cmd_index: actor_node}
        """
        from recorder.vehicle import Vehicle, OtherVehicle
        from recorder.infrastructure import Infrastructure
        from recorder.actor_factory import NodeType

        logger.info("Phase 3: Building actor nodes...")

        responses = spawn_result['all_responses']
        vehicle_cmd_indices = spawn_result['vehicle_cmd_indices']

        actor_nodes_map = {}  # {cmd_index: actor_node}

        # Build vehicle nodes from spawn responses
        for cmd_index, batch_index in vehicle_cmd_indices:
            response = responses[batch_index]

            if response.error:
                logger.warning(f"Skipping failed vehicle spawn (cmd {cmd_index}, batch {batch_index})")
                continue

            # Get spawned vehicle actor
            carla_actor = self.world.get_actor(response.actor_id)
            cmd = self.spawn_commands[cmd_index]

            # Create vehicle object
            if cmd['type'] == 'vehicle':
                vehicle_object = Vehicle(
                    uid=self.actor_factory.generate_uid(),
                    name=cmd['vehicle_name'],
                    base_save_dir=self.actor_factory.base_save_dir,
                    carla_actor=carla_actor,
                    route_config=cmd['route_config']
                )
                vehicle_node = Node(vehicle_object, NodeType.VEHICLE)

            elif cmd['type'] == 'other_vehicle':
                vehicle_object = OtherVehicle(
                    uid=self.actor_factory.generate_uid(),
                    name='',
                    base_save_dir="/tmp",
                    carla_actor=carla_actor
                )
                vehicle_node = Node(vehicle_object, NodeType.OTHER_VEHICLE)

            # Save to map and add to tree
            actor_nodes_map[cmd_index] = vehicle_node
            self.root.add_child(vehicle_node)
            self.node_list.append(vehicle_node)

        # Build infrastructure nodes (no spawn needed, virtual actors)
        for cmd_index, cmd in enumerate(self.spawn_commands):
            if cmd['type'] == 'infrastructure':
                # Infrastructure is virtual (no CARLA actor), create directly
                infrastructure_object = Infrastructure(
                    uid=self.actor_factory.generate_uid(),
                    name=cmd['infrastructure_name'],
                    base_save_dir=self.actor_factory.base_save_dir,
                    transform=cmd['transform']
                )
                infrastructure_node = Node(infrastructure_object, NodeType.INFRASTRUCTURE)

                # Save to map and add to tree
                actor_nodes_map[cmd_index] = infrastructure_node
                self.root.add_child(infrastructure_node)
                self.node_list.append(infrastructure_node)
                logger.debug(f"Built infrastructure node: {cmd['infrastructure_name']}")

        vehicle_count = sum(1 for cmd in self.spawn_commands if cmd['type'] in ['vehicle', 'other_vehicle'])
        infra_count = sum(1 for cmd in self.spawn_commands if cmd['type'] == 'infrastructure')
        logger.info(f"✓ Built {vehicle_count} vehicle nodes, {infra_count} infrastructure nodes")
        return actor_nodes_map

    def spawn_sensors_batch(self, client, actor_nodes_map):
        """
        Phase 4: Batch spawn sensors attached to actors (vehicles and infrastructures)

        Args:
            client: carla.Client instance
            actor_nodes_map: Map of {cmd_index: actor_node}

        Returns:
            dict: Sensor spawn result
            {
                'all_responses': [response1, response2, ...],
                'sensor_info_list': [(response_idx, cmd_idx, sensor_cmd, parent_node), ...]
            }
        """
        from carla import command
        import carla

        logger.info("Phase 4: Batch spawning sensors...")

        batch = []
        sensor_info_list = []  # Track sensor info for node building

        # Build sensor batch commands
        for cmd_idx, parent_node in actor_nodes_map.items():
            cmd = self.spawn_commands[cmd_idx]

            # Add sensors for this actor
            if 'sensors' in cmd and cmd['sensors']:
                for sensor_cmd in cmd['sensors']:
                    response_idx = len(batch)

                    # Check if parent has a real CARLA actor
                    parent_actor = parent_node.get_actor()
                    carla_actor = getattr(parent_actor, 'carla_actor', None)

                    if carla_actor is not None:
                        # Vehicle: spawn sensor attached to vehicle actor
                        sensor_spawn_cmd = command.SpawnActor(
                            sensor_cmd['blueprint'],
                            sensor_cmd['transform'],
                            carla_actor
                        )
                    else:
                        # Infrastructure: spawn sensor at absolute position
                        # Calculate world position by combining infrastructure transform and sensor relative transform
                        parent_transform = parent_actor.get_carla_transform()
                        sensor_relative_transform = sensor_cmd['transform']

                        # Transform sensor location to world coordinates
                        sensor_location = parent_transform.transform(sensor_relative_transform.location)
                        sensor_world_transform = carla.Transform(
                            sensor_location,
                            sensor_relative_transform.rotation
                        )

                        sensor_spawn_cmd = command.SpawnActor(
                            sensor_cmd['blueprint'],
                            sensor_world_transform
                        )

                    batch.append(sensor_spawn_cmd)
                    sensor_info_list.append((response_idx, cmd_idx, sensor_cmd, parent_node))

        # Execute batch
        if not batch:
            logger.info("No sensors to spawn")
            return {
                'all_responses': [],
                'sensor_info_list': []
            }

        logger.info(f"Spawning {len(batch)} sensors...")
        responses = client.apply_batch_sync(batch, True)  # True = auto tick

        # Check responses
        success_count = sum(1 for r in responses if not r.error)
        logger.info(f"✓ Spawned {success_count}/{len(responses)} sensors successfully")

        # Log failures
        for i, response in enumerate(responses):
            if response.error:
                logger.error(f"Failed to spawn sensor at batch index {i}: {response.error}")

        return {
            'all_responses': responses,
            'sensor_info_list': sensor_info_list
        }

    def build_sensor_nodes(self, sensor_result, actor_nodes_map):
        """
        Phase 5: Build sensor nodes and attach to parent actors

        Args:
            sensor_result: Result from spawn_sensors_batch
            actor_nodes_map: Map of {cmd_index: actor_node}
        """
        from recorder.actor_factory import NodeType
        from recorder.infrastructure import Infrastructure

        logger.info("Phase 5: Building sensor nodes...")

        responses = sensor_result['all_responses']
        sensor_info_list = sensor_result['sensor_info_list']

        sensor_count = 0
        for response_idx, cmd_idx, sensor_cmd, parent_node in sensor_info_list:
            response = responses[response_idx]

            if response.error:
                logger.warning(f"Skipping failed sensor: {sensor_cmd['sensor_name']}")
                continue

            # Get spawned sensor actor
            carla_sensor = self.world.get_actor(response.actor_id)

            # Get parent actor object
            parent_object = parent_node.get_actor()

            # Create sensor object
            sensor_object = self.actor_factory.create_sensor_object(
                sensor_cmd, carla_sensor, parent_object
            )

            # Create sensor node and attach to parent
            sensor_node = Node(sensor_object, NodeType.SENSOR)
            parent_node.add_child(sensor_node)
            self.node_list.append(sensor_node)
            sensor_count += 1

            # Register V2X Custom sensors to parent actor for message broadcasting
            # V2X Custom sensors need to be registered so parent can send messages
            if sensor_cmd['sensor_type'] == 'sensor.other.v2x_custom':
                logger.info(f"Found V2X Custom sensor '{sensor_cmd['sensor_name']}', parent type: {type(parent_object).__name__}")

                # Register to Infrastructure for one-way broadcasting
                if isinstance(parent_object, Infrastructure):
                    logger.info(f"  Infrastructure V2X sensor spawn details:")
                    logger.info(f"    Actor ID: {response.actor_id}")
                    logger.info(f"    CARLA Actor: {carla_sensor}")
                    logger.info(f"    Sensor Type: {carla_sensor.type_id if carla_sensor else 'None'}")
                    logger.info(f"    Is Alive: {carla_sensor.is_alive if carla_sensor else False}")

                    parent_object.register_v2x_sensor(sensor_object)
                    logger.info(f"✓ Registered V2X Custom sensor '{sensor_cmd['sensor_name']}' to infrastructure '{parent_object.name}'")
                else:
                    # Register to Vehicle for V2V communication
                    from recorder.vehicle import Vehicle
                    if isinstance(parent_object, Vehicle):
                        logger.info(f"  Vehicle V2X sensor spawn details:")
                        logger.info(f"    Actor ID: {response.actor_id}")
                        logger.info(f"    CARLA Actor: {carla_sensor}")
                        logger.info(f"    Is Alive: {carla_sensor.is_alive if carla_sensor else False}")

                        parent_object.register_v2x_sensor(sensor_object)
                        logger.info(f"✓ Registered V2X Custom sensor '{sensor_cmd['sensor_name']}' to vehicle '{parent_object.name}'")
                    else:
                        logger.info(f"V2X Custom sensor '{sensor_cmd['sensor_name']}' attached to {type(parent_object).__name__}, no registration")

        # Add root to node list
        self.node_list.append(self.root)
        logger.info(f"✓ Built {sensor_count} sensor nodes")
        logger.info(f"✓ Actor tree complete: {len(self.node_list)} total nodes")

    def enable_autopilot_batch(self, client, tm_port, vehicle_nodes_map):
        """
        Batch enable autopilot for spawned vehicles

        This method should be called by data_recorder after stabilization ticks.

        Args:
            client: carla.Client instance
            tm_port: Traffic Manager port
            vehicle_nodes_map: Map of {cmd_index: vehicle_node} from build_actor_nodes

        Returns:
            int: Number of vehicles with autopilot enabled successfully
        """
        from carla import command

        logger.info("Batch enabling autopilot for vehicles...")

        autopilot_batch = []
        autopilot_info = []  # For logging (vehicle_name, actor_id)

        for cmd_idx, vehicle_node in vehicle_nodes_map.items():
            cmd = self.spawn_commands[cmd_idx]

            if cmd.get('use_autopilot', False):
                vehicle_actor = vehicle_node.get_actor().carla_actor
                autopilot_batch.append(
                    command.SetAutopilot(vehicle_actor, True, tm_port)
                )
                vehicle_name = cmd.get('vehicle_name', 'other_vehicle')
                autopilot_info.append((vehicle_name, vehicle_actor.id))

        if not autopilot_batch:
            logger.info("No vehicles need autopilot")
            return 0

        logger.info(f"Enabling autopilot for {len(autopilot_batch)} vehicles...")
        responses = client.apply_batch_sync(autopilot_batch, True)

        # Check results
        success_count = sum(1 for r in responses if not r.error)
        logger.info(f"✓ Enabled autopilot for {success_count}/{len(autopilot_batch)} vehicles")

        # Log failures
        for i, response in enumerate(responses):
            if response.error:
                vehicle_name, actor_id = autopilot_info[i]
                logger.error(f"Failed to enable autopilot for {vehicle_name} (ID {actor_id}): {response.error}")

        return success_count

    def destroy(self):
        """Cleanup resources including thread pool and actors"""
        # Cleanup thread pool first to ensure no pending tasks
        if hasattr(self, 'thread_pool'):
            logger.info("Closing thread pool...")
            self.thread_pool.close()
            self.thread_pool.join()
            logger.info("Thread pool closed successfully")

        # Then destroy actors
        self.root.destroy()

    def add_node(self, node):
        self.root.add_child(node)

    def tick_controller(self):
        infrastructure_count = 0
        for v2i_layer_node in self.root.get_children():
            v2i_layer_node.tick_controller()
            # Count infrastructure nodes
            from recorder.infrastructure import Infrastructure
            if isinstance(v2i_layer_node.get_actor(), Infrastructure):
                infrastructure_count += 1

        if infrastructure_count > 0:
            logger.debug(f"tick_controller called on {infrastructure_count} infrastructure nodes")

    def tick_data_saving(self, frame_id, timestamp: float):
        """
        Save data from all nodes with complete error handling

        Uses persistent thread pool for efficient parallel processing.

        Args:
            frame_id: Absolute CARLA frame ID (for file naming and synchronization)
            timestamp: Current timestamp

        Returns:
            dict: Collected save information from all actors
                  Format: {actor_name: actor_save_info, ...}

        Raises:
            RuntimeError: If any node fails to save data (strict mode)
        """
        frame_id_list = [frame_id] * len(self.node_list)
        timestamp_list = [timestamp] * len(self.node_list)

        # Use persistent thread pool - no need to create/destroy on each frame
        results = self.thread_pool.starmap(
            self._safe_save_data,
            zip(frame_id_list, timestamp_list, self.node_list)
        )

        # Check for failed nodes
        failed = [r for r in results if not r['success']]
        if failed:
            # Log all failure details
            logger.error(
                f"Frame {frame_id}: {len(failed)}/{len(self.node_list)} nodes failed to save"
            )
            for fail_info in failed:
                logger.error(
                    f"  - Node '{fail_info['node']}' failed: {fail_info['error']}"
                )

            # Strict mode: immediately raise exception to abort recording
            raise RuntimeError(
                f"Data save failed: {len(failed)} node(s) failed. "
                f"See logs above for details. Aborting to ensure data integrity."
            )

        # Collect save information from all nodes
        actors_info = {}
        for result in results:
            if result['success'] and result.get('save_info'):
                save_info = result['save_info']
                actor_name = save_info.get('name')
                if actor_name:
                    # For vehicles with sensors, group sensors under vehicle
                    if save_info['type'] == 'sensor':
                        parent_name = self._get_parent_name_for_sensor(result['node'])
                        if parent_name:
                            if parent_name not in actors_info:
                                actors_info[parent_name] = {
                                    'type': 'vehicle',
                                    'name': parent_name,
                                    'sensors': {}
                                }
                            # Ensure 'sensors' key exists (in case vehicle was added first)
                            if 'sensors' not in actors_info[parent_name]:
                                actors_info[parent_name]['sensors'] = {}
                            actors_info[parent_name]['sensors'][actor_name] = save_info
                        else:
                            # Standalone sensor (no vehicle parent)
                            actors_info[actor_name] = save_info
                    else:
                        # Vehicle or World actor
                        if actor_name in actors_info:
                            # Merge with existing (has sensors)
                            # Preserve existing 'sensors' dict if it exists
                            existing_sensors = actors_info[actor_name].get('sensors', {})
                            actors_info[actor_name].update(save_info)
                            # Restore sensors (in case save_info overwrote it)
                            if existing_sensors:
                                if 'sensors' not in actors_info[actor_name]:
                                    actors_info[actor_name]['sensors'] = existing_sensors
                                else:
                                    # Merge sensors if both exist
                                    actors_info[actor_name]['sensors'].update(existing_sensors)
                        else:
                            actors_info[actor_name] = save_info

        return actors_info

    def _safe_save_data(self, frame_id, timestamp: float, node: Node) -> dict:
        """
        Safe data saving wrapper that catches exceptions and returns results

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Timestamp
            node: Node to save

        Returns:
            dict: Contains success status, node name, save_info, and possible error info
        """
        try:
            save_info = node.tick_data_saving(frame_id, timestamp)
            return {
                'success': True,
                'node': self._get_node_name(node),
                'save_info': save_info
            }
        except Exception as e:
            node_name = self._get_node_name(node)
            logger.exception(f"Node '{node_name}' failed to save data")
            return {
                'success': False,
                'node': node_name,
                'error': str(e),
                'save_info': None
            }

    def _get_parent_name_for_sensor(self, node_name: str) -> str:
        """
        Get parent vehicle name for a sensor node.

        Searches the actor tree to find which vehicle (or actor) a sensor
        belongs to. Used for organizing save information by parent actor.

        Args:
            node_name: Sensor node name to find parent for

        Returns:
            str: Parent vehicle/actor name, or None if no parent found
        """
        # Try to find the parent vehicle by looking at node tree structure
        for node in self.node_list:
            actor = node.get_actor()
            if actor and hasattr(actor, 'name'):
                # Check if this node has children that match the sensor name
                for child in node.get_children():
                    child_actor = child.get_actor()
                    if child_actor and hasattr(child_actor, 'name'):
                        if child_actor.name == node_name:
                            return actor.name
        return None

    def _get_node_name(self, node: Node) -> str:
        """
        Get node name safely with error handling.

        Args:
            node: Node to get name from

        Returns:
            str: Node name, or 'unknown' if name cannot be retrieved
        """
        try:
            if node.get_actor():
                return node.get_actor().name
        except:
            pass
        return 'unknown'

    def save_data(self, frame_id, timestamp: float, node: Node):
        """
        Save single node data (deprecated).

        This method is deprecated and kept only for backward compatibility.
        Use tick_data_saving() for parallel saving of all nodes.

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Simulation timestamp
            node: Node to save data for
        """
        node.tick_data_saving(frame_id, timestamp)

    def print_tree(self):
        logger.info("------ Actor Tree BEGIN ------")
        for node in self.root.get_children():
            logger.info(f"- {node.get_actor().name}")
            for child_node in node.get_children():
                if child_node is not None:
                    logger.info(f"|- {child_node.get_actor().name}")
        logger.info("------ Actor Tree END ------")

    def clear_sensor_queues(self) -> int:
        """
        Clear accumulated sensor data from initialization phase.

        During initialization (spawn, stabilization, autopilot enabling),
        sensors may accumulate data in their queues. This method drains all
        sensor queues to free memory and ensure a clean state before recording
        starts. This prevents old initialization data from being saved as
        part of the recording.

        Process:
            - Iterates through all sensor nodes in the tree
            - Drains each sensor's queue using non-blocking get_nowait()
            - Counts total frames cleared across all sensors

        Returns:
            int: Total number of frames cleared from all sensor queues
        """
        import queue as queue_module
        from recorder.actor_factory import NodeType

        total_cleared = 0
        for node in self.node_list:
            if node.get_node_type() == NodeType.SENSOR:
                sensor = node._actor  # Get Sensor object
                cleared = 0
                # Drain queue using get_nowait (non-blocking)
                try:
                    while True:
                        sensor.queue.get_nowait()
                        cleared += 1
                except queue_module.Empty:
                    pass  # Queue is now empty

                total_cleared += cleared
                if cleared > 0:
                    logger.debug(f"Cleared {cleared} frames from sensor {sensor.name}")

        return total_cleared
