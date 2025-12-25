#!/usr/bin/python3
import os
import logging

import carla
from recorder.actor_factory import ActorFactory, Node
from multiprocessing.dummy import Pool as ThreadPool

# 获取logger实例
logger = logging.getLogger(__name__)


class ActorTree(object):
    def __init__(self, world: carla.World, config=None, base_save_dir=None):
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
        Initialize actor tree (spawn only, autopilot handled by caller)

        Phase 1: Prepare spawn commands
        Phase 2: Batch spawn actors (without autopilot)
        Phase 3: Build actor nodes
        Phase 4: Batch spawn sensors
        Phase 5: Build sensor nodes

        NOTE: Autopilot should be enabled by caller using enable_autopilot_batch()

        Args:
            client: carla.Client instance
            tm_port: Traffic Manager port
        """
        # Phase 1: Prepare spawn commands
        self.prepare_spawn_commands()

        # Phase 2: Batch spawn actors (without autopilot)
        actor_responses = self.spawn_actors_batch(client, tm_port)

        # Phase 3: Build actor nodes (vehicles and infrastructure)
        actor_nodes_result = self.build_actor_nodes(actor_responses)
        self.vehicle_nodes_map = actor_nodes_result['vehicle_nodes_map']
        self.infrastructure_nodes_map = actor_nodes_result.get('infrastructure_nodes_map', {})

        # Phase 4: Batch spawn sensors (attached to vehicles and infrastructure)
        sensor_responses = self.spawn_sensors_batch(
            client, 
            self.vehicle_nodes_map, 
            self.infrastructure_nodes_map
        )

        # Phase 5: Build sensor nodes and attach
        # Combine both node maps for sensor node building
        all_parent_nodes_map = {**self.vehicle_nodes_map, **self.infrastructure_nodes_map}
        self.build_sensor_nodes(sensor_responses, all_parent_nodes_map)

    def prepare_spawn_commands(self):
        """
        Phase 1: Prepare all spawn commands (不执行spawn)
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

        # Collect vehicle and infrastructure spawn commands
        for actor_info in self.config["actors"]:
            actor_type = str(actor_info["type"])

            if actor_type.startswith("vehicle"):
                cmd = self.actor_factory.create_vehicle_spawn_command(actor_info)
                self.spawn_commands.append(cmd)
            elif actor_type.startswith("infrastructure"):
                cmd = self.actor_factory.create_infrastructure_spawn_command(actor_info)
                self.spawn_commands.append(cmd)

        # Collect other_vehicles spawn commands
        other_vehicle_info = self.config.get("other_vehicles", {})
        if other_vehicle_info:
            other_cmds = self.actor_factory.create_other_vehicle_spawn_commands(other_vehicle_info)
            self.spawn_commands.extend(other_cmds)

        logger.info(f"✓ Prepared {len(self.spawn_commands)} spawn commands")

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
        infrastructure_cmd_indices = []  # List of (cmd_index, batch_index)

        # Build batch commands for vehicles and infrastructure (without autopilot)
        for cmd_idx, cmd in enumerate(self.spawn_commands):
            if cmd['type'] in ['vehicle', 'other_vehicle']:
                batch_index = len(batch)
                vehicle_cmd_indices.append((cmd_idx, batch_index))
                spawn_cmd = command.SpawnActor(cmd['blueprint'], cmd['transform'])
                batch.append(spawn_cmd)
            elif cmd['type'] == 'infrastructure' and cmd['blueprint'] is not None:
                batch_index = len(batch)
                infrastructure_cmd_indices.append((cmd_idx, batch_index))
                spawn_cmd = command.SpawnActor(cmd['blueprint'], cmd['transform'])
                batch.append(spawn_cmd)
            elif cmd['type'] == 'infrastructure':
                logger.warning(f"Infrastructure '{cmd.get('infrastructure_name', 'unknown')}' will not spawn attachment actor (no blueprint)")

        # Execute batch
        total_actors = len(batch)
        logger.info(f"Spawning {total_actors} actors ({len(vehicle_cmd_indices)} vehicles, {len(infrastructure_cmd_indices)} infrastructure)...")
        responses = client.apply_batch_sync(batch, True)  # True = auto tick

        # Check responses
        success_count = sum(1 for r in responses if not r.error)
        logger.info(f"✓ Spawned {success_count}/{total_actors} actors")

        # Log failures
        for i, response in enumerate(responses):
            if response.error:
                logger.error(f"Failed to spawn actor at batch index {i}: {response.error}")

        return {
            'all_responses': responses,
            'vehicle_cmd_indices': vehicle_cmd_indices,
            'infrastructure_cmd_indices': infrastructure_cmd_indices
        }

    def build_actor_nodes(self, spawn_result):
        """
        Phase 3: Build actor nodes from spawn responses (vehicles and infrastructure)

        Args:
            spawn_result: Result from spawn_actors_batch

        Returns:
            dict: vehicle_nodes_map {cmd_index: vehicle_node}
        """
        from recorder.vehicle import Vehicle, OtherVehicle
        from recorder.actor_factory import NodeType

        logger.info("Phase 3: Building actor nodes (vehicles and infrastructure)...")

        responses = spawn_result['all_responses']
        vehicle_cmd_indices = spawn_result['vehicle_cmd_indices']
        infrastructure_cmd_indices = spawn_result.get('infrastructure_cmd_indices', [])

        vehicle_nodes_map = {}  # {cmd_index: vehicle_node}
        infrastructure_nodes_map = {}  # {cmd_index: infrastructure_node}

        # Build vehicle nodes
        for cmd_index, batch_index in vehicle_cmd_indices:
            response = responses[batch_index]
            if response.error:
                logger.warning(f"Skipping failed vehicle spawn (cmd {cmd_index}, batch {batch_index})")
                continue

            carla_actor = self.world.get_actor(response.actor_id)
            cmd = self.spawn_commands[cmd_index]

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
            else:
                continue

            vehicle_nodes_map[cmd_index] = vehicle_node
            self.root.add_child(vehicle_node)
            self.node_list.append(vehicle_node)

        # Build infrastructure nodes
        for cmd_index, batch_index in infrastructure_cmd_indices:
            response = responses[batch_index]
            if response.error:
                logger.warning(f"Skipping failed infrastructure spawn (cmd {cmd_index}, batch {batch_index})")
                continue

            carla_actor = self.world.get_actor(response.actor_id)
            cmd = self.spawn_commands[cmd_index]
            infrastructure_node = self.actor_factory.create_infrastructure_node(
                cmd['actor_info'],
                carla_actor=carla_actor
            )

            infrastructure_nodes_map[cmd_index] = infrastructure_node
            self.root.add_child(infrastructure_node)
            self.node_list.append(infrastructure_node)

        logger.info(f"✓ Built {len(vehicle_nodes_map)} vehicle nodes and {len(infrastructure_nodes_map)} infrastructure nodes")
        return {
            'vehicle_nodes_map': vehicle_nodes_map,
            'infrastructure_nodes_map': infrastructure_nodes_map
        }

    def spawn_sensors_batch(self, client, vehicle_nodes_map, infrastructure_nodes_map=None):
        """
        Phase 4: Batch spawn sensors attached to vehicles and infrastructure

        Args:
            client: carla.Client instance
            vehicle_nodes_map: Map of {cmd_index: vehicle_node}
            infrastructure_nodes_map: Map of {cmd_index: infrastructure_node}

        Returns:
            dict: Sensor spawn result
            {
                'all_responses': [response1, response2, ...],
                'sensor_info_list': [(response_idx, cmd_idx, sensor_cmd, parent_node), ...]
            }
        """
        from carla import command

        logger.info("Phase 4: Batch spawning sensors...")

        batch = []
        sensor_info_list = []  # Track sensor info for node building

        if infrastructure_nodes_map is None:
            infrastructure_nodes_map = {}

        # Helper function to add sensors for a parent node
        def add_sensors_for_parent(cmd_idx, parent_node, parent_actor):
            """Add sensor spawn commands for a parent node"""
            cmd = self.spawn_commands[cmd_idx]
            if 'sensors' not in cmd or not cmd['sensors']:
                return
            
            for sensor_cmd in cmd['sensors']:
                response_idx = len(batch)
                if parent_actor is not None:
                    # Attach to parent actor
                    sensor_spawn_cmd = command.SpawnActor(
                        sensor_cmd['blueprint'],
                        sensor_cmd['transform'],
                        parent_actor
                    )
                else:
                    # Fallback: spawn to world (calculate world transform)
                    infra_transform = cmd['transform']
                    sensor_transform = sensor_cmd['transform']
                    world_location = infra_transform.transform(sensor_transform.location)
                    world_transform = carla.Transform(world_location, sensor_transform.rotation)
                    sensor_spawn_cmd = command.SpawnActor(
                        sensor_cmd['blueprint'],
                        world_transform
                    )
                batch.append(sensor_spawn_cmd)
                sensor_info_list.append((response_idx, cmd_idx, sensor_cmd, parent_node))

        # Build sensor batch commands for vehicles
        for cmd_idx, vehicle_node in vehicle_nodes_map.items():
            add_sensors_for_parent(cmd_idx, vehicle_node, vehicle_node.get_actor().carla_actor)

        # Build sensor batch commands for infrastructure
        for cmd_idx, infrastructure_node in infrastructure_nodes_map.items():
            cmd = self.spawn_commands[cmd_idx]
            infrastructure_actor = infrastructure_node.get_actor().get_carla_actor()
            if infrastructure_actor is None:
                logger.warning(f"Infrastructure '{cmd.get('infrastructure_name', 'unknown')}' has no attachment actor, sensors will spawn to world")
            add_sensors_for_parent(cmd_idx, infrastructure_node, infrastructure_actor)

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

    def build_sensor_nodes(self, sensor_result, parent_nodes_map):
        """
        Phase 5: Build sensor nodes and attach to parent actors (vehicles/infrastructure)

        Args:
            sensor_result: Result from spawn_sensors_batch
            parent_nodes_map: Map of {cmd_index: parent_node} (vehicles or infrastructure)
        """
        from recorder.actor_factory import NodeType

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

            # Get parent actor object (vehicle or infrastructure)
            parent_actor = parent_node.get_actor()

            # Create sensor object
            sensor_object = self.actor_factory.create_sensor_object(
                sensor_cmd, carla_sensor, parent_actor
            )

            # Create sensor node and attach to parent
            sensor_node = Node(sensor_object, NodeType.SENSOR)
            parent_node.add_child(sensor_node)
            self.node_list.append(sensor_node)
            sensor_count += 1

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
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            logger.info("Closing thread pool...")
            try:
                self.thread_pool.close()
                self.thread_pool.join()
                logger.info("Thread pool closed successfully")
            except Exception as e:
                logger.error(f"Error closing thread pool: {e}")
            finally:
                self.thread_pool = None

        # Then destroy actors
        if hasattr(self, 'root') and self.root is not None:
            self.root.destroy()

    def add_node(self, node):
        self.root.add_child(node)

    def tick_controller(self):
        for v2i_layer_node in self.root.get_children():
            v2i_layer_node.tick_controller()

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
        Get parent vehicle name for a sensor node

        Args:
            node_name: Sensor node name

        Returns:
            Parent vehicle name, or None if no parent found
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
        """Get node name (safe)"""
        try:
            if node.get_actor():
                return node.get_actor().name
        except:
            pass
        return 'unknown'

    def save_data(self, frame_id, timestamp: float, node: Node):
        """
        Save single node data (deprecated, replaced by _safe_save_data)
        Kept for backward compatibility
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

        This removes old sensor data that accumulated during initialization
        (spawn, stabilization, autopilot enabling) to free memory and
        ensure clean state before recording starts.

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
