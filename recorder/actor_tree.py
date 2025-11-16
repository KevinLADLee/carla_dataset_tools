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
        Initialize actor tree using 3-batch spawning (5-phase process)

        Following CARLA official pattern:
        - Batch 1: Spawn vehicles WITH immediate autopilot
        - Batch 2: Spawn sensors
        (No separate stabilization or autopilot batches needed)

        Args:
            client: carla.Client instance
            tm_port: Traffic Manager port
        """
        # Phase 1: Prepare spawn commands
        self.prepare_spawn_commands()

        # Phase 2: Batch spawn actors WITH autopilot (official pattern)
        actor_responses = self.spawn_actors_batch(client, tm_port)

        # Phase 3: Build actor nodes (vehicles)
        vehicle_nodes_map = self.build_actor_nodes(actor_responses)

        # Phase 4: Batch spawn sensors (attached to vehicles)
        sensor_responses = self.spawn_sensors_batch(client, vehicle_nodes_map)

        # Phase 5: Build sensor nodes and attach
        self.build_sensor_nodes(sensor_responses, vehicle_nodes_map)

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

        # Collect vehicle spawn commands
        for actor_info in self.config["actors"]:
            actor_type = str(actor_info["type"])

            if actor_type.startswith("vehicle"):
                cmd = self.actor_factory.create_vehicle_spawn_command(actor_info)
                self.spawn_commands.append(cmd)
            elif actor_type.startswith("infrastructure"):
                # TODO: Add infrastructure support
                logger.warning(f"Infrastructure type not yet supported in batch mode: {actor_type}")

        # Collect other_vehicles spawn commands
        other_vehicle_info = self.config.get("other_vehicles", {})
        if other_vehicle_info:
            other_cmds = self.actor_factory.create_other_vehicle_spawn_commands(other_vehicle_info)
            self.spawn_commands.extend(other_cmds)

        logger.info(f"✓ Prepared {len(self.spawn_commands)} spawn commands")

    def spawn_actors_batch(self, client, tm_port):
        """
        Phase 2: Batch spawn actors WITH immediate autopilot (CARLA official pattern)

        Args:
            client: carla.Client instance
            tm_port: Traffic Manager port

        Returns:
            dict: Spawn result containing responses and command indices
            {
                'all_responses': [response1, response2, ...],
                'vehicle_cmd_indices': [(cmd_idx, batch_idx), ...]
            }
        """
        from carla import command

        logger.info("Phase 2: Batch spawning actors WITH autopilot (official pattern)...")

        batch = []
        vehicle_cmd_indices = []  # List of (cmd_index, batch_index)

        # Build batch commands for vehicles
        for cmd_idx, cmd in enumerate(self.spawn_commands):
            if cmd['type'] in ['vehicle', 'other_vehicle']:
                batch_index = len(batch)
                vehicle_cmd_indices.append((cmd_idx, batch_index))

                # Official pattern: Spawn + SetAutopilot in ONE command
                spawn_cmd = command.SpawnActor(cmd['blueprint'], cmd['transform'])

                if cmd.get('use_autopilot', False):
                    # Chain SetAutopilot to spawn command (official pattern)
                    spawn_cmd = spawn_cmd.then(
                        command.SetAutopilot(command.FutureActor, True, tm_port)
                    )

                batch.append(spawn_cmd)

        # Execute batch
        logger.info(f"Spawning {len(batch)} vehicles with autopilot...")
        responses = client.apply_batch_sync(batch, True)  # True = auto tick

        # Check responses
        success_count = sum(1 for r in responses if not r.error)
        logger.info(f"✓ Spawned {success_count}/{len(responses)} vehicles successfully")

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
        Phase 3: Build actor nodes from spawn responses (vehicles only)

        Args:
            spawn_result: Result from spawn_actors_batch

        Returns:
            dict: vehicle_nodes_map {cmd_index: vehicle_node}
        """
        from recorder.vehicle import Vehicle, OtherVehicle
        from recorder.actor_factory import NodeType

        logger.info("Phase 3: Building actor nodes (vehicles)...")

        responses = spawn_result['all_responses']
        vehicle_cmd_indices = spawn_result['vehicle_cmd_indices']

        vehicle_nodes_map = {}  # {cmd_index: vehicle_node}

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
            vehicle_nodes_map[cmd_index] = vehicle_node
            self.root.add_child(vehicle_node)
            self.node_list.append(vehicle_node)

        logger.info(f"✓ Built {len(vehicle_nodes_map)} vehicle nodes")
        return vehicle_nodes_map

    def spawn_sensors_batch(self, client, vehicle_nodes_map):
        """
        Phase 4: Batch spawn sensors attached to vehicles

        Args:
            client: carla.Client instance
            vehicle_nodes_map: Map of {cmd_index: vehicle_node}

        Returns:
            dict: Sensor spawn result
            {
                'all_responses': [response1, response2, ...],
                'sensor_info_list': [(response_idx, cmd_idx, sensor_cmd, vehicle_node), ...]
            }
        """
        from carla import command

        logger.info("Phase 4: Batch spawning sensors...")

        batch = []
        sensor_info_list = []  # Track sensor info for node building

        # Build sensor batch commands
        for cmd_idx, vehicle_node in vehicle_nodes_map.items():
            cmd = self.spawn_commands[cmd_idx]

            # Get real vehicle actor for attachment
            vehicle_actor = vehicle_node.get_actor().carla_actor

            # Add sensors for this vehicle
            if 'sensors' in cmd and cmd['sensors']:
                for sensor_cmd in cmd['sensors']:
                    response_idx = len(batch)

                    # Spawn sensor attached to REAL vehicle actor (not Response!)
                    sensor_spawn_cmd = command.SpawnActor(
                        sensor_cmd['blueprint'],
                        sensor_cmd['transform'],
                        vehicle_actor  # ✅ Real actor, not command.Response
                    )

                    batch.append(sensor_spawn_cmd)
                    sensor_info_list.append((response_idx, cmd_idx, sensor_cmd, vehicle_node))

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

    def build_sensor_nodes(self, sensor_result, vehicle_nodes_map):
        """
        Phase 5: Build sensor nodes and attach to vehicles

        Args:
            sensor_result: Result from spawn_sensors_batch
            vehicle_nodes_map: Map of {cmd_index: vehicle_node}
        """
        from recorder.actor_factory import NodeType

        logger.info("Phase 5: Building sensor nodes...")

        responses = sensor_result['all_responses']
        sensor_info_list = sensor_result['sensor_info_list']

        sensor_count = 0
        for response_idx, cmd_idx, sensor_cmd, vehicle_node in sensor_info_list:
            response = responses[response_idx]

            if response.error:
                logger.warning(f"Skipping failed sensor: {sensor_cmd['sensor_name']}")
                continue

            # Get spawned sensor actor
            carla_sensor = self.world.get_actor(response.actor_id)

            # Get parent vehicle object
            vehicle_object = vehicle_node.get_actor()

            # Create sensor object
            sensor_object = self.actor_factory.create_sensor_object(
                sensor_cmd, carla_sensor, vehicle_object
            )

            # Create sensor node and attach to vehicle
            sensor_node = Node(sensor_object, NodeType.SENSOR)
            vehicle_node.add_child(sensor_node)
            self.node_list.append(sensor_node)
            sensor_count += 1

        # Add root to node list
        self.node_list.append(self.root)
        logger.info(f"✓ Built {sensor_count} sensor nodes")
        logger.info(f"✓ Actor tree complete: {len(self.node_list)} total nodes")

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
        for v2i_layer_node in self.root.get_children():
            v2i_layer_node.tick_controller()

    def tick_data_saving(self, frame_id, timestamp: float):
        """
        Save data from all nodes with complete error handling

        Uses persistent thread pool for efficient parallel processing.

        Args:
            frame_id: Current frame ID
            timestamp: Current timestamp

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

    def _safe_save_data(self, frame_id, timestamp: float, node: Node) -> dict:
        """
        Safe data saving wrapper that catches exceptions and returns results

        Args:
            frame_id: Frame ID
            timestamp: Timestamp
            node: Node to save

        Returns:
            dict: Contains success status, node name, and possible error info
        """
        try:
            node.tick_data_saving(frame_id, timestamp)
            return {
                'success': True,
                'node': self._get_node_name(node)
            }
        except Exception as e:
            node_name = self._get_node_name(node)
            logger.exception(f"Node '{node_name}' failed to save data")
            return {
                'success': False,
                'node': node_name,
                'error': str(e)
            }

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
