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

    def init(self):
        self.root = self.actor_factory.create_actor_tree(self.config)
        self.node_list.append(self.root)
        for node in self.root.get_children():
            self.node_list.append(node)
            for sensor_node in node.get_children():
                self.node_list.append(sensor_node)

    def destroy(self):
        self.root.destroy()

    def add_node(self, node):
        self.root.add_child(node)

    def tick_controller(self):
        for v2i_layer_node in self.root.get_children():
            v2i_layer_node.tick_controller()

    def tick_data_saving(self, frame_id, timestamp: float):
        """
        Save data from all nodes with complete error handling

        Args:
            frame_id: Current frame ID
            timestamp: Current timestamp

        Raises:
            RuntimeError: If any node fails to save data (strict mode)
        """
        thread_pool = ThreadPool()
        frame_id_list = [frame_id] * len(self.node_list)
        timestamp_list = [timestamp] * len(self.node_list)

        try:
            # Use safe wrapper method to collect all results
            results = thread_pool.starmap(
                self._safe_save_data,
                zip(frame_id_list, timestamp_list, self.node_list)
            )
            thread_pool.close()
            thread_pool.join()

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

        except Exception as e:
            # Ensure thread pool is properly closed
            logger.exception(f"Critical error during data saving: {e}")
            thread_pool.terminate()
            raise

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
