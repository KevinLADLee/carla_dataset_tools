#!/usr/bin/env python3
"""
Unit tests for ActorTree thread pool management
Tests fix for Issue #1: Thread Pool Resource Leak
"""
import unittest
import time
from unittest.mock import Mock, MagicMock, patch
from multiprocessing.dummy import Pool as ThreadPool


class TestActorTreeThreadPool(unittest.TestCase):
    """Test thread pool lifecycle management in ActorTree"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_world = Mock()
        self.mock_config = {
            'actors': [],
            'recording': {'frame_total': 10, 'frame_step': 1, 'map': 'Town01'},
            'world_settings': {},
            'traffic_lights': {}
        }

    @patch('recorder.actor_tree.ActorFactory')
    def test_thread_pool_created_on_init(self, mock_factory):
        """Test that thread pool is created during initialization"""
        from recorder.actor_tree import ActorTree

        # Create ActorTree instance
        actor_tree = ActorTree(self.mock_world, self.mock_config, '/tmp/test')

        # Verify thread pool exists
        self.assertTrue(hasattr(actor_tree, 'thread_pool'))
        self.assertIsInstance(actor_tree.thread_pool, ThreadPool)

    @patch('recorder.actor_tree.ActorFactory')
    def test_thread_pool_reused_across_saves(self, mock_factory):
        """Test that same thread pool is reused for multiple saves"""
        from recorder.actor_tree import ActorTree

        actor_tree = ActorTree(self.mock_world, self.mock_config, '/tmp/test')

        # Get reference to thread pool
        thread_pool_id_before = id(actor_tree.thread_pool)

        # Simulate multiple data saves
        mock_node = Mock()
        mock_node.tick_data_saving = Mock()
        actor_tree.node_list = [mock_node]

        try:
            actor_tree.tick_data_saving(frame_id=1, timestamp=1.0)
            actor_tree.tick_data_saving(frame_id=2, timestamp=2.0)
        except:
            pass  # Expected to fail with mock nodes

        # Verify same thread pool instance is used
        thread_pool_id_after = id(actor_tree.thread_pool)
        self.assertEqual(
            thread_pool_id_before,
            thread_pool_id_after,
            "Thread pool should be reused, not recreated"
        )

    @patch('recorder.actor_tree.ActorFactory')
    def test_thread_pool_properly_closed(self, mock_factory):
        """Test that thread pool is properly closed on destroy"""
        from recorder.actor_tree import ActorTree

        actor_tree = ActorTree(self.mock_world, self.mock_config, '/tmp/test')

        # Mock the thread pool methods
        actor_tree.thread_pool.close = Mock()
        actor_tree.thread_pool.join = Mock()

        # Mock root.destroy
        actor_tree.root.destroy = Mock()

        # Call destroy
        actor_tree.destroy()

        # Verify thread pool cleanup was called
        actor_tree.thread_pool.close.assert_called_once()
        actor_tree.thread_pool.join.assert_called_once()
        actor_tree.root.destroy.assert_called_once()

    @patch('recorder.actor_tree.ActorFactory')
    def test_destroy_handles_missing_thread_pool(self, mock_factory):
        """Test that destroy() handles case where thread_pool doesn't exist"""
        from recorder.actor_tree import ActorTree

        actor_tree = ActorTree(self.mock_world, self.mock_config, '/tmp/test')

        # Remove thread pool to simulate edge case
        delattr(actor_tree, 'thread_pool')

        # Mock root.destroy
        actor_tree.root.destroy = Mock()

        # Should not raise exception
        try:
            actor_tree.destroy()
        except AttributeError:
            self.fail("destroy() should handle missing thread_pool gracefully")

        # Verify root was still destroyed
        actor_tree.root.destroy.assert_called_once()

    @patch('recorder.actor_tree.ActorFactory')
    def test_no_thread_pool_leaks(self, mock_factory):
        """Test that no thread pools are leaked during normal operation"""
        from recorder.actor_tree import ActorTree
        import gc

        # Force garbage collection
        gc.collect()

        # Count ThreadPool instances before
        initial_pools = sum(
            1 for obj in gc.get_objects()
            if isinstance(obj, ThreadPool)
        )

        # Create and destroy ActorTree
        actor_tree = ActorTree(self.mock_world, self.mock_config, '/tmp/test')
        actor_tree.root.destroy = Mock()
        actor_tree.destroy()

        # Force garbage collection
        del actor_tree
        gc.collect()

        # Count ThreadPool instances after
        final_pools = sum(
            1 for obj in gc.get_objects()
            if isinstance(obj, ThreadPool)
        )

        # Should not have more pools than before
        self.assertEqual(
            initial_pools,
            final_pools,
            f"Thread pool leaked: {final_pools - initial_pools} pools remaining"
        )


class TestActorTreeDataSaving(unittest.TestCase):
    """Test data saving functionality"""

    @patch('recorder.actor_tree.ActorFactory')
    def test_tick_data_saving_uses_thread_pool(self, mock_factory):
        """Test that tick_data_saving uses the persistent thread pool"""
        from recorder.actor_tree import ActorTree

        mock_world = Mock()
        actor_tree = ActorTree(mock_world, None, '/tmp/test')

        # Mock node
        mock_node = Mock()
        mock_node.tick_data_saving = Mock()
        actor_tree.node_list = [mock_node]

        # Mock thread pool starmap
        actor_tree.thread_pool.starmap = Mock(return_value=[
            {'success': True, 'node': 'test'}
        ])

        # Execute data saving
        actor_tree.tick_data_saving(frame_id=1, timestamp=1.0)

        # Verify thread pool starmap was called
        actor_tree.thread_pool.starmap.assert_called_once()

        # Verify starmap was called with correct method
        call_args = actor_tree.thread_pool.starmap.call_args
        self.assertEqual(call_args[0][0], actor_tree._safe_save_data)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
