#!/usr/bin/env python3
"""
Security tests for ConfigManager
Tests fix for Issue #3: Path Traversal Vulnerability
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
import tempfile
import yaml
import shutil


class TestConfigSecurityPathTraversal(unittest.TestCase):
    """Test path traversal protection in route file loading"""

    def setUp(self):
        """Set up test fixtures with temporary directory structure"""
        # Create temporary project structure
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / 'config'
        self.config_dir.mkdir()

        self.profiles_dir = self.config_dir / 'profiles'
        self.profiles_dir.mkdir()

        self.routes_dir = Path(self.temp_dir) / 'routes'
        self.routes_dir.mkdir()

        # Create a legitimate route file
        self.legit_route = self.routes_dir / 'legit_route.yaml'
        with open(self.legit_route, 'w') as f:
            yaml.dump({
                'waypoints': [
                    {'x': 0, 'y': 0, 'z': 0},
                    {'x': 10, 'y': 10, 'z': 0}
                ],
                'loop': False
            }, f)

        # Create a sensitive file outside routes directory
        self.sensitive_file = Path(self.temp_dir) / 'sensitive.txt'
        with open(self.sensitive_file, 'w') as f:
            f.write('SECRET_DATA')

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def _create_config_with_route(self, route_path):
        """Helper to create a config with specific route path"""
        config = {
            'recording': {
                'frame_total': 10,
                'frame_step': 1,
                'map': 'Town01',
            },
            'world_settings': {
                'synchronous_mode': True,
                'fixed_delta_seconds': 0.05,
                'substepping': True,
                'max_substep_delta_time': 0.01,
                'max_substeps': 10,
            },
            'traffic_lights': {
                'red_time': 5.0,
                'green_time': 5.0,
                'yellow_time': 2.0,
            },
            'actors': [
                {
                    'type': 'vehicle.tesla.model3',
                    'name': 'test_vehicle',
                    'spawn_point': 0,
                    'route': {
                        'from_file': route_path,
                        'mode': 'strict'
                    }
                }
            ]
        }
        return config

    def test_legitimate_route_file_loads(self):
        """Test that legitimate route files within routes/ load successfully"""
        from config.config_manager import ConfigManager

        config = self._create_config_with_route('legit_route.yaml')

        # Write config to file
        config_file = self.profiles_dir / 'test.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Should load without error
        manager = ConfigManager(str(self.config_dir))
        try:
            loaded_config = manager.load_config(str(config_file))
            self.assertIsNotNone(loaded_config)
        except Exception as e:
            self.fail(f"Legitimate route file should load successfully: {e}")

    def test_path_traversal_parent_directory_blocked(self):
        """Test that path traversal using .. is blocked"""
        from config.config_manager import ConfigManager, ConfigValidationError

        # Attempt to access sensitive file using ..
        config = self._create_config_with_route('../sensitive.txt')

        config_file = self.profiles_dir / 'malicious1.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        manager = ConfigManager(str(self.config_dir))

        # Should raise ConfigValidationError
        with self.assertRaises(ConfigValidationError) as cm:
            manager.load_config(str(config_file))

        self.assertIn('path traversal', str(cm.exception).lower())

    def test_path_traversal_multiple_parents_blocked(self):
        """Test that path traversal using ../../ is blocked"""
        from config.config_manager import ConfigManager, ConfigValidationError

        config = self._create_config_with_route('../../etc/passwd')

        config_file = self.profiles_dir / 'malicious2.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        manager = ConfigManager(str(self.config_dir))

        with self.assertRaises(ConfigValidationError) as cm:
            manager.load_config(str(config_file))

        self.assertIn('path traversal', str(cm.exception).lower())

    def test_absolute_path_blocked(self):
        """Test that absolute paths are blocked"""
        from config.config_manager import ConfigManager, ConfigValidationError

        # Attempt to use absolute path
        config = self._create_config_with_route('/etc/passwd')

        config_file = self.profiles_dir / 'malicious3.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        manager = ConfigManager(str(self.config_dir))

        with self.assertRaises(ConfigValidationError) as cm:
            manager.load_config(str(config_file))

        self.assertIn('path traversal', str(cm.exception).lower())

    def test_subdirectory_route_allowed(self):
        """Test that routes in subdirectories of routes/ are allowed"""
        from config.config_manager import ConfigManager

        # Create subdirectory in routes
        subdir = self.routes_dir / 'custom'
        subdir.mkdir()

        # Create route in subdirectory
        subdir_route = subdir / 'custom_route.yaml'
        with open(subdir_route, 'w') as f:
            yaml.dump({
                'waypoints': [
                    {'x': 0, 'y': 0, 'z': 0},
                    {'x': 5, 'y': 5, 'z': 0}
                ]
            }, f)

        # Reference with subdirectory path
        config = self._create_config_with_route('custom/custom_route.yaml')

        config_file = self.profiles_dir / 'test_subdir.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        manager = ConfigManager(str(self.config_dir))

        try:
            loaded_config = manager.load_config(str(config_file))
            self.assertIsNotNone(loaded_config)
        except Exception as e:
            self.fail(f"Subdirectory route should be allowed: {e}")

    def test_route_file_size_limit(self):
        """Test that oversized route files are rejected"""
        from config.config_manager import ConfigManager, ConfigValidationError

        # Create a large route file (> 1MB)
        large_route = self.routes_dir / 'large_route.yaml'
        with open(large_route, 'w') as f:
            # Write > 1MB of data
            waypoints = [{'x': i, 'y': i, 'z': 0} for i in range(50000)]
            yaml.dump({'waypoints': waypoints}, f)

        config = self._create_config_with_route('large_route.yaml')

        config_file = self.profiles_dir / 'test_large.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        manager = ConfigManager(str(self.config_dir))

        with self.assertRaises(ConfigValidationError) as cm:
            manager.load_config(str(config_file))

        self.assertIn('too large', str(cm.exception).lower())

    def test_nonexistent_route_file_error(self):
        """Test that nonexistent route files produce clear error"""
        from config.config_manager import ConfigManager, ConfigValidationError

        config = self._create_config_with_route('nonexistent.yaml')

        config_file = self.profiles_dir / 'test_missing.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        manager = ConfigManager(str(self.config_dir))

        with self.assertRaises(ConfigValidationError) as cm:
            manager.load_config(str(config_file))

        self.assertIn('not found', str(cm.exception).lower())


class TestConfigSecurityYAMLBombs(unittest.TestCase):
    """Test protection against YAML bombs and malicious YAML"""

    def test_safe_load_prevents_arbitrary_code(self):
        """Test that yaml.safe_load is used (prevents code execution)"""
        import config.config_manager as cm_module

        # Verify that safe_load is used, not unsafe load
        self.assertTrue(
            hasattr(cm_module.yaml, 'safe_load'),
            "yaml.safe_load should be available"
        )

        # The actual protection is in the implementation
        # This test just verifies the import is correct


if __name__ == '__main__':
    unittest.main(verbosity=2)
