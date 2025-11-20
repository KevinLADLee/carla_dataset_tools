#!/usr/bin/python3
"""
Configuration Manager for CARLA Dataset Tools
Provides unified YAML configuration loading and validation
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigManager:
    """Manages loading and validation of YAML configuration files"""

    # Valid sensor types according to CARLA 0.9.16
    VALID_SENSOR_TYPES = {
        'sensor.camera.rgb',
        'sensor.camera.depth',
        'sensor.camera.semantic_segmentation',
        'sensor.camera.instance_segmentation',  # Instance segmentation camera
        'sensor.camera.dvs',                    # Dynamic Vision Sensor (event camera)
        'sensor.camera.optical_flow',          # Optical flow camera
        'sensor.lidar.ray_cast',
        'sensor.lidar.ray_cast_semantic',
        'sensor.other.radar',
        'sensor.other.imu',                     # Inertial Measurement Unit
        'sensor.other.gnss',                    # Global Navigation Satellite System
        'sensor.other.v2x',                     # V2X CAM sensor (Cooperative Awareness Message)
        'sensor.other.v2x_custom',              # V2X Custom message sensor
    }

    # Available maps in CARLA 0.9.16
    VALID_MAPS = {
        'AnnotationColorLandscape',
        'Town01', 'Town01_Opt',
        'Town02', 'Town02_Opt',
        'Town03', 'Town03_Opt',
        'Town04', 'Town04_Opt',
        'Town05', 'Town05_Opt',
        'Town06', 'Town06_Opt',
        'Town07', 'Town07_Opt',
        'Town10HD', 'Town10HD_Opt',
        'Town11', 'Town12', 'Town13', 'Town15',
    }

    # Available weather presets in CARLA 0.9.16
    VALID_WEATHER_PRESETS = {
        'Default',
        'ClearNoon', 'ClearSunset', 'ClearNight',
        'CloudyNoon', 'CloudySunset', 'CloudyNight',
        'WetNoon', 'WetSunset', 'WetNight',
        'WetCloudyNoon', 'WetCloudySunset', 'WetCloudyNight',
        'SoftRainNoon', 'SoftRainSunset', 'SoftRainNight',
        'MidRainyNoon', 'MidRainSunset', 'MidRainyNight',
        'HardRainNoon', 'HardRainSunset', 'HardRainNight',
        'DustStorm',
    }

    # Required configuration keys
    REQUIRED_KEYS = {
        'recording': ['frame_total', 'frame_step', 'map'],
        'world_settings': ['synchronous_mode', 'fixed_delta_seconds',
                          'substepping', 'max_substep_delta_time', 'max_substeps'],
        'traffic_lights': ['red_time', 'green_time', 'yellow_time'],
        'actors': [],  # List, will be validated separately
    }

    # Maximum config file size (10MB for security)
    MAX_CONFIG_SIZE = 10 * 1024 * 1024

    def __init__(self, config_root: Optional[str] = None):
        """
        Initialize ConfigManager

        Args:
            config_root: Root directory for config files.
                        Defaults to project_root/config
        """
        if config_root is None:
            # Get project root (parent of config directory)
            self.config_root = Path(__file__).parent
        else:
            self.config_root = Path(config_root)

        self.profiles_dir = self.config_root / 'profiles'
        self.config = None

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Load a configuration profile by name

        Args:
            profile_name: Name of the profile (without .yaml extension)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If profile doesn't exist
            ConfigValidationError: If configuration is invalid
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        return self.load_config(str(profile_path))

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If configuration is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Available profiles: {self.list_profiles()}"
            )

        # Security: Check file size to prevent DoS attacks
        file_size = os.path.getsize(config_path)
        if file_size > self.MAX_CONFIG_SIZE:
            raise ConfigValidationError(
                f"Configuration file too large: {file_size} bytes "
                f"(max: {self.MAX_CONFIG_SIZE} bytes / {self.MAX_CONFIG_SIZE // 1024 // 1024}MB)"
            )

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Failed to parse YAML configuration: {config_path}\n"
                f"Error: {e}"
            )

        # Validate and apply environment variable overrides
        config = self._apply_env_overrides(config)
        self._validate_config(config, str(config_path))

        self.config = config
        return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration

        Environment variables:
            CARLA_HOST: Override host
            CARLA_PORT: Override port
            CARLA_MAP: Override map name

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with overrides applied
        """
        # Host and port are handled at runtime level, not in config file
        # But we can override map if needed
        if 'CARLA_MAP' in os.environ:
            if 'recording' not in config:
                config['recording'] = {}
            config['recording']['map'] = os.environ['CARLA_MAP']
            print(f"Config override: map = {os.environ['CARLA_MAP']} (from CARLA_MAP env)")

        return config

    def _validate_config(self, config: Dict[str, Any], config_path: str):
        """
        Validate configuration structure and values

        Args:
            config: Configuration dictionary to validate
            config_path: Path to config file (for error messages)

        Raises:
            ConfigValidationError: If validation fails
        """
        # Check top-level required sections
        for section, required_keys in self.REQUIRED_KEYS.items():
            if section not in config:
                raise ConfigValidationError(
                    f"Missing required section '{section}' in {config_path}"
                )

            # Check required keys in section
            if isinstance(required_keys, list) and required_keys:
                for key in required_keys:
                    if key not in config[section]:
                        raise ConfigValidationError(
                            f"Missing required key '{section}.{key}' in {config_path}"
                        )

        # Validate recording settings
        self._validate_recording_settings(config['recording'], config_path)

        # Validate world settings
        self._validate_world_settings(config['world_settings'], config_path)

        # Validate actors
        self._validate_actors(config['actors'], config_path)

    def _validate_recording_settings(self, recording: Dict[str, Any], config_path: str):
        """Validate recording section"""
        if recording['frame_total'] <= 0:
            raise ConfigValidationError(
                f"frame_total must be > 0 in {config_path}"
            )

        if recording['frame_step'] <= 0:
            raise ConfigValidationError(
                f"frame_step must be > 0 in {config_path}"
            )

        # Validate map name
        map_name = recording['map']
        if not map_name:
            raise ConfigValidationError(
                f"map name cannot be empty in {config_path}"
            )

        if map_name not in self.VALID_MAPS:
            raise ConfigValidationError(
                f"Invalid map '{map_name}' in {config_path}.\n"
                f"Available maps: {sorted(self.VALID_MAPS)}"
            )

        # Validate weather preset if specified
        if 'weather' in recording and recording['weather'] is not None:
            weather = recording['weather']
            if weather not in self.VALID_WEATHER_PRESETS:
                raise ConfigValidationError(
                    f"Invalid weather preset '{weather}' in {config_path}.\n"
                    f"Available presets: {sorted(self.VALID_WEATHER_PRESETS)}"
                )

    def _validate_world_settings(self, world_settings: Dict[str, Any], config_path: str):
        """Validate world_settings section according to CARLA 0.9.16 API"""
        # Validate fixed_delta_seconds
        if world_settings['fixed_delta_seconds'] < 0:
            raise ConfigValidationError(
                f"fixed_delta_seconds must be >= 0 in {config_path}"
            )

        # Validate substepping parameters
        if world_settings['substepping']:
            if world_settings['max_substep_delta_time'] <= 0:
                raise ConfigValidationError(
                    f"max_substep_delta_time must be > 0 when substepping is enabled in {config_path}"
                )

            if world_settings['max_substeps'] <= 0:
                raise ConfigValidationError(
                    f"max_substeps must be > 0 when substepping is enabled in {config_path}"
                )

            # Check constraint: fixed_delta_seconds <= max_substep_delta_time * max_substeps
            max_allowed = world_settings['max_substep_delta_time'] * world_settings['max_substeps']
            if world_settings['fixed_delta_seconds'] > max_allowed:
                raise ConfigValidationError(
                    f"fixed_delta_seconds ({world_settings['fixed_delta_seconds']}) must be <= "
                    f"max_substep_delta_time * max_substeps ({max_allowed}) in {config_path}"
                )

    def _validate_actors(self, actors: list, config_path: str):
        """Validate actors configuration"""
        if not isinstance(actors, list):
            raise ConfigValidationError(
                f"'actors' must be a list in {config_path}"
            )

        for i, actor in enumerate(actors):
            if 'type' not in actor:
                raise ConfigValidationError(
                    f"Actor {i} missing 'type' in {config_path}"
                )

            # Validate spawn_point
            if 'spawn_point' not in actor:
                raise ConfigValidationError(
                    f"Actor {i} missing 'spawn_point' in {config_path}"
                )

            # Validate route if present
            if 'route' in actor:
                self._validate_route(actor['route'], i, config_path)

            # Validate sensors if present
            if 'sensors' in actor:
                self._validate_sensors(actor['sensors'], i, config_path)

    def _validate_route(self, route: Dict[str, Any], actor_index: int, config_path: str):
        """Validate route configuration for an actor with path traversal protection"""
        if not isinstance(route, dict):
            raise ConfigValidationError(
                f"Actor {actor_index} route must be a dictionary in {config_path}"
            )

        # Check if loading from file
        if 'from_file' in route:
            route_file_str = route['from_file']

            # Security: Sanitize path to prevent path traversal attacks
            # Normalize path to resolve .. and . components
            route_file_str = os.path.normpath(route_file_str)

            # Reject absolute paths and paths starting with ..
            if os.path.isabs(route_file_str) or route_file_str.startswith('..'):
                raise ConfigValidationError(
                    f"Actor {actor_index} route path traversal detected: {route['from_file']}. "
                    f"Route files must be relative paths within the routes/ directory. "
                    f"Example: 'Town02_example_route.yaml' or 'custom/my_route.yaml'"
                )

            # Resolve relative to routes directory (not config root)
            routes_dir = self.config_root.parent / 'routes'
            route_file = routes_dir / route_file_str

            # Verify resolved path is within routes directory (prevent symlink attacks)
            try:
                route_file_resolved = route_file.resolve(strict=True)
                routes_dir_resolved = routes_dir.resolve()

                # Check if resolved path starts with routes directory
                if not str(route_file_resolved).startswith(str(routes_dir_resolved)):
                    raise ConfigValidationError(
                        f"Actor {actor_index} route file must be within routes/ directory. "
                        f"Attempted to access: {route['from_file']}"
                    )
            except FileNotFoundError:
                raise ConfigValidationError(
                    f"Actor {actor_index} route file not found: {route['from_file']}. "
                    f"Place route files in the routes/ directory."
                )
            except OSError as e:
                raise ConfigValidationError(
                    f"Actor {actor_index} route file error: {route['from_file']}: {e}"
                )

            # Security: Check file size to prevent resource exhaustion
            file_size = route_file_resolved.stat().st_size
            max_route_size = 1024 * 1024  # 1MB limit for route files
            if file_size > max_route_size:
                raise ConfigValidationError(
                    f"Actor {actor_index} route file too large: {route['from_file']} "
                    f"({file_size} bytes, max: {max_route_size} bytes / 1MB)"
                )

            # Load and validate the route file
            try:
                with open(route_file_resolved, 'r', encoding='utf-8') as f:
                    route_data = yaml.safe_load(f)
                    if 'waypoints' in route_data:
                        self._validate_waypoints(route_data['waypoints'], actor_index, config_path)
            except yaml.YAMLError as e:
                raise ConfigValidationError(
                    f"Actor {actor_index} failed to parse route YAML from {route['from_file']}: {e}"
                )
            except Exception as e:
                raise ConfigValidationError(
                    f"Actor {actor_index} failed to load route from {route['from_file']}: {e}"
                )

        # Validate mode if specified
        if 'mode' in route:
            valid_modes = {'strict', 'disabled'}
            if route['mode'] not in valid_modes:
                raise ConfigValidationError(
                    f"Actor {actor_index} route mode must be one of {valid_modes}, "
                    f"got '{route['mode']}' in {config_path}"
                )

        # Validate waypoints if present
        if 'waypoints' in route:
            self._validate_waypoints(route['waypoints'], actor_index, config_path)

    def _validate_waypoints(self, waypoints: list, actor_index: int, config_path: str):
        """Validate waypoints list"""
        if not isinstance(waypoints, list):
            raise ConfigValidationError(
                f"Actor {actor_index} route waypoints must be a list in {config_path}"
            )

        if len(waypoints) < 2:
            raise ConfigValidationError(
                f"Actor {actor_index} route must have at least 2 waypoints in {config_path}"
            )

        for i, wp in enumerate(waypoints):
            if not isinstance(wp, dict):
                raise ConfigValidationError(
                    f"Actor {actor_index} waypoint {i} must be a dictionary in {config_path}"
                )

            # Check required fields
            required_fields = {'x', 'y', 'z'}
            missing_fields = required_fields - set(wp.keys())
            if missing_fields:
                raise ConfigValidationError(
                    f"Actor {actor_index} waypoint {i} missing fields {missing_fields} in {config_path}"
                )

            # Validate that coordinates are numbers
            for field in required_fields:
                if not isinstance(wp[field], (int, float)):
                    raise ConfigValidationError(
                        f"Actor {actor_index} waypoint {i} field '{field}' must be a number in {config_path}"
                    )

    def _validate_sensors(self, sensors: list, actor_index: int, config_path: str):
        """Validate sensor configurations"""
        if not isinstance(sensors, list):
            raise ConfigValidationError(
                f"Actor {actor_index} sensors must be a list in {config_path}"
            )

        for j, sensor in enumerate(sensors):
            if 'type' not in sensor:
                raise ConfigValidationError(
                    f"Actor {actor_index} sensor {j} missing 'type' in {config_path}"
                )

            sensor_type = sensor['type']
            if sensor_type not in self.VALID_SENSOR_TYPES:
                raise ConfigValidationError(
                    f"Actor {actor_index} sensor {j} has invalid type '{sensor_type}'. "
                    f"Valid types: {self.VALID_SENSOR_TYPES} in {config_path}"
                )

            if 'name' not in sensor:
                raise ConfigValidationError(
                    f"Actor {actor_index} sensor {j} missing 'name' in {config_path}"
                )

            if 'spawn_point' not in sensor:
                raise ConfigValidationError(
                    f"Actor {actor_index} sensor {j} missing 'spawn_point' in {config_path}"
                )

    def list_profiles(self) -> list:
        """
        List all available configuration profiles

        Returns:
            List of profile names (without .yaml extension)
        """
        if not self.profiles_dir.exists():
            return []

        profiles = []
        for yaml_file in self.profiles_dir.glob('*.yaml'):
            profiles.append(yaml_file.stem)

        return sorted(profiles)

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get the currently loaded configuration"""
        return self.config


# Convenience function for simple usage
def load_config(profile_or_path: str, config_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from profile name or file path

    Args:
        profile_or_path: Profile name or path to YAML file
        config_root: Optional config root directory

    Returns:
        Configuration dictionary
    """
    manager = ConfigManager(config_root)

    # Check if it's a file path
    if os.path.exists(profile_or_path):
        return manager.load_config(profile_or_path)

    # Try as profile name
    return manager.load_profile(profile_or_path)
