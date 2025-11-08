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
        'sensor.lidar.ray_cast',
        'sensor.lidar.ray_cast_semantic',
        'sensor.other.radar',
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

            # Validate sensors if present
            if 'sensors' in actor:
                self._validate_sensors(actor['sensors'], i, config_path)

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
