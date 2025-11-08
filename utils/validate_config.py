#!/usr/bin/python3
"""
Configuration Validation Tool
Validates YAML configuration files for CARLA dataset recording
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import config_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager, ConfigValidationError


def main():
    parser = argparse.ArgumentParser(
        description='Validate CARLA dataset recording configuration'
    )
    parser.add_argument(
        'config',
        nargs='?',
        help='Configuration file path or profile name'
    )
    parser.add_argument(
        '--profile',
        help='Profile name to validate'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available profiles'
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    config_root = project_root / 'config'

    config_manager = ConfigManager(str(config_root))

    # List profiles
    if args.list:
        profiles = config_manager.list_profiles()
        print("Available configuration profiles:")
        for profile in profiles:
            print(f"  - {profile}")
        return 0

    # Determine what to validate
    config_to_validate = args.config or args.profile

    if not config_to_validate:
        parser.print_help()
        print("\nError: Please specify a configuration file or profile name")
        return 1

    # Validate
    try:
        # Try as file path first
        if Path(config_to_validate).exists():
            print(f"Validating configuration file: {config_to_validate}")
            config = config_manager.load_config(config_to_validate)
        else:
            # Try as profile name
            print(f"Validating configuration profile: {config_to_validate}")
            config = config_manager.load_profile(config_to_validate)

        # If we get here, validation passed
        print("\n✓ Configuration is valid!")
        print("\nConfiguration summary:")
        print(f"  Map: {config['recording']['map']}")
        print(f"  Total frames: {config['recording']['frame_total']}")
        print(f"  Frame step: {config['recording']['frame_step']}")
        print(f"  Synchronous mode: {config['world_settings']['synchronous_mode']}")
        print(f"  Fixed delta seconds: {config['world_settings']['fixed_delta_seconds']}")
        print(f"  Number of actors: {len(config['actors'])}")

        if 'other_vehicles' in config:
            ov = config['other_vehicles']
            print(f"  Background vehicles: {ov.get('count', 0)}")

        # Show sensor count
        total_sensors = sum(len(actor.get('sensors', [])) for actor in config['actors'])
        print(f"  Total sensors: {total_sensors}")

        return 0

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        available = config_manager.list_profiles()
        if available:
            print(f"\nAvailable profiles: {', '.join(available)}")
        return 1

    except ConfigValidationError as e:
        print(f"\n✗ Validation failed:")
        print(f"  {e}")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
