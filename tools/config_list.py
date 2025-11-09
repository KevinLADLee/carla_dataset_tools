#!/usr/bin/python3
"""
List Available Configuration Profiles
Shows all available YAML configuration profiles
"""
import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager


def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    config_root = project_root / 'config'

    config_manager = ConfigManager(str(config_root))
    profiles = config_manager.list_profiles()

    if not profiles:
        print("No configuration profiles found in config/profiles/")
        return 1

    print("Available Configuration Profiles:")
    print("=" * 70)

    for profile in profiles:
        try:
            # Load profile to get info
            config = config_manager.load_profile(profile)

            print(f"\n{profile}")
            print("-" * 70)
            print(f"  Map: {config['recording']['map']}")
            print(f"  Frames: {config['recording']['frame_total']} (step: {config['recording']['frame_step']})")
            print(f"  Actors: {len(config['actors'])}")

            # Count sensors
            total_sensors = sum(len(actor.get('sensors', [])) for actor in config['actors'])
            print(f"  Sensors: {total_sensors}")

            # Background vehicles
            if 'other_vehicles' in config:
                ov_count = config['other_vehicles'].get('count', 0)
                print(f"  Background vehicles: {ov_count}")

            # Show profile file location
            profile_path = config_manager.profiles_dir / f"{profile}.yaml"
            print(f"  File: {profile_path}")

        except Exception as e:
            print(f"\n{profile}")
            print(f"  Error loading profile: {e}")

    print("\n" + "=" * 70)
    print(f"\nUsage:")
    print(f"  python3 data_recorder.py --profile <profile_name>")
    print(f"  python3 utils/validate_config.py --profile <profile_name>")

    return 0


if __name__ == '__main__':
    exit(main())
