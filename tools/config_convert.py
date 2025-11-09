#!/usr/bin/python3
"""
JSON to YAML Configuration Converter
Converts old JSON configuration files to new YAML format
"""
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def convert_json_to_yaml(json_path: str) -> Dict[str, Any]:
    """
    Convert JSON configuration to YAML format

    Args:
        json_path: Path to JSON file

    Returns:
        Configuration dictionary
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def load_nested_json_config(world_config_path: str, config_root: str) -> Dict[str, Any]:
    """
    Load the old nested JSON configuration structure

    Args:
        world_config_path: Path to world config JSON file
        config_root: Root config directory

    Returns:
        Merged configuration dictionary
    """
    # Load world config
    with open(world_config_path, 'r') as f:
        world_config = json.load(f)

    # Load actor config
    actor_config_file = world_config.get('actor_settings')
    if actor_config_file:
        actor_config_path = Path(config_root) / actor_config_file
        with open(actor_config_path, 'r') as f:
            actor_config = json.load(f)
    else:
        actor_config = {'actors': [], 'other_vehicles': {}}

    # Merge sensors into actors
    merged_actors = []
    for actor in actor_config.get('actors', []):
        sensor_config_file = actor.pop('sensors_setting', None)

        if sensor_config_file:
            sensor_config_path = Path(config_root) / sensor_config_file
            try:
                with open(sensor_config_path, 'r') as f:
                    sensor_config = json.load(f)
                actor['sensors'] = sensor_config.get('sensors', [])
            except FileNotFoundError:
                print(f"Warning: Sensor config not found: {sensor_config_path}")
                actor['sensors'] = []
        else:
            actor['sensors'] = []

        merged_actors.append(actor)

    # Build unified config
    unified_config = {
        'recording': {
            'frame_total': world_config.get('frame_total', 12000),
            'frame_step': world_config.get('frame_step', 3),
            'map': world_config.get('map', 'Town02'),
        },
        'spectator': world_config.get('spectator_pose', {
            'x': 0, 'y': 0, 'z': 100, 'roll': 0, 'pitch': 45, 'yaw': 0
        }),
        'world_settings': world_config.get('world_settings', {}),
        'traffic_lights': world_config.get('traffic_light_setting', {}),
        'actors': merged_actors,
        'other_vehicles': actor_config.get('other_vehicles', {'count': 0}),
    }

    # Ensure other_vehicles has correct keys
    other_vehicles = unified_config['other_vehicles']
    if 'vehicle_num' in other_vehicles:
        other_vehicles['count'] = other_vehicles.pop('vehicle_num')
    if 'spawn_points' not in other_vehicles:
        other_vehicles['spawn_points'] = []

    return unified_config


def save_yaml(config: Dict[str, Any], output_path: str, add_comments: bool = True):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        output_path: Output YAML file path
        add_comments: Whether to add descriptive comments
    """
    # Prepare YAML content
    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Add header comment
    header = (
        "# CARLA Dataset Recording Configuration\n"
        "# Compatible with CARLA 0.9.16\n"
        "# Converted from JSON configuration\n\n"
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(yaml_content)

    print(f"✓ Converted to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON configuration to YAML format'
    )
    parser.add_argument(
        'json_file',
        help='Path to world config JSON file or config directory'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output YAML file path (default: same name with .yaml extension)'
    )
    parser.add_argument(
        '--config-root',
        default=None,
        help='Config root directory (for resolving nested configs)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch convert all JSON files in directory'
    )

    args = parser.parse_args()

    json_path = Path(args.json_file)

    # Determine config root
    if args.config_root:
        config_root = args.config_root
    elif json_path.is_dir():
        config_root = str(json_path)
    else:
        config_root = str(json_path.parent)

    # Batch mode
    if args.batch or json_path.is_dir():
        if json_path.is_dir():
            json_files = list(json_path.glob('**/*config*.json'))
        else:
            json_files = [json_path]

        print(f"Found {len(json_files)} JSON config files to convert")

        for json_file in json_files:
            try:
                # Try to load as nested config
                config = load_nested_json_config(str(json_file), config_root)

                # Determine output path
                if args.output:
                    output_path = args.output
                else:
                    output_path = str(json_file.with_suffix('.yaml'))

                save_yaml(config, output_path)

            except Exception as e:
                print(f"✗ Failed to convert {json_file}: {e}")

    # Single file mode
    else:
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            return 1

        try:
            # Try to load as nested config
            config = load_nested_json_config(str(json_path), config_root)

            # Determine output path
            if args.output:
                output_path = args.output
            else:
                output_path = str(json_path.with_suffix('.yaml'))

            save_yaml(config, output_path)

            print("\nConversion successful!")
            print(f"You can now use: python3 data_recorder.py --config {output_path}")

        except Exception as e:
            print(f"Error during conversion: {e}")
            return 1

    return 0


if __name__ == '__main__':
    exit(main())
