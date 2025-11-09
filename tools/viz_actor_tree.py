#!/usr/bin/python3
"""
Actor Tree Visualization Tool
Visualizes actor hierarchy from configuration files using Graphviz
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager, ConfigValidationError


def check_graphviz():
    """Check if graphviz is installed"""
    try:
        import graphviz
        return True
    except ImportError:
        return False


class ActorTreeVisualizer:
    """Visualizes actor tree from configuration"""

    # Color scheme for different node types
    COLORS = {
        'world': '#87CEEB',           # Sky blue
        'vehicle': '#90EE90',         # Light green
        'infrastructure': '#FFB347',  # Light orange
        'camera': '#B0E0E6',          # Powder blue
        'lidar': '#DDA0DD',           # Plum
        'radar': '#FFD700',           # Gold
        'other_sensor': '#D3D3D3',    # Light gray
        'other_vehicle': '#FFE4B5',   # Moccasin
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer with configuration

        Args:
            config: Configuration dictionary from ConfigManager
        """
        self.config = config
        self.node_id = 0

        # Import graphviz here (after dependency check)
        import graphviz
        self.graphviz = graphviz

        # Create directed graph
        self.graph = graphviz.Digraph(
            name='Actor_Tree',
            comment='CARLA Actor Tree Visualization',
            format='png',
            graph_attr={
                'rankdir': 'TB',  # Top to Bottom
                'splines': 'ortho',
                'nodesep': '0.5',
                'ranksep': '0.8',
                'bgcolor': 'white',
                'fontname': 'Arial',
                'fontsize': '12',
            },
            node_attr={
                'shape': 'box',
                'style': 'filled,rounded',
                'fontname': 'Arial',
                'fontsize': '10',
                'margin': '0.2,0.1',
            },
            edge_attr={
                'fontname': 'Arial',
                'fontsize': '9',
                'color': '#666666',
            }
        )

    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        node_id = f"node_{self.node_id}"
        self.node_id += 1
        return node_id

    def _get_sensor_color(self, sensor_type: str) -> str:
        """Get color for sensor based on type"""
        if 'camera' in sensor_type:
            return self.COLORS['camera']
        elif 'lidar' in sensor_type:
            return self.COLORS['lidar']
        elif 'radar' in sensor_type:
            return self.COLORS['radar']
        else:
            return self.COLORS['other_sensor']

    def _format_spawn_point(self, spawn_point: Any) -> str:
        """Format spawn point for display"""
        if isinstance(spawn_point, int):
            return f"Spawn #{spawn_point}"
        elif isinstance(spawn_point, dict):
            x = spawn_point.get('x', 0)
            y = spawn_point.get('y', 0)
            z = spawn_point.get('z', 0)
            roll = spawn_point.get('roll', 0)
            pitch = spawn_point.get('pitch', 0)
            yaw = spawn_point.get('yaw', 0)

            pos_str = f"pos: ({x:.1f}, {y:.1f}, {z:.1f})"
            if roll != 0 or pitch != 0 or yaw != 0:
                rot_str = f"\\nrot: ({roll:.1f}, {pitch:.1f}, {yaw:.1f})"
            else:
                rot_str = ""
            return pos_str + rot_str
        return "Unknown"

    def _format_sensor_params(self, sensor_info: Dict[str, Any]) -> List[str]:
        """Format sensor parameters for display"""
        params = []
        sensor_type = sensor_info.get('type', '')

        # Common camera parameters
        if 'camera' in sensor_type:
            if 'image_size_x' in sensor_info:
                width = sensor_info['image_size_x']
                height = sensor_info.get('image_size_y', 600)
                params.append(f"resolution: {width}x{height}")
            if 'fov' in sensor_info:
                params.append(f"FOV: {sensor_info['fov']}°")

        # LiDAR parameters
        elif 'lidar' in sensor_type:
            if 'range' in sensor_info:
                params.append(f"range: {sensor_info['range']}m")
            if 'channels' in sensor_info:
                params.append(f"channels: {sensor_info['channels']}")
            if 'points_per_second' in sensor_info:
                pps = sensor_info['points_per_second']
                params.append(f"pps: {pps:,}")
            if 'rotation_frequency' in sensor_info:
                params.append(f"freq: {sensor_info['rotation_frequency']}Hz")

        # Radar parameters
        elif 'radar' in sensor_type:
            if 'horizontal_fov' in sensor_info:
                params.append(f"H-FOV: {sensor_info['horizontal_fov']}°")
            if 'vertical_fov' in sensor_info:
                params.append(f"V-FOV: {sensor_info['vertical_fov']}°")
            if 'range' in sensor_info:
                params.append(f"range: {sensor_info['range']}m")

        return params

    def _create_summary_node(self):
        """Create summary node with statistics"""
        summary_lines = [
            "=== Configuration Summary ===",
            f"Map: {self.config['recording']['map']}",
        ]

        # Weather if specified
        if 'weather' in self.config['recording'] and self.config['recording']['weather']:
            summary_lines.append(f"Weather: {self.config['recording']['weather']}")

        # Recording settings
        summary_lines.extend([
            f"Frames: {self.config['recording']['frame_total']} (step: {self.config['recording']['frame_step']})",
            f"Delta seconds: {self.config['world_settings']['fixed_delta_seconds']}s",
            "",
        ])

        # Count actors and sensors
        num_actors = len(self.config['actors'])
        num_vehicles = sum(1 for a in self.config['actors'] if a['type'].startswith('vehicle'))
        num_infra = sum(1 for a in self.config['actors'] if a['type'].startswith('infrastructure'))
        total_sensors = sum(len(a.get('sensors', [])) for a in self.config['actors'])

        summary_lines.extend([
            f"Total Actors: {num_actors}",
            f"  Vehicles: {num_vehicles}",
            f"  Infrastructure: {num_infra}",
            f"  Total Sensors: {total_sensors}",
        ])

        # Background vehicles
        if 'other_vehicles' in self.config:
            ov = self.config['other_vehicles']
            ov_count = ov.get('count', 0)
            ov_spawns = len(ov.get('spawn_points', []))
            total_bg = ov_count + ov_spawns
            summary_lines.append(f"  Background Vehicles: {total_bg}")

        summary_text = "\\n".join(summary_lines)

        summary_id = self._generate_node_id()
        self.graph.node(
            summary_id,
            summary_text,
            shape='note',
            fillcolor='#FFFACD',  # Lemon chiffon
            style='filled',
            fontsize='11',
        )

        return summary_id

    def _create_world_node(self) -> str:
        """Create world root node"""
        world_label = "World\\n" + f"(Map: {self.config['recording']['map']})"
        world_id = self._generate_node_id()

        self.graph.node(
            world_id,
            world_label,
            fillcolor=self.COLORS['world'],
            shape='ellipse',
            fontsize='12',
            style='filled',
        )

        return world_id

    def _create_vehicle_node(self, actor_info: Dict[str, Any], parent_id: str) -> str:
        """Create vehicle node"""
        vehicle_type = actor_info['type']
        vehicle_name = actor_info.get('name', vehicle_type)
        spawn_point = self._format_spawn_point(actor_info['spawn_point'])

        label_lines = [
            f"🚗 {vehicle_name}",
            f"type: {vehicle_type}",
            spawn_point,
        ]

        # Add route information if present
        if 'route' in actor_info:
            route = actor_info['route']
            if 'from_file' in route:
                label_lines.append(f"route: {route['from_file']}")
            elif 'waypoints' in route:
                wp_count = len(route['waypoints'])
                mode = route.get('mode', 'strict')
                label_lines.append(f"route: {wp_count} waypoints ({mode})")

        label = "\\n".join(label_lines)
        vehicle_id = self._generate_node_id()

        self.graph.node(
            vehicle_id,
            label,
            fillcolor=self.COLORS['vehicle'],
        )

        self.graph.edge(parent_id, vehicle_id)

        return vehicle_id

    def _create_infrastructure_node(self, actor_info: Dict[str, Any], parent_id: str) -> str:
        """Create infrastructure node"""
        infra_name = actor_info.get('name', 'Infrastructure')
        spawn_point = self._format_spawn_point(actor_info['spawn_point'])

        label_lines = [
            f"🚨 {infra_name}",
            "type: infrastructure",
            spawn_point,
        ]

        label = "\\n".join(label_lines)
        infra_id = self._generate_node_id()

        self.graph.node(
            infra_id,
            label,
            fillcolor=self.COLORS['infrastructure'],
        )

        self.graph.edge(parent_id, infra_id)

        return infra_id

    def _create_sensor_node(self, sensor_info: Dict[str, Any], parent_id: str):
        """Create sensor node"""
        sensor_type = sensor_info['type']
        sensor_name = sensor_info['name']
        spawn_point = self._format_spawn_point(sensor_info['spawn_point'])

        # Get sensor icon
        if 'camera' in sensor_type:
            icon = '📷'
        elif 'lidar' in sensor_type:
            icon = '🚨'
        elif 'radar' in sensor_type:
            icon = '📡'
        else:
            icon = '🔌'

        label_lines = [
            f"{icon} {sensor_name}",
            f"type: {sensor_type}",
        ]

        # Add sensor parameters
        params = self._format_sensor_params(sensor_info)
        if params:
            label_lines.extend(params)

        # Add spawn point
        label_lines.append(spawn_point)

        label = "\\n".join(label_lines)
        sensor_id = self._generate_node_id()
        sensor_color = self._get_sensor_color(sensor_type)

        self.graph.node(
            sensor_id,
            label,
            fillcolor=sensor_color,
        )

        self.graph.edge(parent_id, sensor_id)

    def _create_other_vehicles_node(self, parent_id: str):
        """Create node for background vehicles"""
        if 'other_vehicles' not in self.config:
            return

        ov = self.config['other_vehicles']
        ov_count = ov.get('count', 0)
        ov_spawns = ov.get('spawn_points', [])

        if ov_count == 0 and not ov_spawns:
            return

        label_lines = [
            "🚙 Background Vehicles",
        ]

        if ov_count > 0:
            label_lines.append(f"random: {ov_count}")

        if ov_spawns:
            label_lines.append(f"fixed: {len(ov_spawns)} positions")
            label_lines.append(f"spawns: {ov_spawns[:5]}{'...' if len(ov_spawns) > 5 else ''}")

        label = "\\n".join(label_lines)
        ov_id = self._generate_node_id()

        self.graph.node(
            ov_id,
            label,
            fillcolor=self.COLORS['other_vehicle'],
        )

        self.graph.edge(parent_id, ov_id)

    def build_tree(self):
        """Build the complete actor tree"""
        # Create summary node
        summary_id = self._create_summary_node()

        # Create world node
        world_id = self._create_world_node()

        # Connect summary to world
        self.graph.edge(summary_id, world_id, style='invis')

        # Create actor nodes
        for actor_info in self.config['actors']:
            actor_type = actor_info['type']

            if actor_type.startswith('vehicle'):
                vehicle_id = self._create_vehicle_node(actor_info, world_id)

                # Add sensors if present
                if 'sensors' in actor_info:
                    for sensor_info in actor_info['sensors']:
                        self._create_sensor_node(sensor_info, vehicle_id)

            elif actor_type.startswith('infrastructure'):
                infra_id = self._create_infrastructure_node(actor_info, world_id)

                # Add sensors if present
                if 'sensors' in actor_info:
                    for sensor_info in actor_info['sensors']:
                        self._create_sensor_node(sensor_info, infra_id)

        # Add background vehicles
        self._create_other_vehicles_node(world_id)

    def render(self, output_path: str, format: str = 'png', view: bool = False):
        """
        Render the graph to file

        Args:
            output_path: Output file path (without extension)
            format: Output format (png, svg, pdf)
            view: Whether to open the file after rendering
        """
        self.graph.format = format

        try:
            output_file = self.graph.render(
                output_path,
                cleanup=True,
                view=view
            )
            return output_file
        except Exception as e:
            raise RuntimeError(f"Failed to render graph: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Visualize CARLA actor tree from configuration file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize default profile
  python3 tools/viz_actor_tree.py default

  # Visualize with custom output
  python3 tools/viz_actor_tree.py simple --output my_tree --format svg

  # Visualize and open automatically
  python3 tools/viz_actor_tree.py argoverse --view

  # List available profiles
  python3 tools/viz_actor_tree.py --list
        """
    )

    parser.add_argument(
        'config',
        nargs='?',
        help='Configuration file path or profile name'
    )
    parser.add_argument(
        '--output', '-o',
        default='actor_tree',
        help='Output file path (without extension, default: actor_tree)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['png', 'svg', 'pdf'],
        default='png',
        help='Output format (default: png)'
    )
    parser.add_argument(
        '--view', '-v',
        action='store_true',
        help='Open the visualization after generation'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available configuration profiles'
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    config_root = project_root / 'config'

    config_manager = ConfigManager(str(config_root))

    # List profiles (doesn't require graphviz)
    if args.list:
        profiles = config_manager.list_profiles()
        print("Available configuration profiles:")
        for profile in profiles:
            print(f"  - {profile}")
        return 0

    # Require config argument
    if not args.config:
        parser.print_help()
        print("\nError: Please specify a configuration file or profile name")
        print("Use --list to see available profiles")
        return 1

    # Load configuration
    try:
        # Try as file path first
        if Path(args.config).exists():
            print(f"Loading configuration file: {args.config}")
            config = config_manager.load_config(args.config)
        else:
            # Try as profile name
            print(f"Loading configuration profile: {args.config}")
            config = config_manager.load_profile(args.config)

        print("Configuration loaded successfully\n")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        available = config_manager.list_profiles()
        if available:
            print(f"\nAvailable profiles: {', '.join(available)}")
        return 1

    except ConfigValidationError as e:
        print(f"Configuration validation failed:")
        print(f"  {e}")
        return 1

    # Check graphviz dependency before creating visualizer
    if not check_graphviz():
        print("\nError: graphviz Python package is not installed")
        print("\nTo install:")
        print("  pip install graphviz")
        print("\nYou may also need to install the graphviz system package:")
        print("  Ubuntu/Debian: sudo apt-get install graphviz")
        print("  Fedora/RHEL: sudo dnf install graphviz")
        print("  macOS: brew install graphviz")
        return 1

    # Create visualizer
    try:
        print("Building actor tree visualization...")
        visualizer = ActorTreeVisualizer(config)
        visualizer.build_tree()

        # Render
        print(f"Rendering to {args.format.upper()} format...")
        output_file = visualizer.render(
            args.output,
            format=args.format,
            view=args.view
        )

        print(f"\n✓ Visualization saved to: {output_file}")

        if args.view:
            print("Opening visualization...")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
