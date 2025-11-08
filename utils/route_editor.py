#!/usr/bin/env python3
"""
Interactive Route Editor for CARLA Dataset Tools

This tool provides an interactive matplotlib-based interface to define vehicle routes
by clicking on the map. The routes can be saved as YAML configuration files and
used in the data recording process.

Usage:
    python3 route_editor.py --map Town02 --name my_route_01
"""

import argparse
import pickle
import yaml
from pathlib import Path

import carla
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


class RouteEditor:
    """Interactive route editor with matplotlib UI"""

    def __init__(self, args):
        self.args = args
        self.carla_client = carla.Client(args.host, args.port, worker_threads=1)
        self.world = self.carla_client.get_world()

        # Load the specified map if different from current
        if args.map and self.world.get_map().name.split('/')[-1] != args.map:
            print(f"Loading map: {args.map}")
            self.world = self.carla_client.load_world(args.map)

        self.map = self.world.get_map()
        self.map_name = self.map.name.split('/')[-1]

        # Route data
        self.waypoints = []  # List of {x, y, z} dicts
        self.waypoint_circles = []  # Visual elements
        self.route_lines = []  # Visual elements
        self.selected_waypoint = None

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title(f'Route Editor - {self.map_name}')

        # Draw map
        self._draw_roads()
        self._draw_spawn_points()

        # Setup event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Instructions
        self._show_instructions()

    def _show_instructions(self):
        """Display usage instructions"""
        instructions = (
            "Route Editor Controls:\n"
            "- Left Click: Add waypoint\n"
            "- Right Click on waypoint: Delete waypoint\n"
            "- Drag waypoint: Move waypoint\n"
            "- Enter: Save and exit\n"
            "- Escape: Cancel and exit"
        )
        self.ax.text(0.02, 0.98, instructions,
                    transform=self.ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    @staticmethod
    def _lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def _draw_roads(self):
        """Draw road network"""
        print("Drawing road network...")
        precision = 0.1
        topology = self.map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)

        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break

            # Draw lane boundaries
            road_left = [self._lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            road_right = [self._lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            if len(road_left) > 2:
                x = [p.x for p in road_left]
                y = [-p.y for p in road_left]
                self.ax.plot(x, y, color='darkslategrey', linewidth=1, alpha=0.5)

            if len(road_right) > 2:
                x = [p.x for p in road_right]
                y = [-p.y for p in road_right]
                self.ax.plot(x, y, color='darkslategrey', linewidth=1, alpha=0.5)

    def _draw_spawn_points(self):
        """Draw spawn point markers"""
        spawn_points = self.map.get_spawn_points()
        for i, sp in enumerate(spawn_points):
            x = sp.location.x
            y = -sp.location.y
            self.ax.text(x, y, str(i),
                        fontsize=6,
                        color='darkorange',
                        va='center',
                        ha='center',
                        weight='bold',
                        alpha=0.7)

    def _add_waypoint(self, x, y):
        """Add a new waypoint at the clicked position"""
        # Convert matplotlib coords to CARLA coords
        carla_x = x
        carla_y = -y

        # Get the nearest waypoint on the road
        location = carla.Location(x=carla_x, y=carla_y, z=0.0)
        waypoint = self.map.get_waypoint(location, project_to_road=True)

        if waypoint:
            wp_dict = {
                'x': waypoint.transform.location.x,
                'y': waypoint.transform.location.y,
                'z': waypoint.transform.location.z
            }
            self.waypoints.append(wp_dict)

            # Draw waypoint circle
            circle = Circle((wp_dict['x'], -wp_dict['y']), 2.0,
                          color='red', zorder=10, picker=5)
            self.ax.add_patch(circle)
            self.waypoint_circles.append(circle)

            # Update route lines
            self._update_route_lines()
            self.fig.canvas.draw_idle()

            print(f"Added waypoint {len(self.waypoints)}: ({wp_dict['x']:.2f}, {wp_dict['y']:.2f}, {wp_dict['z']:.2f})")

    def _remove_waypoint(self, index):
        """Remove waypoint at given index"""
        if 0 <= index < len(self.waypoints):
            removed = self.waypoints.pop(index)
            circle = self.waypoint_circles.pop(index)
            circle.remove()

            self._update_route_lines()
            self.fig.canvas.draw_idle()

            print(f"Removed waypoint: ({removed['x']:.2f}, {removed['y']:.2f}, {removed['z']:.2f})")

    def _update_route_lines(self):
        """Update the route line visualization"""
        # Remove old lines
        for line in self.route_lines:
            line.remove()
        self.route_lines.clear()

        # Draw new lines connecting waypoints
        if len(self.waypoints) > 1:
            x = [wp['x'] for wp in self.waypoints]
            y = [-wp['y'] for wp in self.waypoints]
            line, = self.ax.plot(x, y, 'b-', linewidth=2, zorder=5, marker='o', markersize=4)
            self.route_lines.append(line)

    def _find_waypoint_at_position(self, x, y, tolerance=3.0):
        """Find waypoint index near the clicked position"""
        for i, wp in enumerate(self.waypoints):
            dist = np.sqrt((wp['x'] - x)**2 + (-wp['y'] - y)**2)
            if dist < tolerance:
                return i
        return None

    def _on_click(self, event):
        """Handle mouse button press"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            # Check if clicking on existing waypoint to drag
            idx = self._find_waypoint_at_position(event.xdata, event.ydata)
            if idx is not None:
                self.selected_waypoint = idx
            else:
                # Add new waypoint
                self._add_waypoint(event.xdata, event.ydata)

        elif event.button == 3:  # Right click - delete waypoint
            idx = self._find_waypoint_at_position(event.xdata, event.ydata)
            if idx is not None:
                self._remove_waypoint(idx)

    def _on_release(self, event):
        """Handle mouse button release"""
        self.selected_waypoint = None

    def _on_motion(self, event):
        """Handle mouse motion for dragging"""
        if event.inaxes != self.ax:
            return

        if self.selected_waypoint is not None and event.button == 1:
            # Update waypoint position
            carla_x = event.xdata
            carla_y = -event.ydata

            # Snap to road
            location = carla.Location(x=carla_x, y=carla_y, z=0.0)
            waypoint = self.map.get_waypoint(location, project_to_road=True)

            if waypoint:
                self.waypoints[self.selected_waypoint] = {
                    'x': waypoint.transform.location.x,
                    'y': waypoint.transform.location.y,
                    'z': waypoint.transform.location.z
                }

                # Update visual
                wp = self.waypoints[self.selected_waypoint]
                self.waypoint_circles[self.selected_waypoint].center = (wp['x'], -wp['y'])

                self._update_route_lines()
                self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'enter':
            self._save_and_exit()
        elif event.key == 'escape':
            print("Cancelled. No route saved.")
            plt.close(self.fig)

    def _save_and_exit(self):
        """Save the route and exit"""
        if len(self.waypoints) < 2:
            print("Error: Route must have at least 2 waypoints!")
            return

        # Create routes directory if it doesn't exist
        routes_dir = Path('routes')
        routes_dir.mkdir(exist_ok=True)

        # Generate filename
        route_name = self.args.name or f"{self.map_name}_route_01"
        yaml_path = routes_dir / f"{route_name}.yaml"
        pkl_path = routes_dir / f"{route_name}.pkl"

        # Prepare route data
        route_data = {
            'map': self.map_name,
            'waypoints': self.waypoints,
            'mode': 'strict',
            'metadata': {
                'waypoint_count': len(self.waypoints),
                'created_with': 'route_editor.py'
            }
        }

        # Save as YAML
        with open(yaml_path, 'w') as f:
            yaml.dump(route_data, f, default_flow_style=False, sort_keys=False)
        print(f"\nRoute saved to: {yaml_path}")

        # Save as pickle (includes full carla.Waypoint objects for future use)
        with open(pkl_path, 'wb') as f:
            pickle.dump(route_data, f)
        print(f"Route saved to: {pkl_path}")

        # Print YAML snippet for easy copying to config files
        print("\n" + "="*60)
        print("YAML Configuration Snippet (copy to actors config):")
        print("="*60)
        print("route:")
        print(f"  mode: strict")
        print(f"  waypoints:")
        for wp in self.waypoints:
            print(f"    - {{x: {wp['x']:.2f}, y: {wp['y']:.2f}, z: {wp['z']:.2f}}}")
        print("\n# Or load from file:")
        print(f"# route:")
        print(f"#   from_file: {yaml_path}")
        print("="*60)

        plt.close(self.fig)

    def run(self):
        """Start the interactive editor"""
        plt.axis('equal')
        plt.tight_layout()
        print(f"\nRoute editor ready for map: {self.map_name}")
        print("Click on the map to add waypoints...")
        plt.show()


def main():
    argparser = argparse.ArgumentParser(
        description='Interactive Route Editor for CARLA Dataset Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a route for Town02
    python3 route_editor.py --map Town02 --name my_route_01

    # Edit route on current map
    python3 route_editor.py --name test_route
        """
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '-m', '--map',
        default=None,
        help='CARLA map to use (e.g., Town02). If not specified, uses current map.')
    argparser.add_argument(
        '-n', '--name',
        default=None,
        help='Name for the route file (default: <map>_route_01)')

    args = argparser.parse_args()

    try:
        editor = RouteEditor(args)
        editor.run()
    except KeyboardInterrupt:
        print('\nCancelled by user.')
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
