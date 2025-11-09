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
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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

        # Initialize GlobalRoutePlanner for topology-aware path display
        from recorder.agents.navigation.global_route_planner import GlobalRoutePlanner
        print("Initializing route planner...")
        self.grp = GlobalRoutePlanner(self.map, 2.0)
        print("Route planner ready!")

        # Route data
        self.waypoints = []  # List of {x, y, z} dicts
        self.waypoint_circles = []  # Visual elements
        self.route_lines = []  # Visual elements (reference lines)
        self.topology_lines = []  # Topology-aware route visualization
        self.selected_waypoint = None
        self.is_loop = False  # Whether this is a closed loop route

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(14, 12), dpi=150)
        self.fig.canvas.manager.set_window_title(f'Route Editor - {self.map_name}')

        # Add grid for better positioning
        self.ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # Draw map
        self._draw_roads()
        self._draw_spawn_points()

        # Setup event handlers (no motion handler for dragging)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Instructions
        self._show_instructions()

    def _show_instructions(self):
        """Display usage instructions"""
        instructions = (
            "Route Editor Controls:\n"
            "- Left Click: Add waypoint (allows repeats)\n"
            "- Right Click on circle: Delete waypoint\n"
            "- Ctrl+Z: Undo last waypoint\n"
            "- Enter: Save and exit\n"
            "- Escape: Cancel and exit\n"
            "\n"
            "Green: Road topology path\n"
            "Blue dashed: Direct waypoint links\n"
            "Loop auto-detected if first & last are close"
        )
        self.ax.text(0.02, 0.98, instructions,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

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
        """Add a new waypoint at the clicked position (allows duplicates)"""
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
            # Always allow adding waypoint (no duplicate check)
            self.waypoints.append(wp_dict)

            # Draw waypoint circle (larger and more visible)
            circle = Circle((wp_dict['x'], -wp_dict['y']), 3.0,
                          color='red', linewidth=2, fill=True,
                          alpha=0.7, zorder=10, picker=True)
            self.ax.add_patch(circle)
            self.waypoint_circles.append(circle)

            # Update route lines (will auto-detect loop)
            self._update_route_lines()
            self.fig.canvas.draw_idle()

            print(f"Added waypoint {len(self.waypoints)}: ({wp_dict['x']:.2f}, {wp_dict['y']:.2f}, {wp_dict['z']:.2f})")

            # Check if this creates a loop
            if self._is_loop_route():
                print("  → Loop detected: First and last waypoints are close!")

    def _remove_waypoint(self, index):
        """Remove waypoint at given index"""
        if 0 <= index < len(self.waypoints):
            removed = self.waypoints.pop(index)
            circle = self.waypoint_circles.pop(index)
            circle.remove()

            self._update_route_lines()
            self.fig.canvas.draw_idle()

            print(f"Removed waypoint: ({removed['x']:.2f}, {removed['y']:.2f}, {removed['z']:.2f})")

    def _is_loop_route(self, tolerance=5.0):
        """Check if first and last waypoints are close enough to form a loop"""
        if len(self.waypoints) < 3:  # Need at least 3 points for a meaningful loop
            return False

        first = self.waypoints[0]
        last = self.waypoints[-1]

        # Calculate distance between first and last waypoint
        dist = np.sqrt(
            (first['x'] - last['x'])**2 +
            (first['y'] - last['y'])**2 +
            (first['z'] - last['z'])**2
        )

        return dist <= tolerance

    def _update_route_lines(self):
        """Update the route line visualization with topology-aware paths"""
        # Remove old lines
        for line in self.route_lines:
            line.remove()
        self.route_lines.clear()

        for line in self.topology_lines:
            line.remove()
        self.topology_lines.clear()

        if len(self.waypoints) < 2:
            return

        # Auto-detect if this is a loop route
        self.is_loop = self._is_loop_route()

        # Update window title to show loop status
        title = f'Route Editor - {self.map_name}'
        if self.is_loop:
            title += ' [LOOP DETECTED]'
        self.fig.canvas.manager.set_window_title(title)

        # Determine how many segments to draw
        num_segments = len(self.waypoints) - 1
        if self.is_loop:
            num_segments = len(self.waypoints)  # Include segment from last to first

        # Draw topology-aware route segments (green, thick)
        print(f"Computing road topology for {num_segments} segments...")

        for i in range(num_segments):
            start_wp = self.waypoints[i]
            end_wp = self.waypoints[(i + 1) % len(self.waypoints)]  # Wrap around for loop

            start_loc = carla.Location(
                x=float(start_wp['x']),
                y=float(start_wp['y']),
                z=float(start_wp['z'])
            )
            end_loc = carla.Location(
                x=float(end_wp['x']),
                y=float(end_wp['y']),
                z=float(end_wp['z'])
            )

            try:
                # Calculate road path using GlobalRoutePlanner
                segment_route = self.grp.trace_route(start_loc, end_loc)

                if segment_route:
                    # Extract waypoint positions
                    path_x = [wp[0].transform.location.x for wp in segment_route]
                    path_y = [-wp[0].transform.location.y for wp in segment_route]

                    # Draw topology path (brighter green, thicker)
                    line, = self.ax.plot(path_x, path_y, 'lime',
                                        linewidth=3, alpha=0.9, zorder=6,
                                        label='Road Path' if i == 0 else '')
                    self.topology_lines.append(line)

                    print(f"  Segment {i+1}/{num_segments}: {len(segment_route)} waypoints")
                else:
                    print(f"  Warning: No path found for segment {i+1}")
            except Exception as e:
                print(f"  Error computing segment {i+1}: {e}")

        # Draw reference lines connecting waypoints directly (blue dashed, thin)
        waypoint_indices = list(range(len(self.waypoints)))
        if self.is_loop:
            waypoint_indices.append(0)  # Close the loop

        x = [self.waypoints[i]['x'] for i in waypoint_indices]
        y = [-self.waypoints[i]['y'] for i in waypoint_indices]
        line, = self.ax.plot(x, y, 'b--', linewidth=1.5, alpha=0.5, zorder=5,
                            marker='o', markersize=5,
                            label='Waypoint Links' if not self.route_lines else '')
        self.route_lines.append(line)

        # Update legend and status
        if len(self.topology_lines) > 0:
            handles, labels = self.ax.get_legend_handles_labels()
            # Remove duplicate labels
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(),
                          loc='upper right', fontsize=10)

        # Add status text showing waypoint count
        status_text = f"Waypoints: {len(self.waypoints)}"
        if self.is_loop:
            status_text += " (LOOP)"

        # Remove old status text if exists
        if hasattr(self, 'status_text_obj'):
            self.status_text_obj.remove()

        self.status_text_obj = self.ax.text(0.98, 0.02, status_text,
                                            transform=self.ax.transAxes,
                                            fontsize=11,
                                            weight='bold',
                                            ha='right',
                                            va='bottom',
                                            bbox=dict(boxstyle='round',
                                                     facecolor='lightblue',
                                                     alpha=0.8))

    def _on_click(self, event):
        """Handle mouse button press - simplified, no dragging"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click - always add new waypoint
            self._add_waypoint(event.xdata, event.ydata)

        elif event.button == 3:  # Right click - delete waypoint if clicking on circle
            # Check if clicking on a waypoint circle
            for i, circle in enumerate(self.waypoint_circles):
                contains, _ = circle.contains(event)
                if contains:
                    self._remove_waypoint(i)
                    break

    def _on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'enter':
            self._save_and_exit()
        elif event.key == 'escape':
            print("Cancelled. No route saved.")
            plt.close(self.fig)
        elif event.key == 'ctrl+z' or event.key == 'cmd+z':
            # Undo last waypoint
            if len(self.waypoints) > 0:
                self._remove_waypoint(len(self.waypoints) - 1)
                print("Undo: Removed last waypoint")

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
            'loop': self.is_loop,  # Add loop flag
            'metadata': {
                'waypoint_count': len(self.waypoints),
                'is_loop': self.is_loop,
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
        if self.is_loop:
            print(f"  loop: true  # Closed loop route")
        print(f"  waypoints:")
        for wp in self.waypoints:
            print(f"    - {{x: {wp['x']:.2f}, y: {wp['y']:.2f}, z: {wp['z']:.2f}}}")
        print("\n# Or load from file:")
        print(f"# route:")
        print(f"#   from_file: {yaml_path}")
        print("="*60)

        if self.is_loop:
            print("\nNote: This is a LOOP route - vehicle will return to start and repeat.")

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
