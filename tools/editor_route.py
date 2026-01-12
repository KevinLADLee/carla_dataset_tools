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
from collections import deque

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
        self.grp_resolution = 2.0
        self.grp = GlobalRoutePlanner(self.map, self.grp_resolution)
        print("Route planner ready!")

        # Route data
        self.waypoints = []  # List of {x, y, z} dicts
        self.waypoint_circles = []  # Visual elements
        self.waypoint_labels = []  # Visual elements (order labels)
        self.waypoint_arrows = []  # Visual elements (direction arrows)
        self.route_lines = []  # Visual elements (reference lines)
        self.topology_lines = []  # Topology-aware route visualization
        self.selected_waypoint = None
        self.is_loop = False  # Whether this is a closed loop route
        self.auto_route_step = 5.0
        self.auto_graph_resolution = 2.0
        self.auto_label_stride = 25
        self.waypoint_marker_size = 0.8
        self.waypoint_marker_linewidth = 1.0

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
            "- G: Auto-generate rightmost-lane traversal from last waypoint\n"
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
            self._add_waypoint_visual(wp_dict)

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
            try:
                self.status_text_obj.remove()
            except ValueError:
                pass

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
        elif event.key == 'g':
            if len(self.waypoints) == 0:
                print("Click to add a start waypoint, then press G to auto-generate.")
                return
            start_wp = self.waypoints[-1]
            start_loc = carla.Location(
                x=float(start_wp['x']),
                y=float(start_wp['y']),
                z=float(start_wp['z'])
            )
            start_waypoint = self.map.get_waypoint(start_loc, project_to_road=True)
            if start_waypoint is None:
                print("Unable to resolve start waypoint on road.")
                return
            self._generate_rightmost_route(start_waypoint)

    def _clear_route(self):
        """Clear all waypoints and visuals."""
        for circle in self.waypoint_circles:
            circle.remove()
        self.waypoint_circles.clear()
        for label in self.waypoint_labels:
            label.remove()
        self.waypoint_labels.clear()
        for arrows in self.waypoint_arrows:
            try:
                arrows.remove()
            except ValueError:
                pass
        self.waypoint_arrows.clear()
        self.waypoints.clear()

        for line in self.route_lines:
            line.remove()
        self.route_lines.clear()

        for line in self.topology_lines:
            line.remove()
        self.topology_lines.clear()

        if hasattr(self, 'status_text_obj'):
            try:
                self.status_text_obj.remove()
            except ValueError:
                pass
            self.status_text_obj = None

    def _add_waypoint_visual(self, wp_dict, color='red', label=None, draw_circle=True):
        """Add waypoint marker and store waypoint data."""
        self.waypoints.append(wp_dict)
        if draw_circle:
            circle = Circle((wp_dict['x'], -wp_dict['y']), self.waypoint_marker_size,
                            color=color, linewidth=self.waypoint_marker_linewidth, fill=True,
                            alpha=0.7, zorder=10, picker=True)
            self.ax.add_patch(circle)
            self.waypoint_circles.append(circle)
        if label is not None:
            text = self.ax.text(
                wp_dict['x'],
                -wp_dict['y'],
                str(label),
                fontsize=6,
                color='black',
                ha='center',
                va='center',
                zorder=11
            )
            self.waypoint_labels.append(text)

    def _add_waypoint_arrows(self, locations):
        """Add direction arrows for a sequence of locations."""
        if len(locations) < 2:
            return
        xs = []
        ys = []
        us = []
        vs = []
        for start, end in zip(locations[:-1], locations[1:]):
            dx = end.x - start.x
            dy = end.y - start.y
            length = np.hypot(dx, dy)
            if length < 0.5:
                continue
            xs.append(start.x)
            ys.append(-start.y)
            us.append(dx)
            vs.append(-dy)

        if not xs:
            return

        arrows = self.ax.quiver(
            xs, ys, us, vs,
            angles='xy', scale_units='xy', scale=1.0,
            width=0.0025, color='red', alpha=0.8, zorder=9
        )
        self.waypoint_arrows.append(arrows)

    def _get_rightmost_waypoint(self, waypoint):
        """Get the rightmost reachable lane waypoint respecting lane change rules."""
        if waypoint.lane_type != carla.LaneType.Driving:
            return None
        current = waypoint
        while True:
            right = current.get_right_lane()
            if right is None or right.lane_type != carla.LaneType.Driving:
                return current
            current = right

    def _get_lane_end(self, waypoint):
        """Get the last waypoint before lane end."""
        segment = waypoint.next_until_lane_end(self.auto_graph_resolution)
        if segment:
            return segment[-1]
        return waypoint

    def _build_rightmost_lane_graph(self, extra_start=None):
        """Build a directed graph of rightmost lanes using topology."""
        topology = self.map.get_topology()
        nodes = {}
        adjacency = {}
        queue = deque()

        for w0, _ in topology:
            rm = self._get_rightmost_waypoint(w0)
            if rm is None:
                continue
            if rm.id not in nodes:
                nodes[rm.id] = rm
                adjacency[rm.id] = set()
                queue.append(rm.id)

        if extra_start is not None:
            rm = self._get_rightmost_waypoint(extra_start)
            if rm is not None and rm.id not in nodes:
                nodes[rm.id] = rm
                adjacency[rm.id] = set()
                queue.append(rm.id)

        processed = set()
        while queue:
            node_id = queue.popleft()
            if node_id in processed:
                continue
            processed.add(node_id)

            node_wp = nodes[node_id]
            end_wp = self._get_lane_end(node_wp)
            next_wps = end_wp.next(self.auto_graph_resolution)

            for next_wp in next_wps:
                if next_wp.lane_type != carla.LaneType.Driving:
                    continue
                rm_next = self._get_rightmost_waypoint(next_wp)
                if rm_next is None:
                    continue
                if rm_next.id not in nodes:
                    nodes[rm_next.id] = rm_next
                    adjacency[rm_next.id] = set()
                    queue.append(rm_next.id)
                adjacency[node_id].add(rm_next.id)

        return nodes, adjacency

    def _find_path_to_unvisited(self, start_id, adjacency, unvisited_edges):
        """Find shortest directed path to any node with unvisited outgoing edges."""
        targets = set(edge[0] for edge in unvisited_edges)
        if not targets:
            return None

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            node_id, path = queue.popleft()
            if node_id in targets and node_id != start_id:
                return path
            for neighbor in adjacency.get(node_id, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return None

    def _append_location(self, route_locations, location, threshold=0.5):
        """Append location if far enough from the last entry."""
        if not route_locations:
            route_locations.append(location)
            return
        if route_locations[-1].distance(location) > threshold:
            route_locations.append(location)

    def _append_route_segment(self, route_locations, start_wp, end_wp, max_points=3):
        """Append a route segment using GlobalRoutePlanner sampling."""
        start_loc = start_wp.transform.location
        end_loc = end_wp.transform.location
        try:
            segment_route = self.grp.trace_route(start_loc, end_loc)
        except Exception as e:
            print(f"  Error computing segment: {e}")
            segment_route = []

        if not segment_route:
            self._append_location(route_locations, start_loc)
            self._append_location(route_locations, end_loc)
            return

        route_len = len(segment_route)
        if max_points <= 2 or route_len <= 2:
            self._append_location(route_locations, segment_route[0][0].transform.location)
            self._append_location(route_locations, segment_route[-1][0].transform.location)
            return

        mid_idx = route_len // 2
        indices = [0, mid_idx, route_len - 1]
        for idx in indices:
            wp = segment_route[idx][0]
            self._append_location(route_locations, wp.transform.location)

    def _generate_rightmost_route(self, start_waypoint):
        """Generate a continuous route covering rightmost lanes with minimal repeats."""
        print("Generating rightmost-lane traversal route...")
        nodes, adjacency = self._build_rightmost_lane_graph(extra_start=start_waypoint)
        start_rm = self._get_rightmost_waypoint(start_waypoint)

        if start_rm is None:
            print("Start waypoint is not on a driving lane. Route not generated.")
            return

        unvisited_edges = set()
        for node_id, next_ids in adjacency.items():
            for next_id in next_ids:
                unvisited_edges.add((node_id, next_id))

        if not unvisited_edges:
            print("No rightmost lane edges found. Route not generated.")
            return

        current_id = start_rm.id
        if current_id not in nodes:
            nodes[current_id] = start_rm
            adjacency[current_id] = set()

        route_locations = []
        self._append_location(route_locations, start_rm.transform.location)

        safety_limit = len(unvisited_edges) * 10 + 50
        steps = 0

        while unvisited_edges and steps < safety_limit:
            steps += 1
            outgoing_unvisited = [
                next_id for next_id in adjacency.get(current_id, [])
                if (current_id, next_id) in unvisited_edges
            ]

            if outgoing_unvisited:
                next_id = sorted(outgoing_unvisited)[0]
                self._append_route_segment(route_locations, nodes[current_id], nodes[next_id])
                unvisited_edges.discard((current_id, next_id))
                current_id = next_id
                continue

            path = self._find_path_to_unvisited(current_id, adjacency, unvisited_edges)
            if not path:
                break

            for prev_id, next_id in zip(path[:-1], path[1:]):
                self._append_route_segment(route_locations, nodes[prev_id], nodes[next_id])
                unvisited_edges.discard((prev_id, next_id))
            current_id = path[-1]

        if unvisited_edges:
            print(f"Warning: {len(unvisited_edges)} rightmost-lane edges are unreachable from start.")

        if len(route_locations) < 2:
            print("Generated route is too short. Route not generated.")
            return

        self._clear_route()
        total = len(route_locations)
        for idx, loc in enumerate(route_locations):
            wp_dict = {'x': loc.x, 'y': loc.y, 'z': loc.z}
            label = idx if idx % self.auto_label_stride == 0 else None
            self._add_waypoint_visual(wp_dict, color='red', label=label, draw_circle=False)

        self._add_waypoint_arrows(route_locations)

        self._update_route_lines()
        self.fig.canvas.draw_idle()

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
