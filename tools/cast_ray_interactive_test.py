#!/usr/bin/env python3

"""
CARLA Interactive Cast Ray Test Program

This program provides an interactive interface for testing carla.world.cast_ray
using spectator position selection and single-key commands.

Commands:
  s - Set current spectator position as start point
  e - Set current spectator position as end point
  r - Cast ray from start to end and visualize results
  c - Clear all debug visualization
  h - Show help information
  q - Quit the program

The visualization elements persist until new commands are issued or cleared.
"""

import carla
import time
import sys
import argparse


class RayCastTester:
    def __init__(self, host='localhost', port=2000, map_name=None):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = None
        self.debug = None
        self.spectator = None
        self.map_name = map_name

        # Ray points
        self.start_point = None
        self.end_point = None

        # Visualization elements tracking
        self.active_visualizations = []

        self.connect_to_world()

    def connect_to_world(self):
        """Connect to CARLA world and initialize components"""
        try:
            if self.map_name:
                print(f"🗺️ Loading map: {self.map_name}")
                self.client.load_world(self.map_name)
                print(f"✓ Map '{self.map_name}' loaded successfully")
            else:
                print("🗺️ Using current CARLA world")

            self.world = self.client.get_world()
            self.debug = self.world.debug
            self.spectator = self.world.get_spectator()

            # Get current map name
            current_map = self.world.get_map()
            map_name = current_map.name if current_map else "Unknown"
            print(f"✓ Connected to CARLA world (Map: {map_name})")

        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            sys.exit(1)

    def get_current_spectator_location(self):
        """Get current spectator location"""
        transform = self.spectator.get_transform()
        return transform.location

    def clear_visualizations(self):
        """Clear all persistent debug visualizations"""
        # Clear by drawing empty points with very short lifetime
        for viz_data in self.active_visualizations:
            if viz_data['type'] == 'point':
                self.debug.draw_point(viz_data['location'], 0.01,
                                     carla.Color(0, 0, 0), 0.01)
            elif viz_data['type'] == 'line':
                self.debug.draw_line(viz_data['start'], viz_data['end'], 0.01,
                                    carla.Color(0, 0, 0), 0.01)
            elif viz_data['type'] == 'string':
                self.debug.draw_string(viz_data['location'], '', False,
                                      carla.Color(0, 0, 0), 0.01)

        self.active_visualizations = []
        print("✓ Cleared all visualizations")

    def add_persistent_point(self, location, size, color, label=""):
        """Add a persistent point visualization"""
        self.debug.draw_point(location, size, color, 0)  # life_time=0 for persistent
        self.active_visualizations.append({
            'type': 'point',
            'location': location,
            'size': size,
            'color': color
        })

        if label:
            self.add_persistent_string(location + carla.Location(z=1.0), label)

    def add_persistent_line(self, start, end, thickness, color):
        """Add a persistent line visualization"""
        self.debug.draw_line(start, end, thickness, color, 0)  # life_time=0 for persistent
        self.active_visualizations.append({
            'type': 'line',
            'start': start,
            'end': end,
            'thickness': thickness,
            'color': color
        })

    def add_persistent_string(self, location, text):
        """Add a persistent string visualization"""
        self.debug.draw_string(location, text, False, carla.Color(255, 255, 255), 0)
        self.active_visualizations.append({
            'type': 'string',
            'location': location,
            'text': text
        })

    def set_start_point(self):
        """Set current spectator position as start point"""
        self.start_point = self.get_current_spectator_location()
        print(f"✓ Start point set to: {self.start_point}")

        # Draw start point marker (white)
        self.add_persistent_point(self.start_point, 0.3, carla.Color(255, 255, 255), "START")

    def set_end_point(self):
        """Set current spectator position as end point"""
        self.end_point = self.get_current_spectator_location()
        print(f"✓ End point set to: {self.end_point}")

        # Draw end point marker (blue)
        self.add_persistent_point(self.end_point, 0.3, carla.Color(0, 0, 255), "END")

    def get_color_for_label(self, label):
        """Get color based on semantic label"""
        # Map common semantic labels to colors
        label_colors = {
            'Road': carla.Color(64, 64, 64, 255),          # Dark gray
            'Sidewalk': carla.Color(128, 128, 128, 255),   # Light gray
            'Building': carla.Color(139, 69, 19, 255),     # Brown
            'Vegetation': carla.Color(0, 128, 0, 255),     # Green
            'Vehicle': carla.Color(255, 0, 0, 255),        # Red
            'Pedestrian': carla.Color(255, 165, 0, 255),   # Orange
            'TrafficLight': carla.Color(255, 255, 0, 255), # Yellow
            'TrafficSign': carla.Color(0, 0, 255, 255),    # Blue
            'Sky': carla.Color(135, 206, 235, 255),        # Sky blue
            'Ground': carla.Color(101, 67, 33, 255),       # Ground brown
            'None': carla.Color(255, 255, 255, 255),       # White
        }

        # Return color if label matches, otherwise random color based on label
        if label in label_colors:
            return label_colors[label]
        else:
            # Generate a pseudo-random color based on label string
            hash_val = sum(ord(c) for c in str(label))
            return carla.Color(
                (hash_val * 137) % 256,
                (hash_val * 89) % 256,
                (hash_val * 43) % 256,
                255
            )

    def cast_ray(self):
        """Cast ray from start to end and visualize results"""
        if not self.start_point:
            print("✗ Start point not set! Use 's' to set start point first.")
            return

        if not self.end_point:
            print("✗ End point not set! Use 'e' to set end point first.")
            return

        print(f"\n🎯 Casting ray from {self.start_point} to {self.end_point}")

        try:
            # Cast the ray
            labelled_points = self.world.cast_ray(self.start_point, self.end_point)

            # Process and visualize results
            if labelled_points:
                print(f"✓ Found {len(labelled_points)} intersections:")

                # Draw segmented ray with different colors
                self.draw_segmented_ray(labelled_points)

                # Draw enhanced intersection points
                for i, point in enumerate(labelled_points):
                    color = self.get_color_for_label(point.label)

                    # Draw large, highly visible intersection point with glow effect
                    # Outer glow (larger, semi-transparent)
                    outer_color = carla.Color(
                        max(0, min(255, color.r // 2)),
                        max(0, min(255, color.g // 2)),
                        max(0, min(255, color.b // 2)),
                        128
                    )
                    self.debug.draw_point(point.location, 0.5, outer_color, 0)

                    # Middle layer (medium size, semi-transparent)
                    middle_color = carla.Color(
                        max(0, min(255, int(color.r * 0.7))),
                        max(0, min(255, int(color.g * 0.7))),
                        max(0, min(255, int(color.b * 0.7))),
                        180
                    )
                    self.debug.draw_point(point.location, 0.35, middle_color, 0)

                    # Core point (smaller, solid color) - track for cleanup
                    self.add_persistent_point(point.location, 0.2, color)

                    # Add detailed information in CARLA 3D world
                    info_lines = [
                        f"#{i+1}: {point.label}",
                        f"({point.location.x:.2f}, {point.location.y:.2f}, {point.location.z:.2f})"
                    ]

                    # Draw multi-line text in 3D world with shadow for better visibility
                    text_offset = 1.5
                    for j, line in enumerate(info_lines):
                        text_pos = point.location + carla.Location(x=0.5, y=0, z=text_offset + j*0.8)

                        # Draw shadow text for better readability
                        shadow_pos = text_pos + carla.Location(x=0.1, y=0.1, z=0)
                        self.debug.draw_string(shadow_pos, line, True,
                                             carla.Color(0, 0, 0), 0)  # Black shadow

                        # Draw main text
                        self.add_persistent_string(text_pos, line)

                    # Draw connecting line from point to text
                    text_start = point.location + carla.Location(x=0.25, y=0, z=0.3)
                    text_end = point.location + carla.Location(x=0.5, y=0, z=text_offset)
                    self.debug.draw_line(text_start, text_end, 0.05, color, 0)

                    print(f"  {i+1}: Location={point.location}, Label='{point.label}'")
            else:
                print("  No intersections detected")
                # Draw single ray line when no intersections
                self.add_persistent_line(self.start_point, self.end_point, 0.15, carla.Color(0, 255, 0))

        except Exception as e:
            print(f"✗ Error casting ray: {e}")

    def draw_segmented_ray(self, labelled_points):
        """Draw ray with segments colored by intersection labels"""
        current_point = self.start_point

        for labelled_point in labelled_points:
            # Get color for this segment based on the label
            segment_color = self.get_color_for_label(labelled_point.label)

            # Draw segment from current point to intersection point
            self.add_persistent_line(current_point, labelled_point.location, 0.2, segment_color)

            # Update current point for next segment
            current_point = labelled_point.location

        # Draw final segment from last intersection to end point
        if current_point:
            self.add_persistent_line(current_point, self.end_point, 0.2, carla.Color(255, 255, 255))

    def show_status(self):
        """Display current status"""
        current_loc = self.get_current_spectator_location()
        print(f"\n📍 Current spectator location: {current_loc}")

        if self.start_point:
            print(f"🎯 Start point: {self.start_point}")
        else:
            print("🎯 Start point: Not set")

        if self.end_point:
            print(f"🏁 End point: {self.end_point}")
        else:
            print("🏁 End point: Not set")

    def show_help(self):
        """Show help information"""
        print("\n📖 Commands:")
        print("  s - Set current spectator position as start point")
        print("  e - Set current spectator position as end point")
        print("  r - Cast ray from start to end and visualize results")
        print("  c - Clear all debug visualization")
        print("  h - Show this help information")
        print("  q - Quit the program")
        print("\n💡 Usage:")
        print("  1. Move spectator in CARLA to desired start position")
        print("  2. Press 's' to set start point (white marker)")
        print("  3. Move spectator to desired end position")
        print("  4. Press 'e' to set end point (blue marker)")
        print("  5. Press 'r' to cast ray and see results")
        print("  6. Use 'c' to clear visualizations when needed")
        print("\n🎨 Visualization Features:")
        print("  • Segmented rays colored by semantic labels")
        print("  • Enhanced intersection points with glow effects")
        print("  • 3D world text showing coordinates and labels")
        print("  • Color-coded semantic objects:")
        print("    - Road: Gray  • Building: Brown  • Vegetation: Green")
        print("    - Vehicle: Red  • Sky: Blue  • Ground: Dark Brown")

    def run(self):
        """Main program loop"""
        print("🚗 CARLA Interactive Cast Ray Test Program")
        print("=" * 50)

        self.show_help()
        print("\n" + "=" * 50)

        try:
            while True:
                self.show_status()
                print("\nEnter command (h for help): ", end="")

                try:
                    command = input().lower().strip()
                except EOFError:
                    command = 'q'

                if command == 's':
                    self.set_start_point()
                elif command == 'e':
                    self.set_end_point()
                elif command == 'r':
                    self.cast_ray()
                elif command == 'c':
                    self.clear_visualizations()
                elif command == 'h':
                    self.show_help()
                elif command == 'q':
                    print("\n👋 Goodbye!")
                    break
                elif command == '':
                    continue  # Empty input, just show status again
                else:
                    print(f"✗ Unknown command: '{command}'. Type 'h' for help.")

                print("\n" + "-" * 30)

        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")

        # Clean up visualizations before exit
        self.clear_visualizations()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CARLA Interactive Cast Ray Test Program',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use current CARLA world
  %(prog)s -m Town01                 # Load Town01 map
  %(prog)s --map Town02              # Load Town02 map
  %(prog)s --host 192.168.1.100      # Connect to remote CARLA server

Available CARLA maps:
  Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10, Town10HD
        """)

    parser.add_argument('-H', '--host', default='localhost',
                        help='CARLA server host (default: localhost)')
    parser.add_argument('-p', '--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('-m', '--map', dest='map_name',
                        help='CARLA map to load (e.g., Town01, Town02, etc.)')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    tester = RayCastTester(host=args.host, port=args.port, map_name=args.map_name)
    tester.run()


if __name__ == '__main__':
    main()