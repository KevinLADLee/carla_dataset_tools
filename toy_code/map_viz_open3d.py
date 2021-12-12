import carla
import open3d as o3d

LINE_COLOR = [0.5, 0.5, 0.5]


class CarlaMapVisualization:
    """
    Pseudo opendrive sensor
    """

    def __init__(self):
        self.world = None
        self.connect_to_carla()
        self.map = self.world.get_map()
        self.map_name = self.map.name
        print("Map Visualization Node: Loading {} map!".format(self.map_name))
        self.id = 0
        self.o3d_markers = []
        self.vis = o3d.visualization.Visualizer()

    def connect_to_carla(self):
        host = '127.0.0.1'
        port = 2000
        timeout = 10
        print("Map Visualization Node: CARLA world available. "
              "Trying to connect to {host}:{port}".format(host=host, port=port))
        carla_client = carla.Client(host=host, port=port)
        carla_client.set_timeout(timeout)
        try:
            self.world = carla_client.get_world()
        except RuntimeError as e:
            print("Error while connecting to Carla: {}".format(e))
            raise e
        print("Connected to Carla.")

    def visualize_in_o3d(self):
        self.draw_map()
        self.vis.create_window("CarlaMapViz")
        for m in self.o3d_markers:
            self.vis.add_geometry(m)
        vis_ctl = self.vis.get_view_control()
        vis_ctl.translate(x=0, y=0)
        vis_ctl.set_zoom(50)
        self.vis.run()
        self.vis.destroy_window()
        # o3d.visualization.draw_geometries(self.o3d_markers)
        #web viz
        # o3d.visualization.webrtc_server.enable_webrtc()
        # o3d.visualization.draw(self.o3d_markers)

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def set_marker_id(self):
        self.id += 1
        return self.id - 1

    def points_to_lineset(self, points_set: list):
        size = len(points_set)
        if size <= 1:
            return None
        line = []
        for i in range(size-1):
            line.append([i, i+1])
        return line

    def add_line_strip_marker(self, points=None):
        o3d_points = []
        if points is not None:
            for p in points:
                o3d_point = [p.x, -p.y, p.z]
                o3d_points.append(o3d_point)
        lines = self.points_to_lineset(o3d_points)
        colors = [LINE_COLOR for i in range(len(lines))]
        o3d_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(o3d_points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        o3d_lineset.colors = o3d.utility.Vector3dVector(colors)
        self.o3d_markers.append(o3d_lineset)
        return

    def draw_map(self):
        precision = 0.1
        topology = self.map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
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
            set_waypoints.append(waypoints)

        for waypoints in set_waypoints:
            waypoint = waypoints[0]
            road_left_side = [self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            road_right_side = [self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            # road_points = road_left_side + [x for x in reversed(road_right_side)]
            # self.add_line_strip_marker(points=road_points)

            if len(road_left_side) > 2:
                self.add_line_strip_marker(points=road_left_side)
            if len(road_right_side) > 2:
                self.add_line_strip_marker(points=road_right_side)

            # if not waypoint.is_junction:
            #     for n, wp in enumerate(waypoints):
            #         if ((n + 1) % 400) == 0:
            #             self.add_arrow_line_marker(wp.transform)


def main(args=None):
    """
    main function
    """

    carla_map_visualization = None
    try:
        carla_visualization = CarlaMapVisualization()
        carla_visualization.visualize_in_o3d()

    except KeyboardInterrupt:
        print("User requested shut down.")
    finally:
        print("Shutting down.")


if __name__ == "__main__":
    main()
