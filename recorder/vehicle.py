#!/usr/bin/python3
import copy
import csv
import os
import logging
import carla

from recorder.actor import Actor
from recorder.agents.navigation.behavior_agent import BasicAgent
from recorder.agents.navigation.behavior_agent import BehaviorAgent

# Get logger instance
logger = logging.getLogger(__name__)


class OtherVehicle(Actor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 carla_actor: carla.Vehicle):
        super().__init__(uid=uid, name=name, parent=None, carla_actor=carla_actor)
        self.vehicle_type = copy.deepcopy(carla_actor.type_id)
        self.save_dir = '{}/{}_{}'.format(base_save_dir, self.vehicle_type, self.get_uid())
        self.first_tick = True
        # For vehicle control
        self.auto_pilot = True
        self.vehicle_agent = None

    def get_type_id(self):
        return 'others.other_vehicle'

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Other vehicle not saving data

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Timestamp
            debug: Debug flag
        """
        return

    def get_save_dir(self):
        return self.save_dir

    def control_step(self):
        """
        Control step for other vehicles.
        Autopilot已在batch spawn时通过SetAutopilot命令设置，
        Traffic Manager自动控制，无需手动操作。
        """
        pass


class Vehicle(Actor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 carla_actor: carla.Vehicle,
                 route_config=None):
        super().__init__(uid=uid, name=name, parent=None, carla_actor=carla_actor)
        self.vehicle_type = copy.deepcopy(carla_actor.type_id)
        self.save_dir = '{}/{}'.format(base_save_dir, self.name)
        self.first_tick = True

        # Route configuration
        self.route_config = route_config

        # For vehicle control
        if self.route_config is None:
            # No route specified, use default autopilot
            self.use_auto_pilot = True
            self.vehicle_agent = BasicAgent(self.carla_actor)
        else:
            # Route specified, setup agent for route following
            self.use_auto_pilot = False
            self._setup_route()

        # self.vehicle_agent = BehaviorAgent(self.carla_actor)

    def _setup_route(self):
        """Setup vehicle agent to follow configured route"""
        if self.route_config is None:
            return

        mode = self.route_config.get('mode', 'strict')
        waypoints = self.route_config.get('waypoints', [])
        is_loop = self.route_config.get('loop', False)  # Check if it's a loop route

        if mode == 'strict' and len(waypoints) >= 2:
            # Create BasicAgent for waypoint following
            self.vehicle_agent = BasicAgent(self.carla_actor)

            # Convert waypoint dicts to carla.Location objects
            world = self.carla_actor.get_world()
            carla_map = world.get_map()

            # Build complete route through all waypoints using GlobalRoutePlanner
            from recorder.agents.navigation.global_route_planner import GlobalRoutePlanner

            # Create route planner with 2.0m sampling resolution
            grp = GlobalRoutePlanner(carla_map, 2.0)

            # Build complete plan by connecting waypoints with road topology
            complete_plan = []

            # Determine number of segments
            num_segments = len(waypoints) - 1
            if is_loop:
                num_segments = len(waypoints)  # Include segment from last back to first

            for i in range(num_segments):
                # Current waypoint
                start_loc = carla.Location(
                    x=float(waypoints[i]['x']),
                    y=float(waypoints[i]['y']),
                    z=float(waypoints[i]['z'])
                )

                # Next waypoint (wrap around for loop routes)
                next_idx = (i + 1) % len(waypoints)
                end_loc = carla.Location(
                    x=float(waypoints[next_idx]['x']),
                    y=float(waypoints[next_idx]['y']),
                    z=float(waypoints[next_idx]['z'])
                )

                # Use GlobalRoutePlanner to find the route between waypoints
                # This respects road topology, lanes, and intersections
                segment_route = grp.trace_route(start_loc, end_loc)

                # Add segment to complete plan
                if i == 0:
                    # First segment: add all waypoints
                    complete_plan.extend(segment_route)
                else:
                    # Subsequent segments: skip first waypoint to avoid duplicates
                    complete_plan.extend(segment_route[1:])

            # Set the global plan for the agent
            if len(complete_plan) >= 2:
                self.vehicle_agent.set_global_plan(complete_plan, clean_queue=True)
                loop_info = " (LOOP)" if is_loop else ""
                logger.info(
                    f"Vehicle '{self.name}' configured with route: {len(waypoints)} waypoints "
                    f"expanded to {len(complete_plan)} road waypoints following topology{loop_info}"
                )
            else:
                logger.warning(f"Vehicle '{self.name}' route planning failed. Using autopilot.")
                self.use_auto_pilot = True

        elif mode == 'disabled':
            # Explicitly disabled route, use autopilot
            self.use_auto_pilot = True
            self.vehicle_agent = BasicAgent(self.carla_actor)
        else:
            logger.warning(f"Unknown route mode '{mode}' for vehicle '{self.name}'. Using autopilot.")
            self.use_auto_pilot = True
            self.vehicle_agent = BasicAgent(self.carla_actor)

    def get_save_dir(self):
        return self.save_dir

    def get_carla_bbox(self):
        return self.carla_actor.bounding_box

    def get_carla_transform(self):
        return self.carla_actor.get_transform()

    def get_control(self):
        """
        Get vehicle control command.
        :return: vehicle control command.
        """
        return self.carla_actor.get_control()

    @staticmethod
    def vehicle_control_to_dict(vehicle_control: carla.VehicleControl) -> dict:
        return {'throttle': vehicle_control.throttle,
                'brake': vehicle_control.brake,
                'steer': vehicle_control.steer,
                'reverse': vehicle_control.reverse,
                'gear': vehicle_control.gear}

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save vehicle state to disk

        Args:
            frame_id: Absolute CARLA frame ID (for file naming)
            timestamp: Timestamp
            debug: Debug flag

        Returns:
            dict: Vehicle state information
        """
        os.makedirs(self.save_dir, exist_ok=True)
        fieldnames = ['frame',
                      'timestamp',
                      'x', 'y', 'z',
                      'roll', 'pitch', 'yaw',
                      'speed',
                      'vx', 'vy', 'vz',
                      'ax', 'ay', 'az',
                      'throttle', 'brake',
                      'steer', 'reverse', 'gear']

        if self.first_tick:
            self.save_vehicle_info()
            with open('{}/vehicle_status.csv'.format(self.save_dir), 'w', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if self.first_tick:
                    writer.writeheader()
                    self.first_tick = False

        # Save vehicle status to csv file
        try:
            csv_line = {'frame': frame_id,
                        'timestamp': timestamp,
                        'speed': self.get_speed()}
            csv_line.update(self.get_acceleration().to_dict(prefix='a'))
            csv_line.update(self.get_velocity().to_dict(prefix='v'))
            csv_line.update(self.get_transform().to_dict())
            csv_line.update(self.vehicle_control_to_dict(self.get_control()))

            with open('{}/vehicle_status.csv'.format(self.save_dir), 'a', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(csv_line)

            if debug:
                logger.debug(f"Vehicle status recorded: uid={self.uid} name={self.name}")

            # Prepare return info
            return {
                'type': 'vehicle',
                'name': self.name,
                'vehicle_state': csv_line
            }

        except IOError as e:
            logger.error(f"Failed to write vehicle status to {self.save_dir}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error saving vehicle {self.uid} status: {e}")
            raise

    def save_vehicle_info(self):
        # TODO: Save vehicle physics info here
        pass

    def control_step(self):
        """Execute one control step for the vehicle"""
        if self.route_config is not None and not self.use_auto_pilot:
            # Using agent-based route following
            control = self.vehicle_agent.run_step()
            self.carla_actor.apply_control(control)
        else:
            pass