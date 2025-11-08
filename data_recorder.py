import argparse
import os
import signal
import time

import carla

from param import *
from config.config_manager import ConfigManager, ConfigValidationError
from recorder.actor_tree import ActorTree
from utils.transform import Transform, Location, Rotation
from utils.transform import transform_to_carla_transform

sig_interrupt = False


def signal_handler(signal, frame):
    global sig_interrupt
    sig_interrupt = True


class DataRecorder:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.carla_client = carla.Client(self.host, self.port)
        self.carla_client.set_timeout(10.0)
        self.world = self._get_world()
        self.tm = self.carla_client.get_trafficmanager()
        self.debug_helper = self.world.debug
        self.record_name = None
        self.base_save_dir = None
        self.config = None
        self.actor_tree = ActorTree(self.world)
        self.frame_total = -1
        self.frame_step = 1

    def _get_world(self) -> carla.World:
        return self.carla_client.get_world()

    def destroy(self):
        self.actor_tree.destroy()

    def set_traffic_light_time(self, traffic_light_setting):
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                actor.set_red_time(traffic_light_setting["red_time"])
                actor.set_green_time(traffic_light_setting["green_time"])
                actor.set_yellow_time(traffic_light_setting["yellow_time"])

    def setting_world_and_actors(self, config):
        """
        Configure world and actors from unified configuration

        Args:
            config: Configuration dictionary from ConfigManager
        """
        # Load map
        self.carla_client.load_world(config['recording']['map'])

        # Configure world settings
        settings = self.world.get_settings()
        settings.synchronous_mode = config['world_settings']['synchronous_mode']
        settings.fixed_delta_seconds = config['world_settings']['fixed_delta_seconds']
        settings.substepping = config['world_settings']['substepping']
        settings.max_substep_delta_time = config['world_settings']['max_substep_delta_time']
        settings.max_substeps = config['world_settings']['max_substeps']

        print("World settings:", settings)
        self.world.apply_settings(settings)

        # Set spectator position if specified
        if 'spectator' in config and config['spectator'] is not None:
            pose = config['spectator']
            spectator = self.world.get_spectator()
            spectator_transform = Transform(
                Location(pose['x'], pose['y'], pose['z']),
                Rotation(roll=pose.get('roll', 0.0),
                        pitch=pose.get('pitch', 0.0),
                        yaw=pose.get('yaw', 0.0))
            )
            spectator.set_transform(transform_to_carla_transform(spectator_transform))

        # Set synchronous mode for traffic manager
        print("Set synchronous mode now...")
        self.tm.set_synchronous_mode(True)

        # Set recording parameters
        self.frame_total = config['recording']['frame_total']
        self.frame_step = config['recording']['frame_step']

        # Set traffic light timings
        traffic_light_settings = config.get('traffic_lights', {})
        self.set_traffic_light_time(traffic_light_settings)

        # Create save directory
        self.record_name = time.strftime("%Y_%m%d_%H%M", time.localtime())
        self.base_save_dir = "{}/record_{}".format(RAW_DATA_PATH, self.record_name)

        # Initialize actor tree with configuration
        self.actor_tree = ActorTree(self.world, config, self.base_save_dir)
        self.actor_tree.init()

        # Store config
        self.config = config

    def start_record(self, config):
        """
        Start recording with the given configuration

        Args:
            config: Configuration dictionary
        """
        self.setting_world_and_actors(config)
        print("Recording to folder: {}".format(self.base_save_dir))
        os.makedirs(self.base_save_dir, exist_ok=True)
        carla_logfile = "{}/carla_raw_record.log".format(self.base_save_dir)
        print("Start recording to {}".format(carla_logfile))
        self.carla_client.start_recorder(carla_logfile)
        try:
            total_frame_count = 0
            while True:
                print("----------")
                # Tick Control
                self.actor_tree.tick_controller()
                # Tick World
                tick_s = time.time()
                frame_id = self.world.tick(seconds=60.0)
                world_snapshot = self.world.get_snapshot()
                timestamp = world_snapshot.timestamp.elapsed_seconds
                print("World Tick -> FrameID: {} Timestamp: {} Cost: {:.3f}s".format(frame_id,
                                                                                     timestamp,
                                                                                     time.time()-tick_s))
                # Save data to disk
                if total_frame_count % self.frame_step == 0:
                    save_s = time.time()
                    self.actor_tree.tick_data_saving(frame_id, timestamp)
                    print("Raw data saved, cost {:.3f}s".format(time.time()-save_s))

                global sig_interrupt
                if sig_interrupt:
                    print("Exit step, wait 2 seconds...")
                    time.sleep(2.0)
                    break

                total_frame_count += 1
                if total_frame_count >= self.frame_total:
                    time.sleep(2.0)
                    break

        except KeyboardInterrupt:
            print("User interrupt, exit...")
        else:
            print("Unhandled error: reload the world and exit...")
        self.destroy()
        self.carla_client.reload_world()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--profile',
        default='default',
        type=str,
        help='Configuration profile name (default: default). Available: default, kitti, argoverse, simple')
    argparser.add_argument(
        '--config',
        type=str,
        help='Path to custom YAML configuration file (overrides --profile)')

    args = argparser.parse_args()

    # Load configuration
    try:
        config_manager = ConfigManager("{}/config".format(ROOT_PATH))

        if args.config:
            # Load custom config file
            print(f"Loading configuration from: {args.config}")
            config = config_manager.load_config(args.config)
        else:
            # Load profile
            print(f"Loading configuration profile: {args.profile}")
            available_profiles = config_manager.list_profiles()
            if args.profile not in available_profiles:
                print(f"Error: Profile '{args.profile}' not found")
                print(f"Available profiles: {', '.join(available_profiles)}")
                return 1
            config = config_manager.load_profile(args.profile)

        print(f"Configuration loaded successfully!")
        print(f"  Map: {config['recording']['map']}")
        print(f"  Frames: {config['recording']['frame_total']}")
        print(f"  Frame step: {config['recording']['frame_step']}")
        print(f"  Actors: {len(config['actors'])}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ConfigValidationError as e:
        print(f"Configuration validation error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Start recording
    data_recorder = DataRecorder(args)
    data_recorder.start_record(config)


if __name__ == "__main__":
    # execute only if run as a script
    main()
