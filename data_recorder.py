import argparse
import os
import signal
import time
import logging

import carla

from param import *
from config.config_manager import ConfigManager, ConfigValidationError
from recorder.actor_tree import ActorTree
from core.transform import Transform, Location, Rotation
from core.transform import transform_to_carla_transform
from core.logger import configure_global_logging, get_logger


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
        self.interrupted = False
        self.logger = get_logger(__name__)

        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal (Ctrl+C)"""
        self.logger.info("Interrupt signal received (Ctrl+C), preparing graceful shutdown...")
        self.interrupted = True

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

        self.logger.info(f"World settings: {settings}")
        self.world.apply_settings(settings)

        # Set weather if specified
        if 'weather' in config['recording'] and config['recording']['weather']:
            weather_preset = config['recording']['weather']
            self.logger.info(f"Setting weather to: {weather_preset}")
            try:
                weather = getattr(carla.WeatherParameters, weather_preset)
                self.world.set_weather(weather)
                self.logger.info(f"✓ Weather set to {weather_preset}")
            except AttributeError:
                self.logger.warning(f"Weather preset '{weather_preset}' not found, using default")

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
        self.logger.info("Setting synchronous mode for traffic manager...")
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
        self.logger.info(f"Recording to folder: {self.base_save_dir}")
        os.makedirs(self.base_save_dir, exist_ok=True)
        carla_logfile = f"{self.base_save_dir}/carla_raw_record.log"
        self.logger.info(f"Start recording to {carla_logfile}")
        self.carla_client.start_recorder(carla_logfile)

        try:
            total_frame_count = 0
            while True:
                self.logger.info("="*50)
                # Tick Control
                self.actor_tree.tick_controller()

                # Tick World
                tick_s = time.time()
                frame_id = self.world.tick(seconds=60.0)
                world_snapshot = self.world.get_snapshot()
                timestamp = world_snapshot.timestamp.elapsed_seconds
                tick_cost = time.time() - tick_s
                self.logger.info(
                    f"World Tick -> FrameID: {frame_id}, "
                    f"Timestamp: {timestamp:.3f}s, Cost: {tick_cost:.3f}s"
                )

                # Save data to disk
                if total_frame_count % self.frame_step == 0:
                    save_s = time.time()
                    try:
                        self.actor_tree.tick_data_saving(frame_id, timestamp)
                        save_cost = time.time() - save_s
                        self.logger.info(f"Raw data saved, cost {save_cost:.3f}s")
                    except (RuntimeError, TimeoutError) as e:
                        # Data save failed or timeout - strict mode: abort immediately
                        self.logger.error(
                            f"Data save failed, aborting recording: {e}"
                        )
                        raise

                # Check for user interrupt
                if self.interrupted:
                    self.logger.info("User interrupt detected, exiting gracefully...")
                    time.sleep(2.0)
                    break

                total_frame_count += 1
                if total_frame_count >= self.frame_total:
                    self.logger.info(
                        f"Reached target frame count: {self.frame_total}, finishing..."
                    )
                    time.sleep(2.0)
                    break

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, exiting...")
        except (RuntimeError, TimeoutError) as e:
            self.logger.error(f"Recording aborted due to error: {e}")
            self.logger.error("Data integrity may be compromised. Check logs above for details.")
        except Exception as e:
            self.logger.exception(f"Unexpected error during recording: {e}")
        finally:
            self.logger.info("Cleaning up resources...")
            self.destroy()
            self.logger.info("Reloading world...")
            self.carla_client.reload_world()
            self.logger.info("Recording session ended.")


def main():
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
    argparser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging')
    argparser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (optional)')

    args = argparser.parse_args()

    # Configure global logging system
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file if hasattr(args, 'log_file') else None
    configure_global_logging(level=log_level, log_file=log_file)

    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("CARLA Dataset Tools - Data Recorder")
    logger.info("=" * 60)

    # Load configuration
    try:
        config_manager = ConfigManager(f"{ROOT_PATH}/config")

        if args.config:
            # Load custom config file
            logger.info(f"Loading configuration from: {args.config}")
            config = config_manager.load_config(args.config)
        else:
            # Load profile
            logger.info(f"Loading configuration profile: {args.profile}")
            available_profiles = config_manager.list_profiles()
            if args.profile not in available_profiles:
                logger.error(f"Profile '{args.profile}' not found")
                logger.info(f"Available profiles: {', '.join(available_profiles)}")
                return 1
            config = config_manager.load_profile(args.profile)

        logger.info("Configuration loaded successfully!")
        logger.info(f"  Map: {config['recording']['map']}")
        logger.info(f"  Frames: {config['recording']['frame_total']}")
        logger.info(f"  Frame step: {config['recording']['frame_step']}")
        logger.info(f"  Actors: {len(config['actors'])}")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ConfigValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error loading configuration: {e}")
        return 1

    # Start recording
    try:
        data_recorder = DataRecorder(args)
        data_recorder.start_record(config)
    except Exception as e:
        logger.exception(f"Fatal error in data recorder: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # execute only if run as a script
    exit(main())
