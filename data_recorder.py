import argparse
import os
import signal
import time
import logging

import carla

from param import *
from config.config_manager import ConfigManager, ConfigValidationError
from recorder.actor_tree import ActorTree
from recorder.index_manager import IndexManager
from core.transform import Transform, Location, Rotation
from core.transform import transform_to_carla_transform
from core.logger import configure_global_logging, get_logger


class DataRecorder:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.carla_client = carla.Client(self.host, self.port)
        self.carla_client.set_timeout(10.0)

        # Delay initialization until setting_world_and_actors
        # to ensure correct initialization order
        self.world = None
        self.tm = None
        self.debug_helper = None
        self.actor_tree = None

        self.record_name = None
        self.base_save_dir = None
        self.config = None
        self.index_manager = None  # Will be initialized after config is loaded
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
        Initialize world and actors following CARLA official patterns
        Reference: CARLA PythonAPI/util/config.py (official implementation)

        NOTE: We do NOT use reload_world(False) as it causes segfaults on large maps.
        Instead, we follow the official pattern: load_world → get_world → apply_settings

        Args:
            config: Configuration dictionary from ConfigManager
        """
        # ============================================================
        # Phase 1: Load Map
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 1: Loading map...")
        self.logger.info("=" * 60)

        map_name = config['recording']['map']
        self.logger.info(f"Loading map: {map_name}")
        self.carla_client.load_world(map_name)

        time.sleep(2.0)  # Wait for map to load

        # Re-get world object after loading
        self.world = self.carla_client.get_world()
        self.debug_helper = self.world.debug  # Initialize debug helper
        self.logger.info("✓ Map loaded")

        # ============================================================
        # Phase 2: Configure Synchronous Mode
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 2: Configuring synchronous mode...")
        self.logger.info("=" * 60)

        settings = self.world.get_settings()
        settings.synchronous_mode = config['world_settings']['synchronous_mode']
        settings.fixed_delta_seconds = config['world_settings']['fixed_delta_seconds']
        settings.substepping = config['world_settings']['substepping']
        settings.max_substep_delta_time = config['world_settings']['max_substep_delta_time']
        settings.max_substeps = config['world_settings']['max_substeps']

        # Check fixed_delta_seconds must <= max_substep_delta_time * max_substeps
        if (settings.fixed_delta_seconds > 0.1):
            self.logger.error(
                "Warning: fixed_delta_seconds is set to a high value (>0.1s), "
                "which may lead to unstable simulation behavior."
            )
            raise ValueError("fixed_delta_seconds > 0.1s is not allowed.")
        if (settings.fixed_delta_seconds >
                settings.max_substep_delta_time * settings.max_substeps):
            error_msg = (
                f"Invalid world settings: fixed_delta_seconds "
                f"({settings.fixed_delta_seconds}) > "
                f"max_substep_delta_time ({settings.max_substep_delta_time}) * "
                f"max_substeps ({settings.max_substeps})"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"  Synchronous mode: {settings.synchronous_mode}")
        self.logger.info(f"  Fixed delta seconds: {settings.fixed_delta_seconds}")
        self.logger.info(f"  Substepping: {settings.substepping}")
        self.logger.info(f"  Max substep delta time: {settings.max_substep_delta_time}")
        self.logger.info(f"  Max substeps: {settings.max_substeps}")

        self.world.apply_settings(settings)
        self.logger.info("✓ World settings applied")

        # ============================================================
        # Phase 3: Configure Traffic Manager
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 3: Configuring Traffic Manager...")
        self.logger.info("=" * 60)

        # Get TM after world is in synchronous mode (CARLA official requirement)
        self.tm = self.carla_client.get_trafficmanager(8000)  # Explicit port
        tm_port = self.tm.get_port()
        self.logger.info(f"  TM port: {tm_port}")

        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_respawn_dormant_vehicles(True)

        # Official recommended parameters
        self.tm.global_percentage_speed_difference(30.0)  # 30% speed variation
        self.tm.set_random_device_seed(42)  # Deterministic behavior

        self.logger.info("  ✓ TM synchronous mode enabled")
        self.logger.info("  ✓ Global distance: 2.5m")
        self.logger.info("  ✓ Respawn dormant vehicles: True")
        self.logger.info("  ✓ Global speed difference: 30.0%")
        self.logger.info("  ✓ Deterministic seed: 42")
        self.logger.info("  ✓ Hybrid physics: False (full simulation)")

        # ============================================================
        # Phase 4: Set Weather and Spectator
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 4: Setting weather and spectator...")
        self.logger.info("=" * 60)

        if 'weather' in config['recording'] and config['recording']['weather']:
            weather_preset = config['recording']['weather']
            try:
                weather = getattr(carla.WeatherParameters, weather_preset)
                self.world.set_weather(weather)
                self.logger.info(f"✓ Weather set to: {weather_preset}")
            except AttributeError:
                self.logger.warning(f"Weather preset '{weather_preset}' not found")

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
            self.logger.info("✓ Spectator position set")

        # ============================================================
        # Phase 5: Configure Traffic Lights
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 5: Configuring traffic lights...")
        self.logger.info("=" * 60)

        traffic_light_settings = config.get('traffic_lights', {})
        self.set_traffic_light_time(traffic_light_settings)
        self.logger.info("✓ Traffic lights configured")

        # ============================================================
        # Phase 6: Batch Spawn Actors (WITHOUT autopilot)
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 6: Spawning actors (batch mode)...")
        self.logger.info("=" * 60)

        # Create save directory
        self.record_name = time.strftime("%Y_%m%d_%H%M", time.localtime())
        self.base_save_dir = "{}/record_{}".format(RAW_DATA_PATH, self.record_name)

        # Initialize ActorTree (spawn vehicles and sensors, autopilot not enabled yet)
        self.actor_tree = ActorTree(self.world, config, self.base_save_dir)
        self.actor_tree.init(self.carla_client, tm_port)
        self.logger.info("✓ All actors spawned (autopilot not enabled yet)")

        # ============================================================
        # Phase 7: Vehicle Stabilization (wait for vehicles to settle)
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 7: Vehicle stabilization...")
        self.logger.info("=" * 60)

        # Get stabilization config
        stabilization_config = config.get('world_settings', {}).get('vehicle_stabilization', {})
        init_tick_num = stabilization_config.get('init_tick_num', 5)

        self.logger.info(f"Ticking {init_tick_num} times for vehicles to settle on ground...")
        for i in range(init_tick_num):
            self.world.tick()
            if (i + 1) % 2 == 0 or (i + 1) == init_tick_num:  # Log every 2 ticks and last tick
                self.logger.info(f"  Stabilization tick {i + 1}/{init_tick_num}")

        self.logger.info(f"✓ Stabilization complete after {init_tick_num} ticks")

        # ============================================================
        # Phase 8: Enable Autopilot (after stabilization)
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 8: Enabling autopilot...")
        self.logger.info("=" * 60)

        # Enable autopilot for all vehicles
        autopilot_count = self.actor_tree.enable_autopilot_batch(
            self.carla_client,
            tm_port,
            self.actor_tree.vehicle_nodes_map
        )
        self.logger.info(f"✓ Autopilot enabled for {autopilot_count} vehicles")

        # ============================================================
        # Phase 9: Record Initial Frame ID
        # ============================================================
        self.logger.info("=" * 60)
        self.logger.info("Phase 9: Recording initial frame ID...")
        self.logger.info("=" * 60)

        # Get current frame ID as baseline (after stabilization ticks)
        world_snapshot = self.world.get_snapshot()
        init_frame_id = world_snapshot.frame
        self.init_frame_id = init_frame_id  # Store initial frame_id
        self.logger.info(f"✓ Initial frame ID recorded: {init_frame_id}")

        # ============================================================
        # Phase 10: Set Recording Parameters
        # ============================================================
        self.frame_total = config['recording']['frame_total']
        self.frame_step = config['recording']['frame_step']
        self.config = config

        self.logger.info("=" * 60)
        self.logger.info("Initialization Complete")
        self.logger.info("=" * 60)
        self.logger.info(f"  Recording directory: {self.base_save_dir}")
        self.logger.info(f"  Target frames: {self.frame_total}")
        self.logger.info(f"  Frame step: {self.frame_step}")
        self.logger.info(f"  Initial frame ID: {init_frame_id}")
        self.logger.info(f"  Actors: {len(self.actor_tree.node_list)} nodes")
        self.logger.info("=" * 60)

        # Initialize IndexManager after all actors are created
        self.logger.info("Initializing IndexManager...")
        self.index_manager = IndexManager(
            base_save_dir=self.base_save_dir,
            config=config
        )
        self.logger.info("✓ IndexManager initialized")

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
            total_frame_count = 0  # Count from initialization complete
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

                # Calculate relative frame (from initialization complete)
                relative_frame = frame_id - self.init_frame_id

                self.logger.info(
                    f"World Tick -> Absolute FrameID: {frame_id}, "
                    f"Relative Frame: {relative_frame}, "
                    f"Recorded Frames: {total_frame_count}, "
                    f"Timestamp: {timestamp:.3f}s, Cost: {tick_cost:.3f}s"
                )

                # Save data to disk
                if total_frame_count % self.frame_step == 0:
                    save_s = time.time()
                    try:
                        # Use relative_frame as the saved frame_id
                        actors_info = self.actor_tree.tick_data_saving(relative_frame, timestamp)

                        # Collect frame information for indexing
                        self.index_manager.collect_frame_info(relative_frame, timestamp, actors_info)

                        save_cost = time.time() - save_s
                        self.logger.info(f"Data saved (frame {relative_frame}), cost {save_cost:.3f}s")
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

            # Finalize index files
            if self.index_manager:
                try:
                    self.logger.info("Finalizing index files...")
                    self.index_manager.finalize()
                    self.logger.info("✓ Index files finalized")
                except Exception as e:
                    self.logger.error(f"Failed to finalize index files: {e}")

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
