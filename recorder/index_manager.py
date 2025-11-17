#!/usr/bin/python3
"""
Index Manager for CARLA Dataset Recording

This module manages index generation during data recording, creating unified
indices for efficient data querying and access.

Generated index files:
- dataset_info.json: Dataset metadata
- master_index.csv: Global frame-level index
- {actor_name}/sensor_index.csv: Per-actor sensor index
- others.world_X/objects_index.csv: World objects index
"""

import os
import json
import csv
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages index generation during dataset recording.

    Collects frame information during recording and generates multiple
    index files to facilitate data querying and access.
    """

    def __init__(self, base_save_dir: str, config: Dict[str, Any]):
        """
        Initialize IndexManager.

        Args:
            base_save_dir: Base directory for saving dataset
            config: Recording configuration dictionary
        """
        self.base_save_dir = base_save_dir
        self.config = config
        self.start_time = datetime.now()

        # Storage for index data
        self.master_index_data = []  # Global frame index
        self.actor_index_data = {}   # Per-actor index data
        self.world_index_data = []   # World objects index

        # Metadata
        self.dataset_metadata = {
            'dataset_name': os.path.basename(base_save_dir),
            'created_at': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'map': config.get('recording', {}).get('map', 'unknown'),
            'weather': config.get('recording', {}).get('weather', 'default'),
            'frame_rate': 1.0 / config.get('world_settings', {}).get('fixed_delta_seconds', 0.1),
            'actors': {},
            'sensors': {}
        }

        # Extract actor and sensor info from config
        self._extract_metadata_from_config()

        logger.info(f"IndexManager initialized for dataset: {self.dataset_metadata['dataset_name']}")

    def _extract_metadata_from_config(self):
        """Extract actor and sensor metadata from configuration."""
        actors_config = self.config.get('actors', [])

        for actor_cfg in actors_config:
            actor_name = actor_cfg.get('name', 'unknown')
            actor_type = actor_cfg.get('type', 'unknown')

            # Add actor info
            self.dataset_metadata['actors'][actor_name] = {
                'type': actor_type,
                'role': actor_cfg.get('role', 'unknown')
            }

            # Add sensors info
            sensors = actor_cfg.get('sensors', [])
            for sensor_cfg in sensors:
                sensor_name = sensor_cfg.get('name', 'unknown')
                sensor_type = sensor_cfg.get('type', 'unknown')

                self.dataset_metadata['sensors'][sensor_name] = {
                    'parent': actor_name,
                    'type': sensor_type
                }

                # Add type-specific attributes
                if sensor_type.startswith('sensor.camera'):
                    self.dataset_metadata['sensors'][sensor_name].update({
                        'width': sensor_cfg.get('image_size_x', 800),
                        'height': sensor_cfg.get('image_size_y', 600),
                        'fov': sensor_cfg.get('fov', 90.0)
                    })
                elif sensor_type.startswith('sensor.lidar'):
                    self.dataset_metadata['sensors'][sensor_name].update({
                        'channels': sensor_cfg.get('channels', 64),
                        'range': sensor_cfg.get('range', 100.0),
                        'points_per_second': sensor_cfg.get('points_per_second', 100000)
                    })

    def collect_frame_info(self, frame_id: int, timestamp: float, actors_info: Dict[str, Any]):
        """
        Collect information for a single frame.

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Simulation timestamp
            actors_info: Dictionary containing save information from all actors
                Format: {
                    'actor_name': {
                        'type': 'vehicle|sensor|world',
                        'sensors': {
                            'sensor_name': {
                                'file': 'path/to/file',
                                'pose': {'x': ..., 'y': ..., ...},
                                'points_count': 12345,  # for lidar
                                ...
                            }
                        },
                        'vehicle_state': {...},  # for vehicles
                        'objects_count': 27,  # for world
                        ...
                    }
                }
        """
        try:
            # Collect master index entry
            master_entry = {
                'frame': frame_id,
                'timestamp': timestamp,
                'wall_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            }

            # Add per-actor availability flags
            for actor_name, actor_info in actors_info.items():
                if actor_info.get('type') == 'sensor':
                    # For sensors, check if data was saved
                    has_data = actor_info.get('file') is not None
                    master_entry[f'has_{actor_name}'] = has_data
                elif actor_info.get('type') == 'vehicle':
                    # For vehicles, always mark as available if info exists
                    master_entry[f'has_{actor_name}'] = True
                elif actor_info.get('type') == 'world':
                    # For world objects
                    has_objects = actor_info.get('objects_count', 0) > 0
                    master_entry[f'has_{actor_name}_objects'] = has_objects

            self.master_index_data.append(master_entry)

            # Collect per-actor index data
            for actor_name, actor_info in actors_info.items():
                actor_type = actor_info.get('type')

                # Check if this is a vehicle actor (by type or by naming convention)
                is_vehicle = (actor_type == 'vehicle' or
                             actor_name.startswith('vehicle_') or
                             actor_name.startswith('ego'))

                if is_vehicle:
                    self._collect_vehicle_index(frame_id, timestamp, actor_name, actor_info)
                elif actor_type == 'world':
                    self._collect_world_index(frame_id, timestamp, actor_name, actor_info)

            logger.debug(f"Collected index info for frame {frame_id}")

        except Exception as e:
            logger.error(f"Failed to collect frame info for frame {frame_id}: {e}")
            # Don't raise - index collection failure shouldn't stop recording

    def _collect_vehicle_index(self, frame_id: int, timestamp: float,
                               actor_name: str, actor_info: Dict[str, Any]):
        """
        Collect index data for a vehicle and its sensors.

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Timestamp
            actor_name: Vehicle actor name
            actor_info: Actor information dictionary
        """
        if actor_name not in self.actor_index_data:
            self.actor_index_data[actor_name] = []

        index_entry = {
            'frame': frame_id,
            'timestamp': timestamp
        }

        # Add sensor data
        sensors_info = actor_info.get('sensors', {})
        for sensor_name, sensor_info in sensors_info.items():
            prefix = sensor_name.replace(actor_name + '_', '')  # Remove actor prefix

            # Add file path
            if 'file' in sensor_info:
                index_entry[f'{prefix}_file'] = sensor_info['file']

            # Pose data is stored in individual {sensor_name}/poses.csv files
            # No need to duplicate in sensor_index.csv

            # Add sensor-specific metadata
            if 'points_count' in sensor_info:
                index_entry[f'{prefix}_points_count'] = sensor_info['points_count']
            if 'camera_info' in sensor_info:
                # Camera info is static, only add on first frame
                pass

        # Vehicle state is stored in vehicle_status.csv
        # No need to duplicate in sensor_index.csv

        self.actor_index_data[actor_name].append(index_entry)

    def _collect_world_index(self, frame_id: int, timestamp: float,
                            actor_name: str, actor_info: Dict[str, Any]):
        """Collect index data for world objects."""
        world_entry = {
            'frame': frame_id,
            'timestamp': timestamp,
            'pkl_file': actor_info.get('file', ''),
            'object_count': actor_info.get('objects_count', 0),
            'vehicle_count': actor_info.get('vehicle_count', 0),
            'pedestrian_count': actor_info.get('pedestrian_count', 0),
            'static_count': actor_info.get('static_count', 0)
        }
        self.world_index_data.append(world_entry)

    def finalize(self):
        """
        Finalize dataset and save all index files.

        Should be called after recording is complete.
        """
        logger.info("Finalizing indices...")

        # Update metadata with final statistics
        if self.master_index_data:
            self.dataset_metadata['total_frames'] = len(self.master_index_data)
            self.dataset_metadata['duration_seconds'] = (
                self.master_index_data[-1]['timestamp'] -
                self.master_index_data[0]['timestamp']
            )

        # Save all index files
        try:
            self._save_dataset_info()
            self._save_master_index()
            self._save_actor_indices()
            self._save_world_index()

            logger.info("✓ All index files generated successfully")
            logger.info(f"  - Dataset: {self.dataset_metadata['dataset_name']}")
            logger.info(f"  - Frames: {self.dataset_metadata.get('total_frames', 0)}")
            logger.info(f"  - Duration: {self.dataset_metadata.get('duration_seconds', 0):.2f}s")
            logger.info(f"  - Actors: {len(self.actor_index_data)}")

        except Exception as e:
            logger.error(f"Failed to save index files: {e}")
            raise

    def _save_dataset_info(self):
        """Save dataset metadata to JSON file."""
        filepath = os.path.join(self.base_save_dir, 'dataset_info.json')

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.dataset_metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved dataset_info.json")

    def _save_master_index(self):
        """Save master index CSV."""
        if not self.master_index_data:
            logger.warning("No master index data to save")
            return

        filepath = os.path.join(self.base_save_dir, 'master_index.csv')

        # Get all fieldnames from all entries (in case some have different fields)
        fieldnames = set()
        for entry in self.master_index_data:
            fieldnames.update(entry.keys())
        fieldnames = sorted(list(fieldnames))

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.master_index_data)

        logger.info(f"✓ Saved master_index.csv ({len(self.master_index_data)} frames)")

    def _save_actor_indices(self):
        """Save per-actor index CSV files."""
        for actor_name, index_data in self.actor_index_data.items():
            if not index_data:
                continue

            actor_dir = os.path.join(self.base_save_dir, actor_name)
            os.makedirs(actor_dir, exist_ok=True)
            filepath = os.path.join(actor_dir, 'sensor_index.csv')

            # Get all fieldnames
            fieldnames = set()
            for entry in index_data:
                fieldnames.update(entry.keys())
            fieldnames = sorted(list(fieldnames))

            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(index_data)

            logger.info(f"✓ Saved {actor_name}/sensor_index.csv ({len(index_data)} frames)")

    def _save_world_index(self):
        """Save world objects index CSV."""
        if not self.world_index_data:
            logger.warning("No world index data to save")
            return

        # Find world actor directory
        world_dirs = [d for d in os.listdir(self.base_save_dir)
                     if d.startswith('others.world_')]
        if not world_dirs:
            logger.warning("No world directory found, skipping world index")
            return

        world_dir = os.path.join(self.base_save_dir, world_dirs[0])
        filepath = os.path.join(world_dir, 'objects_index.csv')

        fieldnames = ['frame', 'timestamp', 'pkl_file', 'object_count',
                     'vehicle_count', 'pedestrian_count', 'static_count']

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.world_index_data)

        logger.info(f"✓ Saved objects_index.csv ({len(self.world_index_data)} frames)")
