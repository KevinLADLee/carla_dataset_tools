#!/usr/bin/env python3
"""
CARLA Map Bird's Eye View (BEV) Capture Tool

Captures bird's-eye view (BEV) images of entire CARLA maps using multiple cameras
with image stitching. Generates RGB, depth, and semantic segmentation images.

The tool uses a grid of overhead cameras to capture the entire map, with configurable
field of view (FOV) to control perspective distortion. Lower FOV values (e.g., 60°)
reduce edge distortion at the cost of requiring more camera positions.

Usage:
    python tools/capture_map_bev.py \\
        --host localhost \\
        --port 2000 \\
        --map Town02 \\
        --output /path/to/output \\
        --height 150 \\
        --fov 60 \\
        --overlap 0.15 \\
        --resolution 2000
"""

import argparse
import gc
import json
import logging
import math
import os
import queue
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import carla
import cv2 as cv
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SENSOR_QUEUE_TIMEOUT = 10.0  # Timeout for sensor data capture
CARLA_IMAGE_CHANNELS = 4  # BGRA format
CARLA_IMAGE_DTYPE = 'uint8'


class SynchronizedCameraGroup:
    """
    Manages a group of synchronized cameras (RGB, Depth, Semantic) at a single position.

    Handles spawning, data capture, and cleanup of camera sensors.
    """

    def __init__(self, world: carla.World, position: carla.Location,
                 rotation: carla.Rotation, resolution: int, fov: float):
        """
        Initialize camera group.

        Args:
            world: CARLA world object
            position: Camera position in world coordinates
            rotation: Camera rotation (pitch=-90 for overhead view)
            resolution: Image resolution (width and height)
            fov: Field of view in degrees
        """
        self.world = world
        self.position = position
        self.rotation = rotation
        self.resolution = resolution
        self.fov = fov

        # Camera actors
        self.rgb_camera = None
        self.depth_camera = None
        self.semantic_camera = None

        # Image queues
        self.rgb_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        self.semantic_queue = queue.Queue()

        logger.debug(f"SynchronizedCameraGroup initialized at ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")

    def set_transform(self, position: carla.Location):
        """
        Move all cameras to new position without recreating them.

        Args:
            position: New camera position in world coordinates
        """
        self.position = position
        transform = carla.Transform(self.position, self.rotation)

        # Move each camera to the new position
        if self.rgb_camera:
            self.rgb_camera.set_transform(transform)
        if self.depth_camera:
            self.depth_camera.set_transform(transform)
        if self.semantic_camera:
            self.semantic_camera.set_transform(transform)

        logger.debug(f"Cameras moved to ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")

    def clear_queues(self):
        """Clear sensor data queues to prevent old data accumulation."""
        total_cleared = 0
        for queue_name, q in [('RGB', self.rgb_queue), ('Depth', self.depth_queue),
                             ('Semantic', self.semantic_queue)]:
            cleared = 0
            while not q.empty():
                try:
                    q.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            total_cleared += cleared
            if cleared > 0:
                logger.debug(f"Cleared {cleared} items from {queue_name} queue")

        return total_cleared

    def spawn_cameras(self):
        """Spawn all three cameras at the configured position."""
        blueprint_library = self.world.get_blueprint_library()
        transform = carla.Transform(self.position, self.rotation)

        # RGB Camera
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(self.resolution))
        rgb_bp.set_attribute('image_size_y', str(self.resolution))
        rgb_bp.set_attribute('fov', str(self.fov))
        self.rgb_camera = self.world.spawn_actor(rgb_bp, transform)
        self.rgb_camera.listen(lambda image: self.rgb_queue.put(image))

        # Depth Camera
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.resolution))
        depth_bp.set_attribute('image_size_y', str(self.resolution))
        depth_bp.set_attribute('fov', str(self.fov))
        self.depth_camera = self.world.spawn_actor(depth_bp, transform)
        self.depth_camera.listen(lambda image: self.depth_queue.put(image))

        # Semantic Segmentation Camera
        semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', str(self.resolution))
        semantic_bp.set_attribute('image_size_y', str(self.resolution))
        semantic_bp.set_attribute('fov', str(self.fov))
        self.semantic_camera = self.world.spawn_actor(semantic_bp, transform)
        self.semantic_camera.listen(lambda image: self.semantic_queue.put(image))

        logger.debug("All cameras spawned successfully")

    def capture_images(self, temp_dir: str) -> Tuple[str, str, str]:
        """
        Capture synchronized images from all cameras with deep copy and save to temp files.

        Args:
            temp_dir: Temporary directory path to save captured images

        Returns:
            Tuple of (rgb_path, depth_path, semantic_path) as file paths

        Raises:
            TimeoutError: If image capture times out
        """
        # Trigger world tick to generate sensor data
        self.world.tick()

        try:
            # Wait for images from all cameras
            rgb_data = self.rgb_queue.get(timeout=SENSOR_QUEUE_TIMEOUT)
            depth_data = self.depth_queue.get(timeout=SENSOR_QUEUE_TIMEOUT)
            semantic_data = self.semantic_queue.get(timeout=SENSOR_QUEUE_TIMEOUT)

            # CRITICAL: Deep copy immediately to break CARLA memory references
            # Convert RGB image and deep copy
            rgb_array = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
            rgb_image = rgb_array.reshape((rgb_data.height, rgb_data.width, CARLA_IMAGE_CHANNELS)).copy()

            # Convert Depth image with Raw color converter
            # Raw format encodes depth in RGB channels: (R + G*256 + B*256²) / (256³-1) * 1000
            depth_data.convert(carla.ColorConverter.Raw)
            depth_array = np.frombuffer(depth_data.raw_data, dtype=np.uint8)
            depth_image = depth_array.reshape((depth_data.height, depth_data.width, CARLA_IMAGE_CHANNELS)).copy()

            # Convert Semantic image with CityScapesPalette and deep copy
            semantic_data.convert(carla.ColorConverter.CityScapesPalette)
            semantic_array = np.frombuffer(semantic_data.raw_data, dtype=np.uint8)
            semantic_image = semantic_array.reshape((semantic_data.height, semantic_data.width, CARLA_IMAGE_CHANNELS)).copy()

            logger.debug(f"Images deep copied: RGB{rgb_image.shape}, Depth{depth_image.shape}, Semantic{semantic_image.shape}")

            # Generate unique filenames based on position
            position_key = f"{int(self.position.x)}_{int(self.position.y)}"
            rgb_path = os.path.join(temp_dir, f"rgb_{position_key}.png")
            depth_path = os.path.join(temp_dir, f"depth_{position_key}.png")
            semantic_path = os.path.join(temp_dir, f"semantic_{position_key}.png")

            # Save deep-copied arrays to temp files
            cv.imwrite(rgb_path, rgb_image)
            cv.imwrite(depth_path, depth_image)
            cv.imwrite(semantic_path, semantic_image)

            logger.debug(f"Images saved to temp files: {rgb_path}, {depth_path}, {semantic_path}")

            # Explicitly release memory
            del rgb_image, depth_image, semantic_image
            del rgb_array, depth_array, semantic_array

            return rgb_path, depth_path, semantic_path

        except queue.Empty:
            error_msg = f"Timeout waiting for camera data at position ({self.position.x:.1f}, {self.position.y:.1f})"
            logger.error(error_msg)
            raise TimeoutError(error_msg)

    def _decode_depth_from_bgra(self, depth_bgra: np.ndarray) -> np.ndarray:
        """
        Decode depth values from CARLA BGRA encoded depth image.

        CARLA Raw format encodes depth in RGB channels as:
        depth_meters = (R + G*256 + B*256²) / (256³-1) * 1000

        Args:
            depth_bgra: BGRA encoded depth image (height, width, 4)

        Returns:
            Depth values in meters as float32 array (height, width)
        """
        b = depth_bgra[:, :, 0].astype(np.uint32)
        g = depth_bgra[:, :, 1].astype(np.uint32)
        r = depth_bgra[:, :, 2].astype(np.uint32)

        # CARLA formula: depth = (R + G*256 + B*256²) / (256³-1) * 1000
        depth_encoded = r + g * 256 + b * 256 * 256
        depth_meters = depth_encoded.astype(np.float32) / (256**3 - 1) * 1000.0

        return depth_meters

    def _normalize_depth_to_grayscale(self, depth_meters: np.ndarray) -> np.ndarray:
        """
        Normalize depth values to 8-bit grayscale using adaptive percentile-based normalization.

        Args:
            depth_meters: Depth values in meters (height, width)

        Returns:
            Normalized grayscale image (height, width) with values 0-255
        """
        # Find valid depth values (greater than 0)
        valid_depths = depth_meters[depth_meters > 0]

        if len(valid_depths) == 0:
            return np.zeros(depth_meters.shape, dtype=np.uint8)

        # Use 1%-99% percentile range to avoid outliers
        min_depth = np.percentile(valid_depths, 1)
        max_depth = np.percentile(valid_depths, 99)

        # Ensure we have a valid range
        if max_depth <= min_depth:
            max_depth = min_depth + 1.0

        # Normalize to 0-255 range
        depth_normalized = np.clip((depth_meters - min_depth) / (max_depth - min_depth), 0, 1)
        depth_grayscale = (depth_normalized * 255).astype(np.uint8)

        # Set invalid depth values to 0
        depth_grayscale[depth_meters <= 0] = 0

        return depth_grayscale

    def destroy(self):
        """Destroy all camera actors and clean up resources."""
        cameras = [
            ('RGB', self.rgb_camera),
            ('Depth', self.depth_camera),
            ('Semantic', self.semantic_camera)
        ]

        for name, camera in cameras:
            if camera is not None:
                try:
                    camera.stop()
                    camera.destroy()
                    logger.debug(f"{name} camera destroyed")
                except Exception as e:
                    logger.warning(f"Failed to destroy {name} camera: {e}")

        # Clear queues
        for q in [self.rgb_queue, self.depth_queue, self.semantic_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break


class MapBEVCapture:
    """
    Main class for capturing bird's eye view (BEV) of CARLA maps.

    Manages map boundary calculation, camera grid generation, image capture,
    and stitching of final output images.
    """

    def __init__(self, host: str, port: int, map_name: Optional[str] = None):
        """
        Initialize map BEV capture.

        Args:
            host: CARLA server hostname
            port: CARLA server port
            map_name: Name of map to load (None = use current map)
        """
        self.host = host
        self.port = port
        self.map_name = map_name

        # Connect to CARLA
        logger.info(f"Connecting to CARLA server at {host}:{port}")
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)

        # Load map if specified
        if map_name:
            logger.info(f"Loading map: {map_name}")
            self.world = self.client.load_world(map_name)
        else:
            self.world = self.client.get_world()
            self.map_name = self.world.get_map().name.split('/')[-1]

        self.carla_map = self.world.get_map()
        logger.info(f"Using map: {self.map_name}")

        # Enable synchronous mode for consistent captures
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        logger.info("Synchronous mode enabled")

    def calculate_map_bounds(self, margin: float = 50.0) -> Tuple[float, float, float, float]:
        """
        Calculate bounding box of the map based on road topology.

        Args:
            margin: Additional margin around map bounds in meters

        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        logger.info("Calculating map boundaries...")

        topology = self.carla_map.get_topology()

        if not topology:
            logger.warning("No topology found, using spawn points for bounds")
            spawn_points = self.carla_map.get_spawn_points()
            if not spawn_points:
                raise ValueError("Map has no topology or spawn points")

            positions = [sp.location for sp in spawn_points]
        else:
            # Extract all waypoint locations from topology
            positions = []
            for waypoint_pair in topology:
                positions.append(waypoint_pair[0].transform.location)
                if waypoint_pair[1]:
                    positions.append(waypoint_pair[1].transform.location)

        # Calculate bounds
        min_x = min(pos.x for pos in positions) - margin
        max_x = max(pos.x for pos in positions) + margin
        min_y = min(pos.y for pos in positions) - margin
        max_y = max(pos.y for pos in positions) + margin

        logger.info(f"Map bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        logger.info(f"Map size: {max_x - min_x:.1f}m x {max_y - min_y:.1f}m")

        return min_x, max_x, min_y, max_y

    def _decode_depth_from_bgra(self, depth_bgra: np.ndarray) -> np.ndarray:
        """
        Decode depth values from CARLA BGRA encoded depth image.

        CARLA Raw format encodes depth in RGB channels as:
        depth_meters = (R + G*256 + B*256²) / (256³-1) * 1000

        Args:
            depth_bgra: BGRA encoded depth image (height, width, 4)

        Returns:
            Depth values in meters as float32 array (height, width)
        """
        b = depth_bgra[:, :, 0].astype(np.uint32)
        g = depth_bgra[:, :, 1].astype(np.uint32)
        r = depth_bgra[:, :, 2].astype(np.uint32)

        # CARLA formula: depth = (R + G*256 + B*256²) / (256³-1) * 1000
        depth_encoded = r + g * 256 + b * 256 * 256
        depth_meters = depth_encoded.astype(np.float32) / (256**3 - 1) * 1000.0

        return depth_meters

    def normalize_depth(self, depth_stitched: np.ndarray) -> np.ndarray:
        """
        Apply global depth normalization to stitched depth image.

        Normalizes the entire stitched depth image using global percentile-based
        adaptive range for unified depth representation across the entire map.

        Args:
            depth_stitched: Stitched BGRA encoded depth image

        Returns:
            Globally normalized grayscale depth image (height, width) with values 0-255
        """
        logger.info("Applying global depth normalization to stitched image...")

        # Decode BGRA depth image to meters
        depth_meters = self._decode_depth_from_bgra(depth_stitched)

        # Find valid depth values (greater than 0)
        valid_depths = depth_meters[depth_meters > 0]

        if len(valid_depths) == 0:
            logger.warning("No valid depth values found in stitched image")
            return np.zeros(depth_meters.shape[:2], dtype=np.uint8)

        # Use global 1%-99% percentile range to avoid outliers
        min_depth = np.percentile(valid_depths, 1)
        max_depth = np.percentile(valid_depths, 99)

        if max_depth <= min_depth:
            max_depth = min_depth + 1.0
            logger.warning(f"Narrow depth range: [{min_depth:.2f}, {max_depth:.2f}] meters")

        # Global normalization to 0-255 range
        depth_normalized = np.clip((depth_meters - min_depth) / (max_depth - min_depth), 0, 1)
        depth_grayscale = (depth_normalized * 255).astype(np.uint8)
        depth_grayscale[depth_meters <= 0] = 0

        logger.info(f"Global depth normalization complete: range [{min_depth:.2f}, {max_depth:.2f}] meters")

        # Release memory
        del depth_meters, valid_depths, depth_normalized

        return depth_grayscale

    def generate_camera_grid(self, bounds: Tuple[float, float, float, float],
                            height: float, fov: float, resolution: int,
                            overlap: float = 0.15) -> List[Tuple[float, float]]:
        """
        Generate grid of camera positions to cover the entire map.

        Args:
            bounds: Map boundaries (min_x, max_x, min_y, max_y)
            height: Camera height above ground
            fov: Field of view in degrees
            resolution: Camera resolution (used for aspect ratio)
            overlap: Overlap ratio between adjacent cameras (0.0 - 1.0)

        Returns:
            List of (x, y) positions for camera centers
        """
        min_x, max_x, min_y, max_y = bounds

        # Calculate coverage area for a single camera
        fov_rad = math.radians(fov)
        coverage_width = 2.0 * height * math.tan(fov_rad / 2.0)

        # Calculate step size with overlap
        step = coverage_width * (1.0 - overlap)

        logger.info(f"Camera coverage: {coverage_width:.1f}m x {coverage_width:.1f}m per camera")
        logger.info(f"Grid step size: {step:.1f}m (overlap: {overlap*100:.0f}%)")

        # Generate grid positions
        grid_positions = []

        x = min_x + coverage_width / 2
        while x <= max_x + coverage_width / 2:
            y = min_y + coverage_width / 2
            while y <= max_y + coverage_width / 2:
                grid_positions.append((x, y))
                y += step
            x += step

        logger.info(f"Generated grid with {len(grid_positions)} camera positions")
        logger.info(f"Grid layout: {int((max_x - min_x) / step) + 1} x {int((max_y - min_y) / step) + 1}")

        return grid_positions, coverage_width, step

    def capture_at_grid_positions(self, grid_positions: List[Tuple[float, float]],
                    height: float, fov: float, resolution: int) -> Dict:
        """
        Capture images at all grid positions using single camera with set_transform.

        Args:
            grid_positions: List of (x, y) camera positions
            height: Camera height
            fov: Field of view
            resolution: Image resolution

        Returns:
            Dictionary containing image file paths, positions, and temp directory
        """
        logger.info(f"Starting capture of {len(grid_positions)} positions...")

        # Create temporary directory for captured images
        temp_dir = tempfile.mkdtemp(prefix="carla_overhead_")
        logger.info(f"Temporary directory created: {temp_dir}")

        captured_data = {
            'rgb_paths': [],
            'depth_paths': [],
            'semantic_paths': [],
            'positions': [],
            'temp_dir': temp_dir
        }

        rotation = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)  # Overhead view

        # Create single SynchronizedCameraGroup for reuse (KEY OPTIMIZATION)
        first_pos = carla.Location(x=grid_positions[0][0], y=grid_positions[0][1], z=height)
        camera_group = SynchronizedCameraGroup(self.world, first_pos, rotation, resolution, fov)

        try:
            # Spawn cameras once (KEY OPTIMIZATION)
            camera_group.spawn_cameras()
            logger.info("Cameras spawned once for all positions")

            # Wait for cameras to initialize
            time.sleep(0.2)

            for idx, (x, y) in enumerate(grid_positions):
                logger.info(f"Capturing position {idx + 1}/{len(grid_positions)}: ({x:.1f}, {y:.1f})")

                try:
                    # Move camera to new position using set_transform (KEY OPTIMIZATION)
                    position = carla.Location(x=x, y=y, z=height)
                    camera_group.set_transform(position)

                    # Clear queues to remove old data from previous position
                    cleared_count = camera_group.clear_queues()
                    if cleared_count > 0:
                        logger.debug(f"Cleared {cleared_count} old frames from queues")

                    # Wait for position update and stabilization
                    time.sleep(0.05)

                    # Capture images and save to temp files
                    rgb_path, depth_path, semantic_path = \
                        camera_group.capture_images(temp_dir)

                    # Store file paths
                    captured_data['rgb_paths'].append(rgb_path)
                    captured_data['depth_paths'].append(depth_path)
                    captured_data['semantic_paths'].append(semantic_path)
                    captured_data['positions'].append((x, y))

                    logger.info(f"✓ Position {idx + 1} captured successfully")

                except Exception as e:
                    logger.error(f"✗ Failed to capture at position ({x:.1f}, {y:.1f}): {e}")
                    # Add placeholder None paths to maintain grid structure
                    captured_data['rgb_paths'].append(None)
                    captured_data['depth_paths'].append(None)
                    captured_data['semantic_paths'].append(None)
                    captured_data['positions'].append((x, y))

        finally:
            # Destroy cameras once at the end (KEY OPTIMIZATION)
            camera_group.destroy()
            logger.info("Cameras destroyed after all captures")

        logger.info("Grid capture completed")
        return captured_data

    def create_blend_mask(self, img_shape: Tuple[int, int], overlap_pixels: int,
                           sharpness: float = 0.02) -> np.ndarray:
        """
        Create blend weight mask for an image.

        Creates a smooth weight mask that transitions from 1.0 in the center to 0.0
        at the edges, enabling seamless blending in overlap regions.

        Args:
            img_shape: (height, width) of the image
            overlap_pixels: Number of pixels for feather blend at edges
            sharpness: Sharpness of the feather transition (0.02 is typical)

        Returns:
            Weight mask (height, width) with values 0.0-1.0
        """
        height, width = img_shape
        weight = np.ones((height, width), dtype=np.float32)

        # Calculate distance to nearest edge for each pixel
        for y in range(height):
            for x in range(width):
                # Distance to each edge
                dist_to_left = x
                dist_to_right = width - 1 - x
                dist_to_top = y
                dist_to_bottom = height - 1 - y

                # Minimum distance to any edge
                dist_to_edge = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

                # Apply feather blending in the overlap region
                if dist_to_edge < overlap_pixels:
                    # Linear interpolation from 0 to 1
                    alpha = dist_to_edge / overlap_pixels

                    # Apply smooth transition using tanh for softness
                    # This creates a smoother S-curve than linear
                    alpha = (np.tanh((alpha - 0.5) / sharpness) + 1) / 2
                    weight[y, x] = alpha

        return weight

    def stitch_images(self, image_paths: List[str], grid_positions: List[Tuple[float, float]],
                     bounds: Tuple[float, float, float, float], coverage_width: float,
                     step: float, resolution: int, overlap: float = 0.15) -> np.ndarray:
        """
        Stitch captured images into a single overhead view with feather blending.

        Uses weighted blending in overlap regions to create seamless transitions
        between adjacent tiles, eliminating visible seams.

        Args:
            image_paths: List of image file paths (some may be None)
            grid_positions: List of camera (x, y) positions
            bounds: Map boundaries
            coverage_width: Coverage width of each camera
            step: Grid step size
            resolution: Camera resolution
            overlap: Overlap ratio between cameras (default: 0.15)

        Returns:
            Stitched image as numpy array
        """
        min_x, max_x, min_y, max_y = bounds

        # Calculate output image dimensions
        # CRITICAL: In overhead view, axes are swapped:
        # - pixel_x uses cam_y, so output_width corresponds to Y range
        # - pixel_y uses cam_x, so output_height corresponds to X range
        map_width_x = max_x - min_x  # X range in CARLA
        map_width_y = max_y - min_y  # Y range in CARLA

        # Pixels per meter ratio
        pixels_per_meter = resolution / coverage_width

        # Swap width/height to match swapped pixel coordinate mapping
        output_width = int(map_width_y * pixels_per_meter)   # Y range → image width
        output_height = int(map_width_x * pixels_per_meter)  # X range → image height

        logger.info(f"Creating stitched image: {output_width} x {output_height} pixels")

        # Determine number of channels from first valid image
        channels = 4  # Default BGRA
        for img_path in image_paths:
            if img_path is not None:
                temp_img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                if temp_img is not None:
                    channels = temp_img.shape[2] if len(temp_img.shape) == 3 else 1
                    del temp_img
                    break

        # Calculate overlap in pixels
        overlap_pixels = int(coverage_width * overlap * pixels_per_meter)
        logger.info(f"Using feather blending with {overlap_pixels} pixel overlap")

        # Create float32 accumulation buffers for weighted blending
        if channels > 1:
            stitched_float = np.zeros((output_height, output_width, channels), dtype=np.float32)
        else:
            stitched_float = np.zeros((output_height, output_width), dtype=np.float32)

        weight_map = np.zeros((output_height, output_width), dtype=np.float32)

        # Place each image on the canvas with weighted blending
        for img_path, (cam_x, cam_y) in zip(image_paths, grid_positions):
            if img_path is None:
                logger.warning(f"Skipping None image at ({cam_x:.1f}, {cam_y:.1f})")
                continue

            # Load image from file
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Failed to load image from {img_path}")
                continue

            # Calculate pixel position in output image
            # CRITICAL: In overhead view (pitch=-90°), CARLA axes map to image axes as:
            # - CARLA Y-axis (left-right in world) → image X-axis (left-right in image)
            # - CARLA X-axis (forward-back in world) → image Y-axis (top-bottom in image)
            #
            # Mapping verified by user tile layout [2,3 / 0,1]:
            # - Horizontal (pixel_x): smaller Y_carla → left side (0)
            # - Vertical (pixel_y): larger X_carla → top (0), using RIGHT edge of tile
            #
            # Use RIGHT edge (cam_x + coverage_width/2) for pixel_y to align tiles correctly:
            # - tile_2 right edge (503) > max_x (446) → negative pixel_y → clipped to 0 (top)
            # - tile_0 right edge (248) < max_x (446) → positive pixel_y (middle-bottom)
            pixel_x = int((cam_y - coverage_width/2 - min_y) * pixels_per_meter)
            pixel_y = int((max_x - (cam_x + coverage_width/2)) * pixels_per_meter)

            # Calculate how much of the image to use (may be clipped at boundaries)
            img_height, img_width = img.shape[:2]

            # Create blend weight mask for this tile
            tile_weight = self.create_blend_mask(img.shape[:2], overlap_pixels)

            # Source region (from captured image)
            src_x1, src_y1 = 0, 0
            src_x2, src_y2 = img_width, img_height

            # Destination region (in stitched image)
            dst_x1, dst_y1 = pixel_x, pixel_y
            dst_x2, dst_y2 = pixel_x + img_width, pixel_y + img_height

            # Clip to output bounds with safer integer conversion
            if dst_x1 < 0:
                src_x1 = int(src_x1 - dst_x1)
                dst_x1 = 0
            if dst_y1 < 0:
                src_y1 = int(src_y1 - dst_y1)
                dst_y1 = 0
            if dst_x2 > output_width:
                src_x2 = int(src_x2 - (dst_x2 - output_width))
                dst_x2 = output_width
            if dst_y2 > output_height:
                src_y2 = int(src_y2 - (dst_y2 - output_height))
                dst_y2 = output_height

            # Ensure source indices are valid
            src_x1 = max(0, min(src_x1, img_width))
            src_x2 = max(0, min(src_x2, img_width))
            src_y1 = max(0, min(src_y1, img_height))
            src_y2 = max(0, min(src_y2, img_height))

            # Ensure destination indices are valid
            dst_x1 = max(0, min(dst_x1, output_width))
            dst_x2 = max(0, min(dst_x2, output_width))
            dst_y1 = max(0, min(dst_y1, output_height))
            dst_y2 = max(0, min(dst_y2, output_height))

            # Skip if no valid region to copy
            if dst_x1 >= dst_x2 or dst_y1 >= dst_y2 or src_x1 >= src_x2 or src_y1 >= src_y2:
                logger.debug(f"Skipping invalid region at ({cam_x:.1f}, {cam_y:.1f})")
                del img, tile_weight
                continue

            # Extract source region and corresponding weight mask
            try:
                src_region = img[src_y1:src_y2, src_x1:src_x2]
                weight_roi = tile_weight[src_y1:src_y2, src_x1:src_x2]

                # Ensure dimensions match
                dst_h = dst_y2 - dst_y1
                dst_w = dst_x2 - dst_x1
                src_h = src_y2 - src_y1
                src_w = src_x2 - src_x1

                if dst_h != src_h or dst_w != src_w:
                    # Resize if needed (shouldn't happen with correct math, but safety check)
                    src_region = cv.resize(src_region, (dst_w, dst_h), interpolation=cv.INTER_NEAREST)
                    weight_roi = cv.resize(weight_roi, (dst_w, dst_h), interpolation=cv.INTER_LINEAR)

                # Apply weighted blending: accumulate weighted pixels
                if channels > 1:
                    # Multi-channel: expand weight to match channels
                    stitched_float[dst_y1:dst_y2, dst_x1:dst_x2] += src_region.astype(np.float32) * weight_roi[..., np.newaxis]
                else:
                    # Single channel
                    stitched_float[dst_y1:dst_y2, dst_x1:dst_x2] += src_region.astype(np.float32) * weight_roi

                # Accumulate weights
                weight_map[dst_y1:dst_y2, dst_x1:dst_x2] += weight_roi

                logger.debug(f"Blended image at ({cam_x:.1f}, {cam_y:.1f}): dst[{dst_y1}:{dst_y2}, {dst_x1}:{dst_x2}]")

            except Exception as e:
                logger.error(f"Failed to blend image at ({cam_x:.1f}, {cam_y:.1f}): {e}")
                logger.error(f"  src: [{src_y1}:{src_y2}, {src_x1}:{src_x2}], dst: [{dst_y1}:{dst_y2}, {dst_x1}:{dst_x2}]")

            finally:
                # Release image memory after processing this tile
                del img, tile_weight

        # Normalize by weight map to get final blended image
        logger.info("Normalizing weighted accumulation...")
        weight_map[weight_map == 0] = 1  # Avoid division by zero

        if channels > 1:
            stitched = (stitched_float / weight_map[..., np.newaxis]).astype(np.uint8)
        else:
            stitched = (stitched_float / weight_map).astype(np.uint8)

        logger.info("Image stitching with feather blending completed")
        return stitched

    def save_outputs(self, rgb_stitched: np.ndarray, depth_stitched: np.ndarray,
                    depth_normalized_stitched: np.ndarray, semantic_stitched: np.ndarray,
                    output_path: str, metadata: Dict):
        """
        Save stitched images and metadata to disk.

        Args:
            rgb_stitched: Stitched RGB image
            depth_stitched: Stitched depth image
            depth_normalized_stitched: Stitched normalized depth grayscale image
            semantic_stitched: Stitched semantic image
            output_path: Output directory path
            metadata: Metadata dictionary
        """
        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Save images
        rgb_path = os.path.join(output_path, f"{self.map_name}_rgb.png")
        depth_path = os.path.join(output_path, f"{self.map_name}_depth.png")
        depth_normalized_path = os.path.join(output_path, f"{self.map_name}_depth_normalized.png")
        semantic_path = os.path.join(output_path, f"{self.map_name}_semantic.png")

        logger.info(f"Saving RGB image to {rgb_path}")
        # RGB image is in BGRA format, OpenCV saves as-is
        cv.imwrite(rgb_path, rgb_stitched)

        logger.info(f"Saving Depth image to {depth_path}")
        cv.imwrite(depth_path, depth_stitched)

        logger.info(f"Saving Normalized Depth image to {depth_normalized_path}")
        cv.imwrite(depth_normalized_path, depth_normalized_stitched)

        logger.info(f"Saving Semantic image to {semantic_path}")
        cv.imwrite(semantic_path, semantic_stitched)

        # Save metadata
        metadata_path = os.path.join(output_path, "metadata.json")
        logger.info(f"Saving metadata to {metadata_path}")

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info("All outputs saved successfully")

    def cleanup(self):
        """Restore CARLA settings and cleanup."""
        logger.info("Restoring CARLA settings...")
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            logger.info("Synchronous mode disabled")
        except Exception as e:
            logger.warning(f"Failed to restore settings: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture bird's eye view (BEV) of CARLA maps with RGB, Depth, and Semantic images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture Town02 map with default settings (FOV 60°)
  python tools/capture_map_bev.py --map Town02 --output ./output

  # Test different FOV values to control distortion
  python tools/capture_map_bev.py --map Town01 --fov 50 --output ./test_fov50  # Less distortion, more tiles
  python tools/capture_map_bev.py --map Town01 --fov 70 --output ./test_fov70  # More distortion, fewer tiles

  # High resolution capture with custom parameters
  python tools/capture_map_bev.py --map Town01 --output ./output --height 200 --resolution 4000 --overlap 0.2
        """
    )

    parser.add_argument('--host', default='localhost',
                       help='CARLA server IP address (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    parser.add_argument('--map', dest='map_name', default=None,
                       help='Map name to load (default: use current map)')
    parser.add_argument('--output', required=True,
                       help='Output directory path (required)')
    parser.add_argument('--height', type=float, default=120.0,
                       help='Camera height in meters (default: 150)')
    parser.add_argument('--fov', type=float, default=60.0,
                       help='Camera field of view in degrees (default: 60, lower FOV reduces distortion)')
    parser.add_argument('--overlap', type=float, default=0.15,
                       help='Overlap ratio between cameras 0.0-1.0 (default: 0.15)')
    parser.add_argument('--resolution', type=int, default=2000,
                       help='Camera resolution in pixels (default: 2000)')
    parser.add_argument('--margin', type=float, default=50.0,
                       help='Margin around map bounds in meters (default: 50)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.height <= 0:
        parser.error("Height must be positive")
    if not (0.0 <= args.overlap < 1.0):
        parser.error("Overlap must be between 0.0 and 1.0")
    if args.resolution < 100:
        parser.error("Resolution must be at least 100 pixels")
    if not (10.0 <= args.fov <= 120.0):
        parser.error("FOV must be between 10 and 120 degrees")

    # Initialize capture tool
    capture = None
    temp_dir = None
    try:
        logger.info("=" * 70)
        logger.info("CARLA Map Bird's Eye View (BEV) Capture Tool")
        logger.info("=" * 70)

        capture = MapBEVCapture(args.host, args.port, args.map_name)

        # Calculate map bounds
        bounds = capture.calculate_map_bounds(margin=args.margin)

        # Generate camera grid
        grid_positions, coverage_width, step = capture.generate_camera_grid(
            bounds, args.height, args.fov, args.resolution, args.overlap
        )

        # Capture images at all grid positions
        captured_data = capture.capture_at_grid_positions(
            grid_positions, args.height, args.fov, args.resolution
        )

        # Track temp directory for cleanup
        temp_dir = captured_data.get('temp_dir')

        # Explicitly run garbage collection before stitching
        gc.collect()
        logger.info("Garbage collection completed")

        # Save individual camera images for debugging/verification
        logger.info("Saving individual camera images for verification...")
        os.makedirs(args.output, exist_ok=True)

        for idx, (rgb_path, depth_path, semantic_path, (cam_x, cam_y)) in enumerate(zip(
            captured_data['rgb_paths'],
            captured_data['depth_paths'],
            captured_data['semantic_paths'],
            grid_positions
        )):
            # Copy individual images to output directory with position info
            if rgb_path:
                shutil.copy(rgb_path, f"{args.output}/tile_{idx}_x{int(cam_x)}_y{int(cam_y)}_rgb.png")
            if depth_path:
                shutil.copy(depth_path, f"{args.output}/tile_{idx}_x{int(cam_x)}_y{int(cam_y)}_depth.png")
            if semantic_path:
                shutil.copy(semantic_path, f"{args.output}/tile_{idx}_x{int(cam_x)}_y{int(cam_y)}_semantic.png")
            logger.info(f"Saved tile {idx}: position ({cam_x:.1f}, {cam_y:.1f})")

        logger.info(f"Individual tiles saved to {args.output}")

        # Stitch images from file paths
        logger.info("Stitching RGB images...")
        rgb_stitched = capture.stitch_images(
            captured_data['rgb_paths'], grid_positions, bounds,
            coverage_width, step, args.resolution, args.overlap
        )

        logger.info("Stitching Depth images...")
        depth_stitched = capture.stitch_images(
            captured_data['depth_paths'], grid_positions, bounds,
            coverage_width, step, args.resolution, args.overlap
        )

        logger.info("Stitching Semantic images...")
        semantic_stitched = capture.stitch_images(
            captured_data['semantic_paths'], grid_positions, bounds,
            coverage_width, step, args.resolution, args.overlap
        )

        # Apply global depth normalization after stitching
        logger.info("Applying global depth normalization...")
        depth_normalized_stitched = capture.normalize_depth(depth_stitched)

        # Use standard stitched images
        rgb_final = rgb_stitched
        depth_final = depth_stitched
        semantic_final = semantic_stitched
        depth_normalized_final = depth_normalized_stitched

        # Prepare metadata
        metadata = {
            'map_name': capture.map_name,
            'bounds': {
                'min_x': bounds[0],
                'max_x': bounds[1],
                'min_y': bounds[2],
                'max_y': bounds[3]
            },
            'camera_settings': {
                'height': args.height,
                'fov': args.fov,
                'resolution': args.resolution,
                'overlap': args.overlap
            },
            'grid_info': {
                'num_positions': len(grid_positions),
                'coverage_width': coverage_width,
                'step_size': step
            },
            'output_dimensions': {
                'rgb': {'width': rgb_final.shape[1], 'height': rgb_final.shape[0]},
                'depth': {'width': depth_final.shape[1], 'height': depth_final.shape[0]},
                'depth_normalized': {'width': depth_normalized_final.shape[1], 'height': depth_normalized_final.shape[0]},
                'semantic': {'width': semantic_final.shape[1], 'height': semantic_final.shape[0]}
            },
            'depth_normalization': {
                'enabled': True,
                'method': 'global_adaptive_percentile',
                'range': '1%-99%',
                'strategy': 'post_stitching',
                'description': 'Global depth normalization applied after stitching for unified depth representation across entire map'
            }
        }

        # Save outputs
        capture.save_outputs(rgb_final, depth_final, depth_normalized_final, semantic_final,
                           args.output, metadata)

        logger.info("=" * 70)
        logger.info("Capture completed successfully!")
        logger.info(f"Output saved to: {args.output}")
        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1
    finally:
        # Cleanup CARLA resources
        if capture:
            capture.cleanup()

        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Temporary directory cleaned up: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory: {e}")


if __name__ == '__main__':
    sys.exit(main())
