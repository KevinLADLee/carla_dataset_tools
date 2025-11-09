"""
CARLA Dataset Tools - Core Module

This module contains core shared components used across the project:
- geometry: Geometric types (Vector3d, Location, Rotation, Transform, BoundingBox)
- types: Label and object types
- transform: Coordinate transformation utilities
- converters: Data format converters
- logger: Unified logging system
"""

from .geometry import Vector3d, Location, Rotation, Transform, BoundingBox
from .types import ObjectLabel
from .logger import setup_logger, get_logger, configure_global_logging

__all__ = [
    'Vector3d',
    'Location',
    'Rotation',
    'Transform',
    'BoundingBox',
    'ObjectLabel',
    'setup_logger',
    'get_logger',
    'configure_global_logging',
]
