#!/usr/bin/env python3
"""
Unit tests for coordinate transformation utilities

Tests verify:
- Conversion between CARLA (left-handed) and custom (right-handed) coordinate systems
- Location, Rotation, and Transform conversions
- Roundtrip conversion consistency
- Point transformation operations
- Vector3D and BoundingBox conversions
"""

import sys
import unittest
import math
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import carla
import numpy as np

from core.transform import (
    carla_location_to_numpy_vec,
    carla_location_to_location,
    carla_rotation_to_RPY,
    carla_rotation_to_rotation,
    carla_transform_to_transform,
    carla_vec3d_to_numpy_vec,
    carla_vec3d_to_vec3d,
    RPY_to_carla_rotation,
    rotation_to_carla_rotation,
    location_to_carla_location,
    transform_to_carla_transform,
    carla_bbox_to_bbox,
    bbox_to_o3d_bbox
)
from core.geometry import Location, Rotation, Transform, Vector3d, BoundingBox


class TestCoordinateTransformation(unittest.TestCase):
    """Test suite for coordinate transformation between CARLA and custom types"""

    def setUp(self):
        """Set up test fixtures with sample data"""
        # Sample CARLA types (left-handed coordinate system)
        self.carla_location = carla.Location(1.0, 2.0, 3.0)
        self.carla_rotation = carla.Rotation(pitch=30.0, yaw=60.0, roll=90.0)
        self.carla_transform = carla.Transform(self.carla_location, self.carla_rotation)
        self.carla_vec3d = carla.Vector3D(4.0, 5.0, 6.0)

        # Sample custom types (right-handed coordinate system)
        self.location = Location(1.0, -2.0, 3.0)
        self.rotation = Rotation(pitch=-30.0, yaw=-60.0, roll=90.0)
        self.transform = Transform(self.location, self.rotation)

        # Tolerance for floating point comparisons
        self.tolerance = 1e-6

    def test_carla_location_to_location_conversion(self):
        """Test conversion from CARLA Location to custom Location"""
        location = carla_location_to_location(self.carla_location)

        # Verify coordinate system conversion (y-axis is inverted)
        self.assertAlmostEqual(location.x, 1.0, places=6)
        self.assertAlmostEqual(location.y, -2.0, places=6)
        self.assertAlmostEqual(location.z, 3.0, places=6)

    def test_carla_location_to_numpy_vec_conversion(self):
        """Test conversion from CARLA Location to numpy array"""
        vec = carla_location_to_numpy_vec(self.carla_location)

        # Check shape and values
        self.assertEqual(vec.shape, (3, 1))
        self.assertAlmostEqual(vec[0, 0], 1.0, places=6)
        self.assertAlmostEqual(vec[1, 0], -2.0, places=6)
        self.assertAlmostEqual(vec[2, 0], 3.0, places=6)

    def test_carla_rotation_to_RPY(self):
        """Test conversion from CARLA Rotation to RPY tuple"""
        roll, pitch, yaw = carla_rotation_to_RPY(self.carla_rotation)

        # Verify rotation conversion (pitch and yaw are inverted)
        self.assertAlmostEqual(roll, 90.0, places=6)
        self.assertAlmostEqual(pitch, -30.0, places=6)
        self.assertAlmostEqual(yaw, -60.0, places=6)

    def test_carla_rotation_to_rotation_conversion(self):
        """Test conversion from CARLA Rotation to custom Rotation"""
        rotation = carla_rotation_to_rotation(self.carla_rotation)

        # Verify rotation values
        self.assertAlmostEqual(rotation.roll, 90.0, places=6)
        self.assertAlmostEqual(rotation.pitch, -30.0, places=6)
        self.assertAlmostEqual(rotation.yaw, -60.0, places=6)

    def test_carla_transform_to_transform_conversion(self):
        """Test conversion from CARLA Transform to custom Transform"""
        transform = carla_transform_to_transform(self.carla_transform)

        # Verify location conversion
        self.assertAlmostEqual(transform.location.x, 1.0, places=6)
        self.assertAlmostEqual(transform.location.y, -2.0, places=6)
        self.assertAlmostEqual(transform.location.z, 3.0, places=6)

        # Verify rotation conversion
        self.assertAlmostEqual(transform.rotation.roll, 90.0, places=6)
        self.assertAlmostEqual(transform.rotation.pitch, -30.0, places=6)
        self.assertAlmostEqual(transform.rotation.yaw, -60.0, places=6)

    def test_location_roundtrip_conversion(self):
        """Test that Location conversion is reversible (roundtrip)"""
        # CARLA -> Custom -> CARLA
        custom_location = carla_location_to_location(self.carla_location)
        carla_location_back = location_to_carla_location(custom_location)

        # Verify original values are preserved
        self.assertAlmostEqual(carla_location_back.x, self.carla_location.x, places=6)
        self.assertAlmostEqual(carla_location_back.y, self.carla_location.y, places=6)
        self.assertAlmostEqual(carla_location_back.z, self.carla_location.z, places=6)

    def test_rotation_roundtrip_conversion(self):
        """Test that Rotation conversion is reversible (roundtrip)"""
        # CARLA -> Custom -> CARLA
        custom_rotation = carla_rotation_to_rotation(self.carla_rotation)
        carla_rotation_back = rotation_to_carla_rotation(custom_rotation)

        # Verify original values are preserved
        self.assertAlmostEqual(carla_rotation_back.roll, self.carla_rotation.roll, places=6)
        self.assertAlmostEqual(carla_rotation_back.pitch, self.carla_rotation.pitch, places=6)
        self.assertAlmostEqual(carla_rotation_back.yaw, self.carla_rotation.yaw, places=6)

    def test_transform_roundtrip_conversion(self):
        """Test that Transform conversion is reversible (roundtrip)"""
        # CARLA -> Custom -> CARLA
        custom_transform = carla_transform_to_transform(self.carla_transform)
        carla_transform_back = transform_to_carla_transform(custom_transform)

        # Verify location is preserved
        self.assertAlmostEqual(carla_transform_back.location.x, self.carla_transform.location.x, places=6)
        self.assertAlmostEqual(carla_transform_back.location.y, self.carla_transform.location.y, places=6)
        self.assertAlmostEqual(carla_transform_back.location.z, self.carla_transform.location.z, places=6)

        # Verify rotation is preserved
        self.assertAlmostEqual(carla_transform_back.rotation.roll, self.carla_transform.rotation.roll, places=6)
        self.assertAlmostEqual(carla_transform_back.rotation.pitch, self.carla_transform.rotation.pitch, places=6)
        self.assertAlmostEqual(carla_transform_back.rotation.yaw, self.carla_transform.rotation.yaw, places=6)

    def test_coordinate_system_y_axis_inversion(self):
        """Test that y-axis is inverted during coordinate system conversion"""
        # Left-handed (CARLA) to right-handed (custom)
        carla_loc = carla.Location(10.0, 20.0, 30.0)
        custom_loc = carla_location_to_location(carla_loc)

        # Y should be inverted, X and Z unchanged
        self.assertEqual(custom_loc.x, 10.0)
        self.assertEqual(custom_loc.y, -20.0)
        self.assertEqual(custom_loc.z, 30.0)

    def test_transform_point_consistency(self):
        """Test that point transformation is consistent between CARLA and custom types"""
        # Point in reference frame
        point = Location(3.0, 2.0, 1.0)

        # Reference frame in world coordinate
        ref_coord = Transform(
            Location(1.0, 2.0, 3.0),
            Rotation(pitch=30.0, roll=60.0, yaw=90.0)
        )

        # Transform point using custom type
        trans_point_custom = ref_coord.transform(point)

        # Convert to CARLA and transform using CARLA type
        point_carla = location_to_carla_location(point)
        ref_coord_carla = transform_to_carla_transform(ref_coord)
        trans_point_carla = ref_coord_carla.transform(point_carla)

        # Convert back to custom type
        trans_point_from_carla = carla_location_to_location(trans_point_carla)

        # Both transformations should yield the same result
        self.assertAlmostEqual(trans_point_custom.x, trans_point_from_carla.x, places=5)
        self.assertAlmostEqual(trans_point_custom.y, trans_point_from_carla.y, places=5)
        self.assertAlmostEqual(trans_point_custom.z, trans_point_from_carla.z, places=5)

    def test_carla_vec3d_to_numpy_vec_left_to_right(self):
        """Test Vector3D to numpy conversion with coordinate system conversion"""
        vec = carla_vec3d_to_numpy_vec(self.carla_vec3d, left_to_right_hand=True)

        # Check shape and values (y should be inverted)
        self.assertEqual(vec.shape, (3, 1))
        self.assertAlmostEqual(vec[0, 0], 4.0, places=6)
        self.assertAlmostEqual(vec[1, 0], -5.0, places=6)
        self.assertAlmostEqual(vec[2, 0], 6.0, places=6)

    def test_carla_vec3d_to_numpy_vec_no_conversion(self):
        """Test Vector3D to numpy conversion without coordinate system conversion"""
        vec = carla_vec3d_to_numpy_vec(self.carla_vec3d, left_to_right_hand=False)

        # Check shape and values (y should NOT be inverted)
        self.assertEqual(vec.shape, (3, 1))
        self.assertAlmostEqual(vec[0, 0], 4.0, places=6)
        self.assertAlmostEqual(vec[1, 0], 5.0, places=6)
        self.assertAlmostEqual(vec[2, 0], 6.0, places=6)

    def test_carla_vec3d_to_vec3d_conversion(self):
        """Test conversion from CARLA Vector3D to custom Vector3d"""
        vec3d = carla_vec3d_to_vec3d(self.carla_vec3d)

        # Verify coordinate system conversion
        self.assertAlmostEqual(vec3d.x, 4.0, places=6)
        self.assertAlmostEqual(vec3d.y, -5.0, places=6)
        self.assertAlmostEqual(vec3d.z, 6.0, places=6)

    def test_RPY_to_carla_rotation_radians(self):
        """Test conversion from RPY (radians) to CARLA Rotation (degrees)"""
        roll_rad = math.radians(90.0)
        pitch_rad = math.radians(30.0)
        yaw_rad = math.radians(60.0)

        carla_rot = RPY_to_carla_rotation(roll_rad, pitch_rad, yaw_rad)

        # Verify conversion (pitch and yaw inverted, radians to degrees)
        self.assertAlmostEqual(carla_rot.roll, 90.0, places=5)
        self.assertAlmostEqual(carla_rot.pitch, -30.0, places=5)
        self.assertAlmostEqual(carla_rot.yaw, -60.0, places=5)

    def test_carla_bbox_to_bbox_conversion(self):
        """Test conversion from CARLA BoundingBox to custom BoundingBox"""
        carla_bbox = carla.BoundingBox(
            carla.Location(10.0, 20.0, 30.0),
            carla.Vector3D(1.0, 2.0, 3.0)
        )
        carla_bbox.rotation = carla.Rotation(pitch=10.0, yaw=20.0, roll=30.0)

        bbox = carla_bbox_to_bbox(carla_bbox)

        # Verify location conversion
        self.assertAlmostEqual(bbox.location.x, 10.0, places=6)
        self.assertAlmostEqual(bbox.location.y, -20.0, places=6)
        self.assertAlmostEqual(bbox.location.z, 30.0, places=6)

        # Verify extent (no coordinate conversion, just copy)
        self.assertAlmostEqual(bbox.extent.x, 1.0, places=6)
        self.assertAlmostEqual(bbox.extent.y, 2.0, places=6)
        self.assertAlmostEqual(bbox.extent.z, 3.0, places=6)

        # Verify rotation conversion
        self.assertAlmostEqual(bbox.rotation.pitch, -10.0, places=6)
        self.assertAlmostEqual(bbox.rotation.yaw, -20.0, places=6)
        self.assertAlmostEqual(bbox.rotation.roll, 30.0, places=6)

    def test_bbox_to_o3d_bbox_conversion(self):
        """Test conversion from custom BoundingBox to Open3D OrientedBoundingBox"""
        bbox = BoundingBox(
            Location(5.0, 10.0, 15.0),
            Vector3d(1.0, 2.0, 3.0),
            Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        o3d_bbox = bbox_to_o3d_bbox(bbox)

        # Verify center
        center = o3d_bbox.center
        self.assertAlmostEqual(center[0], 5.0, places=6)
        self.assertAlmostEqual(center[1], 10.0, places=6)
        self.assertAlmostEqual(center[2], 15.0, places=6)

        # Verify extent (should be doubled)
        extent = o3d_bbox.extent
        self.assertAlmostEqual(extent[0], 2.0, places=6)
        self.assertAlmostEqual(extent[1], 4.0, places=6)
        self.assertAlmostEqual(extent[2], 6.0, places=6)

    def test_equality_operators(self):
        """Test equality operators for custom types"""
        # Test Location equality
        loc1 = Location(1.0, 2.0, 3.0)
        loc2 = Location(1.0, 2.0, 3.0)
        loc3 = Location(1.0, 2.0, 3.001)

        self.assertEqual(loc1, loc2)
        self.assertNotEqual(loc1, loc3)

        # Test Rotation equality
        rot1 = Rotation(pitch=30.0, yaw=60.0, roll=90.0)
        rot2 = Rotation(pitch=30.0, yaw=60.0, roll=90.0)

        self.assertEqual(rot1, rot2)

        # Test Transform equality
        trans1 = Transform(loc1, rot1)
        trans2 = Transform(loc2, rot2)

        self.assertEqual(trans1, trans2)

    def test_rotation_matrix_consistency(self):
        """Test that rotation matrices are computed consistently"""
        rotation = Rotation(pitch=30.0, yaw=45.0, roll=60.0)

        # Get rotation matrix
        rot_matrix = rotation.get_rotation_matrix()

        # Check matrix shape
        self.assertEqual(rot_matrix.shape, (3, 3))

        # Check that matrix is orthogonal (R * R^T = I)
        identity = np.matmul(rot_matrix, rot_matrix.T)
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=6)

    def test_transform_matrix_consistency(self):
        """Test that transform matrices are computed consistently"""
        transform = Transform(
            Location(10.0, 20.0, 30.0),
            Rotation(pitch=10.0, yaw=20.0, roll=30.0)
        )

        # Get transformation matrix
        trans_matrix = transform.get_matrix()

        # Check matrix shape (4x4 homogeneous)
        self.assertEqual(trans_matrix.shape, (4, 4))

        # Check bottom row is [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(
            trans_matrix[3, :],
            np.array([0.0, 0.0, 0.0, 1.0]),
            decimal=6
        )

        # Check inverse consistency
        inv_matrix = transform.get_inverse_matrix()
        identity = np.matmul(trans_matrix, inv_matrix)
        np.testing.assert_array_almost_equal(identity, np.eye(4), decimal=6)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_zero_location(self):
        """Test conversion of zero location"""
        carla_loc = carla.Location(0.0, 0.0, 0.0)
        custom_loc = carla_location_to_location(carla_loc)

        self.assertEqual(custom_loc.x, 0.0)
        self.assertEqual(custom_loc.y, 0.0)
        self.assertEqual(custom_loc.z, 0.0)

    def test_zero_rotation(self):
        """Test conversion of zero rotation"""
        carla_rot = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        custom_rot = carla_rotation_to_rotation(carla_rot)

        self.assertEqual(custom_rot.pitch, 0.0)
        self.assertEqual(custom_rot.yaw, 0.0)
        self.assertEqual(custom_rot.roll, 0.0)

    def test_large_values(self):
        """Test conversion with large coordinate values"""
        carla_loc = carla.Location(10000.0, 20000.0, 30000.0)
        custom_loc = carla_location_to_location(carla_loc)

        self.assertEqual(custom_loc.x, 10000.0)
        self.assertEqual(custom_loc.y, -20000.0)
        self.assertEqual(custom_loc.z, 30000.0)

    def test_negative_values(self):
        """Test conversion with negative coordinate values"""
        carla_loc = carla.Location(-10.0, -20.0, -30.0)
        custom_loc = carla_location_to_location(carla_loc)

        self.assertEqual(custom_loc.x, -10.0)
        self.assertEqual(custom_loc.y, 20.0)  # Inverted
        self.assertEqual(custom_loc.z, -30.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
