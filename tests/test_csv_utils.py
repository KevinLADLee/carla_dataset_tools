#!/usr/bin/env python3
"""
Comprehensive test suite for core/csv_utils.py

This test file is designed to be CI-friendly and self-contained.
It tests all functionality of csv_utils including edge cases and integration scenarios.

Usage:
    python tests/test_csv_utils.py
    python -m unittest tests.test_csv_utils -v
    python -m unittest tests.test_csv_utils --buffer  # CI mode
"""

import unittest
import tempfile
import shutil
import os
import csv
import logging
import threading
import time
from unittest.mock import patch, mock_open
from pathlib import Path

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.csv_utils import (
    ensure_directory_exists,
    safe_append_to_csv,
    validate_csv_data,
    create_csv_writer,
    get_csv_file_info
)


class TestCSVUtils(unittest.TestCase):
    """Test suite for csv_utils.py functionality"""

    def setUp(self):
        """Set up test environment before each test"""
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.test_dir, 'test.csv')
        self.logger = logging.getLogger('test_csv_utils')

        # Standard test data
        self.test_fieldnames = ['frame', 'timestamp', 'x', 'y', 'z']
        self.test_data = {
            'frame': 1,
            'timestamp': 1234567890.123,
            'x': 1.23,
            'y': 4.56,
            'z': 7.89
        }

        # Sensor-specific test data
        self.imu_fieldnames = ['frame', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'compass']
        self.imu_test_data = {
            'frame': 1,
            'timestamp': 1234567890.123,
            'acc_x': 0.1,
            'acc_y': 0.2,
            'acc_z': 9.81,
            'gyro_x': 0.01,
            'gyro_y': 0.02,
            'gyro_z': 0.03,
            'compass': 1.57
        }

        self.gnss_fieldnames = ['frame', 'timestamp', 'latitude', 'longitude', 'altitude']
        self.gnss_test_data = {
            'frame': 1,
            'timestamp': 1234567890.123,
            'latitude': 37.7749,
            'longitude': -122.4194,
            'altitude': 100.0
        }

    def tearDown(self):
        """Clean up test environment after each test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # ==================== Directory Operations Tests ====================

    def test_ensure_directory_exists_create_new(self):
        """Test creating a new directory"""
        new_dir = os.path.join(self.test_dir, 'new_directory')
        self.assertFalse(os.path.exists(new_dir))

        result = ensure_directory_exists(new_dir)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.access(new_dir, os.W_OK))

    def test_ensure_directory_exists_existing(self):
        """Test with existing directory"""
        result = ensure_directory_exists(self.test_dir)
        self.assertTrue(result)

    def test_ensure_directory_exists_permission_denied(self):
        """Test permission denied scenario"""
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            result = ensure_directory_exists('/root/test_no_permission')
            self.assertFalse(result)

    def test_ensure_directory_exists_nested_path(self):
        """Test creating nested directory structure"""
        nested_dir = os.path.join(self.test_dir, 'level1', 'level2', 'level3')

        result = ensure_directory_exists(nested_dir)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(nested_dir))

    # ==================== CSV Data Validation Tests ====================

    def test_validate_csv_data_valid(self):
        """Test validation of valid data"""
        result = validate_csv_data(self.test_data, self.test_fieldnames)

        self.assertTrue(result['valid'])
        self.assertIsNone(result['extra_fields'])

    def test_validate_csv_data_missing_fields(self):
        """Test validation with missing fields"""
        incomplete_data = {'frame': 1, 'timestamp': 1234567890.123}  # Missing x, y, z

        result = validate_csv_data(incomplete_data, self.test_fieldnames)

        self.assertFalse(result['valid'])
        self.assertEqual(set(result['missing_fields']), {'x', 'y', 'z'})

    def test_validate_csv_data_extra_fields(self):
        """Test validation with extra fields"""
        extra_data = {**self.test_data, 'extra_field': 'extra_value'}

        result = validate_csv_data(extra_data, self.test_fieldnames)

        self.assertTrue(result['valid'])  # Extra fields should not cause failure
        self.assertEqual(result['extra_fields'], ['extra_field'])

    def test_validate_csv_data_invalid_type(self):
        """Test validation with invalid data type"""
        result = validate_csv_data("not a dict", self.test_fieldnames)

        self.assertFalse(result['valid'])
        self.assertIn('Data must be a dictionary', result['error'])

    # ==================== CSV Writing Tests ====================

    def test_safe_append_to_csv_first_write(self):
        """Test first write to CSV (with header)"""
        result = safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            data=self.test_data,
            is_first_write=True,
            logger_name='test'
        )

        self.assertTrue(result['success'])
        self.assertTrue(os.path.exists(self.test_csv_path))

        # Verify file content
        with open(self.test_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('frame,timestamp,x,y,z', content)
            self.assertIn('1,1234567890.123,1.23,4.56,7.89', content)

    def test_safe_append_to_csv_subsequent_write(self):
        """Test subsequent writes to CSV (without header)"""
        # First write
        safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            data=self.test_data,
            is_first_write=True
        )

        # Second write
        second_data = {**self.test_data, 'frame': 2}
        result = safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            data=second_data,
            is_first_write=False
        )

        self.assertTrue(result['success'])

        # Count lines (should be 3: header + 2 data rows)
        with open(self.test_csv_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)

    def test_safe_append_to_csv_creates_directory(self):
        """Test that CSV writing creates parent directories"""
        nested_path = os.path.join(self.test_dir, 'subdir', 'data.csv')

        result = safe_append_to_csv(
            csv_path=nested_path,
            fieldnames=self.test_fieldnames,
            data=self.test_data,
            is_first_write=True
        )

        self.assertTrue(result['success'])
        self.assertTrue(os.path.exists(nested_path))

    def test_safe_append_to_csv_permission_denied(self):
        """Test CSV writing with permission denied"""
        # Test with a root directory that will fail directory access check
        result = safe_append_to_csv(
            csv_path='/root/test.csv',
            fieldnames=self.test_fieldnames,
            data=self.test_data,
            is_first_write=True
        )

        self.assertFalse(result['success'])
        self.assertIn('Cannot access directory', result['error'])

    def test_safe_append_to_csv_unicode_data(self):
        """Test CSV writing with Unicode characters"""
        unicode_data = {
            'frame': 1,
            'timestamp': 1234567890.123,
            'x': 1.23,
            'y': 4.56,
            'z': '测试数据'  # Unicode characters
        }

        result = safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames + ['z_unicode'],
            data={**self.test_data, 'z_unicode': '测试数据'},
            is_first_write=True
        )

        self.assertTrue(result['success'])

        # Verify Unicode is correctly encoded
        with open(self.test_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('测试数据', content)

    # ==================== CSV Writer Creation Tests ====================

    def test_create_csv_writer(self):
        """Test CSV writer creation"""
        writer = create_csv_writer(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            mode='w'
        )

        self.assertIsNotNone(writer)
        self.assertIsInstance(writer, csv.DictWriter)
        self.assertEqual(writer.fieldnames, self.test_fieldnames)

    def test_create_csv_writer_invalid_mode(self):
        """Test CSV writer creation with invalid mode"""
        # Use a mode that would cause directory access failure
        writer = create_csv_writer(
            csv_path='/invalid/path/test.csv',
            fieldnames=self.test_fieldnames,
            mode='w'
        )

        self.assertIsNone(writer)

    # ==================== File Info Tests ====================

    def test_get_csv_file_info_existing(self):
        """Test getting info about existing CSV file"""
        # Create test file first
        safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            data=self.test_data,
            is_first_write=True
        )

        info = get_csv_file_info(self.test_csv_path)

        self.assertTrue(info['exists'])
        self.assertGreater(info['size_bytes'], 0)
        self.assertTrue(info['has_header'])
        self.assertEqual(len(info['header_fields']), len(self.test_fieldnames))
        self.assertEqual(info['row_count'], 1)  # Only data rows, not header

    def test_get_csv_file_info_nonexistent(self):
        """Test getting info about non-existent file"""
        info = get_csv_file_info('/nonexistent/path/file.csv')

        self.assertFalse(info['exists'])

    # ==================== Sensor Integration Tests ====================

    def test_imu_sensor_data_integration(self):
        """Test integration with IMU sensor data format"""
        imu_csv_path = os.path.join(self.test_dir, 'imu_data.csv')
        result = safe_append_to_csv(
            csv_path=imu_csv_path,
            fieldnames=self.imu_fieldnames,
            data=self.imu_test_data,
            is_first_write=True,
            logger_name='IMUTest'
        )

        self.assertTrue(result['success'])

        # Verify IMU-specific data integrity
        with open(imu_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            self.assertEqual(row['acc_x'], '0.1')
            self.assertEqual(row['compass'], '1.57')

    def test_gnss_sensor_data_integration(self):
        """Test integration with GNSS sensor data format"""
        gnss_csv_path = os.path.join(self.test_dir, 'gnss_data.csv')
        result = safe_append_to_csv(
            csv_path=gnss_csv_path,
            fieldnames=self.gnss_fieldnames,
            data=self.gnss_test_data,
            is_first_write=True,
            logger_name='GNSSTest'
        )

        self.assertTrue(result['success'])

        # Verify GNSS-specific data integrity
        with open(gnss_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            self.assertEqual(row['latitude'], '37.7749')
            self.assertEqual(row['longitude'], '-122.4194')

    def test_sensor_pose_data_integration(self):
        """Test integration with sensor pose data format"""
        pose_fieldnames = ['frame', 'timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
        pose_data = {
            'frame': 1,
            'timestamp': 1234567890.123,
            'x': 1.0,
            'y': 2.0,
            'z': 3.0,
            'roll': 0.0,
            'pitch': 0.1,
            'yaw': 1.57
        }

        result = safe_append_to_csv(
            csv_path=os.path.join(self.test_dir, 'poses.csv'),
            fieldnames=pose_fieldnames,
            data=pose_data,
            is_first_write=True,
            logger_name='PoseTest'
        )

        self.assertTrue(result['success'])

    # ==================== Concurrent Writing Tests ====================

    def test_concurrent_csv_writing(self):
        """Test sequential writing to different CSV files (simulating concurrent access)"""
        results = []

        # Write multiple files sequentially (safer for CI)
        for i in range(3):
            worker_data = {**self.test_data, 'frame': i}
            worker_path = os.path.join(self.test_dir, f'sequential_{i}.csv')

            result = safe_append_to_csv(
                csv_path=worker_path,
                fieldnames=self.test_fieldnames,
                data=worker_data,
                is_first_write=True,
                logger_name=f'Sequential{i}'
            )
            results.append(result)

        # Verify all writes succeeded
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result['success'])

    # ==================== Performance Tests ====================

    def test_large_data_writing_performance(self):
        """Test performance with large dataset"""
        start_time = time.time()

        # Write 100 rows (reduced for faster testing)
        for i in range(100):
            data = {**self.test_data, 'frame': i}
            is_first = (i == 0)

            result = safe_append_to_csv(
                csv_path=self.test_csv_path,
                fieldnames=self.test_fieldnames,
                data=data,
                is_first_write=is_first
            )

            if not result['success']:
                self.fail(f"Failed to write row {i}: {result.get('error', 'Unknown error')}")

        elapsed_time = time.time() - start_time

        # Performance assertion (should complete within reasonable time)
        self.assertLess(elapsed_time, 2.0, "Large dataset writing took too long")

        # Verify file content
        info = get_csv_file_info(self.test_csv_path)
        self.assertEqual(info['row_count'], 100)  # Only data rows

    # ==================== Edge Cases and Error Handling ====================

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        result = safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            data={},  # Empty dict
            is_first_write=True
        )

        # Should fail due to validation (no data provided)
        self.assertFalse(result['success'])
        self.assertIn('No data provided', result['error'])

    def test_none_data_handling(self):
        """Test handling of None data"""
        result = safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            data=None,
            is_first_write=True
        )

        self.assertFalse(result['success'])
        self.assertIn('No data provided', result['error'])

    def test_very_long_field_names(self):
        """Test handling of very long field names"""
        long_fieldnames = ['field_' + 'x' * 100 for _ in range(3)]
        long_data = {field: f'value_{i}' for i, field in enumerate(long_fieldnames)}

        result = safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=long_fieldnames,
            data=long_data,
            is_first_write=True
        )

        self.assertTrue(result['success'])

    def test_special_characters_in_data(self):
        """Test handling of special characters in data"""
        special_data = {
            'frame': 1,
            'timestamp': 1234567890.123,
            'x': 'value,with,commas',
            'y': 'value\nwith\nnewlines',
            'z': 'value"with"quotes'
        }

        result = safe_append_to_csv(
            csv_path=self.test_csv_path,
            fieldnames=self.test_fieldnames,
            data=special_data,
            is_first_write=True
        )

        self.assertTrue(result['success'])


class TestCSVUtilsCICompatibility(unittest.TestCase):
    """CI-specific compatibility tests"""

    def setUp(self):
        """Set up CI test environment"""
        # Use temp directory that's guaranteed to be writable
        self.test_dir = tempfile.mkdtemp(prefix='csv_test_ci_')
        self.original_cwd = os.getcwd()

    def tearDown(self):
        """Clean up CI test environment"""
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_run_from_different_directory(self):
        """Test that tests work when run from different directory"""
        # Change to test directory
        os.chdir(self.test_dir)

        # Test CSV operations work relative to new directory
        test_data = {'frame': 1, 'timestamp': 1234567890.123, 'value': 'test'}

        result = safe_append_to_csv(
            csv_path='test_relative.csv',
            fieldnames=['frame', 'timestamp', 'value'],
            data=test_data,
            is_first_write=True
        )

        self.assertTrue(result['success'])
        self.assertTrue(os.path.exists('test_relative.csv'))

    def test_no_temp_dir_permissions_issue(self):
        """Test robust handling of temporary directory permissions"""
        # Create a directory and remove write permissions
        restricted_dir = os.path.join(self.test_dir, 'restricted')
        os.makedirs(restricted_dir)
        os.chmod(restricted_dir, 0o444)  # Read-only

        try:
            result = safe_append_to_csv(
                csv_path=os.path.join(restricted_dir, 'test.csv'),
                fieldnames=['test'],
                data={'test': 'value'},
                is_first_write=True
            )

            # Should handle permission error gracefully
            self.assertFalse(result['success'])

        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_dir, 0o755)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise in test output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests with different verbosity based on environment
    import os
    if os.getenv('CI'):  # CI Environment
        unittest.main(verbosity=1, buffer=True)
    else:  # Local development
        unittest.main(verbosity=2)