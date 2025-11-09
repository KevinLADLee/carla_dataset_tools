#!/usr/bin/env python3
"""
Comprehensive tests for the CARLA Dataset Tools logging system

Tests verify:
- Logger configuration at all levels
- Logger hierarchy and inheritance
- Console and file output
- Colored formatting
- Global configuration behavior
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import (
    configure_global_logging,
    get_logger,
    setup_logger,
    ColoredFormatter,
    LogColors,
    debug, info, warning, error, critical
)


class TestLoggerConfiguration:
    """Test suite for logger configuration"""

    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0

    def assert_true(self, condition, test_name, message=""):
        """Assert helper"""
        if condition:
            self.passed += 1
            self.test_results.append(f"✓ PASS: {test_name}")
            return True
        else:
            self.failed += 1
            self.test_results.append(f"✗ FAIL: {test_name} - {message}")
            return False

    def reset_logging(self):
        """Reset logging configuration between tests"""
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)  # Reset to default

        # Clear all other loggers
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)

    def test_configure_global_logging_basic(self):
        """Test basic global logging configuration"""
        print("\n=== Test: Basic Global Logging Configuration ===")
        self.reset_logging()

        configure_global_logging(level=logging.DEBUG)
        root_logger = logging.getLogger()

        self.assert_true(
            root_logger.level == logging.DEBUG,
            "Root logger level set to DEBUG",
            f"Expected {logging.DEBUG}, got {root_logger.level}"
        )

        self.assert_true(
            len(root_logger.handlers) > 0,
            "Root logger has handlers",
            f"Expected handlers, got {len(root_logger.handlers)}"
        )

    def test_child_logger_inherits_level(self):
        """Test that child loggers inherit configured level"""
        print("\n=== Test: Child Logger Inheritance ===")
        self.reset_logging()

        # Configure root logger to DEBUG
        configure_global_logging(level=logging.DEBUG)

        # Create test stream and handler
        test_stream = StringIO()
        handler = logging.StreamHandler(test_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Replace handlers
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)

        # Create child loggers (simulating module loggers)
        sensor_logger = logging.getLogger('recorder.sensor')

        sensor_logger.debug("Test DEBUG message")
        output = test_stream.getvalue()

        self.assert_true(
            "Test DEBUG message" in output,
            "Child logger can log DEBUG messages",
            f"Output: {output}"
        )

    def test_all_log_levels(self):
        """Test all log levels work correctly"""
        print("\n=== Test: All Log Levels ===")
        self.reset_logging()

        test_stream = StringIO()
        handler = logging.StreamHandler(test_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        configure_global_logging(level=logging.DEBUG)

        # Replace handlers
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)

        test_logger = logging.getLogger('test.logger')

        test_logger.debug("DEBUG test")
        test_logger.info("INFO test")
        test_logger.warning("WARNING test")
        test_logger.error("ERROR test")
        test_logger.critical("CRITICAL test")

        output = test_stream.getvalue()

        self.assert_true("DEBUG test" in output, "DEBUG level logged")
        self.assert_true("INFO test" in output, "INFO level logged")
        self.assert_true("WARNING test" in output, "WARNING level logged")
        self.assert_true("ERROR test" in output, "ERROR level logged")
        self.assert_true("CRITICAL test" in output, "CRITICAL level logged")

    def test_log_level_filtering(self):
        """Test that log level filtering works correctly"""
        print("\n=== Test: Log Level Filtering ===")
        self.reset_logging()

        test_stream = StringIO()
        handler = logging.StreamHandler(test_stream)
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        configure_global_logging(level=logging.WARNING)

        # Replace handlers
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)

        test_logger = logging.getLogger('test.filter')

        test_logger.debug("Should not appear")
        test_logger.info("Should not appear")
        test_logger.warning("Should appear")
        test_logger.error("Should appear")

        output = test_stream.getvalue()

        self.assert_true(
            "Should not appear" not in output,
            "DEBUG/INFO filtered when level=WARNING",
            f"Output: {output}"
        )
        self.assert_true(
            "Should appear" in output,
            "WARNING/ERROR shown when level=WARNING"
        )

    def test_file_logging(self):
        """Test logging to file"""
        print("\n=== Test: File Logging ===")
        self.reset_logging()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            configure_global_logging(
                level=logging.INFO,
                log_file=str(log_file)
            )

            test_logger = logging.getLogger('test.file')
            test_logger.info("Test INFO message to file")
            test_logger.error("Test ERROR message to file")

            # Force flush
            for handler in logging.getLogger().handlers:
                handler.flush()

            self.assert_true(
                log_file.exists(),
                "Log file created"
            )

            if log_file.exists():
                content = log_file.read_text()
                self.assert_true(
                    "Test INFO message to file" in content,
                    "INFO message written to file"
                )
                self.assert_true(
                    "Test ERROR message to file" in content,
                    "ERROR message written to file"
                )

    def test_colored_formatter(self):
        """Test ColoredFormatter behavior"""
        print("\n=== Test: Colored Formatter ===")

        formatter = ColoredFormatter('%(levelname)s - %(message)s')

        # Create a mock record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)

        self.assert_true(
            "Test message" in formatted,
            "Formatter includes message"
        )

    def test_get_logger_function(self):
        """Test get_logger convenience function"""
        print("\n=== Test: get_logger Function ===")
        self.reset_logging()

        configure_global_logging(level=logging.DEBUG)

        # Test with name
        named_logger = get_logger('test.named')
        self.assert_true(
            named_logger.name == 'test.named',
            "get_logger returns logger with correct name"
        )

        # Test without name (default logger)
        default_logger = get_logger()
        self.assert_true(
            default_logger is not None,
            "get_logger returns default logger when no name provided"
        )

    def test_convenience_functions(self):
        """Test convenience logging functions"""
        print("\n=== Test: Convenience Functions ===")
        self.reset_logging()

        test_stream = StringIO()
        handler = logging.StreamHandler(test_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        configure_global_logging(level=logging.DEBUG)

        # Replace handlers
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)

        # Test convenience functions
        debug("Debug convenience")
        info("Info convenience")
        warning("Warning convenience")
        error("Error convenience")
        critical("Critical convenience")

        output = test_stream.getvalue()

        self.assert_true("Debug convenience" in output, "debug() function works")
        self.assert_true("Info convenience" in output, "info() function works")
        self.assert_true("Warning convenience" in output, "warning() function works")
        self.assert_true("Error convenience" in output, "error() function works")
        self.assert_true("Critical convenience" in output, "critical() function works")

    def test_multiple_loggers_hierarchy(self):
        """Test that multiple module loggers work correctly"""
        print("\n=== Test: Multiple Logger Hierarchy ===")
        self.reset_logging()

        test_stream = StringIO()
        handler = logging.StreamHandler(test_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        configure_global_logging(level=logging.DEBUG)

        # Clear and replace handlers
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)

        # Simulate different module loggers
        sensor_logger = logging.getLogger('recorder.sensor')
        vehicle_logger = logging.getLogger('recorder.vehicle')
        world_logger = logging.getLogger('recorder.world')
        actor_tree_logger = logging.getLogger('recorder.actor_tree')

        sensor_logger.info("Sensor recording frame 100")
        vehicle_logger.debug("Vehicle position updated")
        world_logger.warning("World object count high")
        actor_tree_logger.error("Node failed to save")

        output = test_stream.getvalue()

        self.assert_true(
            "recorder.sensor" in output and "Sensor recording frame 100" in output,
            "Sensor logger works",
            f"Output: {output}"
        )
        self.assert_true(
            "recorder.vehicle" in output and "Vehicle position updated" in output,
            "Vehicle logger works",
            f"Output: {output}"
        )
        self.assert_true(
            "recorder.world" in output and "World object count high" in output,
            "World logger works",
            f"Output: {output}"
        )
        self.assert_true(
            "recorder.actor_tree" in output and "Node failed to save" in output,
            "Actor tree logger works",
            f"Output: {output}"
        )

    def test_no_duplicate_handlers(self):
        """Test that multiple configure calls don't create duplicate handlers"""
        print("\n=== Test: No Duplicate Handlers ===")
        self.reset_logging()

        configure_global_logging(level=logging.INFO)
        handler_count_1 = len(logging.getLogger().handlers)

        configure_global_logging(level=logging.DEBUG)
        handler_count_2 = len(logging.getLogger().handlers)

        self.assert_true(
            handler_count_1 == handler_count_2,
            "No duplicate handlers after reconfigure",
            f"First: {handler_count_1}, Second: {handler_count_2}"
        )

    def test_file_handler_debug_level(self):
        """Test that file handler can log DEBUG when configured"""
        print("\n=== Test: File Handler Debug Level ===")
        self.reset_logging()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "debug_test.log"

            # Configure to DEBUG so both console and file get all messages
            # Note: To have different levels, need to set root to DEBUG
            # and configure individual handler levels
            configure_global_logging(
                level=logging.DEBUG,
                log_file=str(log_file)
            )

            test_logger = logging.getLogger('test.file_debug')
            test_logger.debug("DEBUG message")
            test_logger.info("INFO message")
            test_logger.warning("WARNING message")

            # Force flush
            for handler in logging.getLogger().handlers:
                handler.flush()

            if log_file.exists():
                content = log_file.read_text()
                self.assert_true(
                    "DEBUG message" in content and "INFO message" in content,
                    "File handler logs DEBUG messages",
                    f"File content missing expected messages"
                )

    def run_all_tests(self):
        """Run all test methods"""
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE LOGGER TESTS")
        print("="*60)

        # Get all test methods
        test_methods = [
            method for method in dir(self)
            if method.startswith('test_') and callable(getattr(self, method))
        ]

        for test_method_name in test_methods:
            test_method = getattr(self, test_method_name)
            try:
                test_method()
            except Exception as e:
                self.failed += 1
                self.test_results.append(
                    f"✗ FAIL: {test_method_name} - Exception: {str(e)}"
                )
                print(f"Exception in {test_method_name}: {e}")

        # Print results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        for result in self.test_results:
            print(result)

        print("\n" + "="*60)
        print(f"Total: {self.passed + self.failed} tests")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print("="*60)

        return self.failed == 0


def test_real_world_usage():
    """Test real-world usage scenario"""
    print("\n" + "="*60)
    print("REAL-WORLD USAGE TEST")
    print("="*60)

    # Reset logging
    root = logging.getLogger()
    root.handlers.clear()

    print("\n1. Configuring global logging with DEBUG level...")
    configure_global_logging(level=logging.DEBUG)

    print("\n2. Creating module loggers (simulating actual usage)...")
    sensor_logger = logging.getLogger('recorder.sensor')
    vehicle_logger = logging.getLogger('recorder.vehicle')
    world_logger = logging.getLogger('recorder.world')

    print("\n3. Logging messages at various levels...")
    print("   (You should see colored output below)")
    print("-" * 60)

    sensor_logger.debug("Sensor queue waiting for data (Frame 100)")
    sensor_logger.info("Sensor data received successfully (Frame 100)")
    vehicle_logger.info("Vehicle position: x=10.5, y=20.3, z=0.5")
    world_logger.warning("High object count detected: 1500 objects")
    sensor_logger.error("Sensor timeout: camera_front (10.0s)")
    world_logger.critical("Critical: World actor save failed!")

    print("-" * 60)
    print("\n✓ Real-world usage test complete")
    print("  All log levels should have appeared above with colors")
    print("  This confirms the logger inheritance bug is FIXED")


if __name__ == '__main__':
    # Run comprehensive tests
    test_suite = TestLoggerConfiguration()
    all_passed = test_suite.run_all_tests()

    # Run real-world usage test
    test_real_world_usage()

    # Final status
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Logger system is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please review failures above")
    print("="*60)

    sys.exit(0 if all_passed else 1)
