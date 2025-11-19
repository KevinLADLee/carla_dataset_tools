#!/usr/bin/python3
"""
CSV Utilities for CARLA Dataset Tools

Provides unified and safe CSV handling functions for sensor data recording.
All error messages and logs are in English as required.
"""
import csv
import os
import logging
from typing import Dict, List, Any, Optional


def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure directory exists and has write permissions.

    Args:
        directory: Directory path to check/create

    Returns:
        bool: True if directory is accessible, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)

        # Test write permissions by creating a test file
        test_file = os.path.join(directory, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)

        return True
    except (OSError, PermissionError) as e:
        logging.error(f"Failed to access directory {directory}: {e}")
        return False


def safe_append_to_csv(csv_path: str, fieldnames: List[str],
                        data: Dict[str, Any], is_first_write: bool,
                        logger_name: str = None) -> Dict[str, Any]:
    """
    Safely append data to CSV file with header handling and error processing.

    Args:
        csv_path: Path to CSV file
        fieldnames: List of CSV column names
        data: Dictionary of data to write
        is_first_write: Whether this is the first write (include header)
        logger_name: Optional logger name for context

    Returns:
        dict: Result with success status and details
    """
    logger = logging.getLogger(logger_name or __name__)

    # Validate input data
    if not data:
        return {
            'success': False,
            'error': 'No data provided for CSV write',
            'file': os.path.basename(csv_path)
        }

    # Ensure directory exists
    directory = os.path.dirname(csv_path)
    if directory and not ensure_directory_exists(directory):
        return {
            'success': False,
            'error': f'Cannot access directory: {directory}',
            'file': os.path.basename(csv_path)
        }

    try:
        mode = 'w' if is_first_write else 'a'
        write_header = is_first_write

        with open(csv_path, mode, newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerow(data)

        logger.debug(f"Successfully appended data to {os.path.basename(csv_path)}")

        return {
            'success': True,
            'file': os.path.basename(csv_path),
            'rows_written': 1,
            'mode': mode
        }

    except (IOError, OSError) as e:
        error_msg = f"File I/O error writing to {csv_path}: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'file': os.path.basename(csv_path)
        }
    except csv.Error as e:
        error_msg = f"CSV formatting error writing to {csv_path}: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'file': os.path.basename(csv_path)
        }
    except Exception as e:
        error_msg = f"Unexpected error writing to {csv_path}: {e}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'file': os.path.basename(csv_path)
        }


def create_csv_writer(csv_path: str, fieldnames: List[str],
                      mode: str = 'w', logger_name: str = None) -> Optional[csv.DictWriter]:
    """
    Create a CSV writer with error handling.

    Args:
        csv_path: Path to CSV file
        fieldnames: List of CSV column names
        mode: File open mode ('w' for write, 'a' for append)
        logger_name: Optional logger name for context

    Returns:
        csv.DictWriter: Writer instance or None if failed
    """
    logger = logging.getLogger(logger_name or __name__)

    # Ensure directory exists
    directory = os.path.dirname(csv_path)
    if directory and not ensure_directory_exists(directory):
        logger.error(f"Cannot access directory for CSV file: {directory}")
        return None

    try:
        csv_file = open(csv_path, mode, newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header if in write mode
        if mode == 'w':
            writer.writeheader()

        return writer

    except (IOError, OSError, csv.Error) as e:
        logger.error(f"Failed to create CSV writer for {csv_path}: {e}")
        return None


def validate_csv_data(data: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    """
    Validate CSV data against expected fieldnames.

    Args:
        data: Dictionary of data to validate
        fieldnames: Expected field names

    Returns:
        dict: Validation result with status and details
    """
    if not isinstance(data, dict):
        return {
            'valid': False,
            'error': 'Data must be a dictionary'
        }

    # Check for missing fields
    missing_fields = [field for field in fieldnames if field not in data]
    if missing_fields:
        return {
            'valid': False,
            'error': f'Missing required fields: {missing_fields}',
            'missing_fields': missing_fields
        }

    # Check for extra fields (not critical, but good to know)
    extra_fields = [field for field in data if field not in fieldnames]

    return {
        'valid': True,
        'extra_fields': extra_fields if extra_fields else None,
        'field_count': len(data)
    }


def get_csv_file_info(csv_path: str) -> Dict[str, Any]:
    """
    Get information about a CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        dict: File information
    """
    if not os.path.exists(csv_path):
        return {
            'exists': False,
            'path': csv_path
        }

    try:
        stat = os.stat(csv_path)
        file_size = stat.st_size

        # Count rows (excluding header)
        row_count = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for _ in reader:
                row_count += 1

        return {
            'exists': True,
            'path': csv_path,
            'size_bytes': file_size,
            'row_count': row_count,
            'has_header': header is not None,
            'header_fields': header if header is not None else None
        }

    except Exception as e:
        return {
            'exists': True,
            'path': csv_path,
            'error': str(e)
        }