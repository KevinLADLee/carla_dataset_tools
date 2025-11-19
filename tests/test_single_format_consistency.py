#!/usr/bin/env python3

"""
测试单格式传感器数据一致性
Test data consistency for single format sensors
验证所有传感器都使用单一文件格式，没有多格式冗余保存
"""

import os
import sys
import json
import glob
from pathlib import Path

# Add parent directory to path to import from recorder module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_sensor_format_consistency(data_dir):
    """
    测试传感器数据格式一致性

    Args:
        data_dir: 数据目录路径

    Returns:
        dict: 测试结果
    """
    print(f"=== 传感器单格式一致性测试 ===")
    print(f"测试目录: {data_dir}")
    print()

    results = {
        'total_sensors': 0,
        'consistent_sensors': 0,
        'inconsistencies': [],
        'format_summary': {}
    }

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        return results

    # 查找所有传感器目录
    sensor_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    print(f"发现 {len(sensor_dirs)} 个传感器目录")

    for sensor_dir in sensor_dirs:
        sensor_name = sensor_dir.name
        print(f"\n--- 检查传感器: {sensor_name} ---")

        # 检查文件格式
        format_analysis = analyze_sensor_files(sensor_dir)

        if format_analysis['is_single_format']:
            print(f"✓ 单一格式: {format_analysis['primary_format']}")
            results['consistent_sensors'] += 1
        else:
            print(f"✗ 多格式问题: {format_analysis['formats_found']}")
            results['inconsistencies'].append({
                'sensor': sensor_name,
                'formats': format_analysis['formats_found'],
                'files': format_analysis['file_details']
            })

        # 检查元数据文件
        metadata_check = check_metadata(sensor_dir)

        results['format_summary'][sensor_name] = {
            'format_analysis': format_analysis,
            'metadata_check': metadata_check
        }

        results['total_sensors'] += 1

    # 输出总结
    print(f"\n=== 测试总结 ===")
    print(f"总传感器数: {results['total_sensors']}")
    print(f"符合单格式要求: {results['consistent_sensors']}")
    print(f"存在多格式问题: {len(results['inconsistencies'])}")

    if results['inconsistencies']:
        print(f"\n=== 格式问题详情 ===")
        for issue in results['inconsistencies']:
            print(f"传感器: {issue['sensor']}")
            print(f"  多格式: {issue['formats']}")
            print(f"  文件: {issue['files']}")

    return results

def analyze_sensor_files(sensor_dir):
    """
    分析传感器目录中的文件格式

    Args:
        sensor_dir: 传感器目录路径

    Returns:
        dict: 格式分析结果
    """
    files = list(sensor_dir.glob("*"))
    data_files = [f for f in files if f.is_file() and f.name not in ['sensor_metadata.json', 'poses.csv']]

    if not data_files:
        return {
            'is_single_format': True,
            'primary_format': 'no_data_files',
            'formats_found': [],
            'file_details': []
        }

    # 按扩展名分类
    formats = set()
    file_details = []

    for file_path in data_files:
        ext = file_path.suffix.lower()
        formats.add(ext)
        file_details.append({
            'name': file_path.name,
            'extension': ext,
            'size': file_path.stat().st_size
        })

    # 检查是否符合单格式原则
    is_single_format = len(formats) <= 1  # 允许无数据文件的情况
    primary_format = list(formats)[0] if formats else 'no_data_files'

    return {
        'is_single_format': is_single_format,
        'primary_format': primary_format,
        'formats_found': list(formats),
        'file_details': file_details
    }

def check_metadata(sensor_dir):
    """
    检查传感器元数据文件

    Args:
        sensor_dir: 传感器目录路径

    Returns:
        dict: 元数据检查结果
    """
    metadata_file = sensor_dir / 'sensor_metadata.json'
    poses_file = sensor_dir / 'poses.csv'

    result = {
        'has_metadata': metadata_file.exists(),
        'has_poses': poses_file.exists(),
        'metadata_valid': False
    }

    if result['has_metadata']:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 检查必要的元数据字段
            required_fields = ['sensor_type', 'attributes', 'data_format']
            missing_fields = [field for field in required_fields if field not in metadata]

            result['metadata_valid'] = len(missing_fields) == 0
            result['missing_fields'] = missing_fields
            result['sensor_type'] = metadata.get('sensor_type', 'unknown')
            result['data_format'] = metadata.get('data_format', 'unknown')

        except Exception as e:
            result['metadata_error'] = str(e)

    return result

def validate_expected_formats():
    """
    验证预期的传感器格式映射

    Returns:
        dict: 验证结果
    """
    expected_formats = {
        # 图像类传感器 -> PNG
        'sensor.camera.rgb': 'png',
        'sensor.camera.depth': 'png',
        'sensor.camera.semantic_segmentation': 'png',
        'sensor.camera.instance_segmentation': 'png',
        'sensor.camera.dvs': 'png',
        'sensor.camera.optical_flow': 'png',

        # 点云类传感器 -> PLY
        'sensor.lidar.ray_cast': 'ply',
        'sensor.lidar.ray_cast_semantic': 'ply',

        # 结构化数据 -> CSV
        'sensor.other.radar': 'csv',
        'sensor.other.imu': 'csv',
        'sensor.other.gnss': 'csv'
    }

    print("\n=== 预期格式验证 ===")
    print("单格式传感器映射:")
    for sensor_type, expected_ext in expected_formats.items():
        print(f"  {sensor_type} -> {expected_ext}")

    return expected_formats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test single format sensor data consistency')
    parser.add_argument('data_dir', nargs='?',
                       help='Data directory to test (default: current working directory)')
    parser.add_argument('--validate-expected', action='store_true',
                       help='Validate expected sensor formats')

    args = parser.parse_args()

    # 确定测试目录
    if args.data_dir:
        test_dir = args.data_dir
    else:
        test_dir = os.getcwd()

    # 运行一致性测试
    results = test_sensor_format_consistency(test_dir)

    # 可选：验证预期格式
    if args.validate_expected:
        expected_formats = validate_expected_formats()

    # 退出码
    if results['inconsistencies']:
        print(f"\n❌ 发现 {len(results['inconsistencies'])} 个传感器存在多格式问题")
        return 1
    else:
        print(f"\n✅ 所有传感器都符合单格式要求")
        return 0

if __name__ == "__main__":
    import argparse
    success = main()
    sys.exit(0 if success else 1)