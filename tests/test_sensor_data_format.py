#!/usr/bin/env python3

"""
测试传感器数据格式验证
Test sensor data format validation
验证传感器数据的单格式原则和格式正确性
"""

import os
import sys
import json
import glob
import re
from pathlib import Path

# Add parent directory to path to import from recorder module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def find_sensor_data_directories(root_dir):
    """
    查找传感器数据目录

    Args:
        root_dir: 根目录

    Returns:
        list: 传感器数据目录列表
    """
    sensor_dirs = []

    # 查找 raw_data 下的子目录
    raw_data_dir = Path(root_dir) / "raw_data"
    if raw_data_dir.exists():
        # 查找包含传感器数据的目录（通常包含 poses.csv 或数据文件）
        for item in raw_data_dir.iterdir():
            if item.is_dir():
                # 检查是否是传感器目录（包含 poses.csv 或有子传感器目录）
                poses_file = item / "poses.csv"
                sensor_subdirs = [d for d in item.iterdir() if d.is_dir()]

                if poses_file.exists() or sensor_subdirs:
                    sensor_dirs.append(item)
                    # 也检查子目录
                    sensor_dirs.extend(sensor_subdirs)

    return sensor_dirs

def test_sensor_single_format(sensor_dir):
    """
    测试单个传感器的单格式原则

    Args:
        sensor_dir: 传感器目录路径

    Returns:
        dict: 测试结果
    """
    print(f"检查传感器: {sensor_dir.name}")

    # 查找数据文件（排除元数据文件）
    files = []
    for pattern in ["**/*.png", "**/*.ply", "**/*.csv", "**/*.json", "**/*.tiff", "**/*.npy"]:
        files.extend(sensor_dir.glob(pattern))

    # 过滤掉元数据文件
    data_files = []
    for file_path in files:
        relative_path = file_path.relative_to(sensor_dir)
        # 排除元数据文件
        if (relative_path.name not in ['sensor_metadata.json', 'poses.csv'] and
            'metadata.json' not in str(relative_path) and
            'camera_info.csv' not in str(relative_path) and
            '_stats.json' not in str(relative_path)):
            data_files.append(file_path)

    if not data_files:
        return {
            'sensor': sensor_dir.name,
            'has_data': False,
            'is_single_format': True,
            'formats': [],
            'primary_format': None,
            'file_count': 0,
            'data_files': []
        }

    # 按扩展名分类
    formats = set()
    file_info = []

    for file_path in data_files:
        ext = file_path.suffix.lower()
        formats.add(ext)
        file_info.append({
            'name': file_path.relative_to(sensor_dir).as_posix(),
            'extension': ext,
            'size': file_path.stat().st_size
        })

    return {
        'sensor': sensor_dir.name,
        'has_data': True,
        'is_single_format': len(formats) <= 1,
        'formats': sorted(list(formats)),
        'primary_format': list(formats)[0] if formats else None,
        'file_count': len(data_files),
        'data_files': file_info
    }

def validate_sensor_format_map():
    """
    验证传感器格式映射是否符合预期

    Args:
        sensor_results: 传感器测试结果列表

    Returns:
        dict: 格式映射验证结果
    """
    expected_formats = {
        # 图像类传感器 -> PNG
        'rgb': 'png',
        'camera': 'png',
        'depth': 'png',
        'semantic': 'png',
        'instance': 'png',
        'dvs': 'csv',  # DVS使用CSV格式

        # 点云类传感器 -> PLY
        'lidar': 'ply',

        # 结构化数据 -> CSV
        'radar': 'csv',
        'imu': 'csv',
        'gnss': 'csv',

        # 光流 -> PNG（特殊打包）
        'optical': 'png',
        'flow': 'png'
    }

    return expected_formats

def check_sensor_format_compliance(sensor_result, expected_formats):
    """
    检查传感器格式是否符合预期

    Args:
        sensor_result: 传感器测试结果
        expected_formats: 预期格式映射

    Returns:
        dict: 合规性检查结果
    """
    if not sensor_result['has_data']:
        return {'compliant': True, 'reason': 'no_data'}

    sensor_name = sensor_result['sensor'].lower()
    actual_format = sensor_result['primary_format']

    # 检查传感器名称与预期格式的匹配
    for keyword, expected_format in expected_formats.items():
        if keyword in sensor_name:
            return {
                'compliant': actual_format == expected_format,
                'expected': expected_format,
                'actual': actual_format,
                'keyword': keyword
            }

    # 如果没有匹配的关键词，假设符合
    return {
        'compliant': True,
        'reason': 'no_specific_expectation',
        'actual': actual_format
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test sensor data format compliance')
    parser.add_argument('--data-dir', default='.',
                       help='Root data directory (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    print("=== 传感器数据格式验证 ===")
    print(f"根目录: {os.path.abspath(args.data_dir)}")
    print()

    # 查找传感器数据目录
    sensor_dirs = find_sensor_data_directories(args.data_dir)
    print(f"发现 {len(sensor_dirs)} 个传感器数据目录")

    if not sensor_dirs:
        print("未找到传感器数据目录")
        return 1

    # 测试每个传感器
    sensor_results = []
    for sensor_dir in sensor_dirs:
        if args.verbose:
            print(f"\n--- {sensor_dir} ---")

        result = test_sensor_single_format(sensor_dir)
        sensor_results.append(result)

    # 验证格式映射
    expected_formats = validate_sensor_format_map()

    print(f"\n=== 预期格式映射 ===")
    for keyword, format_ext in expected_formats.items():
        print(f"  {keyword:10s} -> {format_ext}")

    # 合规性检查
    compliance_results = []
    format_summary = {}

    print(f"\n=== 传感器格式合规性检查 ===")
    for result in sensor_results:
        if result['has_data']:
            compliance = check_sensor_format_compliance(result, expected_formats)
            compliance_results.append(compliance)

            # 格式统计
            format_key = result['primary_format']
            if format_key:
                format_summary[format_key] = format_summary.get(format_key, 0) + 1

            status = "✓" if compliance['compliant'] else "✗"
            print(f"{status} {result['sensor']:25s} -> {result['primary_format']:6s} ({len(result['data_files'])} files)")

            if not compliance['compliant']:
                if 'expected' in compliance:
                    print(f"    预期: {compliance['expected']}, 实际: {compliance['actual']}")
                else:
                    print(f"    问题: {compliance.get('reason', 'unknown')}")
        else:
            print(f"  {result['sensor']:25s} -> 无数据文件")

    # 统计总结
    total_with_data = len([r for r in sensor_results if r['has_data']])
    compliant_count = len([c for c in compliance_results if c['compliant']])
    single_format_count = len([r for r in sensor_results if r['has_data'] and r['is_single_format']])

    print(f"\n=== 验证总结 ===")
    print(f"总传感器数: {len(sensor_results)}")
    print(f"有数据的传感器: {total_with_data}")
    print(f"符合单格式: {single_format_count}")
    print(f"格式合规: {compliant_count}")
    print(f"格式分布: {dict(format_summary)}")

    # 成功条件
    if single_format_count == total_with_data and compliant_count == total_with_data:
        print("\n✅ 所有传感器都符合单格式要求和格式预期")
        return 0
    else:
        print(f"\n❌ 存在格式问题:")
        if single_format_count < total_with_data:
            print(f"  - {total_with_data - single_format_count} 个传感器违反单格式原则")
        if compliant_count < total_with_data:
            print(f"  - {total_with_data - compliant_count} 个传感器格式不符合预期")
        return 1

if __name__ == "__main__":
    import argparse
    success = main()
    sys.exit(0 if success else 1)