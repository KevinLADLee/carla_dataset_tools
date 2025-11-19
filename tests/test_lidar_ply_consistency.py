#!/usr/bin/env python3

"""
测试NPY和PLY格式数据一致性
Test data consistency between NPY and PLY formats for CARLA LiDAR data
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import from recorder module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def read_npy_lidar_file(filepath):
    """读取NPY格式的Regular LiDAR数据"""
    try:
        data = np.load(filepath)
        return data
    except Exception as e:
        print(f"Error reading NPY file {filepath}: {e}")
        return None

def read_npy_semantic_lidar_file(filepath):
    """读取NPY格式的Semantic LiDAR数据"""
    try:
        data = np.load(filepath)
        return data
    except Exception as e:
        print(f"Error reading NPY file {filepath}: {e}")
        return None

def read_ply_lidar_file(filepath):
    """读取PLY格式的Regular LiDAR数据"""
    try:
        from plyfile import PlyData
        ply_data = PlyData.read(filepath)

        # 检查是否为Regular LiDAR格式
        vertex_props = [p.name for p in ply_data['vertex'].properties]

        if 'intensity' in vertex_props and 'cos_angle' not in vertex_props:
            # Regular LiDAR格式: [x, y, z, intensity]
            points = np.column_stack([
                ply_data['vertex']['x'],
                ply_data['vertex']['y'],
                ply_data['vertex']['z'],
                ply_data['vertex']['intensity']
            ])
            return points, 'regular'
        elif 'cos_angle' in vertex_props:
            # Semantic LiDAR格式: [x, y, z, cos_angle, obj_idx, obj_tag]
            points = np.column_stack([
                ply_data['vertex']['x'],
                ply_data['vertex']['y'],
                ply_data['vertex']['z'],
                ply_data['vertex']['cos_angle'],
                ply_data['vertex']['obj_idx'],
                ply_data['vertex']['obj_tag']
            ])
            return points, 'semantic'
        else:
            print(f"Unknown PLY format in {filepath}")
            print(f"Available properties: {vertex_props}")
            return None, None

    except ImportError:
        print("Error: plyfile not available. Install with: pip install plyfile")
        return None, None
    except Exception as e:
        print(f"Error reading PLY file {filepath}: {e}")
        return None, None

def compare_regular_lidar_files(npy_file, ply_file, tolerance=1e-6):
    """比较Regular LiDAR的NPY和PLY文件数据一致性"""
    print(f"Comparing Regular LiDAR files:")
    print(f"  NPY: {npy_file}")
    print(f"  PLY: {ply_file}")

    # 读取NPY文件
    npy_data = read_npy_lidar_file(npy_file)
    if npy_data is None:
        return False

    # 读取PLY文件
    ply_data, ply_type = read_ply_lidar_file(ply_file)
    if ply_data is None:
        return False

    # 检查PLY类型是否正确
    if ply_type != 'regular':
        print(f"Error: PLY file {ply_file} is not Regular LiDAR format")
        return False

    # 检查数据形状
    if npy_data.shape != ply_data.shape:
        print(f"Error: Shape mismatch - NPY: {npy_data.shape}, PLY: {ply_data.shape}")
        return False

    # 比较数据
    try:
        np.testing.assert_allclose(npy_data, ply_data, rtol=tolerance, atol=tolerance)
        print(f"✓ Data一致性验证通过 ({npy_data.shape[0]} points)")

        # 显示统计信息
        print(f"  NPY数据范围: x=[{npy_data[:, 0].min():.3f}, {npy_data[:, 0].max():.3f}], "
              f"y=[{npy_data[:, 1].min():.3f}, {npy_data[:, 1].max():.3f}], "
              f"z=[{npy_data[:, 2].min():.3f}, {npy_data[:, 2].max():.3f}], "
              f"intensity=[{npy_data[:, 3].min():.3f}, {npy_data[:, 3].max():.3f}]")

        # 文件大小对比
        npy_size = os.path.getsize(npy_file)
        ply_size = os.path.getsize(ply_file)
        print(f"  文件大小: NPY={npy_size} bytes, PLY={ply_size} bytes "
              f"(PLY/NPY ratio: {ply_size/npy_size:.3f})")

        return True

    except AssertionError as e:
        print(f"✗ Data不一致: {e}")
        return False

def compare_semantic_lidar_files(npy_file, ply_file, tolerance=1e-6):
    """比较Semantic LiDAR的NPY和PLY文件数据一致性"""
    print(f"Comparing Semantic LiDAR files:")
    print(f"  NPY: {npy_file}")
    print(f"  PLY: {ply_file}")

    # 读取NPY文件
    npy_data = read_npy_semantic_lidar_file(npy_file)
    if npy_data is None:
        return False

    # 读取PLY文件
    ply_data, ply_type = read_ply_lidar_file(ply_file)
    if ply_data is None:
        return False

    # 检查PLY类型是否正确
    if ply_type != 'semantic':
        print(f"Error: PLY file {ply_file} is not Semantic LiDAR format")
        return False

    # 检查数据形状
    if npy_data.shape[0] != ply_data.shape[0]:
        print(f"Error: Point count mismatch - NPY: {npy_data.shape[0]}, PLY: {ply_data.shape[0]}")
        return False

    # 比较坐标数据
    try:
        # NPY是结构化数组，PLY是二维数组，需要分别比较字段
        np.testing.assert_allclose(npy_data['x'], ply_data[:, 0], rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(npy_data['y'], ply_data[:, 1], rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(npy_data['z'], ply_data[:, 2], rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(npy_data['CosAngle'], ply_data[:, 3], rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(npy_data['ObjIdx'], ply_data[:, 4])
        np.testing.assert_array_equal(npy_data['ObjTag'], ply_data[:, 5])

        print(f"✓ Data一致性验证通过 ({npy_data.shape[0]} points)")

        # 显示统计信息
        print(f"  NPY坐标范围: x=[{npy_data['x'].min():.3f}, {npy_data['x'].max():.3f}], "
              f"y=[{npy_data['y'].min():.3f}, {npy_data['y'].max():.3f}], "
              f"z=[{npy_data['z'].min():.3f}, {npy_data['z'].max():.3f}]")
        print(f"  NPY语义字段: CosAngle=[{npy_data['CosAngle'].min():.3f}, {npy_data['CosAngle'].max():.3f}], "
              f"ObjIdx=[{npy_data['ObjIdx'].min()}, {npy_data['ObjIdx'].max()}], "
              f"ObjTag=[{npy_data['ObjTag'].min()}, {npy_data['ObjTag'].max()}]")

        # 文件大小对比
        npy_size = os.path.getsize(npy_file)
        ply_size = os.path.getsize(ply_file)
        print(f"  文件大小: NPY={npy_size} bytes, PLY={ply_size} bytes "
              f"(PLY/NPY ratio: {ply_size/npy_size:.3f})")

        return True

    except AssertionError as e:
        print(f"✗ Data不一致: {e}")
        return False

def auto_detect_and_compare(npy_file, tolerance=1e-6):
    """自动检测NPY文件类型并寻找对应的PLY文件进行比较"""
    print(f"Auto-detecting file type for: {npy_file}")

    # 检查NPY文件是否存在
    if not os.path.exists(npy_file):
        print(f"Error: NPY file {npy_file} does not exist")
        return False

    # 寻找对应的PLY文件
    base_path = Path(npy_file)
    ply_file = str(base_path.with_suffix('.ply'))

    if not os.path.exists(ply_file):
        print(f"Error: Corresponding PLY file {ply_file} does not exist")
        return False

    # 读取NPY文件判断类型
    try:
        npy_data = np.load(npy_file)

        if npy_data.dtype == np.float32 and len(npy_data.shape) == 2 and npy_data.shape[1] == 4:
            # Regular LiDAR: (N, 4) float32 array
            return compare_regular_lidar_files(npy_file, ply_file, tolerance)
        elif hasattr(npy_data, 'dtype') and hasattr(npy_data.dtype, 'names') and npy_data.dtype.names:
            # Semantic LiDAR: structured array
            expected_fields = {'x', 'y', 'z', 'CosAngle', 'ObjIdx', 'ObjTag'}
            actual_fields = set(npy_data.dtype.names)
            if expected_fields.issubset(actual_fields):
                return compare_semantic_lidar_files(npy_file, ply_file, tolerance)

        print(f"Error: Unknown NPY file format for {npy_file}")
        print(f"NPY shape: {npy_data.shape}, dtype: {npy_data.dtype}")
        if hasattr(npy_data, 'dtype') and hasattr(npy_data.dtype, 'names'):
            print(f"NPY fields: {npy_data.dtype.names}")
        return False

    except Exception as e:
        print(f"Error reading NPY file {npy_file}: {e}")
        return False

def find_and_compare_directory(directory, tolerance=1e-6):
    """在目录中查找并比较所有NPY/PLY文件对"""
    print(f"Scanning directory: {directory}")

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return False

    # 查找所有NPY文件
    npy_files = list(Path(directory).glob("*.npy"))
    if not npy_files:
        print(f"No NPY files found in {directory}")
        return False

    print(f"Found {len(npy_files)} NPY files")

    results = []
    for npy_file in npy_files:
        ply_file = npy_file.with_suffix('.ply')
        if ply_file.exists():
            result = auto_detect_and_compare(str(npy_file), tolerance)
            results.append(result)
            print("-" * 50)
        else:
            print(f"Warning: No corresponding PLY file for {npy_file}")
            results.append(False)
            print("-" * 50)

    passed = sum(results)
    total = len(results)
    print(f"Directory comparison results: {passed}/{total} passed")
    return passed == total

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Compare NPY and PLY LiDAR data consistency')
    parser.add_argument('input', nargs='?', help='NPY file or directory to compare')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Numerical tolerance for comparison (default: 1e-6)')
    parser.add_argument('--regular', action='store_true',
                       help='Force treat as Regular LiDAR')
    parser.add_argument('--semantic', action='store_true',
                       help='Force treat as Semantic LiDAR')

    args = parser.parse_args()

    print("=== CARLA LiDAR NPY/PLY Consistency Test ===")
    print()

    if args.input:
        if os.path.isfile(args.input):
            # 单个文件
            if args.input.endswith('.npy'):
                result = auto_detect_and_compare(args.input, args.tolerance)
            else:
                print("Error: Please provide an NPY file")
                return False
        elif os.path.isdir(args.input):
            # 目录
            result = find_and_compare_directory(args.input, args.tolerance)
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            return False
    else:
        # 默认创建测试文件进行比较
        print("No input provided, creating test files...")
        result = create_and_compare_test_files(args.tolerance)

    print()
    if result:
        print("🎉 All comparisons passed!")
    else:
        print("❌ Some comparisons failed")

    return result

def create_and_compare_test_files(tolerance=1e-6):
    """创建测试文件并进行比较（用于演示）"""
    print("Creating test files...")

    # 创建Regular LiDAR测试数据
    num_points = 1000
    regular_data = np.random.randn(num_points, 4).astype(np.float32)
    regular_data[:, 1] *= -1  # 坐标系转换

    npy_regular = 'test_regular.npy'
    ply_regular = 'test_regular.ply'

    np.save(npy_regular, regular_data)

    # 保存Regular LiDAR PLY
    try:
        from plyfile import PlyData, PlyElement
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
        structured_array = np.zeros(len(regular_data), dtype=dtype)
        structured_array['x'] = regular_data[:, 0]
        structured_array['y'] = regular_data[:, 1]
        structured_array['z'] = regular_data[:, 2]
        structured_array['intensity'] = regular_data[:, 3]
        vertex = PlyElement.describe(structured_array, 'vertex')
        PlyData([vertex]).write(ply_regular)
    except ImportError:
        print("Error: plyfile not available")
        return False

    # 比较Regular LiDAR
    result1 = compare_regular_lidar_files(npy_regular, ply_regular, tolerance)
    print("-" * 50)

    # 创建Semantic LiDAR测试数据
    dtype_semantic = [
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
    ]
    semantic_data = np.zeros(num_points, dtype=dtype_semantic)
    semantic_data['x'] = np.random.randn(num_points).astype(np.float32)
    semantic_data['y'] = np.random.randn(num_points).astype(np.float32)
    semantic_data['z'] = np.random.randn(num_points).astype(np.float32)
    semantic_data['CosAngle'] = np.random.rand(num_points).astype(np.float32)
    semantic_data['ObjIdx'] = np.random.randint(0, 100, num_points).astype(np.uint32)
    semantic_data['ObjTag'] = np.random.randint(0, 50, num_points).astype(np.uint32)
    semantic_data['y'] *= -1  # 坐标系转换

    npy_semantic = 'test_semantic.npy'
    ply_semantic = 'test_semantic.ply'

    np.save(npy_semantic, semantic_data)

    # 保存Semantic LiDAR PLY
    try:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('cos_angle', 'f4'), ('obj_idx', 'i4'), ('obj_tag', 'i4')]
        structured_array = np.zeros(len(semantic_data), dtype=dtype)
        structured_array['x'] = semantic_data['x']
        structured_array['y'] = semantic_data['y']
        structured_array['z'] = semantic_data['z']
        structured_array['cos_angle'] = semantic_data['CosAngle']
        structured_array['obj_idx'] = semantic_data['ObjIdx']
        structured_array['obj_tag'] = semantic_data['ObjTag']
        vertex = PlyElement.describe(structured_array, 'vertex')
        PlyData([vertex]).write(ply_semantic)
    except ImportError:
        print("Error: plyfile not available")
        return False

    # 比较Semantic LiDAR
    result2 = compare_semantic_lidar_files(npy_semantic, ply_semantic, tolerance)

    # 清理测试文件
    for f in [npy_regular, ply_regular, npy_semantic, ply_semantic]:
        if os.path.exists(f):
            os.remove(f)

    return result1 and result2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)