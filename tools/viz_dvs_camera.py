#!/usr/bin/env python3
"""
DVS相机可视化工具
参考manual_control.py的可视化逻辑 + viz_lidar.py的设计模式
支持NPY格式DVS事件数据可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DVSVisualizer:
    def __init__(self, data_source: str, display_size: tuple = (800, 600)):
        self.data_source = Path(data_source)
        self.display_size = display_size
        self.positive_color = [0, 0, 255]  # 蓝色（正事件）
        self.negative_color = [255, 0, 0]  # 红色（负事件）
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """加载传感器元数据"""
        metadata_file = self.data_source / 'sensor_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata: image_size={metadata.get('image_size', 'unknown')}")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def get_image_size(self):
        """获取图像尺寸（从元数据或使用默认值）"""
        if self.metadata and 'image_size' in self.metadata:
            width, height = self.metadata['image_size']
            return (height, width)  # 返回 (height, width) 格式
        return (1080, 1920)  # 默认尺寸

    def load_npy_file(self, npy_path: Path) -> np.ndarray:
        """加载NPY事件数据"""
        try:
            events = np.load(npy_path)
            logger.info(f"Loaded {len(events)} events from {npy_path.name}")
            return events
        except Exception as e:
            logger.error(f"Failed to load {npy_path}: {e}")
            return np.array([])

    def render_events(self, events: np.ndarray, image_size: tuple = None) -> np.ndarray:
        """
        渲染DVS事件为图像
        基于manual_control.py：蓝色正事件，红色负事件
        """
        if image_size is None:
            image_size = self.get_image_size()

        height, width = image_size
        vis_img = np.zeros((height, width, 3), dtype=np.uint8)

        if len(events) == 0:
            return vis_img

        # CARLA DVS格式：结构化数组 [x, y, t, pol]
        try:
            # 如果是结构化数组
            if events.dtype.fields:
                x_coords = events['x']
                y_coords = events['y']
                polarities = events['pol']
            else:
                # 如果是普通数组
                x_coords = events[:, 0]
                y_coords = events[:, 1]
                polarities = events[:, 3]

            # 过滤有效坐标
            valid_mask = (x_coords < width) & (y_coords < height)
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            polarities = polarities[valid_mask]

            # 渲染事件
            for x, y, pol in zip(x_coords, y_coords, polarities):
                color = self.positive_color if pol else self.negative_color
                vis_img[int(y), int(x)] = color

        except (IndexError, KeyError) as e:
            logger.error(f"Failed to render events: {e}")

        return vis_img

    def visualize_single_file(self, npy_path: Path):
        """可视化单个NPY文件"""
        logger.info(f"Visualizing single file: {npy_path}")

        events = self.load_npy_file(npy_path)
        image_size = self.get_image_size()
        vis_img = self.render_events(events, image_size)

        # 统计信息
        if len(events) > 0:
            try:
                if events.dtype.fields:
                    positive_count = np.sum(events['pol'] == True)
                    negative_count = np.sum(events['pol'] == False)
                else:
                    positive_count = np.sum(events[:, 3] == 1)
                    negative_count = np.sum(events[:, 3] == 0)
            except (IndexError, KeyError):
                positive_count = negative_count = 0
        else:
            positive_count = negative_count = 0

        plt.figure(figsize=(12, 8))
        plt.imshow(vis_img)
        plt.title(f'DVS Events - {npy_path.name}\\n'
                 f'Positive: {positive_count}, Negative: {negative_count}, Total: {len(events)}')
        plt.axis('off')

        # 添加文本信息
        info_text = f'Image Size: {image_size[1]}x{image_size[0]}\\n'
        if self.metadata:
            info_text += f"Sensor: {self.metadata.get('sensor_type', 'unknown')}\\n"
            info_text += f"Data Format: {self.metadata.get('data_format', 'unknown')}"

        plt.figtext(0.02, 0.02, info_text, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

        plt.tight_layout()
        plt.show()

    def visualize_directory(self, fps: int = 30):
        """目录动画播放"""
        npy_files = sorted(self.data_source.glob("*.npy"))
        if not npy_files:
            logger.error(f"No NPY files found in {self.data_source}")
            return

        logger.info(f"Found {len(npy_files)} NPY files for animation")

        image_size = self.get_image_size()

        # 加载第一帧初始化显示
        first_events = self.load_npy_file(npy_files[0])
        vis_img = self.render_events(first_events, image_size)

        fig, ax = plt.subplots(figsize=(12, 8))
        img_plot = ax.imshow(vis_img)
        ax.axis('off')

        # 动画更新函数
        def update_frame(frame_idx):
            if frame_idx < len(npy_files):
                events = self.load_npy_file(npy_files[frame_idx])
                vis_img = self.render_events(events, image_size)
                img_plot.set_array(vis_img)

                # 统计当前帧信息
                if len(events) > 0:
                    try:
                        if events.dtype.fields:
                            positive_count = np.sum(events['pol'] == True)
                            negative_count = np.sum(events['pol'] == False)
                        else:
                            positive_count = np.sum(events[:, 3] == 1)
                            negative_count = np.sum(events[:, 3] == 0)
                    except (IndexError, KeyError):
                        positive_count = negative_count = 0
                else:
                    positive_count = negative_count = 0

                ax.set_title(f'DVS Events - Frame {frame_idx+1}/{len(npy_files)} - {npy_files[frame_idx].name}\\n'
                           f'Positive: {positive_count}, Negative: {negative_count}')

            return [img_plot]

        # 创建动画
        ani = animation.FuncAnimation(
            fig, update_frame, frames=len(npy_files),
            interval=1000//fps, blit=True, repeat=True
        )

        plt.tight_layout()
        plt.show()

        return ani

    def print_statistics(self):
        """打印数据统计信息"""
        npy_files = list(self.data_source.glob("*.npy"))
        if not npy_files:
            print("No NPY files found")
            return

        total_events = 0
        positive_events = 0
        negative_events = 0

        print(f"Analyzing {len(npy_files)} NPY files...")

        for npy_file in sorted(npy_files):
            try:
                events = np.load(npy_file)
                total_events += len(events)

                if len(events) > 0:
                    if events.dtype.fields:
                        positive_events += np.sum(events['pol'] == True)
                        negative_events += np.sum(events['pol'] == False)
                    else:
                        positive_events += np.sum(events[:, 3] == 1)
                        negative_events += np.sum(events[:, 3] == 0)

            except Exception as e:
                logger.warning(f"Failed to analyze {npy_file}: {e}")

        print(f"\\n=== DVS Data Statistics ===")
        print(f"Total files: {len(npy_files)}")
        print(f"Total events: {total_events:,}")
        print(f"Positive events: {positive_events:,}")
        print(f"Negative events: {negative_events:,}")
        print(f"Average events per frame: {total_events/len(npy_files):.1f}")

        if self.metadata:
            print(f"\\n=== Sensor Info ===")
            print(f"Sensor type: {self.metadata.get('sensor_type', 'unknown')}")
            print(f"Image size: {self.metadata.get('image_size', 'unknown')}")
            print(f"Data format: {self.metadata.get('data_format', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description='DVS相机可视化工具')
    parser.add_argument('source', help='NPY文件或目录路径')
    parser.add_argument('--fps', type=int, default=30, help='动画播放帧率 (默认: 30)')
    parser.add_argument('--size', nargs=2, type=int, default=[1920, 1080],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='图像尺寸 width height (默认: 1920 1080)')
    parser.add_argument('--stats', action='store_true',
                       help='显示数据统计信息')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别 (默认: INFO)')

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # 创建可视化器
    visualizer = DVSVisualizer(args.source, tuple(args.size))

    # 显示统计信息
    if args.stats:
        visualizer.print_statistics()
        return

    # 执行可视化
    source_path = Path(args.source)
    if source_path.is_file() and source_path.suffix == '.npy':
        visualizer.visualize_single_file(source_path)
    elif source_path.is_dir():
        visualizer.visualize_directory(args.fps)
    else:
        print(f"错误: 路径不存在或不是NPY文件: {args.source}")
        print("使用示例:")
        print("  python tools/viz_dvs_camera.py data/camera_dvs/0000000001.npy")
        print("  python tools/viz_dvs_camera.py data/camera_dvs/ --fps 30")
        print("  python tools/viz_dvs_camera.py data/camera_dvs/ --stats")


if __name__ == "__main__":
    main()