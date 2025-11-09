# CARLA Dataset Tools - 待办事项清单

> 📅 最后更新: 2025-11-09
> 🎯 目标: 提升代码质量、可维护性和可靠性

---

## 📊 项目现状概览

### 代码统计
- **代码总量**: ~14,643 行 Python 代码
- **模块数量**: 47 个 Python 模块
- **测试覆盖率**: ~5% (5个手动测试文件)
- **日志系统**: 0% (178个print语句，无结构化日志)
- **类型注解**: ~10% (最小化使用)

### 评估等级
- **架构设计**: ⭐⭐⭐⭐ (优秀)
- **配置管理**: ⭐⭐⭐⭐⭐ (卓越)
- **文档质量**: ⭐⭐⭐⭐ (良好)
- **测试覆盖**: ⭐ (需要改进)
- **错误处理**: ⭐⭐ (需要改进)
- **日志系统**: ⭐ (需要改进)

---

## 🔥 关键问题（需立即处理）

### 1. 🔴 线程安全问题 - actor_tree.py:36-42

**文件**: `recorder/actor_tree.py`
**严重程度**: 🔴 严重
**影响**: 可能导致数据丢失且不可察觉

**问题描述**:
```python
def tick_data_saving(self, frame_id, timestamp: float):
    thread_pool = ThreadPool()
    # ...
    # 使用线程池保存数据，但没有错误处理
    thread_pool.starmap(self.save_data, zip(frame_id_list, timestamp_list, self.node_list))
    thread_pool.close()
    thread_pool.join()
```

**问题**:
- 多线程保存传感器数据时，如果某个传感器失败，会静默失败
- 无法追踪哪些数据保存失败
- 用户不知道数据集不完整

**解决方案**:
```python
def tick_data_saving(self, frame_id, timestamp: float):
    """保存所有节点数据，带完整的错误处理"""
    import logging
    logger = logging.getLogger(__name__)

    thread_pool = ThreadPool()
    frame_id_list = [frame_id] * len(self.node_list)
    timestamp_list = [timestamp] * len(self.node_list)

    # 使用包装方法进行错误处理
    try:
        results = thread_pool.starmap(
            self._safe_save_data,
            zip(frame_id_list, timestamp_list, self.node_list)
        )
        thread_pool.close()
        thread_pool.join()

        # 检查失败的节点
        failed = [r for r in results if not r['success']]
        if failed:
            logger.error(f"Frame {frame_id}: {len(failed)}个节点保存失败: {failed}")
            # 可选：抛出异常中止录制
            # raise RuntimeError(f"数据保存失败: {failed}")
    except Exception as e:
        logger.exception(f"数据保存过程中发生严重错误: {e}")
        thread_pool.terminate()
        raise

def _safe_save_data(self, frame_id, timestamp: float, node: Node) -> dict:
    """带异常捕获的数据保存包装器"""
    try:
        node.tick_data_saving(frame_id, timestamp)
        return {
            'success': True,
            'node': node.get_actor().name if node.get_actor() else 'unknown'
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        node_name = node.get_actor().name if node.get_actor() else 'unknown'
        logger.exception(f"节点 {node_name} 保存数据失败: {e}")
        return {
            'success': False,
            'node': node_name,
            'error': str(e)
        }
```

**预估工作量**: 4小时
**优先级**: P0 (最高)

---

### 2. 🔴 传感器队列阻塞问题 - sensor.py:39-42

**文件**: `recorder/sensor.py`
**严重程度**: 🔴 严重
**影响**: 可能导致程序永久挂起

**问题描述**:
```python
def save_to_disk(self, frame_id, timestamp, debug=False):
    sensor_frame_id = 0
    while sensor_frame_id < frame_id:
        sensor_data = self.queue.get(True)  # 无超时的阻塞调用！
```

**问题**:
- 如果传感器停止产生数据，`queue.get(True)` 会永久阻塞
- 整个录制过程卡死
- 没有错误提示

**解决方案**:
```python
import queue
import logging

def save_to_disk(self, frame_id, timestamp, debug=False):
    """保存传感器数据到磁盘，带超时保护"""
    logger = logging.getLogger(__name__)
    sensor_frame_id = 0
    timeout = 5.0  # 5秒超时

    while sensor_frame_id < frame_id:
        try:
            sensor_data = self.queue.get(block=True, timeout=timeout)
            sensor_frame_id = sensor_data.frame

            # 丢弃旧帧
            if sensor_frame_id < frame_id:
                logger.debug(
                    f"传感器 {self.name}: 丢弃旧帧 {sensor_frame_id}，"
                    f"等待帧 {frame_id}"
                )
                continue

            # 保存数据
            os.makedirs(self.save_dir, exist_ok=True)
            success = self.save_to_disk_impl(self.save_dir, sensor_data)

            if not success:
                raise IOError(
                    f"传感器 {self.name} 保存帧 {frame_id} 失败"
                )

            self.save_pose(frame_id, timestamp)
            self._first_frame = False

            if debug:
                self.print_debug_info(sensor_data.frame, sensor_data)

        except queue.Empty:
            logger.error(
                f"传感器 {self.name} 等待帧 {frame_id} 超时（{timeout}秒）。"
                f"最后接收帧: {sensor_frame_id}"
            )
            raise TimeoutError(
                f"传感器 {self.name} 数据超时 (frame {frame_id})"
            )
```

**预估工作量**: 3小时
**优先级**: P0 (最高)

---

### 3. 🟡 路径遍历安全风险 - actor_factory.py:278-281

**文件**: `recorder/actor_factory.py`
**严重程度**: 🟡 中等
**影响**: 潜在的路径遍历攻击

**问题描述**:
```python
route_file = Path(route_info["from_file"])
if not route_file.is_absolute():
    route_file = Path(ROOT_PATH) / route_file  # 可能逃逸到ROOT_PATH之外
```

**问题**:
- 恶意用户可以使用 `../../../etc/passwd` 等路径
- 虽然这是本地工具，但仍应遵循安全最佳实践

**解决方案**:
```python
def _parse_route_config(self, route_info):
    """解析路由配置，带路径安全验证"""
    import logging
    from config.config_manager import ConfigValidationError

    logger = logging.getLogger(__name__)
    route_config = {}

    # 检查是否从文件加载
    if "from_file" in route_info:
        route_file = Path(route_info["from_file"])

        # 如果是相对路径，转换为绝对路径
        if not route_file.is_absolute():
            route_file = Path(ROOT_PATH) / route_file

        # 安全检查：确保解析后的路径在项目目录内
        try:
            route_file = route_file.resolve()
            root_path_resolved = Path(ROOT_PATH).resolve()

            # 检查路径是否在项目目录内
            if not str(route_file).startswith(str(root_path_resolved)):
                raise ConfigValidationError(
                    f"安全错误: 路由文件路径在项目目录外: {route_file}\n"
                    f"项目根目录: {root_path_resolved}"
                )

            # 检查文件是否存在
            if not route_file.exists():
                raise ConfigValidationError(
                    f"路由文件不存在: {route_file}"
                )

            # 检查是否是文件（不是目录）
            if not route_file.is_file():
                raise ConfigValidationError(
                    f"路由路径不是文件: {route_file}"
                )

        except (ValueError, OSError) as e:
            raise ConfigValidationError(f"无效的路由文件路径: {e}")

        # 加载路由数据
        try:
            with open(route_file, 'r', encoding='utf-8') as f:
                route_data = yaml.safe_load(f)
                route_config['waypoints'] = route_data.get('waypoints', [])
                route_config['mode'] = route_data.get('mode', 'strict')
                route_config['loop'] = route_data.get('loop', False)
                logger.info(
                    f"从文件加载路由: {route_file} "
                    f"({len(route_config['waypoints'])} 个航点)"
                )
        except Exception as e:
            raise ConfigValidationError(
                f"加载路由文件 {route_file} 失败: {e}"
            )
    else:
        # 直接使用配置中的航点
        route_config['waypoints'] = route_info.get('waypoints', [])
        route_config['mode'] = route_info.get('mode', 'strict')
        route_config['loop'] = route_info.get('loop', False)

    # 验证最小要求
    if len(route_config.get('waypoints', [])) < 2:
        logger.warning("路由必须至少有2个航点。忽略路由配置。")
        return None

    return route_config
```

**预估工作量**: 2小时
**优先级**: P1 (高)

---

## 📝 重要改进（近期处理）

### 4. 🟡 缺少结构化日志系统

**严重程度**: 🟡 中等
**影响**: 难以调试和监控

**现状**:
- 178个 `print()` 语句
- 0个使用 `logging` 模块
- 无法按级别过滤日志
- 无法重定向日志到文件
- 缺少时间戳和上下文信息

**解决方案**: 创建统一的日志系统

**步骤1**: 创建 `utils/logger.py`

```python
#!/usr/bin/env python3
"""
CARLA Dataset Tools 统一日志配置
提供跨模块的结构化日志记录
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 日志颜色（用于终端输出）
class LogColors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅用于终端）"""

    COLORS = {
        'DEBUG': LogColors.CYAN,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.MAGENTA,
    }

    def format(self, record):
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"
            )
        return super().format(record)

def setup_logger(
    name: str = "carla_dataset_tools",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    配置并返回logger实例

    Args:
        name: Logger名称（通常使用 __name__）
        log_file: 日志文件路径（可选）
        level: 日志级别
        console_output: 是否输出到控制台

    Returns:
        配置好的logger实例

    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("开始录制数据")
        >>> logger.error("传感器 camera_01 失败")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler（带颜色）
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # 文件handler（无颜色，包含更多信息）
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件中记录所有级别
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger

# 全局默认logger
_default_logger = None

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger实例（便捷函数）

    Args:
        name: Logger名称，默认使用全局logger

    Returns:
        Logger实例
    """
    global _default_logger

    if name:
        return logging.getLogger(name)

    if _default_logger is None:
        _default_logger = setup_logger()

    return _default_logger

def configure_global_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
):
    """
    配置全局日志设置

    Args:
        level: 日志级别
        log_file: 日志文件路径

    Example:
        >>> from utils.logger import configure_global_logging
        >>> import logging
        >>> configure_global_logging(
        ...     level=logging.DEBUG,
        ...     log_file='logs/recording.log'
        ... )
    """
    global _default_logger
    _default_logger = setup_logger(
        name="carla_dataset_tools",
        level=level,
        log_file=log_file
    )
```

**步骤2**: 替换print语句

优先处理的文件（按重要性排序）：
1. `data_recorder.py` - 主入口
2. `recorder/actor_tree.py` - 数据保存核心
3. `recorder/vehicle.py` - 车辆控制
4. `recorder/sensor.py` - 传感器基类
5. `config/config_manager.py` - 配置管理

示例替换（data_recorder.py）:
```python
# 旧代码
print("World settings:", settings)
print(f"✓ Weather set to {weather_preset}")
print("Start recording to {}".format(carla_logfile))

# 新代码
from utils.logger import get_logger
logger = get_logger(__name__)

logger.info(f"World settings: {settings}")
logger.info(f"Weather set to {weather_preset}")
logger.info(f"Start recording to {carla_logfile}")
```

**步骤3**: 在主程序中配置日志

```python
# data_recorder.py main()
def main():
    # ... 参数解析 ...

    # 配置全局日志
    from utils.logger import configure_global_logging
    import logging

    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = None
    if hasattr(args, 'log_file') and args.log_file:
        log_file = args.log_file

    configure_global_logging(level=log_level, log_file=log_file)
    logger = get_logger(__name__)

    logger.info("="*60)
    logger.info("CARLA Dataset Tools - Data Recorder")
    logger.info("="*60)

    # ... 其余代码 ...
```

**预估工作量**:
- 创建日志系统: 4小时
- 替换print语句: 8小时（178个语句）
- 测试: 2小时
- **总计**: 14小时

**优先级**: P1 (高)

---

### 5. 🟡 过于宽泛的异常处理

**严重程度**: 🟡 中等
**影响**: 隐藏真实错误，难以调试

**发现**: 18处使用 `except Exception:` 的宽泛异常处理

**问题示例1** - data_recorder.py:228-231:
```python
except Exception as e:
    print(f"Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
    return 1
```

**改进**:
```python
except (yaml.YAMLError, ConfigValidationError) as e:
    logger.error(f"配置错误: {e}")
    return 1
except FileNotFoundError as e:
    logger.error(f"配置文件未找到: {e}")
    return 1
except Exception as e:
    # 只捕获真正意外的错误
    logger.exception(f"意外错误: {e}")
    return 1
```

**问题示例2** - recorder/vehicle.py:210-213:
```python
except Exception as e:
    import traceback
    print("\tERROR: Failed to save vehicle status: {}".format(e))
    traceback.print_exc()
```

**改进**:
```python
except IOError as e:
    logger.error(f"无法写入车辆状态到 {self.save_dir}: {e}")
    raise  # 重新抛出，通知调用者
except KeyError as e:
    logger.error(f"车辆状态数据缺少字段: {e}")
    raise
except Exception as e:
    logger.exception(f"保存车辆 {self.uid} 状态时发生意外错误: {e}")
    raise
```

**待修复的文件**:
- [ ] `data_recorder.py` (3处)
- [ ] `recorder/vehicle.py` (2处)
- [ ] `recorder/actor_factory.py` (4处)
- [ ] `config/config_manager.py` (3处)
- [ ] `label_tools/kitti_objects_label.py` (估计6处)

**预估工作量**: 6小时
**优先级**: P1 (高)

---

### 6. 🟡 缺少类型注解

**严重程度**: 🟡 中等
**影响**: 代码可读性差，IDE支持弱

**现状**: 仅约10%的代码有类型注解

**目标**: 为所有公共API添加类型注解

**优先处理的模块**:
1. `config/config_manager.py` - 配置管理
2. `recorder/actor_factory.py` - 工厂类
3. `recorder/vehicle.py` - 车辆类
4. `recorder/sensor.py` - 传感器基类
5. `utils/transform.py` - 坐标转换

**示例改进** - recorder/actor.py:
```python
# 改进前
def get_velocity(self):
    return self.carla_actor.get_velocity()

def get_speed(self):
    v = self.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

# 改进后
from typing import Optional
import carla

def get_velocity(self) -> carla.Vector3D:
    """
    获取actor在世界坐标系中的速度

    Returns:
        速度向量 (m/s)
    """
    return self.carla_actor.get_velocity()

def get_speed(self) -> float:
    """
    获取actor的速度标量

    Returns:
        速度标量 (m/s)
    """
    v = self.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)
```

**步骤**:
1. 安装mypy: `pip install mypy`
2. 创建 `mypy.ini` 配置文件
3. 逐模块添加类型注解
4. 运行 `mypy .` 检查类型错误
5. 修复类型错误

**mypy.ini 配置**:
```ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
warn_redundant_casts = True
warn_unused_ignores = True
disallow_untyped_defs = False
disallow_incomplete_defs = False
check_untyped_defs = True
strict_optional = True
warn_no_return = True

# 逐步迁移策略：先检查，不强制
disallow_untyped_calls = False
disallow_untyped_decorators = False

# 第三方库存根
ignore_missing_imports = True
```

**预估工作量**:
- 配置mypy: 1小时
- 添加类型注解: 20小时（47个模块）
- 修复类型错误: 8小时
- **总计**: 29小时

**优先级**: P2 (中)

---

## 🧪 测试改进（中期目标）

### 7. 🟡 测试覆盖率不足

**严重程度**: 🟡 中等
**影响**: 难以重构，容易引入bug

**现状**:
- 测试覆盖率: ~5%
- 仅有5个手动测试脚本
- 无自动化测试框架
- 无CI/CD流水线

**目标**:
- 测试覆盖率: 70%+
- 自动化单元测试
- 集成测试
- CI/CD自动运行

**步骤1**: 安装测试框架

添加到 `requirements-dev.txt`:
```
# 测试框架
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-timeout>=2.1.0
pytest-xdist>=3.3.0  # 并行测试

# 代码质量
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pylint>=2.17.0
```

**步骤2**: 创建测试目录结构

```
tests/
├── __init__.py
├── conftest.py                      # pytest fixtures
├── unit/                            # 单元测试
│   ├── __init__.py
│   ├── test_config_manager.py       # 配置管理测试
│   ├── test_actor_factory.py        # 工厂测试
│   ├── test_transform.py            # 坐标转换测试
│   ├── test_vehicle.py              # 车辆测试
│   └── test_sensors.py              # 传感器测试
├── integration/                     # 集成测试
│   ├── __init__.py
│   ├── test_data_recording.py       # 录制流程测试
│   └── test_route_following.py      # 路径跟随测试
└── fixtures/                        # 测试数据
    ├── configs/                     # 测试配置文件
    │   ├── valid_config.yaml
    │   ├── invalid_map.yaml
    │   └── missing_fields.yaml
    └── routes/                      # 测试路由文件
        └── test_route.yaml
```

**步骤3**: 创建 pytest 配置

`pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# 详细输出
addopts =
    --verbose
    --strict-markers
    --tb=short
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --cov-report=xml
    --cov-fail-under=70
    -n auto  # 并行运行

# 超时设置
timeout = 300

# 标记
markers =
    slow: 运行时间较长的测试
    integration: 集成测试
    unit: 单元测试
    requires_carla: 需要CARLA服务器的测试
```

**步骤4**: 编写核心模块测试

详见下方"测试用例示例"部分。

**预估工作量**:
- 配置测试框架: 2小时
- 编写ConfigManager测试: 6小时
- 编写ActorFactory测试: 8小时
- 编写Transform测试: 4小时
- 编写Vehicle测试: 8小时
- 编写Sensor测试: 8小时
- 集成测试: 12小时
- **总计**: 48小时

**优先级**: P2 (中)

---

## 📚 文档改进

### 8. 🟢 API文档缺失

**严重程度**: 🟢 低
**影响**: 新开发者上手困难

**目标**: 使用Sphinx生成完整的API文档

**步骤**:

1. 安装Sphinx
```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

2. 初始化文档
```bash
cd docs
sphinx-quickstart
```

3. 配置 `docs/conf.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

html_theme = 'sphinx_rtd_theme'
```

4. 编写docstrings（Google风格）
```python
def create_vehicle_node(self, actor_info: dict) -> Node:
    """
    根据配置创建车辆节点

    Args:
        actor_info: 车辆配置字典，包含:
            type (str): CARLA车辆蓝图ID，如 'vehicle.tesla.model3'
            name (str): 唯一的车辆标识符
            spawn_point (int | dict): 出生点索引或坐标字典
            route (dict, optional): 路径配置
            sensors (list, optional): 传感器配置列表

    Returns:
        车辆节点，包含车辆actor和传感器子节点

    Raises:
        RuntimeError: 如果无法在指定位置生成车辆
        KeyError: 如果缺少必需的配置键

    Example:
        >>> factory = ActorFactory(world, "./data")
        >>> config = {
        ...     'type': 'vehicle.tesla.model3',
        ...     'name': 'ego_vehicle',
        ...     'spawn_point': 50,
        ...     'sensors': []
        ... }
        >>> node = factory.create_vehicle_node(config)
        >>> assert node.get_node_type() == NodeType.VEHICLE
    """
```

5. 构建文档
```bash
cd docs
make html
```

**预估工作量**: 16小时
**优先级**: P3 (低)

---

## 🚀 性能优化

### 9. 🟢 数据保存性能优化

**严重程度**: 🟢 低
**影响**: 高传感器密度时可能掉帧

**优化点**:

1. **使用异步I/O**
```python
import aiofiles
import asyncio

async def async_save_image(path: str, data: np.ndarray):
    """异步保存图像"""
    # OpenCV的imwrite是同步的，使用线程池
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, cv.imwrite, path, data)

async def async_save_lidar(path: str, data: np.ndarray):
    """异步保存点云"""
    async with aiofiles.open(path, 'wb') as f:
        await f.write(data.tobytes())
```

2. **批量写入CSV**
```python
# 当前: 每帧写入一次CSV
with open(csv_path, 'a') as f:
    writer.writerow(data)

# 优化: 缓存后批量写入
class BufferedCSVWriter:
    def __init__(self, path, buffer_size=100):
        self.path = path
        self.buffer = []
        self.buffer_size = buffer_size

    def write_row(self, data):
        self.buffer.append(data)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.buffer:
            with open(self.path, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=...)
                writer.writerows(self.buffer)
            self.buffer.clear()
```

3. **压缩存储**
```python
# LiDAR点云使用压缩
np.savez_compressed(f"{save_dir}/{frame:010d}.npz", points=points)

# 或使用更高效的格式
import pickle
with open(f"{save_dir}/{frame:010d}.pkl", 'wb') as f:
    pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)
```

**预估工作量**: 12小时
**优先级**: P3 (低)

---

## 🔧 代码质量改进

### 10. 🟡 移除全局变量

**文件**: `data_recorder.py:14, 17-19`
**问题**: 使用全局变量进行信号处理

```python
sig_interrupt = False  # ❌ 全局可变状态

def signal_handler(signal, frame):
    global sig_interrupt
    sig_interrupt = True
```

**改进**: 使用类属性
```python
class DataRecorder:
    def __init__(self, args):
        # ...
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        logger = get_logger(__name__)
        logger.info("接收到中断信号 (Ctrl+C)，准备优雅关闭...")
        self.interrupted = True

    def start_record(self, config):
        # ...
        while True:
            # ...
            if self.interrupted:
                logger.info("用户中断，退出录制...")
                break
```

**预估工作量**: 1小时
**优先级**: P2 (中)

---

### 11. 🟡 清理注释代码和TODO

**发现**:
- 3处TODO注释
- 多处注释掉的代码

**待处理**:

1. **recorder/vehicle.py:37-43** - 注释掉的autopilot代码
```python
# TODO: Migration with agents.behavior_agent
# if not self.auto_pilot:
#     self.carla_actor.set_autopilot()
#     self.auto_pilot = True
# else:
#     return
```
**建议**: 删除或实现

2. **recorder/vehicle.py:216-217** - 未实现的方法
```python
def save_vehicle_info(self):
    # TODO: Save vehicle physics info here
    pass
```
**建议**: 实现此功能（见下方实现示例）

3. **recorder/world.py** - 未实现的世界边界框导出
```python
# TODO: Save all object bbox in world
```
**建议**: 创建GitHub issue跟踪

**实现示例 - save_vehicle_info()**:
```python
def save_vehicle_info(self):
    """保存车辆物理信息到JSON文件"""
    import json

    physics = self.carla_actor.get_physics_control()

    vehicle_info = {
        'vehicle_type': self.vehicle_type,
        'physics': {
            'mass': physics.mass,  # kg
            'drag_coefficient': physics.drag_coefficient,
            'max_rpm': physics.max_rpm,
            'moi': physics.moi,  # kg*m^2
            'damping_rate_full_throttle': physics.damping_rate_full_throttle,
            'damping_rate_zero_throttle_clutch_engaged':
                physics.damping_rate_zero_throttle_clutch_engaged,
            'damping_rate_zero_throttle_clutch_disengaged':
                physics.damping_rate_zero_throttle_clutch_disengaged,
            'center_of_mass': {
                'x': physics.center_of_mass.x,
                'y': physics.center_of_mass.y,
                'z': physics.center_of_mass.z
            }
        },
        'bounding_box': {
            'extent': {
                'x': self.carla_actor.bounding_box.extent.x,
                'y': self.carla_actor.bounding_box.extent.y,
                'z': self.carla_actor.bounding_box.extent.z
            },
            'location': {
                'x': self.carla_actor.bounding_box.location.x,
                'y': self.carla_actor.bounding_box.location.y,
                'z': self.carla_actor.bounding_box.location.z
            }
        }
    }

    info_file = f'{self.save_dir}/vehicle_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(vehicle_info, f, indent=2, ensure_ascii=False)

    logger = get_logger(__name__)
    logger.debug(f"车辆物理信息已保存到 {info_file}")
```

**预估工作量**: 4小时
**优先级**: P2 (中)

---

## 🔄 依赖管理改进

### 12. 🟡 精确的依赖版本管理

**现状**:
```
opencv-python>4.0          # 过于宽泛
carla >= 0.9.16            # 缺少上限
numpy<2.0,>=1.24.4         # ✓ 好的！
```

**改进后的 requirements.txt**:
```
# 核心依赖
opencv-python>=4.5.0,<5.0.0
numpy>=1.24.4,<2.0.0
transforms3d>=0.4.2,<0.5.0
pyyaml>=6.0,<7.0

# CARLA API - 锁定到测试的版本
carla==0.9.16

# 可视化
open3d>=0.17.0,<0.18.0
matplotlib>=3.5.0,<4.0.0

# 几何计算
shapely>=2.0.0,<3.0.0
networkx>=3.0,<4.0
```

**创建 requirements-dev.txt**:
```
# 测试框架
pytest>=7.4.0,<8.0.0
pytest-cov>=4.1.0,<5.0.0
pytest-mock>=3.11.0,<4.0.0
pytest-timeout>=2.1.0,<3.0.0
pytest-xdist>=3.3.0,<4.0.0

# 代码质量工具
black>=23.0.0,<24.0.0
flake8>=6.0.0,<7.0.0
mypy>=1.5.0,<2.0.0
pylint>=2.17.0,<3.0.0
isort>=5.12.0,<6.0.0

# 文档生成
sphinx>=7.0.0,<8.0.0
sphinx-rtd-theme>=1.3.0,<2.0.0
sphinx-autodoc-typehints>=1.24.0,<2.0.0
```

**创建 setup.py** 使项目可安装:
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="carla-dataset-tools",
    version="1.0.0",
    author="Kevin LAD Lee",
    description="CARLA模拟器的数据收集和标注工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevinLADLee/carla_dataset_tools",
    packages=find_packages(exclude=["tests", "test_code", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": open("requirements-dev.txt").readlines(),
    },
    entry_points={
        "console_scripts": [
            "carla-record=data_recorder:main",
            "carla-label-kitti=label_tools.kitti_objects_label:main",
            "carla-route-editor=utils.route_editor:main",
        ],
    },
)
```

**预估工作量**: 3小时
**优先级**: P2 (中)

---

## 📋 测试用例示例

### ConfigManager 测试

`tests/unit/test_config_manager.py`:
```python
"""ConfigManager单元测试"""
import pytest
from pathlib import Path
from config.config_manager import ConfigManager, ConfigValidationError

@pytest.fixture
def temp_config_dir(tmp_path):
    """创建临时配置目录"""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    return tmp_path

@pytest.fixture
def config_manager(temp_config_dir):
    """创建ConfigManager实例"""
    return ConfigManager(str(temp_config_dir))

@pytest.fixture
def valid_config_content():
    """有效的配置内容"""
    return """
recording:
  frame_total: 1000
  frame_step: 1
  map: Town02
  weather: ClearNoon

world_settings:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
  substepping: false
  max_substep_delta_time: 0.01
  max_substeps: 10

traffic_lights:
  red_time: 2.0
  green_time: 2.0
  yellow_time: 1.0

actors:
  - type: vehicle.tesla.model3
    name: test_vehicle
    spawn_point: 0
    sensors: []
"""

class TestConfigManagerBasics:
    """基础功能测试"""

    def test_list_profiles_empty(self, config_manager):
        """测试空配置目录"""
        profiles = config_manager.list_profiles()
        assert profiles == []

    def test_load_nonexistent_profile(self, config_manager):
        """测试加载不存在的配置"""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            config_manager.load_profile("nonexistent")

    def test_load_valid_profile(self, temp_config_dir, config_manager, valid_config_content):
        """测试加载有效配置"""
        # 创建配置文件
        profile_file = temp_config_dir / "profiles" / "test.yaml"
        profile_file.write_text(valid_config_content)

        # 加载配置
        config = config_manager.load_profile("test")

        # 验证
        assert config['recording']['map'] == 'Town02'
        assert config['recording']['weather'] == 'ClearNoon'
        assert config['recording']['frame_total'] == 1000
        assert len(config['actors']) == 1

class TestConfigValidation:
    """配置验证测试"""

    def test_invalid_map_name(self, temp_config_dir, config_manager):
        """测试无效的地图名称"""
        invalid_config = """
recording:
  frame_total: 1000
  frame_step: 1
  map: InvalidMapName123
world_settings:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
  substepping: false
  max_substep_delta_time: 0.01
  max_substeps: 10
traffic_lights:
  red_time: 2.0
  green_time: 2.0
  yellow_time: 1.0
actors: []
"""
        profile_file = temp_config_dir / "profiles" / "invalid.yaml"
        profile_file.write_text(invalid_config)

        with pytest.raises(ConfigValidationError, match="Invalid map"):
            config_manager.load_profile("invalid")

    def test_missing_required_section(self, temp_config_dir, config_manager):
        """测试缺少必需部分"""
        incomplete_config = """
recording:
  frame_total: 1000
  frame_step: 1
  map: Town02
"""
        profile_file = temp_config_dir / "profiles" / "incomplete.yaml"
        profile_file.write_text(incomplete_config)

        with pytest.raises(ConfigValidationError, match="Missing required section"):
            config_manager.load_profile("incomplete")

    def test_file_too_large(self, temp_config_dir, config_manager):
        """测试文件大小限制"""
        large_file = temp_config_dir / "profiles" / "large.yaml"
        with open(large_file, 'w') as f:
            # 写入超过10MB的内容
            f.write("x: " + "a" * (11 * 1024 * 1024))

        with pytest.raises(ConfigValidationError, match="too large"):
            config_manager.load_profile("large")

    def test_invalid_weather_preset(self, temp_config_dir, config_manager, valid_config_content):
        """测试无效的天气预设"""
        config = valid_config_content.replace("ClearNoon", "InvalidWeather")
        profile_file = temp_config_dir / "profiles" / "bad_weather.yaml"
        profile_file.write_text(config)

        with pytest.raises(ConfigValidationError, match="Invalid weather preset"):
            config_manager.load_profile("bad_weather")

class TestSensorValidation:
    """传感器配置验证测试"""

    def test_invalid_sensor_type(self, temp_config_dir, config_manager):
        """测试无效的传感器类型"""
        config = """
recording:
  frame_total: 100
  frame_step: 1
  map: Town02
world_settings:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
  substepping: false
  max_substep_delta_time: 0.01
  max_substeps: 10
traffic_lights:
  red_time: 2.0
  green_time: 2.0
  yellow_time: 1.0
actors:
  - type: vehicle.tesla.model3
    name: test
    spawn_point: 0
    sensors:
      - type: sensor.invalid.type
        name: invalid_sensor
        spawn_point: {x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0}
"""
        profile_file = temp_config_dir / "profiles" / "invalid_sensor.yaml"
        profile_file.write_text(config)

        with pytest.raises(ConfigValidationError, match="invalid type"):
            config_manager.load_profile("invalid_sensor")

class TestRouteValidation:
    """路由配置验证测试"""

    def test_route_with_less_than_two_waypoints(self, temp_config_dir, config_manager):
        """测试少于2个航点的路由"""
        config = """
recording:
  frame_total: 100
  frame_step: 1
  map: Town02
world_settings:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
  substepping: false
  max_substep_delta_time: 0.01
  max_substeps: 10
traffic_lights:
  red_time: 2.0
  green_time: 2.0
  yellow_time: 1.0
actors:
  - type: vehicle.tesla.model3
    name: test
    spawn_point: 0
    route:
      mode: strict
      waypoints:
        - {x: 0, y: 0, z: 0}
    sensors: []
"""
        profile_file = temp_config_dir / "profiles" / "bad_route.yaml"
        profile_file.write_text(config)

        with pytest.raises(ConfigValidationError, match="at least 2 waypoints"):
            config_manager.load_profile("bad_route")
```

**预估工作量**: 包含在测试改进任务中
**优先级**: P2 (中)

---

## 📅 开发计划

### 阶段1：关键问题修复（第1-2周）

**目标**: 修复所有严重和高优先级问题

**任务列表**:
- [x] Code review完成
- [ ] **P0-1**: 修复线程安全问题 (actor_tree.py) - 4小时
- [ ] **P0-2**: 修复传感器队列阻塞 (sensor.py) - 3小时
- [ ] **P1-3**: 修复路径遍历风险 (actor_factory.py) - 2小时
- [ ] **P1-4**: 创建日志系统 (utils/logger.py) - 4小时
- [ ] **P1-5**: 替换关键模块的print语句 - 8小时
  - [ ] data_recorder.py
  - [ ] recorder/actor_tree.py
  - [ ] recorder/vehicle.py
  - [ ] recorder/sensor.py
  - [ ] config/config_manager.py
- [ ] **P1-6**: 修复宽泛异常处理（前5个文件） - 6小时
- [ ] **P2-10**: 移除全局变量 - 1小时
- [ ] 测试所有修复
- [ ] 更新文档

**时间估计**: 28小时 → **1-2周**（考虑测试和文档）

**交付物**:
- 无线程安全问题的代码
- 统一的日志系统
- 更好的错误处理
- 更新的开发者文档

---

### 阶段2：代码质量提升（第3-6周）

**目标**: 提升代码可维护性和可读性

**任务列表**:
- [ ] **P1-5**: 完成所有模块的日志迁移 - 6小时
- [ ] **P1-6**: 修复剩余的宽泛异常处理 - 3小时
- [ ] **P2-11**: 清理注释代码和TODO - 4小时
  - [ ] 实现 save_vehicle_info()
  - [ ] 删除过时注释
  - [ ] 为未完成功能创建GitHub issues
- [ ] **P2-12**: 依赖管理改进 - 3小时
  - [ ] 更新 requirements.txt
  - [ ] 创建 requirements-dev.txt
  - [ ] 创建 setup.py
- [ ] **P2-6**: 添加类型注解（优先模块） - 20小时
  - [ ] config/config_manager.py
  - [ ] recorder/actor_factory.py
  - [ ] recorder/vehicle.py
  - [ ] recorder/sensor.py
  - [ ] utils/transform.py
  - [ ] 配置mypy
  - [ ] 修复类型错误
- [ ] 代码格式化（black, isort）
- [ ] 代码检查（flake8, pylint）

**时间估计**: 36小时 → **3-4周**

**交付物**:
- 完整的日志系统
- 清理的代码库
- 类型注解的核心模块
- 标准化的代码格式

---

### 阶段3：测试框架建设（第7-12周）

**目标**: 建立完善的测试体系

**任务列表**:
- [ ] **P2-7**: 测试框架配置 - 2小时
  - [ ] 安装pytest和插件
  - [ ] 配置pytest.ini
  - [ ] 创建tests目录结构
- [ ] **P2-7**: 单元测试开发 - 30小时
  - [ ] ConfigManager测试 - 6小时
  - [ ] ActorFactory测试 - 8小时
  - [ ] Transform测试 - 4小时
  - [ ] Vehicle测试 - 8小时
  - [ ] Sensor测试 - 8小时
  - [ ] 其他核心模块 - 6小时
- [ ] **P2-7**: 集成测试 - 12小时
  - [ ] 数据录制流程测试
  - [ ] 路径跟随测试
  - [ ] 多传感器协同测试
- [ ] **P2-7**: CI/CD配置 - 4小时
  - [ ] GitHub Actions配置
  - [ ] 代码覆盖率报告
  - [ ] 自动化测试流程

**时间估计**: 48小时 → **5-6周**

**交付物**:
- 70%+测试覆盖率
- 自动化测试流程
- CI/CD管道
- 测试文档

---

### 阶段4：文档和优化（第13-16周）

**目标**: 完善文档，优化性能

**任务列表**:
- [ ] **P3-8**: API文档生成 - 16小时
  - [ ] 配置Sphinx
  - [ ] 编写docstrings
  - [ ] 生成HTML文档
  - [ ] 发布到GitHub Pages
- [ ] **P3-9**: 性能优化 - 12小时
  - [ ] 性能分析（profiling）
  - [ ] 异步I/O实现
  - [ ] 批量CSV写入
  - [ ] 数据压缩
- [ ] 创建故障排除指南
- [ ] 录制使用视频教程
- [ ] 更新README和用户指南

**时间估计**: 28小时 → **3-4周**

**交付物**:
- 完整的API文档
- 性能优化的数据保存
- 全面的用户文档
- 视频教程

---

## 📈 里程碑和验收标准

### 里程碑1：稳定版本 v1.1.0（第2周）
**验收标准**:
- ✅ 无线程安全相关bug
- ✅ 所有关键路径有结构化日志
- ✅ 无全局可变状态
- ✅ 通过手动测试（10个传感器同时录制）

### 里程碑2：质量版本 v1.2.0（第6周）
**验收标准**:
- ✅ 代码通过black格式化
- ✅ 代码通过flake8检查
- ✅ 核心模块有类型注解
- ✅ mypy检查无错误
- ✅ 无宽泛异常处理
- ✅ 所有TODO已处理

### 里程碑3：测试版本 v1.3.0（第12周）
**验收标准**:
- ✅ 测试覆盖率 ≥ 70%
- ✅ 所有测试通过
- ✅ CI/CD自动运行
- ✅ 代码覆盖率报告自动生成
- ✅ 无关键bug

### 里程碑4：生产版本 v2.0.0（第16周）
**验收标准**:
- ✅ 完整的API文档
- ✅ 用户指南更新
- ✅ 故障排除文档
- ✅ 性能提升20%+
- ✅ 视频教程发布

---

## 🎯 度量指标

### 代码质量指标

| 指标 | 当前值 | 阶段1目标 | 阶段2目标 | 阶段3目标 | 最终目标 |
|-----|--------|----------|----------|----------|---------|
| 测试覆盖率 | ~5% | 10% | 30% | 70% | 80% |
| 结构化日志 | 0% | 30% | 100% | 100% | 100% |
| 类型注解 | ~10% | 15% | 50% | 70% | 85% |
| 宽泛异常处理 | 18处 | 10处 | 0处 | 0处 | 0处 |
| TODO数量 | 3个 | 5个 | 0个 | 0个 | 0个 |
| 注释代码行数 | ~20行 | 10行 | 0行 | 0行 | 0行 |

### 性能指标

| 指标 | 当前值 | 目标值 |
|-----|--------|--------|
| 10传感器同时录制 | ~8 FPS | 10+ FPS |
| 内存使用 | 基线 | <120%基线 |
| 磁盘写入速度 | 基线 | +20% |

### 文档指标

| 指标 | 当前值 | 目标值 |
|-----|--------|--------|
| API文档覆盖率 | 0% | 90% |
| Docstring覆盖率 | ~20% | 85% |
| 教程数量 | 0个 | 3个视频 |

---

## 🔄 持续改进

### 每周检查清单

**每周一**:
- [ ] 回顾上周完成的任务
- [ ] 更新本TODO.md
- [ ] 计划本周任务
- [ ] 更新GitHub项目看板

**每周五**:
- [ ] 运行完整测试套件
- [ ] 检查代码覆盖率变化
- [ ] 更新CHANGELOG.md
- [ ] 创建周报

### 代码审查检查清单

每次PR都应检查:
- [ ] 所有新代码有单元测试
- [ ] 测试覆盖率未下降
- [ ] 使用logger而非print
- [ ] 有适当的类型注解
- [ ] 有docstrings
- [ ] 通过mypy检查
- [ ] 通过flake8检查
- [ ] 通过black格式化

---

## 📞 联系和反馈

如有疑问或建议，请通过以下方式反馈：
- GitHub Issues: https://github.com/KevinLADLee/carla_dataset_tools/issues
- Pull Requests欢迎！

---

**最后更新**: 2025-11-09
**下次审查**: 2025-11-16 (一周后)
