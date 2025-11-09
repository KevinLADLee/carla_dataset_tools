# CARLA 数据集工具 - 开发者指南

本指南涵盖架构、配置详情、API 参考和开发者高级用法。

## 目录

- [架构概述](#架构概述)
- [配置系统](#配置系统)
- [API 参考](#api-参考)
- [高级用法](#高级用法)
- [扩展工具包](#扩展工具包)
- [开发工作流](#开发工作流)

---

## 架构概述

### 项目结构

```
carla_dataset_tools/
├── config/                      # 配置管理
│   ├── config_manager.py        # YAML 配置加载器和验证器
│   └── profiles/                # 预配置文件
│       ├── default.yaml         # 默认配置
│       ├── kitti.yaml           # KITTI 数据集风格
│       ├── argoverse.yaml       # Argoverse 数据集风格
│       ├── simple.yaml          # 简单测试配置
│       └── route_example.yaml   # 路线配置示例
├── label_tools/                 # 标注脚本
│   ├── kitti_objects_label.py   # KITTI 格式标注
│   ├── yolo_label.py            # YOLOv5 格式标注
│   └── kitti_object/            # KITTI 工具
├── recorder/                    # 核心录制模块
│   ├── actor_tree.py            # Actor 层级管理
│   ├── actor_factory.py         # Actor 和传感器生成
│   ├── vehicle.py               # 车辆录制,带路线跟随
│   ├── sensor.py                # 基础传感器类
│   ├── camera.py                # 相机传感器
│   ├── lidar.py                 # 激光雷达传感器
│   ├── radar.py                 # 雷达传感器
│   └── agents/                  # 自动驾驶和导航代理
│       ├── navigation/          # 导航组件
│       │   └── global_route_planner.py  # 拓扑感知路径规划
│       └── ...
├── routes/                      # 车辆路线定义
│   ├── README.md                # 路线系统文档
│   └── *.yaml, *.pkl            # 路线文件 (YAML + pickle)
├── core/                        # 核心共享模块
│   ├── geometry.py              # 几何类型 (Vector3d, Location, Transform 等)
│   ├── types.py                 # 标签和对象类型
│   ├── transform.py             # 坐标转换
│   ├── converters.py            # 数据格式转换器
│   └── logger.py                # 统一日志系统
├── tools/                       # CLI 工具脚本
│   ├── viz_lidar.py             # 点云可视化
│   ├── viz_map.py               # 地图可视化
│   ├── viz_actor_tree.py        # Actor树可视化（录制前）
│   ├── editor_route.py          # 交互式路线创建工具
│   ├── data_generate_imageset.py # 数据集文件列表生成
│   ├── config_convert.py        # JSON 到 YAML 转换器
│   ├── config_list.py           # 列出可用配置文件
│   ├── config_validate.py       # 配置验证工具
│   └── debug_info.py            # 调试信息显示
├── data_recorder.py             # 主录制脚本
└── param.py                     # 全局参数
```

### 核心组件

#### 1. ConfigManager (config/config_manager.py)

集中式配置管理和验证:

```python
class ConfigManager:
    """
    管理 YAML 配置加载和验证

    特性:
    - 基于配置文件的配置
    - CARLA 0.9.16 API 验证
    - 地图和天气预设验证
    - 安全性: 10MB 文件大小限制
    """
```

#### 2. ActorTree (recorder/actor_tree.py)

Actor 和传感器的层级管理:

```python
class ActorTree:
    """
    管理 actor 层级和数据录制

    结构:
    World
    ├── Vehicle_1
    │   ├── Camera_1
    │   ├── LiDAR_1
    │   └── ...
    ├── Vehicle_2
    └── Infrastructure_1
    """
```

#### 3. ActorFactory (recorder/actor_factory.py)

生成和配置 actor 和传感器:

```python
class ActorFactory:
    """
    创建 CARLA actor 的工厂模式

    职责:
    - 生成车辆和基础设施
    - 将传感器附加到父 actor
    - 配置自动驾驶和交通管理器
    """
```

#### 4. 传感器类 (recorder/camera.py, lidar.py, radar.py)

基础传感器类及具体实现:

```python
class Sensor:
    """带回调处理的基础传感器类"""

class Camera(Sensor):
    """RGB、深度、语义分割相机"""

class Lidar(Sensor):
    """光线投射和语义激光雷达"""

class Radar(Sensor):
    """雷达传感器"""
```

---

## 配置系统

### YAML 结构

配置文件使用 YAML 格式,包含以下部分:

```yaml
# 录制设置
recording:
  frame_total: 12000        # 要录制的总帧数
  frame_step: 3             # 每 N 帧保存一次
  map: Town02               # CARLA 地图名称
  weather: ClearNoon        # 天气预设 (可选)

# 观察者相机位置
spectator:
  x: 100.0
  y: -150.0
  z: 150.0
  pitch: 60.0
  yaw: -90.0
  roll: 0.0

# 世界物理设置
world_settings:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
  substepping: true
  max_substep_delta_time: 0.01
  max_substeps: 16

# 交通灯时间
traffic_lights:
  red_time: 2.0
  green_time: 2.0
  yellow_time: 0.01

# 传感器模板 (YAML 锚点用于重用)
sensor_templates:
  rgb_camera: &rgb_camera
    type: sensor.camera.rgb
    image_size_x: 800
    image_size_y: 600
    fov: 90.0

# 车辆和传感器 actor
actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    sensors:
      - <<: *rgb_camera        # 重用模板
        name: front_camera
        spawn_point:
          x: 2.0
          y: 0.0
          z: 2.0
          roll: 0.0
          pitch: 0.0
          yaw: 0.0

# 背景交通
other_vehicles:
  count: 50
  spawn_points: [44, 55, 64]
```

### 可用地图 (CARLA 0.9.16)

**城镇地图:**
- `Town01`, `Town01_Opt` - 简单城镇,基本道路网络
- `Town02`, `Town02_Opt` - 小城镇,各种交叉口
- `Town03`, `Town03_Opt` - 较大城市区域,带环岛
- `Town04`, `Town04_Opt` - 小城镇,带高速公路
- `Town05`, `Town05_Opt` - 城市区域,带桥梁和隧道
- `Town06`, `Town06_Opt` - 城市区域,多车道高速公路
- `Town07`, `Town07_Opt` - 乡村环境,狭窄道路
- `Town10HD`, `Town10HD_Opt` - 高清城市区域
- `Town11`, `Town12`, `Town13`, `Town15` - 其他城市变体

**特殊地图:**
- `AnnotationColorLandscape` - 测试环境

**注意:** `_Opt` 版本具有优化的几何形状,性能更好。

### 天气预设

使用天气预设控制环境条件:

**晴朗天气:**
- `ClearNoon`, `ClearSunset`, `ClearNight` - 晴空条件

**多云天气:**
- `CloudyNoon`, `CloudySunset`, `CloudyNight` - 阴天条件

**潮湿天气:**
- `WetNoon`, `WetSunset`, `WetNight` - 湿路,无雨
- `WetCloudyNoon`, `WetCloudySunset`, `WetCloudyNight` - 湿润多云

**雨天:**
- `SoftRainNoon`, `SoftRainSunset`, `SoftRainNight` - 小雨
- `MidRainyNoon`, `MidRainSunset`, `MidRainyNight` - 中雨
- `HardRainNoon`, `HardRainSunset`, `HardRainNight` - 大雨

**极端天气:**
- `DustStorm` - 沙漠沙尘暴条件

**默认:**
- `Default` - CARLA 的默认天气

### 支持的传感器类型 (CARLA 0.9.16)

- `sensor.camera.rgb` - RGB 相机
- `sensor.camera.depth` - 深度相机
- `sensor.camera.semantic_segmentation` - 语义分割相机
- `sensor.lidar.ray_cast` - 激光雷达
- `sensor.lidar.ray_cast_semantic` - 语义激光雷达
- `sensor.other.radar` - 雷达

### 路线配置

车辆可以配置为使用拓扑感知路径规划跟随预定义路线:

#### YAML 中的路线配置

```yaml
actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    route:
      # 选项 1: 从文件加载
      from_file: routes/Town02_my_route.yaml

      # 选项 2: 内联路径点
      mode: strict              # strict | disabled
      loop: true                # 可选: 用于循环路线
      waypoints:
        - {x: 107.5, y: -133.2, z: 0.3}
        - {x: 150.0, y: -130.5, z: 0.3}
        - {x: 200.3, y: -128.8, z: 0.3}
    sensors: [...]
```

#### 路线文件格式

路线文件 (YAML) 包含路径点定义:

```yaml
# routes/Town02_my_route.yaml
mode: strict
loop: false
waypoints:
  - {x: 107.5, y: -133.2, z: 0.3}
  - {x: 150.0, y: -130.5, z: 0.3}
  - {x: 200.3, y: -128.8, z: 0.3}
  - {x: 250.8, y: -125.1, z: 0.3}
```

相应的 pickle 文件 (`.pkl`) 会自动生成供内部使用。

#### 路线跟随模式

- **`strict`**: 车辆使用 GlobalRoutePlanner 跟随路径点
  - 路径点扩展为完整的道路拓扑感知路径
  - 尊重车道、交叉口和道路结构
  - 示例: 8 个用户路径点 → 450+ 个道路路径点

- **`disabled`**: 忽略路线,使用默认自动驾驶

- **未指定路线**: 默认自动驾驶行为(向后兼容)

#### 创建路线

使用交互式路线编辑器:

```bash
python3 tools/editor_route.py --map Town02 --name my_route
```

**特性:**
- 在地图上可视化选择路径点
- 使用 GlobalRoutePlanner 实时拓扑感知路径预览
- 自动检测循环路线(首尾路径点在 5 米内)
- 撤销支持 (Ctrl+Z)
- 保存 YAML(人类可读)和 PKL(内部)格式

**控制:**
- 左键点击: 添加路径点
- 右键点击圆圈: 删除路径点
- Ctrl+Z: 撤销
- Enter: 保存并退出
- Escape: 取消

#### 实现细节

**路径规划过程:**

1. **用户输入**: 在路线编辑器或 YAML 中定义 3-8 个关键路径点
2. **GlobalRoutePlanner**: 计算路径点之间的完整道路路径
   - 使用 CARLA 的道路拓扑图
   - 尊重车道标记、转弯限制、交叉口
3. **路线扩展**: 路径点扩展到数百个道路路径点
4. **BasicAgent**: 沿着完整路径导航车辆
   - 使用 LocalPlanner 进行轨迹控制
   - 用于转向、油门、刹车的 PID 控制器

**验证规则:**
- 至少需要 2 个路径点
- 每个路径点必须有 x、y、z 坐标
- 在配置加载期间验证路线文件
- 无效的路线文件会触发 ConfigValidationError

### 配置验证

ConfigManager 验证:

1. **文件大小**: 最大 10MB (安全性)
2. **YAML 语法**: 有效的 YAML 结构
3. **必需字段**: 所有必需字段都存在
4. **传感器类型**: 匹配 CARLA 0.9.16 API
5. **地图**: 有效的地图名称
6. **天气**: 有效的天气预设
7. **物理约束**: `fixed_delta_seconds <= max_substep_delta_time * max_substeps`

### YAML 高级特性

#### 锚点和别名

使用 YAML 锚点重用配置:

```yaml
sensor_templates:
  # 使用锚点定义模板
  base_camera: &base_camera
    type: sensor.camera.rgb
    image_size_x: 800
    image_size_y: 600
    fov: 90.0

actors:
  - type: vehicle.tesla.model3
    sensors:
      # 重用模板并覆盖特定字段
      - <<: *base_camera
        name: front_camera
        spawn_point: {x: 2.0, y: 0.0, z: 2.0}

      - <<: *base_camera
        name: rear_camera
        spawn_point: {x: -2.0, y: 0.0, z: 2.0, yaw: 180.0}
```

#### 注释

YAML 支持内联和块注释:

```yaml
recording:
  frame_total: 12000        # 要录制的总帧数
  frame_step: 3             # 每 3 帧保存一次
```

---

## API 参考

### ConfigManager API

```python
from config.config_manager import ConfigManager, ConfigValidationError

# 初始化
config_manager = ConfigManager(config_root="/path/to/config")

# 加载配置文件
config = config_manager.load_profile("kitti")

# 加载自定义配置文件
config = config_manager.load_config("/path/to/config.yaml")

# 列出可用配置文件
profiles = config_manager.list_profiles()

# 验证配置
try:
    config = config_manager.load_profile("my_profile")
except ConfigValidationError as e:
    print(f"验证错误: {e}")
```

### ActorTree API

```python
from recorder.actor_tree import ActorTree

# 使用世界和配置初始化
actor_tree = ActorTree(world, config, save_dir)
actor_tree.init()

# 刷新控制器 (更新自动驾驶)
actor_tree.tick_controller()

# 保存当前帧的数据
actor_tree.tick_data_saving(frame_id, timestamp)

# 清理
actor_tree.destroy()
```

### 传感器 API

```python
from recorder.camera import Camera
from recorder.lidar import Lidar
from recorder.radar import Radar

# 创建传感器实例
camera = Camera(world, sensor_config, parent_actor, save_dir)

# 传感器自动注册回调
# 当调用 tick_data_saving() 时保存数据

# 访问传感器属性
sensor_transform = camera.get_transform()
```

### 变换工具

```python
from core.transform import Transform, Location, Rotation
from core.transform import transform_to_carla_transform

# 创建变换
transform = Transform(
    Location(x=10.0, y=5.0, z=2.0),
    Rotation(pitch=0.0, yaw=90.0, roll=0.0)
)

# 转换为 CARLA 变换
carla_transform = transform_to_carla_transform(transform)

# 应用到 actor
actor.set_transform(carla_transform)
```

---

## 高级用法

### 自定义生成点

在地图中查找生成点:

```python
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()

for i, point in enumerate(spawn_points):
    print(f"生成点 {i}: {point.location}")
```

### 基础设施 (V2X) 录制

包含路侧传感器用于 V2X 场景:

```yaml
actors:
  - type: infrastructure
    name: rsu_intersection_1
    spawn_point:
      x: 41
      y: -240
      z: 15.0
    sensors:
      - type: sensor.camera.rgb
        name: infra_camera
        spawn_point: {x: 0.0, y: 0.0, z: 0.0}
```

### 多车辆同步录制

配置具有不同传感器设置的多个车辆:

```yaml
actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    sensors: [...]

  - type: vehicle.audi.a2
    name: following_vehicle
    spawn_point: 76
    sensors: [...]
```

所有车辆使用 CARLA 的同步模式同步。

### 基于路线的数据采集

配置车辆跟随特定路径以进行可重复的数据采集:

#### 创建自定义路线

```bash
# 启动 CARLA 服务器
cd $CARLA_ROOT && ./CarlaUE4.sh

# 使用交互式编辑器创建路线
python3 tools/editor_route.py --map Town02 --name highway_loop

# 在地图上点击路径点,沿着您想要的路径
# 按 Enter 保存
```

#### 在配置中使用路线

**方法 1: 从文件加载**

```yaml
actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    route:
      from_file: routes/Town02_highway_loop.yaml
    sensors:
      - type: sensor.camera.rgb
        name: front_camera
        spawn_point: {x: 2.0, y: 0.0, z: 2.0}
      - type: sensor.lidar.ray_cast
        name: lidar
        spawn_point: {x: 0.0, y: 0.0, z: 2.5}
```

**方法 2: 内联路径点**

```yaml
actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    route:
      mode: strict
      loop: true
      waypoints:
        - {x: 107.5, y: -133.2, z: 0.3}
        - {x: 150.0, y: -130.5, z: 0.3}
        - {x: 200.3, y: -128.8, z: 0.3}
        - {x: 180.5, y: -170.8, z: 0.3}
    sensors: [...]
```

#### 路线跟随行为

当路线配置为 `mode: strict` 时:

1. **启动时**: GlobalRoutePlanner 计算完整的道路路径
   - 输入: 用户的 8 个路径点
   - 输出: 450+ 个遵循拓扑的道路路径点
   - 控制台: `Vehicle 'ego_vehicle' configured with route: 8 waypoints expanded to 453 road waypoints (LOOP)`

2. **录制期间**: BasicAgent 跟随路径
   - 保持车道纪律
   - 遵守交通灯(可选)
   - 正确处理交叉口
   - 如果 `loop: true` 则循环回到起点

3. **数据采集**: 车辆每次录制都遵循相同路径
   - 可重复的数据集采集
   - 一致的光照/天气条件
   - 多次采集的相同视角比较

#### 示例: 用于连续录制的循环路线

```yaml
# config/profiles/continuous_loop.yaml
recording:
  frame_total: 50000        # 长时间录制
  frame_step: 1
  map: Town02
  weather: ClearNoon

actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    route:
      from_file: routes/Town02_continuous_loop.yaml
    sensors:
      - type: sensor.camera.rgb
        name: front_camera
        spawn_point: {x: 2.0, y: 0.0, z: 2.0}
        image_size_x: 1920
        image_size_y: 1080
```

运行:
```bash
python3 data_recorder.py --config config/profiles/continuous_loop.yaml
```

车辆将持续循环采集数据,直到达到 `frame_total`。

### 自定义天气条件

以编程方式应用自定义天气参数:

```python
import carla

world = client.get_world()
weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=30.0,
    sun_altitude_angle=70.0
)
world.set_weather(weather)
```

### 交通管理器配置

配置交通行为:

```python
tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)
tm.set_global_distance_to_leading_vehicle(2.5)
tm.set_respawn_dormant_vehicles(True)
tm.set_hybrid_physics_mode(True)  # 优化远处车辆
```

---

## 扩展工具包

### 添加新的传感器类型

1. **在 `recorder/` 中创建传感器类:**

```python
from recorder.sensor import Sensor

class MySensor(Sensor):
    def __init__(self, world, sensor_info, parent_actor, save_dir):
        super().__init__(world, sensor_info, parent_actor, save_dir)
        self._init_sensor()

    def _init_sensor(self):
        blueprint = self.world.get_blueprint_library().find(self.sensor_type)
        # 配置蓝图属性
        self.sensor = self.world.spawn_actor(
            blueprint, self.transform, attach_to=self.parent_actor
        )
        self.sensor.listen(self._on_data)

    def _on_data(self, data):
        # 处理并保存传感器数据
        pass
```

2. **在 ActorFactory 中注册** (`recorder/actor_factory.py`):

```python
from recorder.my_sensor import MySensor

class ActorFactory:
    def create_sensor_node(self, sensor_info, parent_node):
        if sensor_type == "sensor.my.type":
            return MySensor(self.world, sensor_info, parent_actor, save_dir)
```

3. **更新 ConfigManager** 验证:

```python
VALID_SENSOR_TYPES = {
    'sensor.my.type',
    # ... 现有类型
}
```

### 添加新的数据集格式

1. **在 `label_tools/` 中创建标注脚本:**

```python
# label_tools/my_format_label.py

def convert_to_my_format(raw_data_path, output_path):
    # 加载原始数据
    # 转换为数据集格式
    # 写入输出文件
    pass
```

2. **遵循现有模式** 从 `kitti_objects_label.py` 或 `yolo_label.py`

3. **添加文档** 到 USER_GUIDE_CN.md

### 自定义配置文件

为特定场景创建专用配置文件:

```yaml
# config/profiles/urban_night.yaml
recording:
  map: Town03
  weather: ClearNight
  frame_total: 5000

# 高灵敏度夜间相机
sensor_templates:
  night_camera: &night_camera
    type: sensor.camera.rgb
    image_size_x: 1920
    image_size_y: 1080
    fov: 90.0
    exposure_mode: manual
    exposure_compensation: 0.5
```

---

## 开发工作流

### 设置开发环境

```bash
# 克隆代码库
git clone https://github.com/KevinLADLee/carla_dataset_tools.git
cd carla_dataset_tools

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装开发工具
pip install pytest black flake8
```

### 运行测试

```bash
# 验证所有配置文件
python3 tools/config_validate.py --all

# 测试配置加载
python3 -c "from config.config_manager import ConfigManager; \
             cm = ConfigManager('config'); \
             config = cm.load_profile('default'); \
             print('成功!')"
```

### 代码风格

遵循 PEP 8 指南:

```bash
# 格式化代码
black data_recorder.py

# 检查风格
flake8 recorder/ --max-line-length=100
```

### Git 工作流

```bash
# 创建功能分支
git checkout -b feature/my-new-feature

# 进行更改并提交
git add .
git commit -m "Add: 我的新功能"

# 推送到远程
git push origin feature/my-new-feature
```

### 调试技���

1. **启用 CARLA 日志:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **检查传感器回调:**
```python
def _on_data(self, data):
    print(f"接收数据: {data.frame} 在 {data.timestamp}")
    # 处理数据
```

3. **验证生成点:**
```bash
python3 tools/debug_info.py --map Town02
```

4. **监控性能:**
```python
import time
start = time.time()
# ... 操作 ...
print(f"操作耗时 {time.time() - start:.3f}s")
```

---

## 贡献

欢迎贡献! 贡献领域:

- 额外的数据集格式支持 (nuScenes, Waymo 等)
- 增强文档和示例
- 错误修复和性能改进
- 新的传感器类型或功能

请向主代码库提交拉取请求。

---

[← 返回 README](../README_CN.md) | [用户指南 ←](USER_GUIDE_CN.md)
