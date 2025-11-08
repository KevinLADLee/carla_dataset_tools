# 🚗 CARLA 数据集工具

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.16-orange.svg)](https://carla.org/)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

> 📦 一套完整的 CARLA 仿真器数据采集与标注工具

用于 [CARLA 仿真器](https://carla.org/) 的数据采集和标注工具。本工具包提供了高效的流水线，用于生成高质量的自动驾驶数据集，支持多种传感器类型和标准数据集格式。

**⚠️ 重要提示**：本工具生成的所有原始数据使用**右手坐标系**。

🌟 本项目是 [**CarlaFLCAV**](https://github.com/SIAT-INVS/CarlaFLCAV) 项目的一部分。欢迎给我们点个 Star！

---

## 📑 目录

- [功能特性](#-功能特性)
- [环境要求](#-环境要求)
- [安装步骤](#-安装步骤)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [数据格式](#-数据格式)
- [高级用法](#-高级用法)
- [常见问题](#-常见问题)
- [项目结构](#-项目结构)
- [参与贡献](#-参与贡献)
- [引用说明](#-引用说明)
- [致谢](#-致谢)

---

## ✨ 功能特性

- ✅ **多传感器支持**：RGB 相机、语义分割、激光雷达、语义激光雷达、毫米波雷达
- ✅ **多种数据集格式**：KITTI Object、YOLOv5、Argoverse
- ✅ **灵活配置**：基于 JSON 的世界、角色和传感器配置
- ✅ **路侧单元支持**：支持 V2X 场景的路侧设备（RSU）仿真
- ✅ **同步录制**：多车辆和多传感器的同步数据采集
- ✅ **可视化工具**：内置点云和数据可视化工具
- ✅ **自动驾驶集成**：使用 CARLA 交通管理器的自动车辆控制

---

## 🔧 环境要求

开始之前，请确保你已具备以下条件：

- **CARLA 仿真器** >= 0.9.16
- **Python** >= 3.8
- **CARLA Python API**（CARLA 发行版自带）
- **操作系统**：Linux（推荐）/ Windows

> 📥 **下载 CARLA**：[https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)

---

## 📦 安装步骤

### 步骤 1：克隆仓库

```bash
git clone https://github.com/KevinLADLee/carla_dataset_tools.git
cd carla_dataset_tools
```

### 步骤 2：安装依赖

```bash
pip3 install -r requirements.txt
```

**requirements.txt 包含以下依赖：**
- opencv-python > 4.0
- carla >= 0.9.16
- numpy < 2.0, >= 1.24.4
- transforms3d ~= 0.4.2
- open3d
- pandas
- shapely
- networkx

### 步骤 3：配置环境变量

在你的 `~/.bashrc` 或 `~/.zshrc` 中添加以下内容：

```bash
# 设置 CARLA 根目录
export CARLA_ROOT=/path/to/your/carla
```

**请替换：**
- `/path/to/your/carla` 为你实际的 CARLA 安装路径

然后重新加载 shell 配置：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### 步骤 4：验证安装

```bash
python3 -c "import carla; print(f'CARLA 版本: {carla.__version__}')"
```

---

## 🚀 快速开始

### 1. 启动 CARLA 仿真器

首先，启动 CARLA 服务器：

```bash
cd $CARLA_ROOT
./CarlaUE4.sh
```

无界面模式（无渲染）：

```bash
./CarlaUE4.sh -RenderOffScreen
```

### 2. 录制数据

使用默认配置运行数据录制器：

```bash
python3 data_recorder.py
```

使用自定义配置：

```bash
python3 data_recorder.py -w world_config_template.json
```

**🎮 控制选项：**
- 录制器将自动采集数据，直到达到配置的帧数
- 按 `Ctrl+C` 可手动停止录制

**📁 输出位置：**
数据将保存到：`raw_data/record_YYYY_MMDD_HHMM/`

### 3. 生成标签

#### KITTI Object 格式

```bash
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303
```

**可选参数：**
```bash
# 指定车辆
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303 -v vehicle.tesla.model3_1

# 指定传感器
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303 -l velodyne -c image_2

# 自定义输出目录
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303 -o my_dataset
```

#### YOLOv5 格式

```bash
python3 label_tools/yolo_label.py -r record_2022_0119_1303
```

#### Argoverse 格式（非稳定版）

```bash
python3 label_tools/argoverse_label.py -r record_2022_0119_1303
```

### 4. 数据可视化

#### 可视化点云

```bash
# 可视化单个文件
python3 utils/visualize_lidar.py --type lidar --source raw_data/record_2022_0119_1303/vehicle.tesla.model3_1/000001_lidar.npy

# 可视化所有帧（通配模式）
python3 utils/visualize_lidar.py --type lidar --source raw_data/record_2022_0119_1303/vehicle.tesla.model3_1/
```

**支持的类型：**
- `lidar` - 标准激光雷达点云
- `semantic_lidar` - 带类别标签的语义激光雷达
- `radar` - 毫米波雷达检测点

---

## ⚙️ 配置说明

所有配置文件位于 `config/` 目录下。

### 📋 世界配置

**文件**：`config/world_config_template.json`

```json
{
    "frame_total": 12000,        // 录制总帧数
    "frame_step": 3,              // 每 N 帧保存一次数据
    "map": "Town02",              // CARLA 地图名称
    "spectator_pose": {...},      // 观察者视角位置
    "world_settings": {
        "fixed_delta_seconds": 0.1,
        "max_substep_delta_time": 0.01,
        "max_substeps": 16
    },
    "actor_settings": "actor_settings_template.json",
    "traffic_light_setting": {
        "red_time": 2.0,
        "yellow_time": 0.01,
        "green_time": 2.0
    }
}
```

**可用地图**：Town01、Town02、Town03、Town04、Town05、Town06、Town07、Town10HD

### 🚗 角色配置

**文件**：`config/actor_settings_template.json`

定义要生成的车辆和基础设施：

```json
{
    "actors": [
        {
            "type": "vehicle.tesla.model3",
            "name": "vehicle.tesla.model3.master",
            "sensors_setting": "sensor_config_template.json",
            "spawn_point": 73  // 生成点索引
        },
        {
            "type": "infrastructure",  // 路侧单元
            "name": "infra_t_junction",
            "sensors_setting": "sensor_config_infrastructure_template.json",
            "spawn_point": {"x": 41, "y": -240, "z": 2.7}
        }
    ],
    "other_vehicles": {
        "vehicle_num": 50,  // 背景车辆数量
        "spawn_points": [44, 55, 64, ...]
    }
}
```

### 📷 传感器配置

**文件**：`config/sensor_config_template.json`

定义附加到每个角色的传感器：

```json
{
    "sensors": [
        {
            "type": "sensor.camera.rgb",
            "name": "image_2",
            "spawn_point": {"x": 2.0, "y": 0.0, "z": 2.0, ...},
            "image_size_x": 1382,
            "image_size_y": 512,
            "fov": 90.0
        },
        {
            "type": "sensor.lidar.ray_cast",
            "name": "velodyne",
            "spawn_point": {"x": 0.0, "y": 0.0, "z": 2.4, ...},
            "range": 100,
            "channels": 64,
            "points_per_second": 1300000
        }
    ]
}
```

**支持的传感器类型：**
- `sensor.camera.rgb` - RGB 相机
- `sensor.camera.semantic_segmentation` - 语义分割相机
- `sensor.lidar.ray_cast` - 激光雷达
- `sensor.lidar.ray_cast_semantic` - 语义激光雷达
- `sensor.other.radar` - 毫米波雷达

### 📂 示例配置

`config/` 目录包含预设配置：

- `config/kitti_object/` - KITTI 风格数据集配置
- `config/argoverse/` - Argoverse 风格数据集配置

---

## 📊 数据格式

### 原始数据结构

```
raw_data/
└── record_YYYY_MMDD_HHMM/
    ├── carla_raw_record.log           # CARLA 录制日志
    ├── vehicle.tesla.model3_1/
    │   ├── 000001_image_2.png         # RGB 图像
    │   ├── 000001_image_2_semantic.png
    │   ├── 000001_velodyne.npy        # 激光雷达 (Nx4: x,y,z,intensity)
    │   ├── 000001_velodyne_semantic.npy
    │   ├── 000001_radar_front.npy
    │   ├── sensor_data.csv            # 传感器位姿
    │   └── vehicle_data.csv           # 车辆状态
    └── others.world_0/
        └── 000001_objects.pkl         # 物体标签
```

### 标注数据集结构（KITTI 格式）

```
dataset/
└── record_YYYY_MMDD_HHMM/
    └── vehicle.tesla.model3_1/
        └── kitti_object/
            ├── ImageSets/
            │   ├── train.txt
            │   └── val.txt
            └── training/
                ├── calib/         # 标定文件
                ├── image_2/       # RGB 图像
                ├── label_2/       # 3D 边界框标签
                └── velodyne/      # 点云 (.bin)
```

### 坐标系统

- **原始数据**：右手坐标系（X：前向，Y：右侧，Z：向上）
- **KITTI 格式**：相机坐标系（X：右侧，Y：向下，Z：前向）
- **坐标转换**：标注过程中自动转换

---

## 🔬 高级用法

### 自定义生成点

查找地图中的生成点：

```python
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()

for i, point in enumerate(spawn_points):
    print(f"生成点 {i}: {point.location}")
```

### 多车辆录制

编辑 `config/actor_settings_template.json` 添加不同传感器配置的多辆车：

```json
{
    "actors": [
        {
            "type": "vehicle.tesla.model3",
            "name": "ego_vehicle",
            "sensors_setting": "sensor_config_template.json",
            "spawn_point": 73
        },
        {
            "type": "vehicle.audi.a2",
            "name": "following_vehicle",
            "sensors_setting": "sensor_config_simple.json",
            "spawn_point": 76
        }
    ]
}
```

### 基础设施（V2X）录制

为路侧单元添加基础设施传感器：

```json
{
    "type": "infrastructure",
    "name": "rsu_intersection_1",
    "sensors_setting": "sensor_config_infrastructure_template.json",
    "spawn_point": {"x": 41, "y": -240, "z": 15.0}
}
```

---

## 🐛 常见问题

### 问题：`ModuleNotFoundError: No module named 'carla'`

**解决方案：**
1. 验证 CARLA_ROOT 已设置：`echo $CARLA_ROOT`
2. 检查 PYTHONPATH 是否包含正确的 .egg 文件
3. 确保 .egg 文件与你的 Python 版本匹配

### 问题：无法连接到 CARLA 服务器

**解决方案：**
1. 确保 CARLA 服务器正在运行：`./CarlaUE4.sh`
2. 检查端口（默认：2000）：`python3 data_recorder.py -p 2000`
3. 验证防火墙设置

### 问题：低帧率 / 录制缓慢

**解决方案：**
1. 减少配置中的传感器数量
2. 降低传感器分辨率（image_size_x、image_size_y）
3. 增加 frame_step 跳过部分帧
4. 使用无界面模式：`./CarlaUE4.sh -RenderOffScreen`

### 问题：Numpy 版本冲突

**解决方案：**
```bash
pip3 install "numpy>=1.24.4,<2.0"
```

---

## 📁 项目结构

```
carla_dataset_tools/
├── config/                   # 配置文件
│   ├── world_config_template.json
│   ├── actor_settings_template.json
│   ├── sensor_config_template.json
│   ├── kitti_object/        # KITTI 预设
│   └── argoverse/           # Argoverse 预设
├── label_tools/             # 标注脚本
│   ├── kitti_objects_label.py
│   ├── yolo_label.py
│   └── argoverse_label.py
├── recorder/                # 核心录制模块
│   ├── actor_tree.py
│   ├── vehicle.py
│   ├── sensor.py
│   └── agents/              # 自动驾驶代理
├── utils/                   # 工具脚本
│   ├── visualize_lidar.py
│   ├── transform.py
│   └── geometry_types.py
├── data_recorder.py         # 主录制脚本
├── param.py                 # 全局参数
├── requirements.txt
├── README.md                # 英文文档
└── README_CN.md             # 中文文档
```

---

## 🤝 参与贡献

欢迎贡献！请随时提交 Pull Request。

**贡献方向：**
- 支持更多数据集格式（nuScenes、Waymo 等）
- 增强文档和示例
- Bug 修复和性能改进
- 新的传感器类型或功能

---

## 📝 待办事项

- [x] 数据录制工具
- [x] KITTI 格式数据标注工具（目标检测）
- [x] YOLOv5 标注工具
- [x] Argoverse 示例
- [x] 增强文档
- [ ] nuScenes 格式支持
- [ ] 实时可视化
- [ ] 自动化测试套件

---

## 📖 引用说明

如果你在研究中使用了本工具，请引用：

```bibtex
@article{wang2022federated,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Wang, Shuai and Li, Chengyang and Ng, Derrick Wing Kwan and Eldar, Yonina C and Poor, H Vincent and Hao, Qi and Xu, Chengzhong},
  journal={IEEE network},
  volume={37},
  number={3},
  pages={16--25},
  year={2022},
  publisher={IEEE}
}
```

---

## 🙏 致谢

本项目基于以下优秀工作：

- [**CARLA 仿真器**](https://carla.org/) - 开源自动驾驶仿真器
- [**CARLA ROS Bridge**](https://github.com/carla-simulator/ros-bridge) - CARLA 的 ROS 集成
- [**CARLA_INVS**](https://github.com/zijianzhang/CARLA_INVS) - 基础设施和车辆仿真

---

## 📄 许可证

本项目采用 GNU 通用公共许可证 v3.0 - 详见 [LICENSE](LICENSE) 文件。

---

## 📮 联系与支持

- **问题反馈**：[GitHub Issues](https://github.com/KevinLADLee/carla_dataset_tools/issues)
- **相关项目**：[CarlaFLCAV](https://github.com/SIAT-INVS/CarlaFLCAV)

---

<div align="center">

**⭐ 如果这个项目对你的研究有帮助，请给我们点个 Star！⭐**

[🏠 主页](https://github.com/KevinLADLee/carla_dataset_tools) • [📖 文档](#) • [🐛 报告 Bug](https://github.com/KevinLADLee/carla_dataset_tools/issues) • [💡 功能建议](https://github.com/KevinLADLee/carla_dataset_tools/issues)

</div>
