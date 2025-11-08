# CARLA 数据集工具 - 用户指南

本指南涵盖 CARLA 数据集工具的安装、使用和常见工作流程。

## 目录

- [前置要求](#前置要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置文件](#配置文件)
- [录制数据](#录制数据)
- [生成标签](#生成标签)
- [可视化](#可视化)
- [数据格式](#数据格式)
- [故障排除](#故障排除)

---

## 前置要求

在开始之前,请确保您具备以下条件:

- **CARLA 模拟器** >= 0.9.16
- **Python** >= 3.8
- **CARLA Python API** (包含在 CARLA 发行版中)
- **操作系统**: Linux (推荐) / Windows

> 下载 CARLA: [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)

---

## 安装

### 步骤 1: 克隆代码库

```bash
git clone https://github.com/KevinLADLee/carla_dataset_tools.git
cd carla_dataset_tools
```

### 步骤 2: 安装依赖

```bash
pip3 install -r requirements.txt
```

**所需包:**
- opencv-python > 4.0
- carla >= 0.9.16
- numpy < 2.0, >= 1.24.4
- transforms3d ~= 0.4.2
- open3d
- pandas
- shapely
- networkx
- pyyaml >= 6.0

### 步骤 3: 配置环境变量

将以下内容添加到您的 `~/.bashrc` 或 `~/.zshrc`:

```bash
# 设置 CARLA 根目录
export CARLA_ROOT=/path/to/your/carla
```

将 `/path/to/your/carla` 替换为您实际的 CARLA 安装路径。

然后重新加载 shell 配置:

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### 步骤 4: 验证安装

```bash
python3 -c "import carla; print(f'CARLA version: {carla.__version__}')"
```

如果看到打印的 CARLA 版本,说明安装成功!

---

## 快速开始

### 1. 启动 CARLA 模拟器

首先,启动 CARLA 服务器:

```bash
cd $CARLA_ROOT
./CarlaUE4.sh
```

无头模式(无渲染):

```bash
./CarlaUE4.sh -RenderOffScreen
```

### 2. 录制数据

使用配置文件运行数据记录器:

```bash
# 使用默认配置文件
python3 data_recorder.py

# 使用 KITTI 风格配置
python3 data_recorder.py --profile kitti

# 使用 Argoverse 风格配置
python3 data_recorder.py --profile argoverse

# 使用简单配置进行测试
python3 data_recorder.py --profile simple

# 使用自定义 YAML 配置文件
python3 data_recorder.py --config my_custom_config.yaml
```

**控制选项:**
- 记录器将自动收集数据,直到达到配置的帧数
- 按 `Ctrl+C` 手动停止录制

**输出位置:**
数据将保存到: `raw_data/record_YYYY_MMDD_HHMM/`

### 3. 生成标签

录制后,以您想要的格式生成标签:

#### KITTI 对象格式

```bash
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303
```

**选项:**
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

#### Argoverse 格式 (实验性)

```bash
python3 label_tools/argoverse_label.py -r record_2022_0119_1303
```

---

## 配置文件

工具包含多个预配置文件,位于 `config/profiles/` 目录:

### 可用配置

- **`default`** - 多车辆和传感器的通用配置
- **`kitti`** - KITTI 风格数据集配置 (Velodyne HDL-64E, 标准相机)
- **`argoverse`** - Argoverse 风格环形相机和立体设置
- **`simple`** - 快速测试的最小配置

### 使用配置文件

```bash
# 列出可用配置文件
python3 utils/list_profiles.py

# 验证配置文件
python3 utils/validate_config.py --profile kitti

# 使用配置文件进行录制
python3 data_recorder.py --profile kitti
```

### 创建自定义配置

1. **复制现有配置文件:**
   ```bash
   cp config/profiles/default.yaml config/profiles/my_config.yaml
   ```

2. **编辑配置:**
   - 修改传感器参数、车辆类型、生成点
   - YAML 支持注释
   - 使用 YAML 锚点 (`&` 和 `*`) 重用配置

3. **验证您的配置:**
   ```bash
   python3 utils/validate_config.py config/profiles/my_config.yaml
   ```

4. **使用您的配置:**
   ```bash
   python3 data_recorder.py --config config/profiles/my_config.yaml
   ```

---

## 录制数据

### 基本录制工作流

1. **启动 CARLA 服务器** - 启动模拟器
2. **选择配置** - 选择或创建配置文件
3. **运行记录器** - 使用选择的配置执行 `data_recorder.py`
4. **监控进度** - 查看控制台输出了解帧进度
5. **停止录制** - 等待完成或按 `Ctrl+C`

### 录制参数

在 YAML 配置文件中配置这些参数:

```yaml
recording:
  frame_total: 12000        # 要录制的总帧数
  frame_step: 3             # 每 N 帧保存一次
  map: Town02               # CARLA 地图名称
  weather: ClearNoon        # 天气预设 (可选)
```

### 多车辆录制

在 `actors` 部分配置多个车辆:

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

---

## 生成标签

### KITTI 格式

KITTI 标签工具将原始数据转换为 KITTI 对象检测格式:

```bash
python3 label_tools/kitti_objects_label.py -r <record_name> [options]
```

**常用选项:**
- `-r, --record` - 录制文件夹名称 (必需)
- `-v, --vehicle` - 要处理的特定车辆
- `-l, --lidar` - 激光雷达传感器名称 (默认: velodyne)
- `-c, --camera` - 相机传感器名称 (默认: image_2)
- `-o, --output` - 输出目录名称

**输出结构:**
```
dataset/record_YYYY_MMDD_HHMM/vehicle_name/kitti_object/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
└── training/
    ├── calib/         # 标定文件
    ├── image_2/       # RGB 图像
    ├── label_2/       # 3D 边界框标签
    └── velodyne/      # 点云 (.bin)
```

### YOLOv5 格式

为 YOLOv5 生成 2D 边界框标签:

```bash
python3 label_tools/yolo_label.py -r <record_name>
```

输出包括:
- YOLO 格式的图像
- 标签文本文件 (类别 x中心 y中心 宽度 高度)
- 数据集配置文件

---

## 可视化

### 可视化点云

```bash
# 可视化单个文件
python3 utils/visualize_lidar.py --type lidar --source raw_data/record_2022_0119_1303/vehicle.tesla.model3_1/000001_lidar.npy

# 可视化所有帧 (glob 模式)
python3 utils/visualize_lidar.py --type lidar --source raw_data/record_2022_0119_1303/vehicle.tesla.model3_1/
```

**支持的类型:**
- `lidar` - 标准激光雷达点云
- `semantic_lidar` - 带类别标签的语义激光雷达
- `radar` - 雷达检测点

**可视化控制:**
- 鼠标: 旋转和缩放
- 方向键: 在帧之间导航 (glob 模式)
- `Q`: 退出可视化

---

## 数据格式

### 原始数据结构

```
raw_data/
└── record_YYYY_MMDD_HHMM/
    ├── carla_raw_record.log           # CARLA 记录器日志
    ├── vehicle.tesla.model3_1/
    │   ├── 000001_image_2.png         # RGB 图像
    │   ├── 000001_image_2_semantic.png
    │   ├── 000001_velodyne.npy        # 激光雷达 (Nx4: x,y,z,强度)
    │   ├── 000001_velodyne_semantic.npy
    │   ├── 000001_radar_front.npy
    │   ├── sensor_data.csv            # 传感器位姿
    │   └── vehicle_data.csv           # 车辆状态
    └── others.world_0/
        └── 000001_objects.pkl         # 对象标签
```

### 标注数据集结构 (KITTI 格式)

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

**重要**: 所有原始数据使用**右手坐标系**:
- **X**: 前方
- **Y**: 右侧
- **Z**: 上方

**KITTI 格式**: 使用相机坐标系 (X: 右, Y: 下, Z: 前)

**转换**: 在标注过程中自动进行转换。

### 文件格式

- **图像**: PNG 格式 (RGB, 语义分割)
- **激光雷达**: NumPy `.npy` 文件 (Nx4: x, y, z, 强度)
- **语义激光雷达**: NumPy `.npy` 文件 (Nx6: x, y, z, cos角度, 对象索引, 标签)
- **雷达**: NumPy `.npy` 文件 (Nx4: x, y, z, 速度)
- **标签**: Pickle `.pkl` 文件 (原始) 或文本文件 (KITTI/YOLO)

---

## 故障排除

### 问题: `ModuleNotFoundError: No module named 'carla'`

**解决方案:**
1. 验证 CARLA_ROOT 已设置: `echo $CARLA_ROOT`
2. 检查 CARLA Python API 可访问
3. 确保 .egg 文件与您的 Python 版本匹配

### 问题: 连接 CARLA 服务器被拒绝

**解决方案:**
1. 确保 CARLA 服务器正在运行: `./CarlaUE4.sh`
2. 检查端口 (默认: 2000): `python3 data_recorder.py -p 2000`
3. 验证防火墙设置允许连接
4. 尝试连接到不同的主机: `python3 data_recorder.py --host localhost`

### 问题: 低 FPS / 录制缓慢

**解决方案:**
1. 减少配置中的传感器数量
2. 降低传感器分辨率 (image_size_x, image_size_y)
3. 增加 frame_step 以跳过帧
4. 使用无头模式: `./CarlaUE4.sh -RenderOffScreen`
5. 减少背景交通 (other_vehicles 数量)

### 问题: Numpy 版本冲突

**解决方案:**
```bash
pip3 install "numpy>=1.24.4,<2.0"
```

### 问题: 配置验证错误

**解决方案:**
1. 检查 YAML 语法是否有效
2. 确保所有必需字段都存在
3. 验证传感器类型与 CARLA 0.9.16 API 匹配
4. 使用验证工具: `python3 utils/validate_config.py --profile <name>`

### 问题: 生成点碰撞

**解决方案:**
1. 在配置中更改生成点
2. 减少车辆数量
3. 使用有更多生成点的不同地图
4. 使用 `utils/find_spawn_points.py` 发现有效位置

### 问题: 输出中缺少传感器数据

**解决方案:**
1. 检查 YAML 中传感器配置正确
2. 验证配置中的传感器名称匹配
3. 确保有足够的磁盘空间存储数据
4. 检查控制台输出是否有传感器错误

---

## 获取帮助

如果遇到此处未涵盖的问题:

- **GitHub Issues**: [https://github.com/KevinLADLee/carla_dataset_tools/issues](https://github.com/KevinLADLee/carla_dataset_tools/issues)
- **CARLA 文档**: [https://carla.readthedocs.io/](https://carla.readthedocs.io/)
- **开发者指南**: 查看 [DEVELOPER_CN.md](DEVELOPER_CN.md) 了解高级主题

---

[← 返回 README](../README_CN.md) | [开发者指南 →](DEVELOPER_CN.md)
