# 🚗 CARLA 数据集工具

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.16-orange.svg)](https://carla.org/)
[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

> 📦 一套完整的 CARLA 仿真器数据采集与标注工具

用于 [CARLA 仿真器](https://carla.org/) 的数据采集和标注工具。本工具包提供了高效的流水线,用于生成高质量的自动驾驶数据集,支持多种传感器类型和标准数据集格式。

**⚠️ 重要提示**: 本工具生成的所有原始数据使用**右手坐标系**。

🌟 本项目是 [**CarlaFLCAV**](https://github.com/SIAT-INVS/CarlaFLCAV) 项目的一部分。欢迎给我们点个 Star!

---

## ✨ 核心特性

- ✅ **多传感器支持**: RGB 相机、语义分割、激光雷达、语义激光雷达、毫米波雷达
- ✅ **多种数据集格式**: KITTI Object、YOLOv5、Argoverse
- ✅ **灵活配置**: 基于 YAML 的配置系统及验证
- ✅ **路侧单元支持**: 支持 V2X 场景的路侧设备 (RSU) 仿真
- ✅ **同步录制**: 多车辆和多传感器的同步数据采集
- ✅ **可视化工具**: 内置点云和数据可视化工具
- ✅ **自动驾驶集成**: 使用 CARLA 交通管理器的自动车辆控制

---

## 🚀 30 秒快速开始

```bash
# 1. 克隆并安装
git clone https://github.com/KevinLADLee/carla_dataset_tools.git
cd carla_dataset_tools
pip3 install -r requirements.txt

# 2. 设置环境变量
export CARLA_ROOT=/path/to/your/carla

# 3. 启动 CARLA
cd $CARLA_ROOT && ./CarlaUE4.sh

# 4. 录制数据 (在新终端)
cd carla_dataset_tools
python3 data_recorder.py --profile kitti

# 5. 生成标签
python3 label_tools/kitti_objects_label.py -r record_YYYY_MMDD_HHMM
```

---

## 📚 文档

### 用户文档

**[用户指南](docs/USER_GUIDE_CN.md)** - 完整的安装和使用指南

- 安装和前置要求
- 使用不同配置文件录制数据
- 生成标签 (KITTI, YOLOv5, Argoverse)
- 数据可视化
- 故障排除

**[User Guide (English)](docs/USER_GUIDE.md)** - Complete installation and usage guide

### 开发者文档

**[开发者指南](docs/DEVELOPER_CN.md)** - 开发者技术文档

- 架构概述
- 配置系统详解 (地图、天气、传感器)
- API 参考
- 扩展工具包
- 开发工作流

**[Developer Guide (English)](docs/DEVELOPER.md)** - Technical documentation for developers

---

## ⚙️ 配置文件

不同数据集风格的预配置文件:

- **`default`** - 多车辆和传感器的通用配置
- **`kitti`** - KITTI 风格 (Velodyne HDL-64E, 标准相机)
- **`argoverse`** - Argoverse 风格环形相机
- **`simple`** - 测试用最小配置

```bash
# 列出可用配置文件
python3 utils/list_profiles.py

# 使用配置文件
python3 data_recorder.py --profile kitti

# 验证配置
python3 utils/validate_config.py --profile kitti
```

完整配置参考请查看[开发者指南](docs/DEVELOPER_CN.md)。

---

## 📊 支持的数据集格式

- **KITTI 对象检测** - 3D 边界框和标定
- **YOLOv5** - 2D 边界框标注
- **Argoverse** - 环形相机设置 (实验性)

---

## 🔧 环境要求

- **CARLA 仿真器** >= 0.9.16
- **Python** >= 3.8
- **操作系统**: Linux (推荐) / Windows

> 下载 CARLA: [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)

---

## 📁 项目结构

```
carla_dataset_tools/
├── config/                      # 配置管理
│   ├── config_manager.py        # YAML 配置加载器和验证器
│   └── profiles/                # 预配置文件 (default, kitti, argoverse, simple)
├── docs/                        # 文档
│   ├── USER_GUIDE.md            # 用户指南 (英文)
│   ├── USER_GUIDE_CN.md         # 用户指南 (中文)
│   ├── DEVELOPER.md             # 开发者指南 (英文)
│   └── DEVELOPER_CN.md          # 开发者指南 (中文)
├── label_tools/                 # 标注脚本 (KITTI, YOLO, Argoverse)
├── recorder/                    # 核心录制模块
├── utils/                       # 工具脚本
└── data_recorder.py             # 主录制脚本
```

---

## 🤝 参与贡献

欢迎贡献! 贡献方向:

- 额外的数据集格式支持 (nuScenes, Waymo 等)
- 增强文档和示例
- 错误修复和性能改进
- 新的传感器类型或功能

请向主代码库提交拉取请求。

---

## 📖 引用说明

如果您在研究中使用本工具,请引用:

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

本项目基于以下优秀工作:

- [**CARLA Simulator**](https://carla.org/) - 开源自动驾驶仿真器
- [**CARLA ROS Bridge**](https://github.com/carla-simulator/ros-bridge) - CARLA 的 ROS 集成
- [**CARLA_INVS**](https://github.com/zijianzhang/CARLA_INVS) - 基础设施和车辆仿真

---

## 📄 许可证

本项目采用 GNU 通用公共许可证 v3.0 - 详见 [LICENSE](LICENSE) 文件。

---

## 📮 联系与支持

- **问题反馈**: [GitHub Issues](https://github.com/KevinLADLee/carla_dataset_tools/issues)
- **项目主页**: [CarlaFLCAV](https://github.com/SIAT-INVS/CarlaFLCAV)
- **文档**: [用户指南](docs/USER_GUIDE_CN.md) | [开发者指南](docs/DEVELOPER_CN.md)

---

<div align="center">

**⭐ 如果本项目对您的研究有帮助,请给我们一个 Star! ⭐**

[🏠 主页](https://github.com/KevinLADLee/carla_dataset_tools) • [📖 用户指南](docs/USER_GUIDE_CN.md) • [🔧 开发者指南](docs/DEVELOPER_CN.md) • [🐛 报告问题](https://github.com/KevinLADLee/carla_dataset_tools/issues)

</div>
