# CARLA Dataset Tools - User Guide

This guide covers installation, usage, and common workflows for CARLA Dataset Tools.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Route Editor](#route-editor)
- [Configuration Profiles](#configuration-profiles)
- [Recording Data](#recording-data)
- [Generating Labels](#generating-labels)
- [Visualization](#visualization)
- [Data Format](#data-format)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have the following:

- **CARLA Simulator** >= 0.9.16
- **Python** >= 3.8
- **CARLA Python API** (included with CARLA distribution)
- **Operating System**: Linux (recommended) / Windows

> Download CARLA: [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/KevinLADLee/carla_dataset_tools.git
cd carla_dataset_tools
```

### Step 2: Install Dependencies

```bash
pip3 install -r requirements.txt
```

**Required packages:**
- opencv-python > 4.0
- carla >= 0.9.16
- numpy < 2.0, >= 1.24.4
- transforms3d ~= 0.4.2
- open3d
- pandas
- shapely
- networkx
- pyyaml >= 6.0

### Step 3: Configure Environment Variables

Add the following to your `~/.bashrc` or `~/.zshrc`:

```bash
# Set CARLA root directory
export CARLA_ROOT=/path/to/your/carla
```

Replace `/path/to/your/carla` with your actual CARLA installation path.

Then reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Step 4: Verify Installation

```bash
python3 -c "import carla; print(f'CARLA version: {carla.__version__}')"
```

If you see the CARLA version printed, installation is successful!

---

## Quick Start

### 1. Start CARLA Simulator

First, launch the CARLA server:

```bash
cd $CARLA_ROOT
./CarlaUE4.sh
```

For headless mode (no rendering):

```bash
./CarlaUE4.sh -RenderOffScreen
```

### 2. Record Data

Run the data recorder with a configuration profile:

```bash
# Use default configuration profile
python3 data_recorder.py

# Use KITTI-style configuration
python3 data_recorder.py --profile kitti

# Use Argoverse-style configuration
python3 data_recorder.py --profile argoverse

# Use simple configuration for testing
python3 data_recorder.py --profile simple

# Use custom YAML configuration file
python3 data_recorder.py --config my_custom_config.yaml
```

**Control Options:**
- The recorder will automatically collect data until the configured frame count is reached
- Press `Ctrl+C` to stop recording manually

**Output Location:**
Data will be saved to: `raw_data/record_YYYY_MMDD_HHMM/`

### 3. Generate Labels

After recording, generate labels in your desired format:

#### KITTI Object Format

```bash
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303
```

**Options:**
```bash
# Specify vehicle
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303 -v vehicle.tesla.model3_1

# Specify sensors
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303 -l velodyne -c image_2

# Custom output directory
python3 label_tools/kitti_objects_label.py -r record_2022_0119_1303 -o my_dataset
```

#### YOLOv5 Format

```bash
python3 label_tools/yolo_label.py -r record_2022_0119_1303
```

#### Argoverse Format (experimental)

```bash
python3 label_tools/argoverse_label.py -r record_2022_0119_1303
```

---

## Route Editor

The Route Editor is an interactive tool for creating custom vehicle routes. It allows you to visually define waypoints on the map, and the system automatically calculates road-topology-aware paths for realistic vehicle navigation.

### Why Use Route Editor?

- **Precise Control**: Define exact paths for vehicles to follow during data collection
- **Loop Routes**: Create circular routes for continuous data recording
- **Visual Feedback**: See the actual road path vehicles will take (not just straight lines)
- **Topology-Aware**: Routes follow real road networks, lanes, and intersections

### Quick Start

**1. Launch Route Editor**

Make sure CARLA server is running, then:

```bash
python3 tools/editor_route.py --map Town02 --name my_route
```

**2. Create Your Route**

- **Left Click**: Add waypoint (can click same location multiple times)
- **Right Click on Circle**: Delete waypoint
- **Ctrl+Z**: Undo last waypoint
- **Enter**: Save and exit
- **Escape**: Cancel

**3. Visual Elements**

- **Red Circles**: Your selected waypoints
- **Lime Green Path**: Actual road path vehicle will follow (topology-aware)
- **Blue Dashed Line**: Direct connections between waypoints (reference)
- **Status Counter**: Bottom-right shows waypoint count and loop status

### Creating Loop Routes

To create a loop route, simply make your last waypoint close to your first waypoint (within 5 meters). The system will automatically:
- Detect the loop
- Show `[LOOP DETECTED]` in the window title
- Draw the return path from last to first waypoint

### Using Routes in Configuration

After creating a route, use it in your YAML configuration:

**Method 1: Load from File**
```yaml
actors:
  - type: vehicle.tesla.model3
    name: vehicle_1st
    spawn_point: 73
    route:
      from_file: routes/Town02_my_route.yaml
    sensors: [...]
```

**Method 2: Inline Waypoints**
```yaml
actors:
  - type: vehicle.tesla.model3
    name: vehicle_1st
    spawn_point: 73
    route:
      mode: strict
      loop: true  # Optional: for loop routes
      waypoints:
        - {x: 107.5, y: -133.2, z: 0.3}
        - {x: 150.0, y: -130.5, z: 0.3}
        - {x: 200.3, y: -128.8, z: 0.3}
    sensors: [...]
```

### Route Following Modes

- **`strict`**: Vehicle strictly follows waypoint sequence using road topology
- **`disabled`**: Ignores route, uses default CARLA autopilot
- **No route specified**: Default autopilot behavior (backward compatible)

### How It Works

1. **User Input**: You specify 3-8 key waypoints
2. **Path Planning**: GlobalRoutePlanner calculates complete road path between each waypoint pair
3. **Expansion**: Your 8 waypoints → 450+ road waypoints following lanes/intersections
4. **Navigation**: BasicAgent guides vehicle along the complete path

**Example Console Output:**
```
Vehicle 'vehicle_1st' configured with route: 8 waypoints
expanded to 453 road waypoints following topology (LOOP)
```

### Tips

- **For Loop Routes**: Click near the start point at the end (within 5m radius)
- **Waypoint Placement**: Waypoints represent "must pass through" locations, not every turn
- **Road Snapping**: Waypoints automatically snap to nearest valid road position
- **Topology Preview**: Green path shows exactly what vehicle will drive
- **Testing**: Use `--profile route_example` to see a complete working example

### Example Workflow

```bash
# 1. Start CARLA
cd $CARLA_ROOT && ./CarlaUE4.sh

# 2. Create route (in another terminal)
python3 tools/editor_route.py --map Town02 --name downtown_loop

# 3. Click waypoints on the map
# 4. Press Enter to save

# 5. Use the route in recording
python3 data_recorder.py --profile route_example
```

Saved routes are stored in `routes/` directory as both YAML (human-readable) and PKL (internal) formats.

---

## Configuration Profiles

The toolkit includes several pre-configured profiles located in `config/profiles/`:

### Available Profiles

- **`default`** - General purpose configuration with multiple vehicles and sensors
- **`kitti`** - KITTI-style dataset configuration (Velodyne HDL-64E, standard cameras)
- **`argoverse`** - Argoverse-style with ring cameras and stereo setup
- **`simple`** - Minimal configuration for quick testing

### Using Profiles

```bash
# List available profiles
python3 tools/config_list.py

# Validate a profile
python3 tools/config_validate.py --profile kitti

# Use a profile for recording
python3 data_recorder.py --profile kitti
```

### Creating Custom Configurations

1. **Copy an existing profile:**
   ```bash
   cp config/profiles/default.yaml config/profiles/my_config.yaml
   ```

2. **Edit the configuration:**
   - Modify sensor parameters, vehicle types, spawn points
   - YAML supports comments for documentation
   - Use YAML anchors (`&` and `*`) to reuse configurations

3. **Validate your configuration:**
   ```bash
   python3 tools/config_validate.py config/profiles/my_config.yaml
   ```

4. **Use your configuration:**
   ```bash
   python3 data_recorder.py --config config/profiles/my_config.yaml
   ```

---

## Recording Data

### Basic Recording Workflow

1. **Start CARLA server** - Launch the simulator
2. **Choose configuration** - Select or create a profile
3. **Run recorder** - Execute `data_recorder.py` with your chosen profile
4. **Monitor progress** - Watch console output for frame progress
5. **Stop recording** - Wait for completion or press `Ctrl+C`

### Recording Parameters

Configure these in your YAML profile:

```yaml
recording:
  frame_total: 12000        # Total frames to record
  frame_step: 3             # Save every N frames
  map: Town02               # CARLA map name
  weather: ClearNoon        # Weather preset (optional)
```

### Multi-Vehicle Recording

Configure multiple vehicles in the `actors` section:

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

## Generating Labels

### KITTI Format

The KITTI labeling tool converts raw data into KITTI object detection format:

```bash
python3 label_tools/kitti_objects_label.py -r <record_name> [options]
```

**Common Options:**
- `-r, --record` - Record folder name (required)
- `-v, --vehicle` - Specific vehicle to process
- `-l, --lidar` - LiDAR sensor name (default: velodyne)
- `-c, --camera` - Camera sensor name (default: image_2)
- `-o, --output` - Output directory name

**Output Structure:**
```
dataset/record_YYYY_MMDD_HHMM/vehicle_name/kitti_object/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
└── training/
    ├── calib/         # Calibration files
    ├── image_2/       # RGB images
    ├── label_2/       # 3D bounding box labels
    └── velodyne/      # Point clouds (.bin)
```

### YOLOv5 Format

Generate 2D bounding box labels for YOLOv5:

```bash
python3 label_tools/yolo_label.py -r <record_name>
```

Output includes:
- Images in YOLO format
- Label text files (class x_center y_center width height)
- Dataset configuration file

---

## Visualization

### Visualize Actor Tree (Pre-Recording)

Before recording, you can visualize your configuration to verify the actor hierarchy is correct:

```bash
# Visualize default profile
python3 tools/viz_actor_tree.py default

# Visualize custom configuration
python3 tools/viz_actor_tree.py my_config.yaml

# Generate SVG format
python3 tools/viz_actor_tree.py simple --format svg --output my_tree

# Generate PDF format
python3 tools/viz_actor_tree.py argoverse --format pdf

# Auto-open after generation
python3 tools/viz_actor_tree.py default --view

# List available profiles
python3 tools/viz_actor_tree.py --list
```

**Features:**
- **No CARLA Required**: Visualizes directly from configuration file
- **Complete Information**: Shows vehicles, sensors, spawn points, sensor parameters, routes
- **Color Coding**: Different colors for different node types (vehicles, sensors, infrastructure)
- **Multiple Formats**: PNG (default), SVG, PDF
- **Statistics Summary**: Total actors, sensors, background vehicles

**Output:**
The tool generates a graphical tree showing:
- Configuration summary (map, frames, delta time)
- World node with all actors
- Each vehicle/infrastructure with its sensors
- Sensor parameters (resolution, FOV, range, etc.)
- Spawn points and coordinates
- Route information (waypoint count, mode)

**Dependencies:**
```bash
# Install required package
pip install graphviz

# Install system graphviz (if needed)
# Ubuntu/Debian:
sudo apt-get install graphviz

# macOS:
brew install graphviz
```

### Visualize Point Cloud

```bash
# Visualize a single file
python3 tools/viz_lidar.py --type lidar --source raw_data/record_2022_0119_1303/vehicle.tesla.model3_1/000001_lidar.npy

# Visualize all frames (glob mode)
python3 tools/viz_lidar.py --type lidar --source raw_data/record_2022_0119_1303/vehicle.tesla.model3_1/
```

**Supported types:**
- `lidar` - Standard LiDAR point cloud
- `semantic_lidar` - Semantic LiDAR with class labels
- `radar` - Radar detection points

**Visualization Controls:**
- Mouse: Rotate and zoom
- Arrow keys: Navigate between frames (glob mode)
- `Q`: Quit visualization

---

## Data Format

### Raw Data Structure

```
raw_data/
└── record_YYYY_MMDD_HHMM/
    ├── carla_raw_record.log           # CARLA recorder log
    ├── vehicle.tesla.model3_1/
    │   ├── 000001_image_2.png         # RGB images
    │   ├── 000001_image_2_semantic.png
    │   ├── 000001_velodyne.npy        # LiDAR (Nx4: x,y,z,intensity)
    │   ├── 000001_velodyne_semantic.npy
    │   ├── 000001_radar_front.npy
    │   ├── sensor_data.csv            # Sensor poses
    │   └── vehicle_data.csv           # Vehicle state
    └── others.world_0/
        └── 000001_objects.pkl         # Object labels
```

### Labeled Dataset Structure (KITTI Format)

```
dataset/
└── record_YYYY_MMDD_HHMM/
    └── vehicle.tesla.model3_1/
        └── kitti_object/
            ├── ImageSets/
            │   ├── train.txt
            │   └── val.txt
            └── training/
                ├── calib/         # Calibration files
                ├── image_2/       # RGB images
                ├── label_2/       # 3D bounding box labels
                └── velodyne/      # Point clouds (.bin)
```

### Coordinate Systems

**Important**: All raw data uses a **right-hand coordinate system**:
- **X**: Forward
- **Y**: Right
- **Z**: Up

**KITTI Format**: Uses camera coordinate system (X: right, Y: down, Z: forward)

**Transformation**: Automatic conversion happens during the labeling process.

### File Formats

- **Images**: PNG format (RGB, semantic segmentation)
- **LiDAR**: NumPy `.npy` files (Nx4: x, y, z, intensity)
- **Semantic LiDAR**: NumPy `.npy` files (Nx6: x, y, z, cos_angle, object_idx, tag)
- **Radar**: NumPy `.npy` files (Nx4: x, y, z, velocity)
- **Labels**: Pickle `.pkl` files (raw) or text files (KITTI/YOLO)

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'carla'`

**Solution:**
1. Verify CARLA_ROOT is set: `echo $CARLA_ROOT`
2. Check the CARLA Python API is accessible
3. Ensure the .egg file matches your Python version

### Issue: Connection refused to CARLA server

**Solution:**
1. Ensure CARLA server is running: `./CarlaUE4.sh`
2. Check the port (default: 2000): `python3 data_recorder.py -p 2000`
3. Verify firewall settings allow connection
4. Try connecting to a different host: `python3 data_recorder.py --host localhost`

### Issue: Low FPS / Slow recording

**Solution:**
1. Reduce sensor count in configuration
2. Lower sensor resolution (image_size_x, image_size_y)
3. Increase frame_step to skip frames
4. Use headless mode: `./CarlaUE4.sh -RenderOffScreen`
5. Reduce background traffic (other_vehicles count)

### Issue: Numpy version conflict

**Solution:**
```bash
pip3 install "numpy>=1.24.4,<2.0"
```

### Issue: Configuration validation error

**Solution:**
1. Check YAML syntax is valid
2. Ensure all required fields are present
3. Validate sensor types match CARLA 0.9.16 API
4. Use validation tool: `python3 tools/config_validate.py --profile <name>`

### Issue: Spawn point collision

**Solution:**
1. Change spawn points in configuration
2. Reduce number of vehicles
3. Use different map with more spawn points
4. Use `tools/debug_info.py` to discover valid positions

### Issue: Missing sensor data in output

**Solution:**
1. Check sensor is properly configured in YAML
2. Verify sensor name matches in configuration
3. Ensure sufficient disk space for data
4. Check console output for sensor errors

---

## Getting Help

If you encounter issues not covered here:

- **GitHub Issues**: [https://github.com/KevinLADLee/carla_dataset_tools/issues](https://github.com/KevinLADLee/carla_dataset_tools/issues)
- **CARLA Documentation**: [https://carla.readthedocs.io/](https://carla.readthedocs.io/)
- **Developer Guide**: See [DEVELOPER.md](DEVELOPER.md) for advanced topics

---

[← Back to README](../README.md) | [Developer Guide →](DEVELOPER.md)
