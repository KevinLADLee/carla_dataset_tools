# CARLA Dataset Tools - Developer Guide

This guide covers architecture, configuration details, API reference, and advanced usage for developers.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Configuration System](#configuration-system)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Extending the Toolkit](#extending-the-toolkit)
- [Development Workflow](#development-workflow)

---

## Architecture Overview

### Project Structure

```
carla_dataset_tools/
├── config/                      # Configuration management
│   ├── config_manager.py        # YAML config loader and validator
│   └── profiles/                # Pre-configured profiles
│       ├── default.yaml         # Default configuration
│       ├── kitti.yaml           # KITTI dataset style
│       ├── argoverse.yaml       # Argoverse dataset style
│       ├── simple.yaml          # Simple testing config
│       └── route_example.yaml   # Route configuration example
├── label_tools/                 # Labeling scripts
│   ├── kitti_objects_label.py   # KITTI format labeling
│   ├── yolo_label.py            # YOLOv5 format labeling
│   └── kitti_object/            # KITTI utilities
├── recorder/                    # Core recording modules
│   ├── actor_tree.py            # Actor hierarchy management
│   ├── actor_factory.py         # Actor and sensor spawning
│   ├── vehicle.py               # Vehicle recording with route following
│   ├── sensor.py                # Base sensor class
│   ├── camera.py                # Camera sensors
│   ├── lidar.py                 # LiDAR sensors
│   ├── radar.py                 # Radar sensor
│   └── agents/                  # Autopilot and navigation agents
│       ├── navigation/          # Navigation components
│       │   └── global_route_planner.py  # Topology-aware path planning
│       └── ...
├── routes/                      # Vehicle route definitions
│   ├── README.md                # Route system documentation
│   └── *.yaml, *.pkl            # Route files (YAML + pickle)
├── core/                        # Core shared modules
│   ├── geometry.py              # Geometric types (Vector3d, Location, Transform, etc.)
│   ├── types.py                 # Label and object types
│   ├── transform.py             # Coordinate transformations
│   ├── converters.py            # Data format converters
│   └── logger.py                # Unified logging system
├── tools/                       # CLI utility scripts
│   ├── viz_lidar.py             # Point cloud visualization
│   ├── viz_map.py               # Map visualization
│   ├── viz_actor_tree.py        # Actor tree visualization (pre-recording)
│   ├── editor_route.py          # Interactive route creation tool
│   ├── data_generate_imageset.py # Dataset file list generation
│   ├── config_convert.py        # JSON to YAML converter
│   ├── config_list.py           # List available profiles
│   ├── config_validate.py       # Config validation tool
│   └── debug_info.py            # Debug information display
├── data_recorder.py             # Main recording script
└── param.py                     # Global parameters
```

### Core Components

#### 1. ConfigManager (config/config_manager.py)

Centralized configuration management with validation:

```python
class ConfigManager:
    """
    Manages YAML configuration loading and validation

    Features:
    - Profile-based configuration
    - CARLA 0.9.16 API validation
    - Map and weather preset validation
    - Security: 10MB file size limit
    """
```

#### 2. ActorTree (recorder/actor_tree.py)

Hierarchical management of actors and sensors:

```python
class ActorTree:
    """
    Manages actor hierarchy and data recording

    Structure:
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

Spawns and configures actors and sensors:

```python
class ActorFactory:
    """
    Factory pattern for creating CARLA actors

    Responsibilities:
    - Spawn vehicles and infrastructure
    - Attach sensors to parent actors
    - Configure autopilot and traffic manager
    """
```

#### 4. Sensor Classes (recorder/camera.py, lidar.py, radar.py)

Base sensor class with specific implementations:

```python
class Sensor:
    """Base sensor class with callback handling"""

class Camera(Sensor):
    """RGB, depth, semantic segmentation cameras"""

class Lidar(Sensor):
    """Ray-cast and semantic LiDAR"""

class Radar(Sensor):
    """Radar sensor"""
```

---

## Configuration System

### YAML Structure

Configuration files use YAML format with the following sections:

```yaml
# Recording settings
recording:
  frame_total: 12000        # Total frames to record
  frame_step: 3             # Save every N frames
  map: Town02               # CARLA map name
  weather: ClearNoon        # Weather preset (optional)

# Spectator camera position
spectator:
  x: 100.0
  y: -150.0
  z: 150.0
  pitch: 60.0
  yaw: -90.0
  roll: 0.0

# World physics settings
world_settings:
  synchronous_mode: true
  fixed_delta_seconds: 0.1
  substepping: true
  max_substep_delta_time: 0.01
  max_substeps: 16

# Traffic light timings
traffic_lights:
  red_time: 2.0
  green_time: 2.0
  yellow_time: 0.01

# Sensor templates (YAML anchors for reuse)
sensor_templates:
  rgb_camera: &rgb_camera
    type: sensor.camera.rgb
    image_size_x: 800
    image_size_y: 600
    fov: 90.0

# Vehicle and sensor actors
actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    sensors:
      - <<: *rgb_camera        # Reuse template
        name: front_camera
        spawn_point:
          x: 2.0
          y: 0.0
          z: 2.0
          roll: 0.0
          pitch: 0.0
          yaw: 0.0

# Background traffic
other_vehicles:
  count: 50
  spawn_points: [44, 55, 64]
```

### Available Maps (CARLA 0.9.16)

**Town Maps:**
- `Town01`, `Town01_Opt` - Simple town with basic road network
- `Town02`, `Town02_Opt` - Small town with various intersections
- `Town03`, `Town03_Opt` - Larger urban area with roundabout
- `Town04`, `Town04_Opt` - Small town with highway
- `Town05`, `Town05_Opt` - Urban area with bridge and tunnel
- `Town06`, `Town06_Opt` - Urban area with multiple lane highway
- `Town07`, `Town07_Opt` - Rural environment with narrow roads
- `Town10HD`, `Town10HD_Opt` - High-definition urban area
- `Town11`, `Town12`, `Town13`, `Town15` - Additional urban variations

**Special Maps:**
- `AnnotationColorLandscape` - Testing environment

**Note:** `_Opt` versions have optimized geometry for better performance.

### Weather Presets

Control environmental conditions with weather presets:

**Clear Weather:**
- `ClearNoon`, `ClearSunset`, `ClearNight` - Clear sky conditions

**Cloudy Weather:**
- `CloudyNoon`, `CloudySunset`, `CloudyNight` - Overcast conditions

**Wet Weather:**
- `WetNoon`, `WetSunset`, `WetNight` - Wet roads, no rain
- `WetCloudyNoon`, `WetCloudySunset`, `WetCloudyNight` - Wet and cloudy

**Rainy Weather:**
- `SoftRainNoon`, `SoftRainSunset`, `SoftRainNight` - Light rain
- `MidRainyNoon`, `MidRainSunset`, `MidRainyNight` - Moderate rain
- `HardRainNoon`, `HardRainSunset`, `HardRainNight` - Heavy rain

**Extreme Weather:**
- `DustStorm` - Desert dust storm conditions

**Default:**
- `Default` - CARLA's default weather

### Supported Sensor Types (CARLA 0.9.16)

- `sensor.camera.rgb` - RGB Camera
- `sensor.camera.depth` - Depth Camera
- `sensor.camera.semantic_segmentation` - Semantic Segmentation Camera
- `sensor.lidar.ray_cast` - LiDAR
- `sensor.lidar.ray_cast_semantic` - Semantic LiDAR
- `sensor.other.radar` - Radar

### Route Configuration

Vehicles can be configured to follow predefined routes using topology-aware path planning:

#### Route Configuration in YAML

```yaml
actors:
  - type: vehicle.tesla.model3
    name: ego_vehicle
    spawn_point: 73
    route:
      # Option 1: Load from file
      from_file: routes/Town02_my_route.yaml

      # Option 2: Inline waypoints
      mode: strict              # strict | disabled
      loop: true                # Optional: for circular routes
      waypoints:
        - {x: 107.5, y: -133.2, z: 0.3}
        - {x: 150.0, y: -130.5, z: 0.3}
        - {x: 200.3, y: -128.8, z: 0.3}
    sensors: [...]
```

#### Route File Format

Route files (YAML) contain waypoint definitions:

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

Corresponding pickle files (`.pkl`) are automatically generated for internal use.

#### Route Following Modes

- **`strict`**: Vehicle follows waypoints using GlobalRoutePlanner
  - Waypoints are expanded into complete road-topology-aware paths
  - Respects lanes, intersections, and road structure
  - Example: 8 user waypoints → 450+ road waypoints

- **`disabled`**: Route is ignored, uses default autopilot

- **No route specified**: Default autopilot behavior (backward compatible)

#### Creating Routes

Use the interactive route editor:

```bash
python3 tools/editor_route.py --map Town02 --name my_route
```

**Features:**
- Visual waypoint selection on map
- Real-time topology-aware path preview using GlobalRoutePlanner
- Auto-detect loop routes (first and last waypoints within 5m)
- Undo support (Ctrl+Z)
- Saves both YAML (human-readable) and PKL (internal) formats

**Controls:**
- Left click: Add waypoint
- Right click on circle: Delete waypoint
- Ctrl+Z: Undo
- Enter: Save and exit
- Escape: Cancel

#### Implementation Details

**Path Planning Process:**

1. **User Input**: Define 3-8 key waypoints in route editor or YAML
2. **GlobalRoutePlanner**: Calculates complete road paths between waypoints
   - Uses CARLA's road topology graph
   - Respects lane markings, turn restrictions, intersections
3. **Route Expansion**: Waypoints expanded to 100s of road waypoints
4. **BasicAgent**: Navigates vehicle along the complete path
   - Uses LocalPlanner for trajectory control
   - PID controllers for steering, throttle, brake

**Validation Rules:**
- Minimum 2 waypoints required
- Each waypoint must have x, y, z coordinates
- Route files validated during configuration loading
- Invalid route files trigger ConfigValidationError

### Configuration Validation

The ConfigManager validates:

1. **File size**: Maximum 10MB (security)
2. **YAML syntax**: Valid YAML structure
3. **Required fields**: All mandatory fields present
4. **Sensor types**: Match CARLA 0.9.16 API
5. **Maps**: Valid map names
6. **Weather**: Valid weather presets
7. **Physics constraints**: `fixed_delta_seconds <= max_substep_delta_time * max_substeps`

### YAML Advanced Features

#### Anchors and Aliases

Reuse configurations with YAML anchors:

```yaml
sensor_templates:
  # Define template with anchor
  base_camera: &base_camera
    type: sensor.camera.rgb
    image_size_x: 800
    image_size_y: 600
    fov: 90.0

actors:
  - type: vehicle.tesla.model3
    sensors:
      # Reuse template and override specific fields
      - <<: *base_camera
        name: front_camera
        spawn_point: {x: 2.0, y: 0.0, z: 2.0}

      - <<: *base_camera
        name: rear_camera
        spawn_point: {x: -2.0, y: 0.0, z: 2.0, yaw: 180.0}
```

#### Comments

YAML supports inline and block comments:

```yaml
recording:
  frame_total: 12000        # Total frames to record
  frame_step: 3             # Save every 3rd frame
```

---

## API Reference

### ConfigManager API

```python
from config.config_manager import ConfigManager, ConfigValidationError

# Initialize
config_manager = ConfigManager(config_root="/path/to/config")

# Load profile
config = config_manager.load_profile("kitti")

# Load custom config file
config = config_manager.load_config("/path/to/config.yaml")

# List available profiles
profiles = config_manager.list_profiles()

# Validate configuration
try:
    config = config_manager.load_profile("my_profile")
except ConfigValidationError as e:
    print(f"Validation error: {e}")
```

### ActorTree API

```python
from recorder.actor_tree import ActorTree

# Initialize with world and configuration
actor_tree = ActorTree(world, config, save_dir)
actor_tree.init()

# Tick controller (update autopilot)
actor_tree.tick_controller()

# Save data for current frame
actor_tree.tick_data_saving(frame_id, timestamp)

# Cleanup
actor_tree.destroy()
```

### Sensor API

```python
from recorder.camera import Camera
from recorder.lidar import Lidar
from recorder.radar import Radar

# Create sensor instance
camera = Camera(world, sensor_config, parent_actor, save_dir)

# Sensors automatically register callbacks
# Data is saved when tick_data_saving() is called

# Access sensor attributes
sensor_transform = camera.get_transform()
```

### Transform Utilities

```python
from core.transform import Transform, Location, Rotation
from core.transform import transform_to_carla_transform

# Create transform
transform = Transform(
    Location(x=10.0, y=5.0, z=2.0),
    Rotation(pitch=0.0, yaw=90.0, roll=0.0)
)

# Convert to CARLA transform
carla_transform = transform_to_carla_transform(transform)

# Apply to actor
actor.set_transform(carla_transform)
```

---

## Advanced Usage

### Custom Spawn Points

Find spawn points in a map:

```python
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()

for i, point in enumerate(spawn_points):
    print(f"Spawn point {i}: {point.location}")
```

### Infrastructure (V2X) Recording

Include roadside sensors for V2X scenarios:

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

### Multi-Vehicle Synchronized Recording

Configure multiple vehicles with different sensor setups:

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

All vehicles are synchronized using CARLA's synchronous mode.

### Route-Based Data Collection

Configure vehicles to follow specific paths for reproducible data collection:

#### Creating a Custom Route

```bash
# Start CARLA server
cd $CARLA_ROOT && ./CarlaUE4.sh

# Create route using interactive editor
python3 tools/editor_route.py --map Town02 --name highway_loop

# Click waypoints on the map following your desired path
# Press Enter to save
```

#### Using Route in Configuration

**Method 1: Load from file**

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

**Method 2: Inline waypoints**

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

#### Route Following Behavior

When route is configured with `mode: strict`:

1. **At Startup**: GlobalRoutePlanner calculates complete road path
   - Input: User's 8 waypoints
   - Output: 450+ road waypoints following topology
   - Console: `Vehicle 'ego_vehicle' configured with route: 8 waypoints expanded to 453 road waypoints (LOOP)`

2. **During Recording**: BasicAgent follows the path
   - Maintains lane discipline
   - Respects traffic lights (optional)
   - Handles intersections properly
   - Loops back to start if `loop: true`

3. **Data Collection**: Vehicle follows same path every recording
   - Reproducible dataset collection
   - Consistent lighting/weather conditions
   - Same viewpoints for multi-session comparison

#### Example: Loop Route for Continuous Recording

```yaml
# config/profiles/continuous_loop.yaml
recording:
  frame_total: 50000        # Long recording
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

Run with:
```bash
python3 data_recorder.py --config config/profiles/continuous_loop.yaml
```

Vehicle will loop continuously collecting data until `frame_total` is reached.

### Custom Weather Conditions

Apply custom weather parameters programmatically:

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

### Traffic Manager Configuration

Configure traffic behavior:

```python
tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)
tm.set_global_distance_to_leading_vehicle(2.5)
tm.set_respawn_dormant_vehicles(True)
tm.set_hybrid_physics_mode(True)  # Optimize distant vehicles
```

---

## Extending the Toolkit

### Adding New Sensor Types

1. **Create sensor class** in `recorder/`:

```python
from recorder.sensor import Sensor

class MySensor(Sensor):
    def __init__(self, world, sensor_info, parent_actor, save_dir):
        super().__init__(world, sensor_info, parent_actor, save_dir)
        self._init_sensor()

    def _init_sensor(self):
        blueprint = self.world.get_blueprint_library().find(self.sensor_type)
        # Configure blueprint attributes
        self.sensor = self.world.spawn_actor(
            blueprint, self.transform, attach_to=self.parent_actor
        )
        self.sensor.listen(self._on_data)

    def _on_data(self, data):
        # Process and save sensor data
        pass
```

2. **Register in ActorFactory** (`recorder/actor_factory.py`):

```python
from recorder.my_sensor import MySensor

class ActorFactory:
    def create_sensor_node(self, sensor_info, parent_node):
        if sensor_type == "sensor.my.type":
            return MySensor(self.world, sensor_info, parent_actor, save_dir)
```

3. **Update ConfigManager** validation:

```python
VALID_SENSOR_TYPES = {
    'sensor.my.type',
    # ... existing types
}
```

### Adding New Dataset Formats

1. **Create labeling script** in `label_tools/`:

```python
# label_tools/my_format_label.py

def convert_to_my_format(raw_data_path, output_path):
    # Load raw data
    # Transform to dataset format
    # Write output files
    pass
```

2. **Follow existing patterns** from `kitti_objects_label.py` or `yolo_label.py`

3. **Add documentation** to USER_GUIDE.md

### Custom Configuration Profiles

Create specialized profiles for specific scenarios:

```yaml
# config/profiles/urban_night.yaml
recording:
  map: Town03
  weather: ClearNight
  frame_total: 5000

# High-sensitivity night camera
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

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/KevinLADLee/carla_dataset_tools.git
cd carla_dataset_tools

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8
```

### Running Tests

```bash
# Validate all profiles
python3 tools/config_validate.py --all

# Test configuration loading
python3 -c "from config.config_manager import ConfigManager; \
             cm = ConfigManager('config'); \
             config = cm.load_profile('default'); \
             print('Success!')"
```

### Code Style

Follow PEP 8 guidelines:

```bash
# Format code
black data_recorder.py

# Check style
flake8 recorder/ --max-line-length=100
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add .
git commit -m "Add: My new feature"

# Push to remote
git push origin feature/my-new-feature
```

### Debugging Tips

1. **Enable CARLA logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check sensor callbacks:**
```python
def _on_data(self, data):
    print(f"Received data: {data.frame} at {data.timestamp}")
    # Process data
```

3. **Verify spawn points:**
```bash
python3 tools/debug_info.py --map Town02
```

4. **Monitor performance:**
```python
import time
start = time.time()
# ... operation ...
print(f"Operation took {time.time() - start:.3f}s")
```

---

## Contributing

Contributions are welcome! Areas for contribution:

- Additional dataset format support (nuScenes, Waymo, etc.)
- Enhanced documentation and examples
- Bug fixes and performance improvements
- New sensor types or features

Please submit pull requests to the main repository.

---

[← Back to README](../README.md) | [User Guide ←](USER_GUIDE.md)
