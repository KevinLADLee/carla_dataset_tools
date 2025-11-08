# Routes Directory

This directory contains predefined vehicle routes for CARLA Dataset Tools.

## Creating Routes

Use the interactive route editor to create routes:

```bash
# Start CARLA server first
cd $CARLA_ROOT && ./CarlaUE4.sh

# In another terminal, launch the route editor
python3 utils/route_editor.py --map Town02 --name my_route_01
```

## Route File Format

Routes are saved in YAML format:

```yaml
map: Town02
waypoints:
  - {x: 107.5, y: -133.2, z: 0.3}
  - {x: 150.0, y: -130.5, z: 0.3}
  - {x: 200.3, y: -128.8, z: 0.3}
mode: strict
metadata:
  waypoint_count: 3
  created_with: route_editor.py
```

## Using Routes in Configuration

### Method 1: Direct waypoints in config

```yaml
actors:
  - type: vehicle.tesla.model3
    name: vehicle_1st
    spawn_point: 73
    route:
      mode: strict
      waypoints:
        - {x: 107.5, y: -133.2, z: 0.3}
        - {x: 150.0, y: -130.5, z: 0.3}
        - {x: 200.3, y: -128.8, z: 0.3}
    sensors: [...]
```

### Method 2: Load from file

```yaml
actors:
  - type: vehicle.tesla.model3
    name: vehicle_1st
    spawn_point: 73
    route:
      from_file: routes/Town02_my_route_01.yaml
    sensors: [...]
```

## Route Modes

- `strict`: Vehicle follows the exact waypoint sequence without deviation
- `disabled`: Ignores route, uses default autopilot (backward compatible)

## Notes

- Waypoints are automatically snapped to valid road positions
- Routes must have at least 2 waypoints
- Route files are also saved as `.pkl` for internal use
- Coordinates use CARLA's coordinate system (right-hand, Z-up)
