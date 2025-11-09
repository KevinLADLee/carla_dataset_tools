# Routes Directory

This directory contains predefined vehicle routes for CARLA Dataset Tools.

## How Route Following Works

When you specify waypoints for a vehicle route, the system uses CARLA's **GlobalRoutePlanner** to:

1. **Find the shortest path** between consecutive waypoints following road topology
2. **Respect road network** including lanes, intersections, and traffic rules
3. **Generate complete waypoint sequence** with proper turn decisions (left/right/straight)

**Important**: The vehicle will **pass through all specified waypoints** but will **follow the road network** between them, not travel in straight lines.

### Example
If you specify waypoints A → B → C, the vehicle will:
- Calculate the road path from A to B (respecting lanes and intersections)
- Then calculate the road path from B to C
- Follow the complete combined route

## Creating Routes

Use the interactive route editor to create routes:

```bash
# Start CARLA server first
cd $CARLA_ROOT && ./CarlaUE4.sh

# In another terminal, launch the route editor
python3 utils/route_editor.py --map Town02 --name my_route_01
```

### Tips for Creating Routes:
- Click on key intersection points or destinations you want the vehicle to reach
- The vehicle will follow roads between these points automatically
- You don't need to click every meter - just the must-pass-through locations
- Waypoints are automatically snapped to valid road positions

## Route File Format

Routes are saved in YAML format:

```yaml
map: Town02
waypoints:
  - {x: 107.5, y: -133.2, z: 0.3}  # Must pass through point 1
  - {x: 150.0, y: -130.5, z: 0.3}  # Must pass through point 2
  - {x: 200.3, y: -128.8, z: 0.3}  # Must pass through point 3
mode: strict
metadata:
  waypoint_count: 3
  created_with: route_editor.py
```

**Note**: The vehicle will find the road path between these waypoints using GlobalRoutePlanner, ensuring it follows proper lanes and road topology.

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

- **`strict`**: Vehicle must pass through all specified waypoints, following road topology between them
- **`disabled`**: Ignores route, uses default autopilot (backward compatible)

## Route Planning Details

When a route is loaded:
1. System reads your specified waypoints (e.g., 8 waypoints)
2. GlobalRoutePlanner calculates the complete road path between each pair
3. The complete route might expand to hundreds of waypoints (e.g., 8 → 450)
4. Vehicle follows this expanded route using BasicAgent

**Console output example:**
```
Vehicle 'vehicle_1st' configured with route: 8 waypoints expanded to 450 road waypoints following topology
```

This means:
- You specified 8 key locations to pass through
- System calculated a 450-waypoint path following roads
- Vehicle will drive naturally along roads to reach all 8 locations

## Notes

- Waypoints represent **must-pass-through locations**, not direct line segments
- The system automatically finds the **shortest road path** between waypoints
- Routes respect **lane topology**, **intersections**, and **turn restrictions**
- Coordinates use CARLA's coordinate system (right-hand, Z-up)
- Minimum 2 waypoints required per route
- Route files are also saved as `.pkl` for internal use
