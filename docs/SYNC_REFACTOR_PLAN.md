# CARLA 同步初始化流程重构方案

> 📅 创建日期: 2025-11-16
> 🎯 目标: 基于CARLA官方文档重构为完全同步、确定性的初始化流程
> 📚 参考文档: `docs/adv_synchrony_timestep.md`, `docs/adv_traffic_manager.md`

---

## 🔍 问题分析

### 当前问题
1. **segmentation fault**: 在同步模式下，spawn后立即调用 `set_autopilot()` 导致崩溃
2. **车辆偏转撞墙**: 车辆在初始化完成前处于无控制状态
3. **时序混乱**: spawn、tick、autopilot设置的顺序不明确

### 根本原因
根据 `adv_synchrony_timestep.md` 第206-239行的 **Physics Determinism** 要求：
- 同步模式必须在 `load_world` 之前启用
- 必须 `reload_world()` 确保时间戳一致性
- 命令应使用 `apply_batch_sync()` 批量执行并自动tick

根据 `adv_traffic_manager.md` 第500-507行：
- 先设置 world 同步模式
- 再设置 TM 同步模式
- spawn actors
- 设置 autopilot
- 在同一 client 中 tick

---

## ✅ 方案B: 使用 Command Batching（推荐方案）

### 核心思想
使用CARLA的批量命令系统 (`apply_batch_sync`)，在一个事务中完成：
1. Spawn actors
2. Set autopilot
3. 自动触发 world.tick()

### 优势
- ✅ 原子操作：spawn 和 autopilot 在同一批次
- ✅ 自动 tick：`apply_batch_sync(commands, True)` 自动执行tick
- ✅ 错误处理：每个命令都有响应，可检查成功/失败
- ✅ 符合官方示例：`generate_traffic.py` 使用相同模式
- ✅ 确定性：严格按照 Physics Determinism 要求

### 参考代码
来自 `PythonAPI/examples/generate_traffic.py`:
```python
batch = []
for n, transform in enumerate(spawn_points):
    blueprint = random.choice(blueprints)
    blueprint.set_attribute('role_name', 'autopilot')

    # 链式命令：spawn then set_autopilot
    batch.append(SpawnActor(blueprint, transform)
        .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

# 批量执行，due_tick_cue=True 自动触发tick
response = client.apply_batch_sync(batch, True)
```

---

## 🏗️ 实施计划

### Phase 1: 准备工作

#### 1.1 创建新分支
```bash
git checkout -b feature/sync-init-refactor
```

#### 1.2 备份当前实现
确保可以回滚到当前版本。

---

### Phase 2: 重构 ActorFactory

#### 2.1 核心思想：所有spawn操作都生成command
**文件**: `recorder/actor_factory.py`

**关键点**:
- Vehicle和Sensor都生成command，不立即spawn
- Sensor的command需要引用parent vehicle（使用索引关联）
- 所有command在一个batch中执行
- 使用`FutureActor`来引用尚未spawn的parent vehicle

#### 2.2 修改返回值结构

**当前**: 直接 spawn actor 并返回 Node
```python
def create_vehicle_node(self, actor_info):
    carla_actor = self.world.spawn_actor(blueprint, transform)
    vehicle_object = Vehicle(...)
    return Node(vehicle_object, NodeType.VEHICLE)
```

**重构为**: 返回 spawn command（包含sensor commands）
```python
def create_vehicle_spawn_command(self, actor_info):
    """
    创建车辆的spawn命令（不立即执行），包含sensors

    Returns:
        dict: 包含spawn信息的字典
        {
            'type': 'vehicle',
            'blueprint': blueprint,
            'transform': transform,
            'actor_info': actor_info,
            'vehicle_name': str,
            'route_config': dict or None,
            'use_autopilot': bool,
            'sensors': [sensor_cmd1, sensor_cmd2, ...]  # sensor命令列表
        }
    """
    vehicle_type = actor_info["type"]
    vehicle_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
    spawn_point = actor_info["spawn_point"]

    if type(spawn_point) is int:
        transform = self.spawn_points[spawn_point]
    else:
        transform = create_spawn_point(
            spawn_point.pop("x", 0.0),
            spawn_point.pop("y", 0.0),
            spawn_point.pop("z", 0.0),
            spawn_point.pop("roll", 0.0),
            spawn_point.pop("pitch", 0.0),
            spawn_point.pop("yaw", 0.0)
        )

    blueprint = self.blueprint_lib.find(vehicle_type)

    # 判断是否使用autopilot
    route_config = None
    if "route" in actor_info:
        route_config = self._parse_route_config(actor_info["route"])
    use_autopilot = (route_config is None)

    # 创建sensor命令列表
    sensor_commands = []
    if "sensors" in actor_info and actor_info["sensors"]:
        for sensor_info in actor_info["sensors"]:
            sensor_cmd = self.create_sensor_spawn_command(sensor_info, vehicle_name)
            sensor_commands.append(sensor_cmd)

    return {
        'type': 'vehicle',
        'blueprint': blueprint,
        'transform': transform,
        'actor_info': actor_info,
        'vehicle_name': vehicle_name,
        'route_config': route_config,
        'use_autopilot': use_autopilot,
        'sensors': sensor_commands  # 包含所有sensor命令
    }

def create_sensor_spawn_command(self, sensor_info, parent_vehicle_name):
    """
    创建sensor的spawn命令

    Args:
        sensor_info: sensor配置字典
        parent_vehicle_name: 父vehicle的名称（用于后续关联）

    Returns:
        dict: sensor spawn命令
        {
            'type': 'sensor',
            'sensor_type': 'camera.rgb',
            'sensor_name': 'camera_front',
            'blueprint': blueprint,
            'transform': transform,
            'parent_vehicle_name': str,  # 父vehicle名称
            'attributes': dict  # sensor属性
        }
    """
    sensor_type = sensor_info["type"]
    sensor_name = sensor_info["name"]

    # 获取blueprint
    blueprint = self.blueprint_lib.find(sensor_type)

    # 设置属性
    for attr_key, attr_value in sensor_info.items():
        if attr_key not in ['type', 'name', 'spawn_point']:
            if blueprint.has_attribute(attr_key):
                blueprint.set_attribute(attr_key, str(attr_value))

    # 获取相对transform
    spawn_point = sensor_info["spawn_point"]
    transform = create_spawn_point(
        spawn_point.get("x", 0.0),
        spawn_point.get("y", 0.0),
        spawn_point.get("z", 0.0),
        spawn_point.get("roll", 0.0),
        spawn_point.get("pitch", 0.0),
        spawn_point.get("yaw", 0.0)
    )

    return {
        'type': 'sensor',
        'sensor_type': sensor_type,
        'sensor_name': sensor_name,
        'blueprint': blueprint,
        'transform': transform,
        'parent_vehicle_name': parent_vehicle_name,
        'sensor_info': sensor_info  # 保存原始配置
    }
        dict: 包含spawn信息的字典
        {
            'type': 'vehicle',
            'blueprint': blueprint,
            'transform': transform,
            'actor_info': actor_info,  # 保存完整配置用于后续构建Node
            'use_autopilot': bool,
            'tm_port': int
        }
    """
    vehicle_type = actor_info["type"]
    vehicle_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
    spawn_point = actor_info["spawn_point"]

    if type(spawn_point) is int:
        transform = self.spawn_points[spawn_point]
    else:
        transform = create_spawn_point(
            spawn_point.pop("x", 0.0),
            spawn_point.pop("y", 0.0),
            spawn_point.pop("z", 0.0),
            spawn_point.pop("roll", 0.0),
            spawn_point.pop("pitch", 0.0),
            spawn_point.pop("yaw", 0.0)
        )

    blueprint = self.blueprint_lib.find(vehicle_type)

    # 判断是否使用autopilot
    route_config = None
    if "route" in actor_info:
        route_config = self._parse_route_config(actor_info["route"])

    use_autopilot = (route_config is None)

    return {
        'type': 'vehicle',
        'blueprint': blueprint,
        'transform': transform,
        'actor_info': actor_info,
        'vehicle_name': vehicle_name,
        'route_config': route_config,
        'use_autopilot': use_autopilot
    }

def create_infrastructure_spawn_command(self, actor_info):
    """创建infrastructure的spawn命令"""
    # 类似的结构
    pass

def create_other_vehicle_spawn_commands(self, other_vehicles_info):
    """创建背景车辆的spawn命令列表"""
    commands = []
    blueprints = self.blueprint_lib.filter('vehicle.*')

    # 处理指定spawn points
    spawn_points = other_vehicles_info.get('spawn_points', [])
    for spawn_point in spawn_points:
        bp = random.choice(blueprints)
        transform = self.spawn_points[spawn_point]
        commands.append({
            'type': 'other_vehicle',
            'blueprint': bp,
            'transform': transform,
            'use_autopilot': True
        })

    # 处理随机生成
    vehicle_count = other_vehicles_info.get('count', 0)
    all_spawn_points = self.world.get_map().get_spawn_points()
    for i in range(vehicle_count):
        bp = random.choice(blueprints)
        transform = random.choice(all_spawn_points)
        commands.append({
            'type': 'other_vehicle',
            'blueprint': bp,
            'transform': transform,
            'use_autopilot': True
        })

    return commands
```

#### 2.3 添加创建Sensor Object的辅助方法
```python
def create_sensor_object(self, sensor_cmd, carla_sensor, parent_vehicle):
    """
    从sensor命令和spawned actor创建Sensor对象

    Args:
        sensor_cmd: sensor spawn命令字典
        carla_sensor: spawned的CARLA sensor actor
        parent_vehicle: 父vehicle object

    Returns:
        Sensor子类实例（RGBCamera, SemanticCamera, Lidar等）
    """
    from recorder.sensor import RGBCamera, SemanticCamera, Lidar, SemanticLidar

    sensor_type = sensor_cmd['sensor_type']
    sensor_name = sensor_cmd['sensor_name']
    save_dir = f"{parent_vehicle.save_dir}/{sensor_name}"

    # 根据sensor类型创建对应的对象
    if sensor_type == "sensor.camera.rgb":
        sensor_object = RGBCamera(
            uid=self.generate_uid(),
            name=sensor_name,
            parent=parent_vehicle,
            carla_actor=carla_sensor,
            save_dir=save_dir
        )
    elif sensor_type == "sensor.camera.semantic_segmentation":
        sensor_object = SemanticCamera(
            uid=self.generate_uid(),
            name=sensor_name,
            parent=parent_vehicle,
            carla_actor=carla_sensor,
            save_dir=save_dir
        )
    elif sensor_type == "sensor.lidar.ray_cast":
        sensor_object = Lidar(
            uid=self.generate_uid(),
            name=sensor_name,
            parent=parent_vehicle,
            carla_actor=carla_sensor,
            save_dir=save_dir
        )
    elif sensor_type == "sensor.lidar.ray_cast_semantic":
        sensor_object = SemanticLidar(
            uid=self.generate_uid(),
            name=sensor_name,
            parent=parent_vehicle,
            carla_actor=carla_sensor,
            save_dir=save_dir
        )
    else:
        # 默认使用基类Sensor
        from recorder.sensor import Sensor
        sensor_object = Sensor(
            uid=self.generate_uid(),
            name=sensor_name,
            parent=parent_vehicle,
            carla_actor=carla_sensor,
            save_dir=save_dir
        )

    return sensor_object
```

---

### Phase 3: 重构 ActorTree

#### 3.1 分阶段初始化
```python
def build_vehicle_node_from_response(self, spawn_cmd, carla_actor):
    """
    从spawn命令和响应的actor构建Vehicle Node

    Args:
        spawn_cmd: 之前的spawn命令字典
        carla_actor: apply_batch_sync返回的actor

    Returns:
        Node: 包含Vehicle的节点
    """
    vehicle_object = Vehicle(
        uid=self.generate_uid(),
        name=spawn_cmd['vehicle_name'],
        base_save_dir=self.base_save_dir,
        carla_actor=carla_actor,
        route_config=spawn_cmd['route_config']
    )

    vehicle_node = Node(vehicle_object, NodeType.VEHICLE)

    # 创建sensors
    actor_info = spawn_cmd['actor_info']
    if "sensors" in actor_info and actor_info["sensors"]:
        sensor_name_set = set()
        for sensor_info in actor_info["sensors"]:
            sensor_node = self.create_sensor_node(
                sensor_info, vehicle_object, sensor_name_set
            )
            vehicle_node.add_child(sensor_node)

    return vehicle_node
```

---

### Phase 3: 重构 ActorTree

#### 3.1 分阶段初始化
**文件**: `recorder/actor_tree.py`

```python
class ActorTree(object):
    def __init__(self, world: carla.World, config=None, base_save_dir=None):
        self.world = world
        self.config = config
        self.actor_factory = ActorFactory(self.world, base_save_dir)
        self.root = Node(None)
        self.node_list = []
        self.thread_pool = ThreadPool(processes=4)

        # 保存spawn命令，用于后续构建
        self.spawn_commands = []

    def prepare_spawn_commands(self):
        """
        阶段1: 准备所有spawn命令（不执行）
        """
        logger.info("Preparing spawn commands...")

        # 创建world node
        world_actor = WorldActor(
            uid=self.actor_factory.generate_uid(),
            carla_world=self.world,
            base_save_dir=self.actor_factory.base_save_dir
        )
        self.root = Node(world_actor, NodeType.WORLD)

        # 收集vehicle spawn commands
        for actor_info in self.config["actors"]:
            actor_type = str(actor_info["type"])

            if actor_type.startswith("vehicle"):
                cmd = self.actor_factory.create_vehicle_spawn_command(actor_info)
                self.spawn_commands.append(cmd)
            elif actor_type.startswith("infrastructure"):
                cmd = self.actor_factory.create_infrastructure_spawn_command(actor_info)
                self.spawn_commands.append(cmd)

        # 收集other_vehicles spawn commands
        other_vehicle_info = self.config.get("other_vehicles", {})
        other_cmds = self.actor_factory.create_other_vehicle_spawn_commands(other_vehicle_info)
        self.spawn_commands.extend(other_cmds)

        logger.info(f"Prepared {len(self.spawn_commands)} spawn commands")

    def execute_batch_spawn(self, client, tm_port):
        """
        阶段2: 批量执行spawn命令（使用apply_batch_sync）
        **关键**: Vehicle和Sensor在同一个batch中spawn

        Args:
            client: carla.Client实例
            tm_port: Traffic Manager端口

        Returns:
            dict: 返回vehicle和sensor的响应映射
            {
                'vehicle_responses': [(cmd_index, response), ...],
                'sensor_responses': [(cmd_index, response, parent_vehicle_index), ...]
            }
        """
        from carla import command

        logger.info("Executing batch spawn for vehicles and sensors...")

        batch = []
        vehicle_cmd_indices = []  # 记录vehicle在batch中的索引
        sensor_cmd_mapping = []   # 记录(sensor在batch中的索引, parent_vehicle在batch中的索引)

        # 第一步：添加所有vehicle spawn命令到batch
        for i, cmd in enumerate(self.spawn_commands):
            if cmd['type'] in ['vehicle', 'other_vehicle']:
                batch_index = len(batch)
                vehicle_cmd_indices.append((i, batch_index))  # (原始cmd索引, batch索引)

                spawn_cmd = command.SpawnActor(cmd['blueprint'], cmd['transform'])

                # 如果需要autopilot，链式添加SetAutopilot命令
                if cmd.get('use_autopilot', False):
                    batch.append(
                        spawn_cmd.then(command.SetAutopilot(command.FutureActor, True, tm_port))
                    )
                else:
                    batch.append(spawn_cmd)

                # 第二步：为该vehicle添加sensor spawn命令
                if 'sensors' in cmd and cmd['sensors']:
                    parent_batch_index = batch_index
                    for sensor_cmd in cmd['sensors']:
                        sensor_batch_index = len(batch)

                        # 关键：使用FutureActor引用parent vehicle
                        # FutureActor的参数是parent在batch中的索引
                        sensor_spawn_cmd = command.SpawnActor(
                            sensor_cmd['blueprint'],
                            sensor_cmd['transform'],
                            command.Response(parent_batch_index, None)  # 引用parent vehicle的batch索引
                        )

                        batch.append(sensor_spawn_cmd)
                        sensor_cmd_mapping.append((sensor_batch_index, parent_batch_index, i, sensor_cmd))

        # 批量执行，due_tick_cue=True 自动执行tick
        logger.info(f"Spawning {len(batch)} actors ({len(vehicle_cmd_indices)} vehicles, "
                    f"{len(sensor_cmd_mapping)} sensors) with autopilot...")
        responses = client.apply_batch_sync(batch, True)

        # 检查响应
        success_count = sum(1 for r in responses if not r.error)
        logger.info(f"Spawned {success_count}/{len(responses)} actors successfully")

        # 检查失败
        for i, response in enumerate(responses):
            if response.error:
                logger.error(f"Failed to spawn actor at batch index {i}: {response.error}")

        # 返回结构化的响应
        return {
            'all_responses': responses,
            'vehicle_indices': vehicle_cmd_indices,
            'sensor_mapping': sensor_cmd_mapping
        }

    def build_nodes_from_responses(self, spawn_result):
        """
        阶段3: 从spawn响应构建Node树（包含vehicles和sensors）

        Args:
            spawn_result: execute_batch_spawn返回的结果字典
        """
        logger.info("Building actor tree from spawn responses...")

        responses = spawn_result['all_responses']
        vehicle_indices = spawn_result['vehicle_indices']
        sensor_mapping = spawn_result['sensor_mapping']

        # 第一步：构建vehicle nodes（不包含sensors）
        vehicle_nodes_map = {}  # {cmd_index: vehicle_node}

        for cmd_index, batch_index in vehicle_indices:
            response = responses[batch_index]

            if response.error:
                logger.warning(f"Skipping failed vehicle spawn (cmd {cmd_index}, batch {batch_index})")
                continue

            # 获取spawned vehicle actor
            carla_actor = self.world.get_actor(response.actor_id)
            cmd = self.spawn_commands[cmd_index]

            # 根据类型构建vehicle node
            if cmd['type'] == 'vehicle':
                vehicle_object = Vehicle(
                    uid=self.actor_factory.generate_uid(),
                    name=cmd['vehicle_name'],
                    base_save_dir=self.actor_factory.base_save_dir,
                    carla_actor=carla_actor,
                    route_config=cmd['route_config']
                )
                vehicle_node = Node(vehicle_object, NodeType.VEHICLE)

            elif cmd['type'] == 'other_vehicle':
                other_vehicle_object = OtherVehicle(
                    uid=self.actor_factory.generate_uid(),
                    name='',
                    base_save_dir="/tmp",
                    carla_actor=carla_actor
                )
                vehicle_node = Node(other_vehicle_object, NodeType.OTHER_VEHICLE)

            # 保存到map，用于后续添加sensors
            vehicle_nodes_map[cmd_index] = vehicle_node
            self.root.add_child(vehicle_node)
            self.node_list.append(vehicle_node)

        # 第二步：构建sensor nodes并attach到对应的vehicle
        for sensor_batch_idx, parent_batch_idx, parent_cmd_idx, sensor_cmd in sensor_mapping:
            response = responses[sensor_batch_idx]

            if response.error:
                logger.warning(f"Skipping failed sensor spawn: {sensor_cmd['sensor_name']}")
                continue

            # 检查parent vehicle是否成功spawn
            if parent_cmd_idx not in vehicle_nodes_map:
                logger.warning(f"Parent vehicle not found for sensor {sensor_cmd['sensor_name']}")
                continue

            # 获取spawned sensor actor
            carla_sensor = self.world.get_actor(response.actor_id)

            # 获取parent vehicle node
            vehicle_node = vehicle_nodes_map[parent_cmd_idx]
            vehicle_object = vehicle_node.get_actor()

            # 创建sensor object
            sensor_object = self.actor_factory.create_sensor_object(
                sensor_cmd, carla_sensor, vehicle_object
            )

            # 创建sensor node并添加到vehicle
            sensor_node = Node(sensor_object, NodeType.SENSOR)
            vehicle_node.add_child(sensor_node)
            self.node_list.append(sensor_node)

        self.node_list.append(self.root)  # 添加root
        logger.info(f"✓ Actor tree built: {len(self.node_list)} nodes total")

    def init(self, client, tm_port):
        """
        完整初始化流程（三阶段）

        Args:
            client: carla.Client实例
            tm_port: Traffic Manager端口
        """
        # 阶段1: 准备命令
        self.prepare_spawn_commands()

        # 阶段2: 批量spawn + autopilot
        responses = self.execute_batch_spawn(client, tm_port)

        # 阶段3: 构建node树
        self.build_nodes_from_responses(responses)
```

---

### Phase 4: 重构 DataRecorder

#### 4.1 按官方文档顺序初始化
**文件**: `data_recorder.py`

```python
def setting_world_and_actors(self, config):
    """
    严格按照CARLA Physics Determinism流程初始化
    参考: adv_synchrony_timestep.md 第212-239行
    """
    # ============================================================
    # Phase 1: 加载地图
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 1: Loading map...")
    self.logger.info("=" * 60)

    map_name = config['recording']['map']
    self.logger.info(f"Loading map: {map_name}")
    self.carla_client.load_world(map_name)

    # 重新获取world对象
    self.world = self.carla_client.get_world()
    self.logger.info("✓ Map loaded")

    # ============================================================
    # Phase 2: 配置同步模式 + 固定时间步长（在reload之前！）
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 2: Configuring synchronous mode...")
    self.logger.info("=" * 60)

    settings = self.world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = config['world_settings']['fixed_delta_seconds']
    settings.substepping = config['world_settings']['substepping']
    settings.max_substep_delta_time = config['world_settings']['max_substep_delta_time']
    settings.max_substeps = config['world_settings']['max_substeps']

    self.logger.info(f"  Synchronous mode: {settings.synchronous_mode}")
    self.logger.info(f"  Fixed delta seconds: {settings.fixed_delta_seconds}")
    self.logger.info(f"  Substepping: {settings.substepping}")
    self.logger.info(f"  Max substep delta time: {settings.max_substep_delta_time}")
    self.logger.info(f"  Max substeps: {settings.max_substeps}")

    self.world.apply_settings(settings)
    self.logger.info("✓ World settings applied")

    # ============================================================
    # Phase 3: Reload world（关键！确保时间戳一致性）
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 3: Reloading world for determinism...")
    self.logger.info("=" * 60)

    self.carla_client.reload_world(False)  # False = keep world settings
    self.world = self.carla_client.get_world()
    self.logger.info("✓ World reloaded")

    # ============================================================
    # Phase 4: 配置 Traffic Manager（必须在world同步后）
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 4: Configuring Traffic Manager...")
    self.logger.info("=" * 60)

    self.tm = self.carla_client.get_trafficmanager()
    tm_port = self.tm.get_port()
    self.logger.info(f"  TM port: {tm_port}")

    self.tm.set_synchronous_mode(True)
    self.tm.set_global_distance_to_leading_vehicle(2.5)
    self.tm.set_respawn_dormant_vehicles(True)
    # 不使用 hybrid_physics_mode - 完整物理模拟

    self.logger.info("  ✓ TM synchronous mode enabled")
    self.logger.info("  ✓ Global distance: 2.5m")
    self.logger.info("  ✓ Respawn dormant vehicles: True")
    self.logger.info("  ✓ Hybrid physics: False (full simulation)")

    # ============================================================
    # Phase 5: 设置天气和观察者
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 5: Setting weather and spectator...")
    self.logger.info("=" * 60)

    if 'weather' in config['recording'] and config['recording']['weather']:
        weather_preset = config['recording']['weather']
        try:
            weather = getattr(carla.WeatherParameters, weather_preset)
            self.world.set_weather(weather)
            self.logger.info(f"✓ Weather set to: {weather_preset}")
        except AttributeError:
            self.logger.warning(f"Weather preset '{weather_preset}' not found")

    if 'spectator' in config and config['spectator'] is not None:
        pose = config['spectator']
        spectator = self.world.get_spectator()
        spectator_transform = Transform(
            Location(pose['x'], pose['y'], pose['z']),
            Rotation(
                roll=pose.get('roll', 0.0),
                pitch=pose.get('pitch', 0.0),
                yaw=pose.get('yaw', 0.0)
            )
        )
        spectator.set_transform(transform_to_carla_transform(spectator_transform))
        self.logger.info("✓ Spectator position set")

    # ============================================================
    # Phase 6: 配置交通灯
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 6: Configuring traffic lights...")
    self.logger.info("=" * 60)

    traffic_light_settings = config.get('traffic_lights', {})
    self.set_traffic_light_time(traffic_light_settings)
    self.logger.info("✓ Traffic lights configured")

    # ============================================================
    # Phase 7: 批量Spawn actors（使用Command Batching）
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 7: Spawning actors (batch mode)...")
    self.logger.info("=" * 60)

    # 创建保存目录
    self.record_name = time.strftime("%Y_%m%d_%H%M", time.localtime())
    self.base_save_dir = f"{RAW_DATA_PATH}/record_{self.record_name}"

    # 初始化ActorTree（三阶段）
    self.actor_tree = ActorTree(self.world, config, self.base_save_dir)
    self.actor_tree.init(self.carla_client, tm_port)

    self.logger.info("✓ All actors spawned and initialized")

    # ============================================================
    # Phase 8: 初始化Tick（让sensors开始工作）
    # ============================================================
    self.logger.info("=" * 60)
    self.logger.info("Phase 8: Initial tick for sensor initialization...")
    self.logger.info("=" * 60)

    init_frame_id = self.world.tick()
    self.init_frame_id = init_frame_id  # 保存初始frame_id
    self.logger.info(f"✓ Initial frame: {init_frame_id}")

    # ============================================================
    # Phase 9: 设置录制参数
    # ============================================================
    self.frame_total = config['recording']['frame_total']
    self.frame_step = config['recording']['frame_step']
    self.config = config

    self.logger.info("=" * 60)
    self.logger.info("Initialization Complete")
    self.logger.info("=" * 60)
    self.logger.info(f"  Recording directory: {self.base_save_dir}")
    self.logger.info(f"  Target frames: {self.frame_total}")
    self.logger.info(f"  Frame step: {self.frame_step}")
    self.logger.info(f"  Initial frame ID: {init_frame_id}")
    self.logger.info(f"  Actors: {len(self.actor_tree.node_list)} nodes")
    self.logger.info("=" * 60)
```

#### 4.2 修改recording loop的frame计数
```python
def start_record(self, config):
    """开始录制"""
    self.setting_world_and_actors(config)

    # 创建保存目录
    self.logger.info(f"Recording to folder: {self.base_save_dir}")
    os.makedirs(self.base_save_dir, exist_ok=True)

    # 开始CARLA recorder
    carla_logfile = f"{self.base_save_dir}/carla_raw_record.log"
    self.logger.info(f"Start CARLA recorder: {carla_logfile}")
    self.carla_client.start_recorder(carla_logfile)

    try:
        # 从初始化完成后开始计数
        total_frame_count = 0  # 录制的帧数

        while True:
            self.logger.info("=" * 50)

            # Tick Control
            self.actor_tree.tick_controller()

            # Tick World
            tick_s = time.time()
            frame_id = self.world.tick(seconds=60.0)
            world_snapshot = self.world.get_snapshot()
            timestamp = world_snapshot.timestamp.elapsed_seconds
            tick_cost = time.time() - tick_s

            # 计算相对frame（从初始化完成后开始）
            relative_frame = frame_id - self.init_frame_id

            self.logger.info(
                f"World Tick -> Absolute FrameID: {frame_id}, "
                f"Relative Frame: {relative_frame}, "
                f"Recorded Frames: {total_frame_count}, "
                f"Timestamp: {timestamp:.3f}s, "
                f"Cost: {tick_cost:.3f}s"
            )

            # 根据frame_step决定是否保存
            if total_frame_count % self.frame_step == 0:
                save_s = time.time()
                try:
                    # 使用relative_frame作为保存的frame_id
                    self.actor_tree.tick_data_saving(relative_frame, timestamp)
                    save_cost = time.time() - save_s
                    self.logger.info(f"Data saved (frame {relative_frame}), cost {save_cost:.3f}s")
                except (RuntimeError, TimeoutError) as e:
                    self.logger.error(f"Data save failed: {e}")
                    raise

            # 检查中断
            if self.interrupted:
                self.logger.info("User interrupt, exiting...")
                time.sleep(2.0)
                break

            # 检查是否达到目标帧数
            total_frame_count += 1
            if total_frame_count >= self.frame_total:
                self.logger.info(f"Reached target frame count: {self.frame_total}")
                time.sleep(2.0)
                break

    except KeyboardInterrupt:
        self.logger.info("Keyboard interrupt received")
    except (RuntimeError, TimeoutError) as e:
        self.logger.error(f"Recording aborted: {e}")
    except Exception as e:
        self.logger.exception(f"Unexpected error: {e}")
    finally:
        self.logger.info("Cleaning up...")
        self.destroy()
        self.logger.info("Reloading world...")
        self.carla_client.reload_world()
        self.logger.info("Recording session ended")
```

---

### Phase 5: 清理 vehicle.py

#### 5.1 移除control_step中的autopilot逻辑
**文件**: `recorder/vehicle.py`

```python
# OtherVehicle
def control_step(self):
    # Autopilot已在spawn时通过batch命令设置
    # Traffic Manager自动控制，无需手动操作
    pass

# Vehicle
def control_step(self):
    """Execute one control step for the vehicle"""
    if self.route_config is not None and not self.use_auto_pilot:
        # Using agent-based route following
        control = self.vehicle_agent.run_step()
        self.carla_actor.apply_control(control)
    # Autopilot vehicles: 已在spawn时设置，无需操作
```

---

## 📊 执行时序图

```
┌─────────────────┐
│ load_world      │
└────────┬────────┘
         │
         v
┌─────────────────────────┐
│ apply_settings          │
│ - synchronous_mode=True │
│ - fixed_delta=0.1       │
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│ reload_world(False)     │  ← 关键！确保时间戳一致
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│ TM.set_synchronous_mode │
│ TM.configure_params     │
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│ prepare_spawn_commands  │  ← 阶段1：准备命令
└────────┬────────────────┘
         │
         v
┌─────────────────────────────────────┐
│ apply_batch_sync(                   │
│   [SpawnActor.then(SetAutopilot)],  │  ← 阶段2：批量spawn+autopilot
│   due_tick_cue=True                 │      自动触发tick
│ )                                   │
└────────┬────────────────────────────┘
         │ (自动tick)
         v
┌─────────────────────────┐
│ build_nodes_from_resp   │  ← 阶段3：构建树
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│ world.tick()            │  ← 额外tick，让sensors初始化
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│ 开始录制循环             │
└─────────────────────────┘
```

---

## ✅ 验证清单

### 功能验证
- [ ] 无 segmentation fault
- [ ] 车辆不会偏转撞墙
- [ ] Autopilot vehicles正常行驶
- [ ] Route-following vehicles正常跟随路径
- [ ] Sensors正常采集数据
- [ ] Frame计数从0开始（初始化后）
- [ ] 所有actors成功spawn
- [ ] apply_batch_sync响应无error

### 性能验证
- [ ] 初始化时间合理（<10秒）
- [ ] Tick时间稳定（接近fixed_delta_seconds）
- [ ] 内存使用正常

### 代码质量
- [ ] 所有日志使用logger（不使用print）
- [ ] 错误处理完善
- [ ] 代码符合PEP8
- [ ] 通过mypy检查

---

## 🔄 回滚方案

如果方案B出现问题，可以：
1. `git checkout dev` 回到原分支
2. 或使用 `git revert` 回滚特定commit

---

## 📝 后续优化

1. **添加deterministic seed**（可选）
   ```python
   tm.set_random_device_seed(12345)
   ```

2. **添加批量传感器spawn**
   目前sensor仍然单独spawn，可以优化

3. **性能profiling**
   测量初始化各阶段耗时

4. **添加重试机制**
   如果某些actors spawn失败，自动重试

---

## 📚 参考资料

1. **CARLA官方文档**:
   - `docs/adv_synchrony_timestep.md` - 同步模式和时间步长
   - `docs/adv_traffic_manager.md` - Traffic Manager使用

2. **CARLA示例代码**:
   - `PythonAPI/examples/generate_traffic.py` - 批量spawn示例
   - `PythonAPI/examples/synchronous_mode.py` - 同步模式示例

3. **Python API**:
   - `carla.command.SpawnActor`
   - `carla.command.SetAutopilot`
   - `carla.Client.apply_batch_sync`

---

**创建日期**: 2025-11-16
**预估工作量**: 8-12小时
**优先级**: P0 (最高)
