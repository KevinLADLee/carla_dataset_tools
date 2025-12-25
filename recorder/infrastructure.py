#!/usr/bin/python3
"""
Infrastructure (Road Side Unit) implementation for CARLA Dataset Tools

Infrastructure represents static roadside units (RSUs) that can be equipped with
sensors for V2X (Vehicle-to-Everything) communication scenarios. Infrastructure
actors are virtual actors (not spawned in CARLA world) but can have sensors
attached to them for broadcasting V2X messages.

Key Features:
- Static positioning at specified world coordinates
- V2X Custom sensor support for one-way message broadcasting
- JSON-formatted message generation for RSU information
- Integration with V2X communication protocols

Limitations:
- Infrastructure does NOT support V2X CAM sensors (requires vehicle dynamics)
- Only supports V2X Custom sensors for manual message sending
- One-way communication only (can send but not receive messages)
"""

import json
import logging
import carla
from recorder.actor import PseudoActor

# Get logger instance
logger = logging.getLogger(__name__)


class Infrastructure(PseudoActor):
    """
    Infrastructure (Road Side Unit) actor for V2X scenarios.

    Represents a static roadside unit that can broadcast V2X messages to nearby
    vehicles. Infrastructure actors are virtual (not spawned as CARLA actors)
    but can have sensors attached for communication.

    Attributes:
        carla_transform: Static transform position in CARLA world
        save_dir: Directory path for saving infrastructure data
        v2x_custom_sensors: List of registered V2X Custom sensors for broadcasting
        _control_step_count: Internal counter for control step tracking
    """

    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 transform: carla.Transform):
        """
        Initialize Infrastructure actor.

        Args:
            uid: Unique identifier for this infrastructure
            name: Human-readable name for this infrastructure
            base_save_dir: Base directory for saving infrastructure data
            transform: CARLA transform specifying infrastructure position
        """
        super().__init__(uid=uid, name=name, parent=None)
        self.carla_transform = transform
        self.save_dir = '{}/{}'.format(base_save_dir, self.name)

        # V2X Custom sensors for message broadcasting
        self.v2x_custom_sensors = []

        # Control step counter for debugging
        self._control_step_count = 0

    def get_carla_transform(self):
        """
        Get infrastructure transform in CARLA world.

        Returns:
            carla.Transform: Static transform position
        """
        return self.carla_transform

    def get_carla_bbox(self):
        """
        Get infrastructure bounding box.

        Returns:
            carla.BoundingBox: Bounding box at infrastructure location
        """
        return carla.BoundingBox(self.carla_transform.location,
                                 carla.Vector3D(1.0, 1.0, 1.0))

    def get_type_id(self):
        """
        Get infrastructure type identifier.

        Returns:
            str: Type ID string 'others.infrastructure'
        """
        return 'others.infrastructure'

    def get_save_dir(self):
        """
        Get directory path for saving infrastructure data.

        Returns:
            str: Save directory path
        """
        return self.save_dir

    def register_v2x_sensor(self, v2x_sensor):
        """
        Register a V2X Custom sensor for message broadcasting.

        Registered sensors will be used to send V2X messages during control_step().
        Only V2X Custom sensors can be registered (not V2X CAM sensors).

        Args:
            v2x_sensor: CustomV2XSensor object with carla_actor attribute
        """
        self.v2x_custom_sensors.append(v2x_sensor)
        logger.info(f"✓ Registered V2X sensor '{v2x_sensor.name}' to infrastructure '{self.name}'")

    def control_step(self):
        """
        Control step executed before world.tick().

        Generates and broadcasts V2X messages through all registered V2X Custom
        sensors. This method should be called once per simulation frame before
        world.tick() to ensure messages are sent at the correct timing.

        Message Format:
            JSON string containing:
            - type: "RSU" (Road Side Unit)
            - name: Infrastructure name
            - location: {x, y, z} coordinates in CARLA world
        """
        self._control_step_count += 1

        # Log every 10 calls to avoid log spam
        if self._control_step_count % 10 == 1:
            logger.info(f"Infrastructure '{self.name}' control_step called (count: {self._control_step_count})")

        if not self.v2x_custom_sensors:
            if self._control_step_count == 1:  # Only warn once
                logger.warning(f"⚠️  Infrastructure '{self.name}' has no V2X sensors registered!")
            return

        # Generate message
        message = self._generate_v2x_message()

        # Send through all registered V2X Custom sensors
        for sensor in self.v2x_custom_sensors:
            try:
                sensor.carla_actor.send(message)
                if self._control_step_count % 10 == 1:  # Log every 10 calls
                    logger.info(f"✓ Infrastructure '{self.name}' sent V2X message: {message[:60]}...")
            except Exception as e:
                logger.error(f"❌ Failed to send V2X message from '{self.name}': {e}")

    def _generate_v2x_message(self):
        """
        Generate V2X message in JSON format.

        Creates a standardized JSON message containing infrastructure (RSU)
        information that can be broadcast to nearby vehicles via V2X Custom sensors.

        Message Structure:
        {
            "type": "RSU",
            "name": <infrastructure_name>,
            "location": {
                "x": <x_coordinate>,
                "y": <y_coordinate>,
                "z": <z_coordinate>
            }
        }

        Returns:
            str: JSON formatted message string ready for V2X transmission
        """
        location = self.carla_transform.location
        message_data = {
            "type": "RSU",
            "name": self.name,
            "location": {
                "x": float(location.x),
                "y": float(location.y),
                "z": float(location.z)
            }
        }
        return json.dumps(message_data)

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save infrastructure status to disk.

        Currently, infrastructure actors do not save persistent data to disk.
        This method is kept for interface compatibility with the Actor base class.

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Simulation timestamp
            debug: If True, log debug information about the save operation
        """
        if debug:
            logger.debug(f"Infrastructure status recorded: uid={self.uid} name={self.name}")



