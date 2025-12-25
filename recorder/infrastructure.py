#!/usr/bin/python3
import logging
import carla
from recorder.actor import PseudoActor

# Get logger instance
logger = logging.getLogger(__name__)


class Infrastructure(PseudoActor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 transform: carla.Transform,
                 carla_actor: carla.Actor = None):
        super().__init__(uid=uid, name=name, parent=None)
        self.carla_transform = transform
        self.save_dir = '{}/{}'.format(base_save_dir, self.name)
        self.carla_actor = carla_actor
        
        # Disable physics on the attachment actor if it exists
        if self.carla_actor is not None:
            try:
                self.carla_actor.set_simulate_physics(False)
                self.carla_actor.set_enable_gravity(False)
                logger.debug(f"Configured attachment actor for infrastructure '{self.name}': "
                            f"type={self.carla_actor.type_id}, id={self.carla_actor.id}")
            except Exception as e:
                logger.warning(f"Failed to configure attachment actor for infrastructure '{self.name}': {e}")

    def get_carla_transform(self):
        return self.carla_transform

    def get_carla_bbox(self):
        return carla.BoundingBox(self.carla_transform.location,
                                 carla.Vector3D(1.0, 1.0, 1.0))

    def get_type_id(self):
        return 'others.infrastructure'

    def get_save_dir(self):
        return self.save_dir
    
    def get_carla_actor(self):
        """
        Return the CARLA actor used as attachment point for sensors.
        Returns None if actor spawning failed or was not attempted.
        
        Returns:
            carla.Actor or None
        """
        return self.carla_actor
    
    def destroy(self):
        """
        Destroy the CARLA actor if it exists.
        
        Returns:
            bool: True if successful or no actor to destroy, False on error
        """
        if self.carla_actor is not None:
            try:
                status = self.carla_actor.destroy()
                self.carla_actor = None
                return status
            except RuntimeError:
                logger.warning(f"Failed to destroy attachment actor for infrastructure '{self.name}'")
                return False
        return True

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save infrastructure status

        Args:
            frame_id: Absolute CARLA frame ID
            timestamp: Timestamp
            debug: Debug flag
        """
        if debug:
            logger.debug(f"Infrastructure status recorded: uid={self.uid} name={self.name}")



