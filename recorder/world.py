#!/usr/bin/python3
import os
import pickle
import logging
import carla
from dataclasses import dataclass

from recorder.actor import PseudoActor
from core.types import *
from core.transform import carla_bbox_to_bbox, carla_transform_to_transform

# Get logger instance
logger = logging.getLogger(__name__)


class WorldActor(PseudoActor):
    def __init__(self, uid, carla_world: carla.World, base_save_dir: str):
        super().__init__(uid, self.get_type_id(), None)
        self.save_dir = "{}/{}_{}".format(base_save_dir, self.get_type_id(), uid)
        self.carla_world = carla_world

    def save_to_disk(self, frame_id, timestamp, debug=False):
        """
        Save world objects to disk

        Returns:
            dict: World objects information including counts by type
        """
        # TODO: Save all object bbox in world
        # Frame Timestamp CityObjectLabel carla_id location rotation box_location box_extent
        object_labels = []

        # Get environment objects for different vehicle types (CARLA 0.9.16 API)
        # In CARLA 0.9.16, CityObjectLabel.Vehicles was removed, use specific types instead
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Car)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Truck)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Bus)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Motorcycle)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Bicycle)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Pedestrians)

        carla_actors = self.carla_world.get_actors()
        for carla_actor in carla_actors:
            if carla_actor.type_id.startswith('vehicle') \
                    or carla_actor.type_id.startswith('walker'):
                transform = carla_transform_to_transform(carla_actor.get_transform())
                bbox = carla_bbox_to_bbox(carla_actor.bounding_box)
                if carla_actor.type_id.startswith('walker'):
                    label_type = 'pedestrian'
                else:
                    label_type = 'vehicle'
                object_labels.append(ObjectLabel(frame=frame_id,
                                                 timestamp=timestamp,
                                                 label_type=label_type,
                                                 carla_id=carla_actor.id,
                                                 transform=transform,
                                                 bounding_box=bbox))

        if len(object_labels) == 0:
            return {
                'type': 'world',
                'name': self.name,
                'objects_count': 0,
                'vehicle_count': 0,
                'pedestrian_count': 0,
                'static_count': 0,
                'file': None
            }

        # Count by type
        vehicle_count = sum(1 for obj in object_labels if obj.label_type == 'vehicle')
        pedestrian_count = sum(1 for obj in object_labels if obj.label_type == 'pedestrian')
        static_count = len(object_labels) - vehicle_count - pedestrian_count

        os.makedirs(self.save_dir, exist_ok=True)
        filename = '{:0>10d}.pkl'.format(frame_id)
        filepath = '{}/{}'.format(self.save_dir, filename)

        with open(filepath, 'wb') as pkl_file:
            pickle.dump(obj=object_labels, file=pkl_file)

        if debug:
            logger.debug(f"WorldObjectsLabel: Frame: {frame_id} Total counts: {len(object_labels)}")

        return {
            'type': 'world',
            'name': self.name,
            'objects_count': len(object_labels),
            'vehicle_count': vehicle_count,
            'pedestrian_count': pedestrian_count,
            'static_count': static_count,
            'file': filename
        }

    def get_type_id(self):
        return 'others.world'

    def get_save_dir(self):
        return self.save_dir

    def get_carla_transform(self) -> carla.Transform:
        return carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))

    def get_env_objects_labels(self, frame, timestamp, object_type: carla.CityObjectLabel) -> list:
        object_labels = []
        # Map CARLA 0.9.16 CityObjectLabel types to our label types
        if object_type in (carla.CityObjectLabel.Car, carla.CityObjectLabel.Truck,
                           carla.CityObjectLabel.Bus, carla.CityObjectLabel.Motorcycle,
                           carla.CityObjectLabel.Bicycle):
            label_type = 'vehicle'
        elif object_type == carla.CityObjectLabel.Pedestrians:
            label_type = 'pedestrian'
        else:
            label_type = 'any'
        env_objects = self.carla_world.get_environment_objects(object_type=object_type)
        for env_object in env_objects:
            transform = carla_transform_to_transform(env_object.transform)
            bbox_extent = Vector3d(env_object.bounding_box.extent.x,
                                   env_object.bounding_box.extent.y,
                                   env_object.bounding_box.extent.z)
            object_labels.append(ObjectLabel(frame=frame,
                                             timestamp=timestamp,
                                             label_type=label_type,
                                             carla_id=env_object.id,
                                             transform=transform,
                                             bounding_box=BoundingBox(Location(0, 0, 0), bbox_extent)))
        return object_labels
