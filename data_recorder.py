import argparse
import carla
import queue
import time

import cv2
import random
from recorder.vehicle import VehicleAgent


class DataRecorder:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.carla_client = carla.Client(self.host, self.port)
        self.carla_client.set_timeout(2.0)
        self.world = self._get_world()
        self.status = True
        # self.setting_world()
        self.actor_list = []
        self.vehicle_agent_list = []
        self.debug_helper = self.world.debug

    def _get_world(self) -> carla.World:
        return self.carla_client.get_world()

    def destroy(self):
        for v in self.vehicle_agent_list:
            v.destroy()

    def setting_world(self):
        settings = self.world.get_settings()
        # settings = carla.WorldSettings

        # Sync
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        # Make sure fixed_delta_seconds <= max_substep_delta_time * max_substeps

        self.world.apply_settings(settings)

    def spawn_actors(self):
        blueprint_lib = self.world.get_blueprint_library()

        vehicle_bp = blueprint_lib.find('vehicle.tesla.model3')
        transform = random.choice(self.world.get_map().get_spawn_points())
        vehicle_actor = self.world.spawn_actor(vehicle_bp, transform)
        vehicle_actor.set_autopilot(True)

        # self.actor_list.append(vehicle_actor)
        vehicle_agent = VehicleAgent(vehicle_actor)
        self.vehicle_agent_list.append(vehicle_agent)

        transform = carla.Transform(carla.Location(x=2.0, z=2.0))
        camera_bp = blueprint_lib.find('sensor.camera.rgb')
        camera = self.world.spawn_actor(camera_bp,
                                        transform,
                                        attach_to=vehicle_actor)
        vehicle_agent.add_sensor("cam_rgb", camera)

        transform = carla.Transform(carla.Location(x=2.0, z=2.0))
        camera_bp = blueprint_lib.find('sensor.camera.semantic_segmentation')
        camera = self.world.spawn_actor(camera_bp,
                                        transform,
                                        attach_to=vehicle_actor)
        vehicle_agent.add_sensor("cam_seg", camera)

        actor_list = self.world.get_actors()
        tl_actor_list = []
        for actor in actor_list:
            if actor.type_id == 'traffic.traffic_light':
                tl_actor_list.append(actor)
                # try:
                box_list = actor.get_light_boxes()
                for box in box_list:
                        self.debug_helper.draw_box(box, carla.Rotation(0,0,0))
                # except AttributeError:
                #     continue



    def save_data(self):
        for v in self.vehicle_agent_list:
            v.save_to_disk()


    def world_tick_thread(self):
        count = 0
        self.spawn_actors()
        self.carla_client.start_recorder("/home/carla/recording01.log")
        while self.status:
            self.world.tick()
            count = count + 1
            if 100 < count < 200:
                self.save_data()
            if count > 200:
                self.status = False
        self.destroy()

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--recorder_time',
        metavar='T',
        default=0,
        type=int,
        help='recorder duration (auto-stop)')
    args = argparser.parse_args()
    data_recorder = DataRecorder(args.host, args.port)
    data_recorder.world_tick_thread()
    return


if __name__ == "__main__":
    # execute only if run as a script
    main()