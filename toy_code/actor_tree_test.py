from recorder.actor_tree import ActorTree
import carla
from param import *

def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(2.0)
    carla_world = carla_client.get_world()
    actor_tree = ActorTree(carla_world,
                           "{}/config/actor_settings_template.json".format(ROOT_PATH))
    try:
        actor_tree.create_actor_tree()
    finally:
        actor_tree.destroy()
    actor_tree.destroy()


if __name__ == "__main__":
    # execute only if run as a script
    main()